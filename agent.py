"""The LangGraph agent.

Graph:   START -> agent --(tool calls?)--> tools -> agent -> ... -> END
                       `--(no tool calls)--> END

The "agent" node is the LLM. It looks at the conversation and either
(a) asks for one or more tools, or (b) writes the final answer.
The "tools" node runs whatever tools were requested and feeds results back.
This loop is what makes it agentic: the LLM decides the steps, not fixed code.
"""
import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

import config
from tools import financial_data, rag, sentiment

SYSTEM_PROMPT = """You are FinSight, a financial analysis assistant.

Rules:
1. Never state a financial number from memory. Only use numbers returned by tools.
2. For company numbers call get_company_financials; for growth or margins call calculate_ratios.
3. For questions about strategy, risks or management commentary, call search_annual_reports
   and cite the passage numbers like [1], [2].
4. For market mood or news, call analyze_news_sentiment and mention which method was used.
5. If a tool fails, say what failed. Do not guess the missing data.
6. Keep answers short and structured. End with one line on what the data does NOT show.
This is an analysis aid, not investment advice."""

_llm = None   # set in build_agent, also used by the sentiment fallback


@tool
def get_company_financials(ticker: str) -> str:
    """Get yearly revenue, net income, total assets, total liabilities and operating
    cash flow (USD billions) for a stock ticker such as AAPL, MSFT or TSLA."""
    try:
        return financial_data.to_json(financial_data.fetch_financials(ticker), in_billions=True)
    except Exception as e:
        return f"ERROR fetching financials for {ticker}: {e}"


@tool
def calculate_ratios(ticker: str) -> str:
    """Calculate net margin %, revenue growth %, net income growth %, liabilities to
    assets, and cash conversion (operating cash flow / net income) for each year."""
    try:
        fin = financial_data.fetch_financials(ticker)
        return financial_data.to_json(financial_data.compute_ratios(fin))
    except Exception as e:
        return f"ERROR calculating ratios for {ticker}: {e}"


@tool
def search_annual_reports(query: str) -> str:
    """Search the annual reports stored locally and return the most relevant passages
    with their source file and page number."""
    try:
        return rag.search(query)
    except Exception as e:
        return f"ERROR searching reports: {e}"


@tool
def analyze_news_sentiment(ticker: str) -> str:
    """Fetch recent news headlines for a ticker and classify each as positive,
    negative or neutral."""
    try:
        headlines = sentiment.fetch_headlines(ticker)
    except Exception as e:
        return f"ERROR fetching news for {ticker}: {e}"
    if not headlines:
        return f"No recent headlines found for {ticker}."

    if sentiment.fine_tuned_available():
        try:
            res = sentiment.summarise(sentiment.classify_finetuned(headlines),
                                      "fine tuned DistilBERT + LoRA model")
            return json.dumps(res, indent=2)
        except Exception as e:
            note = f"fine tuned model failed ({e}); "
    else:
        note = "fine tuned model not found; "

    # fallback: zero shot classification by the LLM
    prompt = ("Classify each financial headline as positive, negative or neutral. "
              "Reply ONLY with a JSON list of labels in the same order.\n" +
              "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines)))
    try:
        raw = _llm.invoke([HumanMessage(prompt)]).content
        labels = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
        results = [{"headline": h, "label": str(l).lower()} for h, l in zip(headlines, labels)]
    except Exception as e:
        return f"ERROR: {note}LLM fallback also failed: {e}"
    return json.dumps(sentiment.summarise(results, note + "used LLM zero shot fallback"), indent=2)


TOOLS = [get_company_financials, calculate_ratios, search_annual_reports, analyze_news_sentiment]


def make_llm():
    from langchain_groq import ChatGroq
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY missing. Copy .env.example to .env and add your key.")
    return ChatGroq(model_name=config.GROQ_MODEL, temperature=0, groq_api_key=config.GROQ_API_KEY)


def build_agent(llm=None):
    """Build and compile the graph. Pass a fake llm in tests."""
    global _llm
    _llm = llm or make_llm()
    llm_with_tools = _llm.bind_tools(TOOLS)

    def agent_node(state: MessagesState):
        msgs = [SystemMessage(SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(msgs)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)   # tools or END
    graph.add_edge("tools", "agent")
    # InMemorySaver keeps chat history per thread_id, so follow up questions work
    return graph.compile(checkpointer=InMemorySaver())


def ask(app, question: str, thread_id: str = "cli"):
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": config.MAX_AGENT_STEPS * 2}
    return app.invoke({"messages": [HumanMessage(question)]}, cfg)


if __name__ == "__main__":
    app = build_agent()
    print("FinSight agent. Type 'exit' to quit.")
    seen = 0
    while (q := input("\nYou: ").strip()).lower() not in {"exit", "quit"}:
        out = ask(app, q)
        for m in out["messages"][seen:]:
            for tc in getattr(m, "tool_calls", []) or []:
                print(f"  [tool] {tc['name']}({tc['args']})")
        seen = len(out["messages"])
        print("\nFinSight:", out["messages"][-1].content)
