"""Streamlit chat UI. Run with: streamlit run app.py"""
import uuid
import streamlit as st

import config
from agent import build_agent, ask
from tools import rag, sentiment

st.set_page_config(page_title="FinSight Agent", page_icon="📈", layout="wide")
st.title("📈 FinSight: Financial Analysis Agent")
st.caption("LangGraph agent with live financials, ratio tools, RAG over annual reports, "
           "and a LoRA fine tuned sentiment model. Analysis aid only, not investment advice.")

with st.sidebar:
    st.header("Status")
    ok = lambda text: f":green[{text}]"
    bad = lambda text: f":red[{text}]"
    st.markdown("**Groq key:** " + (ok("Connected") if config.GROQ_API_KEY
                                    else bad("Missing, add it to .env")))
    st.markdown("**Sentiment model:** " + (ok("Fine tuned model loaded")
                                           if sentiment.fine_tuned_available()
                                           else bad("Using LLM fallback")))
    st.markdown("**Report index:** " + (ok("Built") if rag.index_exists()
                                        else bad("Not built")))
    reports = [p.name for p in config.REPORTS_DIR.glob("*") if p.suffix in {".pdf", ".txt"}]
    st.write(f"Reports in data/reports: {len(reports)}")
    if st.button("Build / rebuild report index", disabled=not reports):
        with st.spinner("Chunking and embedding reports..."):
            n_docs, n_chunks = rag.build_index()
        st.success(f"{n_docs} pages into {n_chunks} chunks")
    if st.button("New conversation"):
        st.session_state.clear()
        st.rerun()
    st.markdown("**Try asking**")
    st.markdown("- How did AAPL's revenue and margins change over the last few years?\n"
                "- Compare MSFT and TSLA cash conversion\n"
                "- What is the news sentiment on TSLA right now?\n"
                "- What risks does the annual report mention?")


@st.cache_resource(show_spinner="Starting agent...")
def get_app():
    return build_agent()


try:
    app = get_app()
except Exception as e:
    st.error(str(e))
    st.stop()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat = []      # list of (role, text, tool_steps)
    st.session_state.seen = 0

for role, text, steps in st.session_state.chat:
    with st.chat_message(role):
        if steps:
            with st.expander(f"🛠️ Agent used {len(steps)} tool call(s)"):
                for s in steps:
                    st.code(s, language="text")
        st.markdown(text)

if q := st.chat_input("Ask about a company, e.g. 'How profitable is AAPL?'"):
    st.session_state.chat.append(("user", q, []))
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking and calling tools..."):
            try:
                out = ask(app, q, st.session_state.thread_id)
                new = out["messages"][st.session_state.seen:]
                st.session_state.seen = len(out["messages"])
                steps = [f"{tc['name']}({tc['args']})"
                         for m in new for tc in (getattr(m, "tool_calls", None) or [])]
                answer = out["messages"][-1].content
            except Exception as e:
                steps, answer = [], f"⚠️ Error: {e}"
        if steps:
            with st.expander(f"🛠️ Agent used {len(steps)} tool call(s)"):
                for s in steps:
                    st.code(s, language="text")
        st.markdown(answer)
    st.session_state.chat.append(("assistant", answer, steps))
