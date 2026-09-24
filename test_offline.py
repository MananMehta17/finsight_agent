"""Offline tests: no internet, no API key, no GPU needed.
Checks the ratio maths, the RAG pipeline and the LangGraph routing loop.
Run: python test_offline.py
"""
import tempfile
from pathlib import Path

import pandas as pd
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import agent
from tools import financial_data, rag

passed = 0
def check(cond, name):
    global passed
    assert cond, name
    passed += 1
    print("PASS", name)

# 1. ratio maths on made up numbers (not real company data)
fin = pd.DataFrame({"revenue": [100.0, 120.0], "net_income": [10.0, 18.0],
                    "total_assets": [200.0, 250.0], "total_liabilities": [100.0, 150.0],
                    "operating_cash_flow": [12.0, 18.0]}, index=[2023, 2024])
r = financial_data.compute_ratios(fin)
check(r.loc[2024, "revenue_growth_pct"] == 20.0, "revenue growth")
check(r.loc[2024, "net_margin_pct"] == 15.0, "net margin")
check(r.loc[2024, "liabilities_to_assets"] == 0.6, "liabilities to assets")
check(r.loc[2024, "cash_conversion"] == 1.0, "cash conversion")
check(pd.isna(r.loc[2023, "revenue_growth_pct"]), "first year growth is empty")

# 2. RAG: build and search an index with fake embeddings
with tempfile.TemporaryDirectory() as d:
    d = Path(d); (d / "docs").mkdir(); (d / "idx").mkdir()
    (d / "docs" / "report.txt").write_text(
        "Risk factors. Supply chain concentration in a few regions is a key risk. " * 30)
    emb = DeterministicFakeEmbedding(size=384)
    n_docs, n_chunks = rag.build_index(d / "docs", d / "idx", embeddings=emb)
    check(n_docs == 1 and n_chunks > 1, f"indexing ({n_chunks} chunks)")
    rag._store = None
    rag.load_index(d / "idx", embeddings=emb)
    out = rag.search("supply chain risk")
    check("[1]" in out and "report.txt" in out, "search returns cited passages")
    rag._store = None

# 3. LangGraph loop with a fake LLM: tool call first, then final answer
class FakeToolLLM(GenericFakeChatModel):
    def bind_tools(self, tools, **kw):
        return self

financial_data.fetch_financials = lambda t: fin   # no internet in tests
script = iter([
    AIMessage(content="", tool_calls=[{"name": "calculate_ratios", "args": {"ticker": "DEMO"},
                                       "id": "call_1", "type": "tool_call"}]),
    AIMessage(content="Revenue grew 20% and net margin reached 15%."),
])
app = agent.build_agent(llm=FakeToolLLM(messages=script))
out = agent.ask(app, "How did DEMO do?", thread_id="t1")
kinds = [type(m).__name__ for m in out["messages"]]
check(kinds == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"], f"graph path {kinds}")
check('"revenue_growth_pct": 20.0' in out["messages"][2].content, "tool result reached the LLM")
check(out["messages"][-1].content.startswith("Revenue grew"), "final answer returned")

print(f"\nAll {passed} checks passed")
