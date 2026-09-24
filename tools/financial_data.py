"""Tool 1 and 2: fetch company financials and calculate ratios.

Data comes from yfinance (an unofficial Yahoo Finance wrapper). Yahoo can change
its format or block requests, so every step fails with a clear message instead
of crashing the agent.
"""
import json
import pandas as pd
import yfinance as yf

# yfinance row labels can vary slightly, so each metric has a few candidates.
METRIC_ROWS = {
    "revenue": ("income_stmt", ["Total Revenue", "Operating Revenue"]),
    "net_income": ("income_stmt", ["Net Income", "Net Income Common Stockholders"]),
    "total_assets": ("balance_sheet", ["Total Assets"]),
    "total_liabilities": ("balance_sheet", ["Total Liabilities Net Minority Interest", "Total Liabilities"]),
    "operating_cash_flow": ("cashflow", ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]),
}


def _pick_row(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    if df is None or df.empty:
        return None
    for name in candidates:
        if name in df.index:
            return df.loc[name]
    return None


def fetch_financials(ticker: str) -> pd.DataFrame:
    """Return a table: one row per fiscal year, one column per metric (in USD)."""
    t = yf.Ticker(ticker.upper().strip())
    statements = {"income_stmt": t.income_stmt, "balance_sheet": t.balance_sheet, "cashflow": t.cashflow}

    columns = {}
    for metric, (stmt, candidates) in METRIC_ROWS.items():
        row = _pick_row(statements[stmt], candidates)
        if row is not None:
            columns[metric] = row

    if not columns:
        raise ValueError(f"No financial data found for '{ticker}'. Check the ticker symbol.")

    table = pd.DataFrame(columns)
    table.index = [pd.Timestamp(i).year for i in table.index]   # column dates -> fiscal year
    table.index.name = "fiscal_year"
    return table.sort_index().dropna(how="all")


def compute_ratios(fin: pd.DataFrame) -> pd.DataFrame:
    """Plain arithmetic, no AI. Keeping maths out of the LLM avoids made up numbers."""
    r = pd.DataFrame(index=fin.index)
    if {"net_income", "revenue"} <= set(fin.columns):
        r["net_margin_pct"] = fin["net_income"] / fin["revenue"] * 100
    if "revenue" in fin.columns:
        r["revenue_growth_pct"] = fin["revenue"].pct_change() * 100
    if "net_income" in fin.columns:
        r["net_income_growth_pct"] = fin["net_income"].pct_change() * 100
    if {"total_liabilities", "total_assets"} <= set(fin.columns):
        r["liabilities_to_assets"] = fin["total_liabilities"] / fin["total_assets"]
    if {"operating_cash_flow", "net_income"} <= set(fin.columns):
        # above 1 means profits are backed by real cash
        r["cash_conversion"] = fin["operating_cash_flow"] / fin["net_income"]
    return r.round(2)


def to_json(df: pd.DataFrame, in_billions: bool = False) -> str:
    out = df / 1e9 if in_billions else df
    out = out.round(2).where(out.notna(), None)
    return json.dumps({str(k): v for k, v in out.to_dict(orient="index").items()}, indent=2)
