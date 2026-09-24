"""Tool 4: financial news sentiment.

Primary: the DistilBERT model YOU fine tuned with LoRA (finetune/ folder, run in Colab).
Fallback: if that model is not in models/finsent_model yet, ask the LLM instead and
say so clearly in the output, so nobody thinks the fine tuned model was used.
"""
import yfinance as yf
import config

_clf = None   # (tokenizer, model), loaded once


def fine_tuned_available() -> bool:
    return (config.SENTIMENT_MODEL_DIR / "config.json").exists()


def _load_classifier():
    """Load tokenizer and model directly (no transformers pipeline, which pulls in
    unrelated image libraries on some versions)."""
    global _clf
    if _clf is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        path = str(config.SENTIMENT_MODEL_DIR)
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.eval()   # inference mode: no dropout
        _clf = (tok, model)
    return _clf


def fetch_headlines(ticker: str, limit: int = 10) -> list[str]:
    """yfinance has changed its news format before, so handle both known shapes."""
    items = yf.Ticker(ticker.upper().strip()).news or []
    titles = []
    for item in items:
        title = item.get("title") or (item.get("content") or {}).get("title")
        if title:
            titles.append(title.strip())
    return titles[:limit]


def classify_finetuned(headlines: list[str]) -> list[dict]:
    import torch
    tok, model = _load_classifier()
    enc = tok(headlines, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=1)
    conf, idx = probs.max(dim=1)
    return [{"headline": h,
             "label": model.config.id2label[int(i)].lower(),
             "confidence": round(float(c), 3)}
            for h, i, c in zip(headlines, idx, conf)]


def summarise(results: list[dict], method: str) -> dict:
    counts = {}
    for r in results:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    return {"method": method, "counts": counts, "headlines": results}