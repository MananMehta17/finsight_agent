# 📈 FinSight: Financial Analysis Agent

An agentic AI assistant that answers questions about listed companies. A **LangGraph** agent decides on its own which tools to call: live financial statements, a ratio calculator, **RAG** over annual reports (LangChain + FAISS), and a **LoRA fine tuned** DistilBERT model for news sentiment. Answers use only numbers returned by tools and cite report passages.

Built by **Manan Mehta**. Analysis aid only, not investment advice.

## Architecture

```
            ┌──────────────────────────────────────────┐
 question → │ agent (Groq LLM with tools bound)        │ → final answer
            └───────────┬──────────────────▲───────────┘
              tool calls│                  │tool results
            ┌───────────▼──────────────────┴───────────┐
            │ ToolNode                                  │
            │  1. get_company_financials  (yfinance)    │
            │  2. calculate_ratios        (pandas)      │
            │  3. search_annual_reports   (FAISS RAG)   │
            │  4. analyze_news_sentiment  (LoRA model)  │
            └───────────────────────────────────────────┘
```

The loop repeats until the LLM stops asking for tools. `InMemorySaver` keeps chat history per conversation, so follow up questions work.

## Project structure

| File | What it does |
| --- | --- |
| `agent.py` | LangGraph `StateGraph`, the 4 tools, system prompt, CLI chat |
| `app.py` | Streamlit chat UI showing which tools the agent used |
| `tools/financial_data.py` | yfinance statements and ratio maths (no AI in the maths) |
| `tools/rag.py` | Load, chunk, embed (MiniLM, 384 dim), FAISS index and search |
| `tools/sentiment.py` | Fine tuned model inference, headline fetching |
| `finetune/train_sentiment_lora.py` | LoRA fine tuning script for Google Colab |
| `ingest.py` | Builds the FAISS index from `data/reports` |
| `test_offline.py` | 10 offline checks: ratios, RAG, graph routing |

## How to run

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and add a free Groq API key. Check the Groq console for a current model that supports tool calling and set `GROQ_MODEL`.
3. Put one or more annual reports (PDF) in `data/reports/`, then run `python ingest.py`
4. Fine tune the sentiment model: open `finetune/train_sentiment_lora.py` in Google Colab (T4 GPU), run it, download `finsent_model.zip`, unzip into `models/finsent_model/`. Until then the agent uses an LLM fallback and says so.
5. `streamlit run app.py` (or `python agent.py` for terminal chat)
6. Optional: `python test_offline.py`

## Results

Fine tuned DistilBERT + LoRA: **84% validation accuracy, 0.78 macro F1** (3 epochs, T4 GPU, about 2 minutes).

## Design choices

1. **Maths outside the LLM.** Ratios are computed in pandas; the LLM only explains them. This prevents invented numbers.
2. **Explicit StateGraph** instead of a prebuilt agent, so the control flow is visible and easy to extend.
3. **LoRA instead of full fine tuning.** Trains about 1 to 2% of the weights, fits a free GPU, keeps the base model intact.
4. **Honest fallbacks.** Every tool returns a clear error instead of crashing, and the sentiment output states which method was used.
5. **Recursion limit** on the graph so the agent can never loop forever.

## Limitations

1. yfinance is an unofficial Yahoo wrapper and can break or rate limit.
2. The sentiment model is trained on short finance tweets, so long articles may be classified less reliably.
3. Chat memory is in RAM only and resets when the app restarts.
