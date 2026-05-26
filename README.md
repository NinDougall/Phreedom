# Phreedom
AI Financial Planner, Accountant, bookkeeping, and tax planner.

## Streamlit personal financial agent

This repository includes a Streamlit app that works as an in-session personal
financial agent. It can:

- Chat about the user's financial profile.
- Upload and parse CSV bank exports or PDF statements, invoices, and receipts.
- Remember imported transactions and document summaries during the session.
- Summarize business income, expenses, net profit, and major expense categories.
- Recommend how much to set aside for taxes based on a configurable reserve rate.

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app works without an API key by using deterministic local analysis. To add
LLM-backed responses, provide an OpenAI API key through Streamlit secrets or the
environment:

```bash
export OPENAI_API_KEY="sk-..."
streamlit run app.py
```

You can optionally set `OPENAI_MODEL`; otherwise the app uses `gpt-4o-mini`.
