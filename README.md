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
- Track freelance hours, rates, earnings, and monthly target progress in an earnings dashboard.

### Run locally

```bash
pip install -r requirements.txt
python3 -m streamlit run app.py
```

The app works without an API key by using deterministic local analysis. To add
LLM-backed responses, provide an OpenAI API key through Streamlit secrets or the
environment:

```bash
export OPENAI_API_KEY="sk-..."
python3 -m streamlit run app.py
```

You can optionally set `OPENAI_MODEL`; otherwise the app uses `gpt-4o-mini`.

### Earnings dashboard

Use the **Earnings dashboard** tab to enter a work date, project/client, hours
worked, and hourly rate. The app stores those entries as remembered billable
income and compares monthly progress against a configurable target pay and base
hourly rate. It shows total earnings, total hours, average billable rate,
expected earnings by today, hours ahead/behind, and previous-month hour gaps.
