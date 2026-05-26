"""Streamlit personal financial agent.

The app keeps an in-session financial memory built from uploaded CSV and PDF
documents, then uses that profile to answer chat questions and recommend tax
savings. It can use an OpenAI-compatible model when OPENAI_API_KEY is present
and falls back to deterministic local analysis otherwise.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st
from pypdf import PdfReader


LEDGER_COLUMNS = ["date", "description", "amount", "kind", "category", "source"]
CURRENCY_RE = re.compile(r"(?<!\w)[-$]?\$?\s?[\d,]+(?:\.\d{2})?(?!\w)")
DATE_RE = re.compile(
    r"(?P<date>\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b)"
)


@dataclass
class ParsedDocument:
    """Structured output from an uploaded file."""

    source: str
    transactions: pd.DataFrame
    summary: str


def init_state() -> None:
    """Initialize Streamlit session memory."""

    if "ledger" not in st.session_state:
        st.session_state.ledger = pd.DataFrame(columns=LEDGER_COLUMNS)
    if "document_summaries" not in st.session_state:
        st.session_state.document_summaries = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi, I am your financial agent. Upload bank exports, "
                    "expense reports, invoices, or tax PDFs, then ask me about "
                    "income, expenses, cash flow, and tax savings."
                ),
            }
        ]


def infer_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """Find a likely column name by comparing normalized tokens."""

    normalized = {column.lower().strip().replace("_", " "): column for column in columns}
    for candidate in candidates:
        candidate = candidate.lower()
        for normalized_name, original_name in normalized.items():
            if candidate == normalized_name or candidate in normalized_name:
                return original_name
    return None


def coerce_money(value: Any) -> float:
    """Convert common accounting/currency formats to a signed float."""

    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return 0.0

    is_negative = text.startswith("(") and text.endswith(")")
    text = text.replace("(", "").replace(")", "")
    text = re.sub(r"[^0-9.\-]", "", text)
    if text in {"", "-", "."}:
        return 0.0

    amount = float(text)
    return -abs(amount) if is_negative else amount


def classify_kind(amount: float, explicit_value: Any = None) -> str:
    """Classify a transaction as income or expense."""

    if explicit_value is not None and not pd.isna(explicit_value):
        text = str(explicit_value).lower()
        if any(token in text for token in ("income", "revenue", "deposit", "credit")):
            return "income"
        if any(token in text for token in ("expense", "debit", "withdrawal", "payment")):
            return "expense"
    return "income" if amount >= 0 else "expense"


def normalize_csv(file_name: str, data: bytes) -> ParsedDocument:
    """Read a CSV and normalize it into the app ledger schema."""

    raw = pd.read_csv(io.BytesIO(data))
    if raw.empty:
        return ParsedDocument(file_name, pd.DataFrame(columns=LEDGER_COLUMNS), "CSV was empty.")

    columns = list(raw.columns)
    date_col = infer_column(columns, ("date", "posted", "transaction date", "created"))
    description_col = infer_column(
        columns,
        ("description", "memo", "vendor", "merchant", "payee", "name", "details"),
    )
    category_col = infer_column(columns, ("category", "account", "type", "class"))
    kind_col = infer_column(columns, ("kind", "transaction type", "type"))
    amount_col = infer_column(columns, ("amount", "total", "net"))
    debit_col = infer_column(columns, ("debit", "withdrawal", "spent", "charge"))
    credit_col = infer_column(columns, ("credit", "deposit", "received", "income"))

    if amount_col:
        amounts = raw[amount_col].map(coerce_money)
    elif debit_col or credit_col:
        debits = raw[debit_col].map(coerce_money) if debit_col else 0
        credits = raw[credit_col].map(coerce_money) if credit_col else 0
        amounts = pd.Series(credits, index=raw.index).fillna(0) - pd.Series(debits, index=raw.index).fillna(0)
    else:
        numeric_columns = raw.select_dtypes(include="number").columns.tolist()
        if not numeric_columns:
            raise ValueError("No amount, debit/credit, or numeric column could be found.")
        amount_col = numeric_columns[0]
        amounts = raw[amount_col].map(coerce_money)

    dates = pd.to_datetime(raw[date_col], errors="coerce").dt.date if date_col else pd.NaT
    descriptions = raw[description_col].fillna("Imported transaction") if description_col else "Imported transaction"
    categories = raw[category_col].fillna("Uncategorized") if category_col else "Uncategorized"

    ledger = pd.DataFrame(
        {
            "date": dates,
            "description": descriptions,
            "amount": amounts,
            "kind": [
                classify_kind(amount, raw.loc[index, kind_col] if kind_col else None)
                for index, amount in amounts.items()
            ],
            "category": categories,
            "source": file_name,
        }
    )
    ledger["amount"] = ledger.apply(
        lambda row: abs(row["amount"]) if row["kind"] == "income" else -abs(row["amount"]),
        axis=1,
    )

    income = ledger.loc[ledger["kind"] == "income", "amount"].sum()
    expenses = abs(ledger.loc[ledger["kind"] == "expense", "amount"].sum())
    summary = (
        f"Imported {len(ledger):,} CSV transactions from {file_name}: "
        f"${income:,.2f} income and ${expenses:,.2f} expenses."
    )
    return ParsedDocument(file_name, ledger[LEDGER_COLUMNS], summary)


def extract_pdf_text(data: bytes) -> str:
    """Extract text from a PDF upload."""

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def parse_pdf_transactions(file_name: str, text: str) -> pd.DataFrame:
    """Best-effort extraction of dated money lines from statements or invoices."""

    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        date_match = DATE_RE.search(line)
        amounts = CURRENCY_RE.findall(line)
        if not date_match or not amounts:
            continue

        raw_amount = amounts[-1]
        amount = coerce_money(raw_amount)
        lower_line = line.lower()
        if any(token in lower_line for token in ("invoice", "payment received", "deposit", "revenue")):
            kind = "income"
            amount = abs(amount)
        elif any(token in lower_line for token in ("fee", "charge", "purchase", "withdrawal", "debit", "expense")):
            kind = "expense"
            amount = -abs(amount)
        else:
            kind = classify_kind(amount)
            amount = abs(amount) if kind == "income" else -abs(amount)

        description = re.sub(r"\s+", " ", line).strip()
        rows.append(
            {
                "date": pd.to_datetime(date_match.group("date"), errors="coerce").date(),
                "description": description[:240],
                "amount": amount,
                "kind": kind,
                "category": "PDF extracted",
                "source": file_name,
            }
        )

    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def summarize_pdf(file_name: str, text: str, ledger: pd.DataFrame) -> str:
    """Create an in-memory note for an uploaded PDF."""

    amounts = [coerce_money(match) for match in CURRENCY_RE.findall(text)]
    keyword_hits = sorted(
        {
            keyword
            for keyword in (
                "1099",
                "w-2",
                "invoice",
                "receipt",
                "deduction",
                "sales tax",
                "estimated tax",
                "payroll",
                "mileage",
            )
            if keyword in text.lower()
        }
    )
    largest_amounts = sorted((abs(amount) for amount in amounts), reverse=True)[:5]
    amount_summary = ", ".join(f"${amount:,.2f}" for amount in largest_amounts) or "no dollar amounts"
    keyword_summary = ", ".join(keyword_hits) if keyword_hits else "no tax keywords detected"

    return (
        f"Read PDF {file_name}: extracted {len(text):,} characters, "
        f"{len(ledger):,} dated transaction-like rows, largest amounts {amount_summary}, "
        f"and {keyword_summary}."
    )


def parse_upload(uploaded_file: Any) -> ParsedDocument:
    """Parse a Streamlit uploaded file into financial memory."""

    data = uploaded_file.getvalue()
    file_name = uploaded_file.name
    extension = file_name.rsplit(".", 1)[-1].lower()

    if extension == "csv":
        return normalize_csv(file_name, data)
    if extension == "pdf":
        text = extract_pdf_text(data)
        ledger = parse_pdf_transactions(file_name, text)
        return ParsedDocument(file_name, ledger, summarize_pdf(file_name, text, ledger))
    raise ValueError(f"Unsupported file type: {extension}")


def append_transactions(transactions: pd.DataFrame) -> None:
    """Merge new transaction rows into session memory."""

    if transactions.empty:
        return
    st.session_state.ledger = pd.concat(
        [st.session_state.ledger, transactions[LEDGER_COLUMNS]], ignore_index=True
    )


def financial_summary(ledger: pd.DataFrame, tax_rate: float) -> dict[str, Any]:
    """Calculate the user's current financial profile."""

    if ledger.empty:
        return {
            "income": 0.0,
            "expenses": 0.0,
            "profit": 0.0,
            "tax_reserve": 0.0,
            "transactions": 0,
            "top_expenses": pd.DataFrame(columns=["category", "amount"]),
        }

    income = float(ledger.loc[ledger["kind"] == "income", "amount"].sum())
    expenses = float(abs(ledger.loc[ledger["kind"] == "expense", "amount"].sum()))
    profit = income - expenses
    tax_reserve = max(profit, 0) * tax_rate
    top_expenses = (
        ledger.loc[ledger["kind"] == "expense"]
        .assign(amount=lambda frame: frame["amount"].abs())
        .groupby("category", dropna=False)["amount"]
        .sum()
        .sort_values(ascending=False)
        .head(8)
        .reset_index()
    )

    return {
        "income": income,
        "expenses": expenses,
        "profit": profit,
        "tax_reserve": tax_reserve,
        "transactions": len(ledger),
        "top_expenses": top_expenses,
    }


def render_metrics(summary: dict[str, Any]) -> None:
    """Render headline financial metrics."""

    income_col, expense_col, profit_col, tax_col = st.columns(4)
    income_col.metric("Income", f"${summary['income']:,.2f}")
    expense_col.metric("Expenses", f"${summary['expenses']:,.2f}")
    profit_col.metric("Net profit", f"${summary['profit']:,.2f}")
    tax_col.metric("Suggested tax reserve", f"${summary['tax_reserve']:,.2f}")


def build_context(tax_rate: float, business_type: str, tax_notes: str) -> str:
    """Create compact context for LLM or fallback responses."""

    ledger = st.session_state.ledger
    summary = financial_summary(ledger, tax_rate)
    top_expenses = summary["top_expenses"]
    top_expense_text = (
        top_expenses.to_string(index=False, formatters={"amount": "${:,.2f}".format})
        if not top_expenses.empty
        else "No expenses loaded."
    )
    docs = "\n".join(f"- {item}" for item in st.session_state.document_summaries) or "No documents uploaded."

    recent_transactions = (
        ledger.tail(12).to_string(index=False)
        if not ledger.empty
        else "No transaction rows available."
    )

    return f"""
Business type: {business_type or "Not specified"}
Tax notes: {tax_notes or "None"}
Tax reserve rate: {tax_rate:.0%}
Transactions: {summary["transactions"]}
Income: ${summary["income"]:,.2f}
Expenses: ${summary["expenses"]:,.2f}
Net profit: ${summary["profit"]:,.2f}
Suggested tax reserve: ${summary["tax_reserve"]:,.2f}

Top expenses by category:
{top_expense_text}

Document memory:
{docs}

Recent transactions:
{recent_transactions}
""".strip()


def fallback_response(prompt: str, context: str, tax_rate: float) -> str:
    """Rule-based advisor used when no model API key is configured."""

    summary = financial_summary(st.session_state.ledger, tax_rate)
    top_expenses = summary["top_expenses"]
    question = prompt.lower()

    tax_guidance = (
        f"Based on the loaded profile, net profit is ${summary['profit']:,.2f}. "
        f"At a {tax_rate:.0%} reserve rate, set aside ${summary['tax_reserve']:,.2f} "
        "in a separate tax savings account. Revisit the rate with a tax professional "
        "if your filing status, entity type, state taxes, or payroll situation changes."
    )

    if "tax" in question or "set aside" in question or "save" in question:
        return tax_guidance
    if "expense" in question or "spend" in question or "category" in question:
        if top_expenses.empty:
            return "I do not have expense data yet. Upload a CSV or PDF statement so I can analyze categories."
        lines = [
            f"- {row.category}: ${row.amount:,.2f}"
            for row in top_expenses.itertuples(index=False)
        ]
        return "Your largest expense categories are:\n" + "\n".join(lines) + "\n\n" + tax_guidance
    if "income" in question or "revenue" in question:
        return (
            f"I found ${summary['income']:,.2f} in income across "
            f"{summary['transactions']:,} loaded transactions. {tax_guidance}"
        )
    if "summary" in question or "profit" in question or "cash" in question:
        return (
            f"Financial summary: income ${summary['income']:,.2f}, expenses "
            f"${summary['expenses']:,.2f}, net profit ${summary['profit']:,.2f}. "
            f"{tax_guidance}"
        )

    return (
        "Here is what I know from your current financial memory:\n\n"
        f"{context}\n\n"
        "Ask about taxes, expenses, income, profit, or uploaded documents for a more focused answer."
    )


def get_openai_key() -> str | None:
    """Read an OpenAI key from Streamlit secrets or environment variables."""

    try:
        return st.secrets.get("OPENAI_API_KEY")  # type: ignore[union-attr]
    except Exception:
        return os.getenv("OPENAI_API_KEY")


def generate_agent_response(prompt: str, tax_rate: float, business_type: str, tax_notes: str) -> str:
    """Answer a chat prompt using the financial memory."""

    context = build_context(tax_rate, business_type, tax_notes)
    api_key = get_openai_key()
    if not api_key:
        return fallback_response(prompt, context, tax_rate)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful personal financial agent for a small business owner. "
                        "Use only the provided financial memory. Be specific with calculations, "
                        "recommend practical next steps, and include a brief disclaimer that this "
                        "is planning guidance rather than formal tax advice."
                    ),
                },
                {"role": "system", "content": f"Financial memory:\n{context}"},
                *st.session_state.messages[-10:],
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or "I could not generate a response."
    except Exception as exc:
        return (
            fallback_response(prompt, context, tax_rate)
            + f"\n\nModel call failed, so I used local analysis instead. Error: {exc}"
        )


def render_uploads() -> None:
    """Render uploader and ingest new files into memory."""

    uploaded_files = st.file_uploader(
        "Upload financial documents",
        type=["csv", "pdf"],
        accept_multiple_files=True,
        help="CSV bank exports are normalized into transactions. PDFs are text-extracted and scanned for dated dollar rows.",
    )

    if not uploaded_files:
        return

    for uploaded_file in uploaded_files:
        file_key = f"{uploaded_file.name}:{uploaded_file.size}"
        if file_key in st.session_state.processed_files:
            continue
        try:
            parsed = parse_upload(uploaded_file)
            append_transactions(parsed.transactions)
            st.session_state.document_summaries.append(parsed.summary)
            st.session_state.processed_files.add(file_key)
            st.success(parsed.summary)
        except Exception as exc:
            st.error(f"Could not process {uploaded_file.name}: {exc}")


def render_chat(tax_rate: float, business_type: str, tax_notes: str) -> None:
    """Render chat history and process new prompts."""

    st.subheader("Chat with your financial agent")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about expenses, income, taxes, cash flow, or uploaded files")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Reviewing your financial memory..."):
            answer = generate_agent_response(prompt, tax_rate, business_type, tax_notes)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


def main() -> None:
    """Run the Streamlit application."""

    st.set_page_config(page_title="Personal Financial Agent", page_icon=":moneybag:", layout="wide")
    init_state()

    st.title("Personal Financial Agent")
    st.caption(
        "Upload CSV or PDF financial documents, build an in-session financial profile, "
        "and get chat-based planning guidance for expenses, income, and tax savings."
    )

    with st.sidebar:
        st.header("Financial profile")
        business_type = st.text_input("Business or income type", placeholder="Freelancer, LLC, sole proprietor...")
        tax_rate = st.slider("Tax savings reserve rate", 0, 50, 30, format="%d%%") / 100
        tax_notes = st.text_area(
            "Tax notes",
            placeholder="State, filing status, estimated payments, payroll, deductions...",
        )
        if st.button("Clear financial memory"):
            st.session_state.ledger = pd.DataFrame(columns=LEDGER_COLUMNS)
            st.session_state.document_summaries = []
            st.session_state.processed_files = set()
            st.session_state.messages = st.session_state.messages[:1]
            st.rerun()

    render_uploads()

    summary = financial_summary(st.session_state.ledger, tax_rate)
    render_metrics(summary)

    data_tab, memory_tab, chat_tab = st.tabs(["Financial profile", "Document memory", "Agent chat"])

    with data_tab:
        st.subheader("Loaded transaction ledger")
        if st.session_state.ledger.empty:
            st.info("Upload a CSV export or PDF statement to start building your financial profile.")
        else:
            st.dataframe(st.session_state.ledger, use_container_width=True, hide_index=True)
            if not summary["top_expenses"].empty:
                st.subheader("Largest expense categories")
                st.bar_chart(summary["top_expenses"].set_index("category"))

            csv = st.session_state.ledger.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download remembered ledger",
                csv,
                file_name=f"financial-agent-ledger-{datetime.now().date()}.csv",
                mime="text/csv",
            )

    with memory_tab:
        st.subheader("Remembered documents")
        if st.session_state.document_summaries:
            for item in st.session_state.document_summaries:
                st.markdown(f"- {item}")
        else:
            st.info("Document summaries will appear here after upload.")

        st.subheader("Tax savings recommendation")
        if summary["profit"] > 0:
            st.markdown(
                f"Set aside **${summary['tax_reserve']:,.2f}** from current net profit "
                f"using your selected **{tax_rate:.0%}** reserve rate."
            )
        else:
            st.markdown(
                "No positive taxable profit is currently loaded. Keep tracking income and expenses, "
                "and reassess estimated taxes once the profile shows profit."
            )
        st.caption("This app provides planning guidance, not formal tax, legal, or accounting advice.")

    with chat_tab:
        render_chat(tax_rate, business_type, tax_notes)


if __name__ == "__main__":
    main()
