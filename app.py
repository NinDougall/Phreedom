"""Streamlit personal financial agent.

The app keeps an in-session financial memory built from uploaded CSV and PDF
documents, then uses that profile to answer chat questions and recommend tax
savings. It can use an OpenAI-compatible model when OPENAI_API_KEY is present
and falls back to deterministic local analysis otherwise.
"""

from __future__ import annotations

import calendar
import io
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
from pypdf import PdfReader


LEDGER_COLUMNS = ["date", "description", "amount", "kind", "category", "source"]
TIMESHEET_COLUMNS = ["date", "project", "hours", "rate", "total_pay"]
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
    if "timesheet" not in st.session_state:
        st.session_state.timesheet = pd.DataFrame(columns=TIMESHEET_COLUMNS)


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


def remembered_documents() -> list[dict[str, Any]]:
    """Return stored document summaries as structured entries."""

    documents = []
    for index, item in enumerate(st.session_state.document_summaries, start=1):
        if isinstance(item, dict):
            documents.append(item)
        else:
            documents.append(
                {
                    "source": f"Document {index}",
                    "summary": str(item),
                    "transactions": None,
                    "uploaded_at": "Earlier in this session",
                }
            )
    return documents


def remember_document(parsed: ParsedDocument) -> None:
    """Store uploaded document memory as a clickable section entry."""

    st.session_state.document_summaries.append(
        {
            "source": parsed.source,
            "summary": parsed.summary,
            "transactions": len(parsed.transactions),
            "uploaded_at": datetime.now().strftime("%b %d, %Y %I:%M %p"),
        }
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


def section_heading(title: str, eyebrow: str | None = None, body: str | None = None) -> None:
    """Render a minimalist section heading."""

    eyebrow_html = f'<div class="section-eyebrow">{eyebrow}</div>' if eyebrow else ""
    body_html = f'<p class="section-body">{body}</p>' if body else ""
    st.markdown(
        f"""
        <div class="section-heading">
            {eyebrow_html}
            <h2>{title}</h2>
            {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(summary: dict[str, Any]) -> None:
    """Render headline financial metrics with simple cards."""

    section_heading(
        "Financial overview",
        "Snapshot",
        "A clean read on income, expenses, profit, and the current tax reserve recommendation.",
    )
    income_col, expense_col, profit_col, tax_col = st.columns(4)
    with income_col:
        metric_card("Income", format_usd(summary["income"]))
    with expense_col:
        metric_card("Expenses", format_usd(summary["expenses"]))
    with profit_col:
        metric_card("Net profit", format_usd(summary["profit"]), summary["profit"])
    with tax_col:
        metric_card("Tax reserve", format_usd(summary["tax_reserve"]))


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
    docs = (
        "\n".join(f"- {doc.get('source', 'Document')}: {doc.get('summary', '')}" for doc in remembered_documents())
        or "No documents uploaded."
    )

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

    key = None
    try:
        key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[union-attr]
    except Exception:
        pass
    return key or os.getenv("OPENAI_API_KEY")


def generate_agent_response(prompt: str, tax_rate: float, business_type: str, tax_notes: str) -> str:
    """Answer a chat prompt using the financial memory."""

    context = build_context(tax_rate, business_type, tax_notes)
    api_key = get_openai_key()
    if not api_key:
        return fallback_response(prompt, context, tax_rate)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        conversation = st.session_state.messages[-10:]
        if not conversation or conversation[-1].get("role") != "user" or conversation[-1].get("content") != prompt:
            conversation = [*conversation, {"role": "user", "content": prompt}]

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
                *conversation,
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
        "Drop in CSV or PDF files",
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
            remember_document(parsed)
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


def month_start_end(as_of: date) -> tuple[date, date, int]:
    """Return first day, last day, and day count for the selected month."""

    month_days = calendar.monthrange(as_of.year, as_of.month)[1]
    return date(as_of.year, as_of.month, 1), date(as_of.year, as_of.month, month_days), month_days


def normalized_timesheet(timesheet: pd.DataFrame) -> pd.DataFrame:
    """Return timesheet entries with stable types for calculations."""

    if timesheet.empty:
        return pd.DataFrame(columns=TIMESHEET_COLUMNS)

    normalized = timesheet.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.date
    normalized["hours"] = pd.to_numeric(normalized["hours"], errors="coerce").fillna(0.0)
    normalized["rate"] = pd.to_numeric(normalized["rate"], errors="coerce").fillna(0.0)
    normalized["total_pay"] = pd.to_numeric(normalized["total_pay"], errors="coerce").fillna(
        normalized["hours"] * normalized["rate"]
    )
    return normalized.dropna(subset=["date"])


def earnings_dashboard_summary(
    timesheet: pd.DataFrame,
    monthly_target: float,
    base_rate: float,
    as_of: date,
) -> dict[str, Any]:
    """Calculate progress toward monthly earnings and hours targets."""

    entries = normalized_timesheet(timesheet)
    month_start, month_end, month_days = month_start_end(as_of)
    elapsed_days = min(max(as_of.day, 1), month_days)
    elapsed_ratio = elapsed_days / month_days

    current_month = entries[
        (entries["date"] >= month_start) & (entries["date"] <= min(as_of, month_end))
    ]
    ytd_entries = entries[(pd.to_datetime(entries["date"]).dt.year == as_of.year) & (entries["date"] <= as_of)]

    prev_month = 12 if as_of.month == 1 else as_of.month - 1
    prev_year = as_of.year - 1 if as_of.month == 1 else as_of.year
    prev_entries = entries[
        (pd.to_datetime(entries["date"]).dt.year == prev_year)
        & (pd.to_datetime(entries["date"]).dt.month == prev_month)
    ]

    expected_month_hours = monthly_target / base_rate if base_rate > 0 else 0.0
    actual_hours = float(current_month["hours"].sum())
    actual_pay = float(current_month["total_pay"].sum())
    avg_rate = actual_pay / actual_hours if actual_hours else 0.0
    expected_hours_to_date = expected_month_hours * elapsed_ratio
    expected_earnings_to_date = monthly_target * elapsed_ratio
    prev_hours_gap = float(prev_entries["hours"].sum()) - expected_month_hours

    monthly_actual = (
        ytd_entries.assign(month_number=pd.to_datetime(ytd_entries["date"]).dt.month)
        .groupby("month_number")["total_pay"]
        .sum()
        if not ytd_entries.empty
        else pd.Series(dtype="float64")
    )
    monthly_chart = pd.DataFrame(
        {
            "month_number": range(1, 13),
            "month": [date(as_of.year, month, 1).strftime("%b") for month in range(1, 13)],
        }
    )
    monthly_chart["actual"] = monthly_chart["month_number"].map(monthly_actual).fillna(0.0)
    monthly_chart["target"] = monthly_target

    return {
        "month_start": month_start,
        "month_end": month_end,
        "elapsed_ratio": elapsed_ratio,
        "monthly_target": monthly_target,
        "annual_target": monthly_target * 12,
        "base_rate": base_rate,
        "expected_month_hours": expected_month_hours,
        "expected_hours_to_date": expected_hours_to_date,
        "actual_hours": actual_hours,
        "avg_rate": avg_rate,
        "actual_pay": actual_pay,
        "ytd_pay": float(ytd_entries["total_pay"].sum()),
        "expected_earnings_to_date": expected_earnings_to_date,
        "earnings_to_date_gap": actual_pay - expected_earnings_to_date,
        "earnings_vs_base_rate": actual_pay - (actual_hours * base_rate),
        "hours_gap": actual_hours - expected_hours_to_date,
        "prev_month_hours_gap": prev_hours_gap,
        "current_month_entries": current_month.sort_values("date"),
        "monthly_chart": monthly_chart,
    }


def add_timesheet_entry(entry_date: date, project: str, hours: float, rate: float) -> None:
    """Remember a manual earnings entry in both the timesheet and ledger."""

    project_name = project.strip() or "Billable work"
    total_pay = round(hours * rate, 2)
    timesheet_row = pd.DataFrame(
        [
            {
                "date": entry_date,
                "project": project_name,
                "hours": hours,
                "rate": rate,
                "total_pay": total_pay,
            }
        ],
        columns=TIMESHEET_COLUMNS,
    )
    st.session_state.timesheet = pd.concat([st.session_state.timesheet, timesheet_row], ignore_index=True)

    ledger_row = pd.DataFrame(
        [
            {
                "date": entry_date,
                "description": f"Timesheet earnings: {project_name}",
                "amount": total_pay,
                "kind": "income",
                "category": "Billable income",
                "source": "Timesheet",
            }
        ],
        columns=LEDGER_COLUMNS,
    )
    append_transactions(ledger_row)


def format_usd(amount: float) -> str:
    """Format USD values with the minus sign before the dollar sign."""

    return f"-${abs(amount):,.2f}" if amount < 0 else f"${amount:,.2f}"


def metric_card(label: str, value: str, status_value: float | None = None) -> None:
    """Render a quiet metric card using the app palette."""

    status_class = "neutral"
    if status_value is not None:
        status_class = "positive" if status_value >= 0 else "negative"
    st.markdown(
        f"""
        <div class="metric-card {status_class}">
            <div class="metric-card-label">{label}</div>
            <div class="metric-card-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_global_styles() -> None:
    """Apply the N-Deavourservices brand system from the brand guide."""

    st.markdown(
        """
        <style>
            :root {
                --brand-bg: #09041A;
                --brand-surface: #10081F;
                --brand-card: #150D27;
                --brand-card-strong: #1C1233;
                --brand-primary: #372757;
                --brand-secondary: #8679A4;
                --brand-soft: #E0D6F8;
                --brand-pale: #F6ECFF;
                --brand-ink: #F6ECFF;
                --brand-muted: #D7CCEA;
                --brand-subtle: #B7A7D0;
                --brand-line: #463368;
                --brand-line-soft: #2E2148;
                --brand-positive: #A7F3D0;
                --brand-negative: #FCA5A5;
            }
            .stApp {
                background:
                    radial-gradient(circle at 18% -8%, rgba(134, 121, 164, 0.28), transparent 30rem),
                    radial-gradient(circle at 92% 4%, rgba(224, 214, 248, 0.11), transparent 28rem),
                    linear-gradient(180deg, #0D061D 0%, var(--brand-bg) 42%, #05020E 100%);
                color: var(--brand-ink);
            }
            .block-container {
                max-width: 1180px;
                padding: 3rem 3rem 4rem;
            }
            h1, h2, h3, h4, h5, h6, p, label, span {
                color: var(--brand-ink);
            }
            h1, h2, h3 {
                letter-spacing: -0.035em;
            }
            .hero {
                background:
                    linear-gradient(135deg, rgba(28, 18, 51, 0.98), rgba(15, 8, 31, 0.98)),
                    var(--brand-card);
                border: 1px solid var(--brand-line);
                border-radius: 1.35rem;
                margin: 0 0 2.5rem;
                padding: clamp(2rem, 5vw, 4.25rem);
                text-align: left;
            }
            .brand-lockup {
                align-items: center;
                color: var(--brand-soft);
                display: flex;
                gap: 1.1rem;
                margin-bottom: 2rem;
            }
            .brand-mark {
                color: var(--brand-soft);
                flex: 0 0 auto;
                height: 4.2rem;
                width: 4.2rem;
            }
            .brand-wordmark {
                color: var(--brand-soft);
                font-size: clamp(1.8rem, 4vw, 3.4rem);
                font-weight: 850;
                letter-spacing: -0.075em;
                line-height: 0.95;
            }
            .brand-tagline {
                color: var(--brand-secondary);
                font-size: clamp(1rem, 2vw, 1.5rem);
                font-weight: 800;
                letter-spacing: -0.025em;
                margin-top: 0.55rem;
            }
            .hero-kicker {
                color: var(--brand-soft);
                font-size: 0.82rem;
                font-weight: 850;
                letter-spacing: 0.16em;
                margin-bottom: 1rem;
                text-transform: uppercase;
            }
            .hero h1 {
                color: var(--brand-ink);
                font-size: clamp(2.15rem, 4.7vw, 3.85rem);
                font-weight: 680;
                line-height: 1.04;
                margin-bottom: 1rem;
                max-width: 840px;
            }
            .hero p {
                color: var(--brand-muted);
                font-size: 1.08rem;
                line-height: 1.75;
                margin: 0;
                max-width: 700px;
            }
            .hero-palette {
                display: flex;
                gap: 0.45rem;
                margin-top: 1.75rem;
            }
            .hero-palette span {
                border: 1px solid rgba(246, 236, 255, 0.18);
                border-radius: 999px;
                height: 0.5rem;
                width: 4.8rem;
            }
            .swatch-mist { background: var(--brand-primary); }
            .swatch-lavender { background: var(--brand-secondary); }
            .swatch-blue { background: var(--brand-soft); }
            .swatch-green { background: var(--brand-pale); }
            .section-heading { margin: 2.75rem 0 1.15rem; }
            .section-heading h2 {
                color: var(--brand-ink);
                font-size: 1.6rem;
                font-weight: 760;
                margin: 0.2rem 0 0;
            }
            .section-eyebrow {
                color: var(--brand-soft);
                font-size: 0.78rem;
                font-weight: 850;
                letter-spacing: 0.14em;
                text-transform: uppercase;
            }
            .section-body {
                color: var(--brand-muted);
                font-size: 1rem;
                line-height: 1.7;
                margin: 0.5rem 0 0;
                max-width: 760px;
            }
            .metric-card {
                background: var(--brand-card);
                border: 1px solid var(--brand-line);
                border-radius: 1.05rem;
                box-shadow: none;
                min-height: 7rem;
                padding: 1.2rem 1.15rem;
                text-align: left;
            }
            .metric-card-label {
                color: var(--brand-subtle);
                font-size: 0.78rem;
                font-weight: 850;
                letter-spacing: 0.09em;
                margin-bottom: 0.7rem;
                text-transform: uppercase;
            }
            .metric-card-value {
                color: var(--brand-soft);
                font-size: clamp(1.45rem, 2.4vw, 2rem);
                font-weight: 780;
                letter-spacing: -0.035em;
            }
            .metric-card.positive .metric-card-value { color: var(--brand-positive); }
            .metric-card.negative .metric-card-value { color: var(--brand-negative); }
            .tracking-note {
                color: var(--brand-muted);
                font-size: 1rem;
                line-height: 1.65;
                margin: 0.35rem 0 1.3rem;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.35rem;
                padding: 0.25rem;
                background: var(--brand-surface);
                border: 1px solid var(--brand-line);
                border-radius: 0.95rem;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 0.75rem;
                color: var(--brand-muted);
                font-weight: 760;
                padding: 0.65rem 1.05rem;
            }
            .stTabs [data-baseweb="tab"] * { color: inherit; }
            .stTabs [aria-selected="true"] {
                background: var(--brand-soft);
                color: #170431;
            }
            .stTabs [aria-selected="true"] * { color: #170431; }
            div[data-testid="stExpander"] {
                background: var(--brand-card);
                border: 1px solid var(--brand-line);
                border-radius: 1rem;
                box-shadow: none;
                margin-bottom: 0.85rem;
            }
            div[data-testid="stExpander"] details summary p {
                color: var(--brand-ink);
                font-weight: 750;
            }
            div[data-testid="stFileUploader"],
            div[data-testid="stDataFrame"],
            div[data-testid="stDataEditor"] {
                background: var(--brand-card);
                border: 1px solid var(--brand-line-soft);
                border-radius: 1rem;
            }
            .stChatMessage {
                background: var(--brand-card);
                border: 1px solid var(--brand-line);
                border-radius: 1rem;
                padding: 0.75rem;
            }
            div[data-testid="stAlert"] {
                background: var(--brand-card-strong);
                border: 1px solid var(--brand-line);
                color: var(--brand-ink);
            }
            .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
                background: var(--brand-soft);
                border: 1px solid var(--brand-soft);
                color: #170431;
                font-weight: 850;
            }
            .stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
                background: var(--brand-pale);
                border-color: var(--brand-pale);
                color: #170431;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_earnings_styles() -> None:
    """Compatibility wrapper; global styles are rendered once from main."""


def render_timesheet_dashboard() -> None:
    """Render the earnings dashboard and manual timesheet intake."""

    section_heading(
        "Earnings dashboard",
        "Monthly target",
        "Track hours, rates, and pay against the earnings pace needed to hit your monthly goal.",
    )

    settings_col, entry_col = st.columns([1, 1.3], gap="large")
    with settings_col:
        st.markdown("#### Target settings")
        monthly_target = st.number_input(
            "Monthly target pay (USD)",
            min_value=0.0,
            value=4160.0,
            step=100.0,
            format="%.2f",
            help="The monthly earnings goal used to calculate whether you are ahead or behind.",
        )
        base_rate = st.number_input(
            "Target/base hourly rate (USD)",
            min_value=0.01,
            value=26.0,
            step=1.0,
            format="%.2f",
            help="Used to translate the monthly pay target into expected billable hours.",
        )
        as_of = st.date_input("Dashboard as of", value=date.today())

    with entry_col:
        st.markdown("#### Add hours worked")
        with st.form("timesheet_entry_form", clear_on_submit=True):
            work_date = st.date_input("Date", value=date.today())
            project = st.text_input("Project or client", placeholder="Client, contract, or workstream")
            hours = st.number_input("Hours worked", min_value=0.0, value=0.0, step=0.25, format="%.2f")
            rate = st.number_input("Rate (USD)", min_value=0.0, value=base_rate, step=1.0, format="%.2f")
            submitted = st.form_submit_button("Add timesheet entry")

        if submitted:
            if hours <= 0 or rate <= 0:
                st.warning("Enter positive hours and rate before adding a timesheet entry.")
            else:
                add_timesheet_entry(work_date, project, hours, rate)
                st.success(f"Added {hours:.2f} hours at ${rate:,.2f}/hr for ${hours * rate:,.2f}.")

    dashboard = earnings_dashboard_summary(st.session_state.timesheet, monthly_target, base_rate, as_of)

    st.markdown(
        f"<p class='tracking-note'><strong>Monthly Target:</strong> ${monthly_target:,.2f} USD "
        f"from {dashboard['month_start'].strftime('%b %d')} to {dashboard['month_end'].strftime('%b %d')}.</p>",
        unsafe_allow_html=True,
    )

    top_cols = st.columns(4)
    with top_cols[0]:
        metric_card("Month Actual Earnings", format_usd(dashboard["actual_pay"]))
    with top_cols[1]:
        metric_card("Month Target Earnings", format_usd(dashboard["monthly_target"]))
    with top_cols[2]:
        metric_card("YTD Actual Earnings", format_usd(dashboard["ytd_pay"]))
    with top_cols[3]:
        metric_card("Annual Target Earnings", format_usd(dashboard["annual_target"]))

    status_cols = st.columns(4)
    with status_cols[0]:
        metric_card("Hours Ahead/Behind", f"{dashboard['hours_gap']:,.2f}", dashboard["hours_gap"])
    with status_cols[1]:
        metric_card(
            "Earnings - Expected by Today",
            format_usd(dashboard["earnings_to_date_gap"]),
            dashboard["earnings_to_date_gap"],
        )
    with status_cols[2]:
        metric_card(
            "Earnings Above/Below Base Rate",
            format_usd(dashboard["earnings_vs_base_rate"]),
            dashboard["earnings_vs_base_rate"],
        )
    with status_cols[3]:
        metric_card("Prev Month Hours Ahead/Behind", f"{dashboard['prev_month_hours_gap']:,.2f}", dashboard["prev_month_hours_gap"])

    detail_left, detail_right = st.columns([1.2, 1])
    with detail_left:
        st.subheader("Actual vs Target Earnings")
        chart_data = dashboard["monthly_chart"]
        month_sort = chart_data["month"].tolist()
        bars = (
            alt.Chart(chart_data)
            .mark_bar(color="#8679A4")
            .encode(
                x=alt.X("month:N", sort=month_sort, title="Month"),
                y=alt.Y("actual:Q", title="Earnings (USD)"),
                tooltip=["month", alt.Tooltip("actual:Q", format="$,.2f")],
            )
        )
        target_line = (
            alt.Chart(chart_data)
            .mark_line(color="#E0D6F8", strokeWidth=3)
            .encode(
                x=alt.X("month:N", sort=month_sort),
                y="target:Q",
                tooltip=["month", alt.Tooltip("target:Q", format="$,.2f")],
            )
        )
        st.altair_chart((bars + target_line).properties(height=300), use_container_width=True)

    with detail_right:
        st.subheader("Tracking variables")
        tracking_rows = pd.DataFrame(
            [
                ["Base Rate", format_usd(dashboard["base_rate"])],
                ["Expected Hours This Month", f"{dashboard['expected_month_hours']:,.2f}"],
                ["Expected Hours by Today", f"{dashboard['expected_hours_to_date']:,.2f}"],
                ["Actual Hours", f"{dashboard['actual_hours']:,.2f}"],
                ["Avg Billable Rate", format_usd(dashboard["avg_rate"])],
                ["Expected Earnings by Today", format_usd(dashboard["expected_earnings_to_date"])],
                ["Actual Earnings", format_usd(dashboard["actual_pay"])],
                ["Earnings Gap", format_usd(dashboard["earnings_to_date_gap"])],
            ],
            columns=["Metric", "Value"],
        )
        st.dataframe(tracking_rows, use_container_width=True, hide_index=True)

    section_heading("Freelance timesheet", "Entries", "A focused view of this month's remembered billable work.")
    summary_cols = st.columns(3)
    with summary_cols[0]:
        metric_card("Avg Billable Rate", format_usd(dashboard["avg_rate"]))
    with summary_cols[1]:
        metric_card("Total Hours", f"{dashboard['actual_hours']:,.2f}")
    with summary_cols[2]:
        metric_card("Total Pay", format_usd(dashboard["actual_pay"]))

    entries = dashboard["current_month_entries"]
    if entries.empty:
        st.info("Add timesheet entries to see monthly hours and pay here.")
    else:
        display_entries = entries.copy()
        display_entries["date"] = pd.to_datetime(display_entries["date"]).dt.strftime("%Y-%m-%d")
        display_entries = display_entries.rename(
            columns={
                "date": "Date",
                "project": "Project",
                "hours": "Hours",
                "rate": "Rate (USD)",
                "total_pay": "Total Pay",
            }
        )
        st.dataframe(
            display_entries.style.format(
                {"Hours": "{:.2f}", "Rate (USD)": "${:,.2f}", "Total Pay": "${:,.2f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download monthly timesheet",
            entries.to_csv(index=False).encode("utf-8"),
            file_name=f"timesheet-{as_of.strftime('%Y-%m')}.csv",
            mime="text/csv",
        )

    if st.button("Clear timesheet entries"):
        st.session_state.timesheet = pd.DataFrame(columns=TIMESHEET_COLUMNS)
        st.session_state.ledger = st.session_state.ledger[st.session_state.ledger["source"] != "Timesheet"]
        st.rerun()


def render_hero() -> None:
    """Render the N-Deavourservices branded app introduction."""

    st.markdown(
        """
        <div class="hero">
            <div class="brand-lockup" aria-label="N-Deavourservices logo">
                <svg class="brand-mark" viewBox="0 0 120 120" role="img" aria-hidden="true">
                    <path d="M60 6 L106 33 L106 87 L60 114 L14 87 L14 33 Z" fill="none" stroke="currentColor" stroke-width="8" stroke-linejoin="round"/>
                    <path d="M37 44 C38 30 56 28 62 40 C70 28 91 34 88 52 C99 60 94 82 77 82 C71 94 50 94 44 82 C28 84 21 63 34 54 C29 48 31 41 37 44 Z" fill="none" stroke="currentColor" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M42 49 C52 54 59 61 62 73 M79 52 C69 56 62 62 58 75 M43 73 C52 70 66 70 77 73" fill="none" stroke="currentColor" stroke-width="6" stroke-linecap="round"/>
                    <circle cx="60" cy="87" r="7" fill="currentColor"/>
                </svg>
                <div>
                    <div class="brand-wordmark">N-Deavourservices</div>
                    <div class="brand-tagline">Excellence through efficiency</div>
                </div>
            </div>
            <div class="hero-kicker">Private financial agent</div>
            <h1>Automated finance alignment for expenses, income, and tax readiness.</h1>
            <p>Upload documents, track billable work, audit ledger categories, and ask focused questions from one branded workspace built around clarity and efficiency.</p>
            <div class="hero-palette">
                <span class="swatch-mist"></span>
                <span class="swatch-lavender"></span>
                <span class="swatch-blue"></span>
                <span class="swatch-green"></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_profile_controls() -> tuple[str, float, str]:
    """Render compact profile controls and return selected settings."""

    with st.expander("Profile settings", expanded=False):
        settings_col, tax_col, notes_col = st.columns([1.1, 0.9, 1.4], gap="large")
        with settings_col:
            business_type = st.text_input(
                "Business or income type",
                placeholder="Freelancer, LLC, sole proprietor...",
            )
        with tax_col:
            tax_rate = st.slider("Tax savings reserve rate", 0, 50, 30, format="%d%%") / 100
        with notes_col:
            tax_notes = st.text_area(
                "Tax notes",
                placeholder="State, filing status, estimated payments, payroll, deductions...",
            )

        if st.button("Clear all remembered financial data"):
            st.session_state.ledger = pd.DataFrame(columns=LEDGER_COLUMNS)
            st.session_state.document_summaries = []
            st.session_state.processed_files = set()
            st.session_state.messages = st.session_state.messages[:1]
            st.session_state.timesheet = pd.DataFrame(columns=TIMESHEET_COLUMNS)
            st.rerun()

    return business_type, tax_rate, tax_notes


def render_upload_section() -> None:
    """Render the document intake area with clear hierarchy."""

    section_heading(
        "Document intake",
        "Upload",
        "Add bank exports, receipts, invoices, or tax PDFs. Each uploaded document becomes its own remembered section below.",
    )
    render_uploads()


def render_financial_profile(summary: dict[str, Any]) -> None:
    """Render the ledger and expense categories in a clean section."""

    section_heading(
        "Financial profile",
        "Ledger",
        "Transactions remembered from uploads and timesheet entries.",
    )
    if st.session_state.ledger.empty:
        st.info("Upload a CSV/PDF or add timesheet earnings to start building your profile.")
        return

    st.dataframe(st.session_state.ledger, use_container_width=True, hide_index=True)
    if not summary["top_expenses"].empty:
        section_heading("Expense categories", "Breakdown", "Your largest remembered expense categories.")
        st.bar_chart(summary["top_expenses"].set_index("category"))

    csv = st.session_state.ledger.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download remembered ledger",
        csv,
        file_name=f"financial-agent-ledger-{datetime.now().date()}.csv",
        mime="text/csv",
    )


def render_document_memory(summary: dict[str, Any], tax_rate: float) -> None:
    """Render stored documents as separate clickable sections."""

    section_heading(
        "Stored documents",
        "Memory",
        "Each upload is saved as a separate clickable section for quick review.",
    )
    documents = remembered_documents()
    if not documents:
        st.info("Upload documents to create separate memory sections here.")
    else:
        for index, document in enumerate(documents, start=1):
            source = document.get("source") or f"Document {index}"
            rows = document.get("transactions")
            uploaded_at = document.get("uploaded_at") or "This session"
            row_text = "No transaction rows" if rows in (None, 0) else f"{rows:,} transaction rows"
            with st.expander(f"{source} - {row_text}", expanded=index == 1):
                st.caption(f"Remembered {uploaded_at}")
                st.write(document.get("summary", "No summary available."))

    section_heading(
        "Tax savings recommendation",
        "Planning",
        "A simple reserve estimate based on the currently remembered financial profile.",
    )
    if summary["profit"] > 0:
        metric_card("Suggested reserve", format_usd(summary["tax_reserve"]))
        st.markdown(
            f"<p class='tracking-note'>Set aside {format_usd(summary['tax_reserve'])} from current net profit "
            f"using your selected {tax_rate:.0%} reserve rate.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "No positive taxable profit is currently loaded. Keep tracking income and expenses, "
            "and reassess estimated taxes once the profile shows profit."
        )
    st.caption("Planning guidance only; not formal tax, legal, or accounting advice.")


def main() -> None:
    """Run the Streamlit application."""

    st.set_page_config(page_title="N-Deavourservices Financial Agent", page_icon=":moneybag:", layout="wide")
    init_state()
    render_global_styles()
    render_hero()

    business_type, tax_rate, tax_notes = render_profile_controls()
    render_upload_section()

    summary = financial_summary(st.session_state.ledger, tax_rate)
    render_metrics(summary)

    earnings_tab, data_tab, memory_tab, chat_tab = st.tabs(
        ["Dashboard", "Ledger", "Documents", "Chat"]
    )

    with earnings_tab:
        render_timesheet_dashboard()

    with data_tab:
        render_financial_profile(summary)

    with memory_tab:
        render_document_memory(summary, tax_rate)

    with chat_tab:
        section_heading(
            "Ask N-Deavourservices",
            "Chat",
            "Use the remembered ledger, documents, and timesheet to ask direct financial questions.",
        )
        render_chat(tax_rate, business_type, tax_notes)


if __name__ == "__main__":
    main()
