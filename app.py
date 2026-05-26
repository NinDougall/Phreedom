"""N-Deavour Alignment — personal financial agent.

Single-file Streamlit app. All state lives in st.session_state (no database).
Logic layer: CSV/PDF parsing, financial summaries, timesheet calculations, chat.
Presentation layer: brand-aligned dark UI with high negative space and clean grid.
"""

from __future__ import annotations

import calendar
import html
import io
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEDGER_COLUMNS = ["date", "description", "amount", "kind", "category", "source"]
TIMESHEET_COLUMNS = ["date", "project", "hours", "rate", "total_pay"]
APP_PAGES = ["Dashboard", "Timesheet", "Chat"]

CURRENCY_RE = re.compile(r"(?<!\w)[-$]?\$?\s?[\d,]+(?:\.\d{2})?(?!\w)")
DATE_RE = re.compile(
    r"(?P<date>\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b)"
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ParsedDocument:
    """Structured output from an uploaded file."""

    source: str
    transactions: pd.DataFrame
    summary: str


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def init_state() -> None:
    """Initialize all session memory."""

    defaults: dict[str, Any] = {
        "ledger": pd.DataFrame(columns=LEDGER_COLUMNS),
        "document_summaries": [],
        "processed_files": set(),
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "Hi — I'm Phreedom, your financial agent. Upload bank exports, "
                    "expense reports, invoices, or tax PDFs, then ask me about "
                    "income, expenses, cash flow, and tax savings."
                ),
            }
        ],
        "timesheet": pd.DataFrame(columns=TIMESHEET_COLUMNS),
        "active_page": "Dashboard",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def infer_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {c.lower().strip().replace("_", " "): c for c in columns}
    for candidate in candidates:
        candidate = candidate.lower()
        for norm, orig in normalized.items():
            if candidate == norm or candidate in norm:
                return orig
    return None


def coerce_money(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    negative = text.startswith("(") and text.endswith(")")
    text = text.replace(",", "").replace("(", "").replace(")", "")
    text = re.sub(r"[^0-9.\-]", "", text)
    if text in ("", ".", "-"):
        return 0.0
    try:
        amount = float(text)
    except ValueError:
        return 0.0
    return -abs(amount) if negative else amount


def classify_kind(amount: float, explicit_value: Any = None) -> str:
    if explicit_value is not None:
        v = str(explicit_value).lower()
        if any(k in v for k in ("income", "credit", "revenue", "deposit")):
            return "income"
        if any(k in v for k in ("expense", "debit", "withdrawal", "charge")):
            return "expense"
    return "income" if amount > 0 else "expense"


def normalize_csv(file_name: str, data: bytes) -> ParsedDocument:
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as exc:
        return ParsedDocument(file_name, pd.DataFrame(columns=LEDGER_COLUMNS), f"CSV parse error: {exc}")

    if df.empty:
        return ParsedDocument(file_name, pd.DataFrame(columns=LEDGER_COLUMNS), "CSV was empty.")

    cols = list(df.columns)
    date_col = infer_column(cols, ("date", "posted", "transaction date", "trans date", "value date"))
    desc_col = infer_column(cols, ("description", "memo", "details", "narrative", "payee", "reference"))
    amount_col = infer_column(cols, ("amount", "net amount", "total", "value"))
    debit_col = infer_column(cols, ("debit", "withdrawal", "charge", "payment"))
    credit_col = infer_column(cols, ("credit", "deposit", "income"))
    kind_col = infer_column(cols, ("type", "transaction type", "kind"))
    category_col = infer_column(cols, ("category", "class", "classification"))

    if amount_col:
        amounts = df[amount_col].map(coerce_money)
    elif debit_col or credit_col:
        debits = df[debit_col].map(coerce_money) if debit_col else pd.Series(0.0, index=df.index)
        credits = df[credit_col].map(coerce_money) if credit_col else pd.Series(0.0, index=df.index)
        amounts = credits - debits
    else:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return ParsedDocument(file_name, pd.DataFrame(columns=LEDGER_COLUMNS), "No numeric column found.")
        amounts = df[num_cols[0]].map(coerce_money)

    dates = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    descriptions = df[desc_col].fillna("Imported transaction") if desc_col else "Imported transaction"
    categories = df[category_col].fillna("Uncategorized") if category_col else "Uncategorized"
    kinds = (
        [classify_kind(a, df[kind_col].iloc[i] if kind_col else None) for i, a in enumerate(amounts)]
        if True
        else ["income" if a > 0 else "expense" for a in amounts]
    )

    ledger = pd.DataFrame(
        {
            "date": dates.dt.date if hasattr(dates, "dt") else dates,
            "description": descriptions,
            "amount": amounts,
            "kind": kinds,
            "category": categories,
            "source": file_name,
        },
        columns=LEDGER_COLUMNS,
    ).dropna(subset=["date"])

    income = float(ledger.loc[ledger["kind"] == "income", "amount"].sum())
    expenses = float(ledger.loc[ledger["kind"] == "expense", "amount"].abs().sum())
    summary = (
        f"Imported {len(ledger)} CSV transactions from {file_name}: "
        f"${income:,.2f} income and ${expenses:,.2f} expenses."
    )
    return ParsedDocument(file_name, ledger, summary)


def extract_pdf_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def parse_pdf_transactions(file_name: str, text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        date_match = DATE_RE.search(line)
        if not date_match:
            continue
        currency_matches = CURRENCY_RE.findall(line)
        if not currency_matches:
            continue
        try:
            entry_date = pd.to_datetime(date_match.group("date"), dayfirst=False).date()
        except Exception:
            continue
        amount = coerce_money(currency_matches[-1])
        description = line[: date_match.start()].strip() or line.strip()[:80]
        rows.append(
            {
                "date": entry_date,
                "description": description,
                "amount": amount,
                "kind": classify_kind(amount),
                "category": "Uncategorized",
                "source": file_name,
            }
        )
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def summarize_pdf(file_name: str, text: str, ledger: pd.DataFrame) -> str:
    word_count = len(text.split())
    if ledger.empty:
        return (
            f"PDF document '{file_name}' imported ({word_count:,} words extracted). "
            "No dated transaction rows were detected automatically."
        )
    income = float(ledger.loc[ledger["kind"] == "income", "amount"].sum())
    expenses = float(ledger.loc[ledger["kind"] == "expense", "amount"].abs().sum())
    return (
        f"PDF '{file_name}': {len(ledger)} transaction rows detected "
        f"(${income:,.2f} income, ${expenses:,.2f} expenses, {word_count:,} words)."
    )


def parse_upload(uploaded_file: Any) -> ParsedDocument:
    name = uploaded_file.name
    data = uploaded_file.getvalue()
    if name.lower().endswith(".csv"):
        return normalize_csv(name, data)
    text = extract_pdf_text(data)
    ledger = parse_pdf_transactions(name, text)
    summary = summarize_pdf(name, text, ledger)
    return ParsedDocument(name, ledger, summary)


def append_transactions(transactions: pd.DataFrame) -> None:
    if transactions.empty:
        return
    st.session_state.ledger = pd.concat(
        [st.session_state.ledger, transactions[LEDGER_COLUMNS]], ignore_index=True
    )


def remembered_documents() -> list[dict[str, Any]]:
    documents = []
    for idx, item in enumerate(st.session_state.document_summaries, start=1):
        if isinstance(item, dict):
            documents.append(item)
        else:
            documents.append(
                {
                    "source": f"Document {idx}",
                    "summary": str(item),
                    "transactions": None,
                    "uploaded_at": "Earlier in this session",
                }
            )
    return documents


def remember_document(parsed: ParsedDocument) -> None:
    st.session_state.document_summaries.append(
        {
            "source": parsed.source,
            "summary": parsed.summary,
            "transactions": len(parsed.transactions),
            "uploaded_at": datetime.now().strftime("%b %d, %Y %I:%M %p"),
        }
    )


# ---------------------------------------------------------------------------
# Financial calculations
# ---------------------------------------------------------------------------


def financial_summary(ledger: pd.DataFrame, tax_rate: float) -> dict[str, Any]:
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
    expenses = float(ledger.loc[ledger["kind"] == "expense", "amount"].abs().sum())
    profit = income - expenses
    tax_reserve = max(profit, 0.0) * tax_rate
    top_expenses = (
        ledger.loc[ledger["kind"] == "expense"]
        .assign(amount=lambda f: f["amount"].abs())
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


# ---------------------------------------------------------------------------
# Agent / chat
# ---------------------------------------------------------------------------


def build_context(tax_rate: float, business_type: str, tax_notes: str) -> str:
    ledger = st.session_state.ledger
    summary = financial_summary(ledger, tax_rate)
    top_expenses = summary["top_expenses"]
    top_expense_text = (
        top_expenses.to_string(index=False, formatters={"amount": "${:,.2f}".format})
        if not top_expenses.empty
        else "No expenses loaded."
    )
    docs = (
        "\n".join(
            f"- {d.get('source', 'Document')}: {d.get('summary', '')}"
            for d in remembered_documents()
        )
        or "No documents uploaded."
    )
    recent_transactions = (
        ledger.tail(12).to_string(index=False) if not ledger.empty else "No transaction rows available."
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

Recent transactions (last 12):
{recent_transactions}
"""


def fallback_response(prompt: str, context: str, tax_rate: float) -> str:
    ledger = st.session_state.ledger
    summary = financial_summary(ledger, tax_rate)
    p = prompt.lower()

    if any(w in p for w in ("tax", "reserve", "set aside", "owe")):
        if summary["profit"] > 0:
            return (
                f"Based on your current profile, your net profit is **${summary['profit']:,.2f}**. "
                f"At a {tax_rate:.0%} reserve rate, I recommend setting aside "
                f"**${summary['tax_reserve']:,.2f}** for taxes. "
                "This is planning guidance — consult a tax professional for formal advice."
            )
        return (
            "No positive profit is currently loaded. Upload income records or add timesheet "
            "earnings to calculate a tax reserve."
        )

    if any(w in p for w in ("income", "revenue", "earn")):
        return (
            f"Total income in your current profile: **${summary['income']:,.2f}** "
            f"across {summary['transactions']} transactions."
        )

    if any(w in p for w in ("expense", "spend", "cost", "burn")):
        top = summary["top_expenses"]
        if top.empty:
            return "No expense transactions are currently loaded."
        top_text = "\n".join(
            f"- **{row['category']}**: ${row['amount']:,.2f}" for _, row in top.iterrows()
        )
        return (
            f"Total expenses: **${summary['expenses']:,.2f}**.\n\n"
            f"Largest categories:\n{top_text}"
        )

    if any(w in p for w in ("profit", "net", "margin")):
        return (
            f"Net profit (income minus expenses): **${summary['profit']:,.2f}**. "
            f"Income: ${summary['income']:,.2f} | Expenses: ${summary['expenses']:,.2f}."
        )

    if any(w in p for w in ("upload", "document", "file", "pdf", "csv")):
        docs = remembered_documents()
        if not docs:
            return "No documents have been uploaded yet. Use the Dashboard to upload CSV or PDF files."
        doc_list = "\n".join(f"- **{d.get('source')}**: {d.get('summary', '')}" for d in docs)
        return f"I have {len(docs)} remembered document(s):\n\n{doc_list}"

    return (
        f"I can see your financial profile: "
        f"**${summary['income']:,.2f}** income, "
        f"**${summary['expenses']:,.2f}** expenses, "
        f"**${summary['profit']:,.2f}** net profit. "
        "Ask me about taxes, expenses, income, or uploaded documents."
    )


def get_openai_key() -> str | None:
    key = None
    try:
        key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[union-attr]
    except Exception:
        pass
    return key or os.getenv("OPENAI_API_KEY")


def generate_agent_response(prompt: str, tax_rate: float, business_type: str, tax_notes: str) -> str:
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
                        "You are Phreedom, a careful personal financial agent for a small business owner. "
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
        return response.choices[0].message.content or fallback_response(prompt, context, tax_rate)
    except Exception as exc:
        return (
            f"{fallback_response(prompt, context, tax_rate)}\n\n"
            f"*(OpenAI unavailable: {exc})*"
        )


# ---------------------------------------------------------------------------
# Timesheet calculations
# ---------------------------------------------------------------------------


def month_start_end(as_of: date) -> tuple[date, date, int]:
    month_days = calendar.monthrange(as_of.year, as_of.month)[1]
    return date(as_of.year, as_of.month, 1), date(as_of.year, as_of.month, month_days), month_days


def normalized_timesheet(timesheet: pd.DataFrame) -> pd.DataFrame:
    if timesheet.empty:
        return pd.DataFrame(columns=TIMESHEET_COLUMNS)
    ts = timesheet.copy()
    ts["date"] = pd.to_datetime(ts["date"], errors="coerce").dt.date
    ts["hours"] = pd.to_numeric(ts["hours"], errors="coerce").fillna(0.0)
    ts["rate"] = pd.to_numeric(ts["rate"], errors="coerce").fillna(0.0)
    ts["total_pay"] = pd.to_numeric(ts["total_pay"], errors="coerce").fillna(
        ts["hours"] * ts["rate"]
    )
    return ts.dropna(subset=["date"])


def earnings_dashboard_summary(
    timesheet: pd.DataFrame,
    monthly_target: float,
    base_rate: float,
    as_of: date,
) -> dict[str, Any]:
    entries = normalized_timesheet(timesheet)
    month_start, month_end, month_days = month_start_end(as_of)
    elapsed_days = min(max(as_of.day, 1), month_days)
    elapsed_ratio = elapsed_days / month_days

    current_month = entries[
        (entries["date"] >= month_start) & (entries["date"] <= min(as_of, month_end))
    ]
    ytd_entries = entries[
        (pd.to_datetime(entries["date"]).dt.year == as_of.year) & (entries["date"] <= as_of)
    ]

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
            "month": [date(as_of.year, m, 1).strftime("%b") for m in range(1, 13)],
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
    project_name = project.strip() or "Billable work"
    total_pay = round(hours * rate, 2)
    ts_row = pd.DataFrame(
        [{"date": entry_date, "project": project_name, "hours": hours, "rate": rate, "total_pay": total_pay}],
        columns=TIMESHEET_COLUMNS,
    )
    st.session_state.timesheet = pd.concat([st.session_state.timesheet, ts_row], ignore_index=True)
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
    return f"-${abs(amount):,.2f}" if amount < 0 else f"${amount:,.2f}"


# ---------------------------------------------------------------------------
# ── PRESENTATION LAYER ──────────────────────────────────────────────────────
# Everything below is purely visual. No business logic lives here.
# ---------------------------------------------------------------------------

# Brand palette (extracted from N-Deavourservices brand guide PDF)
# Primary: #372757  Secondary: #8679A4  Soft: #E0D6F8  Pale: #F6ECFF
_CSS = """
<style>
/* ── Design tokens ─────────────────────────────────────── */
:root {
    --bg:       #09041A;
    --surface:  #110826;
    --card:     #170E30;
    --border:   #2E2148;
    --border-hi:#463368;
    --ink:      #F6ECFF;
    --muted:    #C4B8DF;
    --subtle:   #8679A4;
    --accent:   #E0D6F8;
    --brand:    #8679A4;
    --pos:      #A7F3D0;
    --neg:      #FCA5A5;
    --radius:   0.85rem;
    --radius-sm:0.55rem;
}

/* ── App shell ─────────────────────────────────────────── */
.stApp {
    background:
        radial-gradient(ellipse 80% 40% at 15% -10%, rgba(134,121,164,.20), transparent),
        radial-gradient(ellipse 60% 30% at 90%  5%, rgba(224,214,248,.07), transparent),
        var(--bg);
    color: var(--ink);
}
.block-container { max-width:1180px; padding:2.25rem 2.5rem 4rem; }

/* ── Typography ────────────────────────────────────────── */
h1,h2,h3,h4,h5,h6 { color:var(--ink); letter-spacing:-0.03em; }
p,li,label,span,td,th { color:var(--ink); }
.st-emotion-cache-ztfqz8 p { color:var(--ink) !important; }

/* ── Sidebar ───────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background:var(--surface);
    border-right:1px solid var(--border);
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span { color:var(--muted); }

/* ── Top navigation bar ────────────────────────────────── */
.nd-nav {
    align-items:center;
    border-bottom:1px solid var(--border);
    display:flex;
    gap:1.5rem;
    margin-bottom:2rem;
    padding-bottom:1rem;
}
.nd-nav-brand {
    align-items:center;
    display:flex;
    gap:0.7rem;
    flex-shrink:0;
}
.nd-nav-wordmark {
    color:var(--accent);
    font-size:1.15rem;
    font-weight:800;
    letter-spacing:-0.04em;
    line-height:1;
}
.nd-nav-tagline {
    color:var(--subtle);
    font-size:0.72rem;
    font-weight:600;
    letter-spacing:0.04em;
    line-height:1;
    margin-top:0.2rem;
    text-transform:uppercase;
}
.nd-nav-page {
    color:var(--subtle);
    font-size:0.8rem;
    font-weight:700;
    letter-spacing:0.12em;
    text-transform:uppercase;
}

/* ── Section headings ──────────────────────────────────── */
.nd-section {
    border-top:1px solid var(--border);
    margin:2.5rem 0 1.25rem;
    padding-top:1.5rem;
}
.nd-section-eyebrow {
    color:var(--accent);
    font-size:0.72rem;
    font-weight:800;
    letter-spacing:0.16em;
    margin-bottom:0.35rem;
    text-transform:uppercase;
}
.nd-section-title {
    color:var(--ink);
    font-size:1.45rem;
    font-weight:700;
    letter-spacing:-0.035em;
    line-height:1.1;
    margin:0;
}
.nd-section-body {
    color:var(--muted);
    font-size:0.95rem;
    line-height:1.65;
    margin:0.5rem 0 0;
    max-width:680px;
}

/* ── Metric cards ──────────────────────────────────────── */
.nd-metric {
    background:var(--card);
    border:1px solid var(--border);
    border-radius:var(--radius);
    padding:1.1rem 1.15rem 1.2rem;
}
.nd-metric-label {
    color:var(--subtle);
    font-size:0.72rem;
    font-weight:800;
    letter-spacing:0.12em;
    margin-bottom:0.6rem;
    text-transform:uppercase;
}
.nd-metric-value {
    color:var(--accent);
    font-size:clamp(1.35rem,2.2vw,1.9rem);
    font-weight:750;
    letter-spacing:-0.04em;
    line-height:1;
}
.nd-metric-value.pos { color:var(--pos); }
.nd-metric-value.neg { color:var(--neg); }

/* ── Tracking note ─────────────────────────────────────── */
.nd-note {
    color:var(--muted);
    font-size:0.92rem;
    line-height:1.65;
    margin:0.4rem 0 1rem;
}

/* ── Buttons ───────────────────────────────────────────── */
.stButton>button,
.stDownloadButton>button,
.stFormSubmitButton>button {
    background:var(--accent);
    border:1px solid var(--accent);
    border-radius:var(--radius-sm);
    color:#170431;
    font-weight:800;
    min-height:2.6rem;
    padding:0 1.1rem;
}
.stButton>button:hover,
.stDownloadButton>button:hover,
.stFormSubmitButton>button:hover {
    background:var(--ink);
    border-color:var(--ink);
    color:#170431;
}

/* ── Popover (menu) ────────────────────────────────────── */
div[data-testid="stPopover"] button {
    background:var(--surface) !important;
    border:1px solid var(--border-hi) !important;
    border-radius:var(--radius-sm) !important;
    color:var(--accent) !important;
    font-weight:800 !important;
    min-height:2.6rem;
    min-width:5.5rem;
}

/* ── Expanders ─────────────────────────────────────────── */
div[data-testid="stExpander"] {
    background:var(--card);
    border:1px solid var(--border);
    border-radius:var(--radius);
    margin-bottom:0.75rem;
}
div[data-testid="stExpander"] details summary p {
    color:var(--ink);
    font-weight:700;
}

/* ── File uploader ─────────────────────────────────────── */
div[data-testid="stFileUploader"] {
    background:var(--card);
    border:1px dashed var(--border-hi);
    border-radius:var(--radius);
    padding:0.5rem;
}

/* ── Data frames & editors ─────────────────────────────── */
div[data-testid="stDataFrame"],
div[data-testid="stDataEditor"] {
    border:1px solid var(--border);
    border-radius:var(--radius);
    overflow:hidden;
}

/* ── Chat messages ─────────────────────────────────────── */
.stChatMessage {
    background:var(--card);
    border:1px solid var(--border);
    border-radius:var(--radius);
    margin-bottom:0.5rem;
    padding:0.75rem;
}

/* ── Form inputs ───────────────────────────────────────── */
input,textarea,select,
div[data-baseweb="select"],
div[data-baseweb="input"] { min-height:2.5rem; }

/* ── Focus outlines (WCAG 2.2) ─────────────────────────── */
button:focus-visible,a:focus-visible,
input:focus-visible,textarea:focus-visible,
[tabindex]:focus-visible {
    outline:3px solid var(--accent) !important;
    outline-offset:3px !important;
}

/* ── Alerts ────────────────────────────────────────────── */
div[data-testid="stAlert"] {
    background:var(--card);
    border:1px solid var(--border-hi);
    border-radius:var(--radius-sm);
    color:var(--ink);
}

/* ── Screen-reader only ────────────────────────────────── */
.sr-only {
    clip:rect(0 0 0 0);
    clip-path:inset(50%);
    height:1px;
    overflow:hidden;
    position:absolute;
    white-space:nowrap;
    width:1px;
}

/* ── Skip link ─────────────────────────────────────────── */
a.skip-link {
    background:var(--accent);
    border-radius:var(--radius-sm);
    color:#170431;
    font-weight:800;
    left:1rem;
    padding:0.65rem 1rem;
    position:absolute;
    top:-5rem;
    z-index:9999;
}
a.skip-link:focus { top:1rem; }

/* ── Altair chart background ───────────────────────────── */
.vega-embed .marks { background:transparent !important; }

/* ── Responsive reflow (WCAG 1.4.10) ──────────────────── */
@media (max-width:640px) {
    .block-container { padding:1rem 1rem 3rem; }
    .nd-nav { flex-wrap:wrap; gap:0.75rem; }
    .nd-nav-wordmark { font-size:1rem; }
}
</style>
"""


def _inject_styles() -> None:
    """Inject brand CSS and accessibility bootstrap once per session."""
    components.html(
        "<script>window.parent.document.documentElement.setAttribute('lang','en');</script>",
        height=0,
        width=0,
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <a class="skip-link" href="#main-content">Skip to main content</a>
        <div id="app-status" class="sr-only" role="status" aria-live="polite">Application ready.</div>
        <main id="main-content" tabindex="-1" aria-label="N-Deavourservices financial agent workspace">
        """,
        unsafe_allow_html=True,
    )


def _close_main() -> None:
    st.markdown("</main>", unsafe_allow_html=True)


def _status(message: str) -> None:
    """Push an accessible live-region announcement."""
    st.markdown(
        f'<div class="sr-only" role="status" aria-live="polite">{html.escape(message)}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Reusable UI primitives
# ---------------------------------------------------------------------------


def _section(title: str, eyebrow: str = "", body: str = "") -> None:
    """Render a labelled section break with optional description."""
    heading_id = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "section"
    eyebrow_html = f'<div class="nd-section-eyebrow">{html.escape(eyebrow)}</div>' if eyebrow else ""
    body_html = f'<p class="nd-section-body">{html.escape(body)}</p>' if body else ""
    st.markdown(
        f"""<section class="nd-section" aria-labelledby="{heading_id}">
{eyebrow_html}
<h2 class="nd-section-title" id="{heading_id}">{html.escape(title)}</h2>
{body_html}
</section>""",
        unsafe_allow_html=True,
    )


def _kpi(label: str, value: str, delta: float | None = None) -> None:
    """Render a single branded KPI tile."""
    val_class = ""
    if delta is not None:
        val_class = " pos" if delta >= 0 else " neg"
    st.markdown(
        f"""<div class="nd-metric">
  <div class="nd-metric-label">{html.escape(label)}</div>
  <div class="nd-metric-value{val_class}">{html.escape(value)}</div>
</div>""",
        unsafe_allow_html=True,
    )


def _divider_space(rem: float = 1.5) -> None:
    st.markdown(f'<div style="height:{rem}rem"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

_NAV_SVG = """<svg viewBox="0 0 40 40" width="32" height="32" fill="none" aria-hidden="true">
  <path d="M20 3L36 12.5V31.5L20 41L4 31.5V12.5Z" stroke="currentColor" stroke-width="3.5"
        stroke-linejoin="round"/>
  <path d="M12 17C14 12 19 11.5 21 16C23 11.5 30 13.5 28 19C30.5 21.5 29 29 22 29
           C20.5 32.5 15 32.5 13.5 29C6 29.5 4 22 9 18C7 15.5 10 12 12 17Z"
        stroke="currentColor" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="20" cy="28.5" r="2.5" fill="currentColor"/>
</svg>"""


def render_nav() -> str:
    """Render the persistent top navbar with a left-side popover menu."""
    left_col, _, right_col = st.columns([0.46, 0.34, 0.20], gap="small")

    with left_col:
        brand_col, menu_col = st.columns([0.72, 0.28], gap="small", vertical_alignment="center")
        with brand_col:
            st.markdown(
                f"""<div class="nd-nav-brand">
  {_NAV_SVG}
  <div>
    <div class="nd-nav-wordmark">N-Deavourservices</div>
    <div class="nd-nav-tagline">Excellence through efficiency</div>
  </div>
</div>""",
                unsafe_allow_html=True,
            )
        with menu_col:
            with st.popover("☰  Menu"):
                st.markdown("**Navigate**")
                st.divider()
                for page in APP_PAGES:
                    is_active = page == st.session_state.active_page
                    btn_label = f"→  {page}" if is_active else f"   {page}"
                    if st.button(
                        btn_label,
                        key=f"_nav_{page}",
                        use_container_width=True,
                        disabled=is_active,
                    ):
                        st.session_state.active_page = page
                        _status(f"Navigated to {page}.")
                        st.rerun()

    with right_col:
        st.markdown(
            f'<p class="nd-nav-page" style="text-align:right;margin:0.4rem 0 0">'
            f'{html.escape(st.session_state.active_page)}</p>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:0 0 1.75rem"/>', unsafe_allow_html=True)
    return st.session_state.active_page


# ---------------------------------------------------------------------------
# Shared controls (profile settings, file uploads)
# ---------------------------------------------------------------------------


def render_profile_controls() -> tuple[str, float, str]:
    """Compact profile settings hidden inside an expander."""
    with st.expander("⚙  Profile & tax settings", expanded=False):
        c1, c2, c3 = st.columns([1.2, 0.9, 1.5], gap="large")
        with c1:
            business_type = st.text_input(
                "Business or income type",
                placeholder="Freelancer, LLC, sole proprietor…",
                help="Optional — used by the chat agent when personalizing responses.",
            )
        with c2:
            tax_rate = (
                st.slider(
                    "Tax reserve rate",
                    0, 50, 30,
                    format="%d%%",
                    help="Percentage of positive net profit to set aside for taxes.",
                )
                / 100
            )
        with c3:
            tax_notes = st.text_area(
                "Tax notes",
                placeholder="State, filing status, estimated payments…",
                help="Optional context that improves Phreedom's tax recommendations.",
                height=80,
            )
        st.markdown("")
        if st.button(
            "Clear all financial data",
            help="Removes all uploads, ledger memory, chat history, and timesheet entries.",
        ):
            _status("All financial data cleared.")
            for key, val in {
                "ledger": pd.DataFrame(columns=LEDGER_COLUMNS),
                "document_summaries": [],
                "processed_files": set(),
                "messages": st.session_state.messages[:1],
                "timesheet": pd.DataFrame(columns=TIMESHEET_COLUMNS),
            }.items():
                st.session_state[key] = val
            st.rerun()
    return business_type, tax_rate, tax_notes


def render_uploads() -> None:
    """File intake area — ingest new CSV/PDF files into session memory."""
    uploaded = st.file_uploader(
        "Drop CSV or PDF files here",
        type=["csv", "pdf"],
        accept_multiple_files=True,
        help="CSV bank exports are parsed into transactions. PDFs are text-extracted.",
    )
    if not uploaded:
        return
    for f in uploaded:
        key = f"{f.name}:{f.size}"
        if key in st.session_state.processed_files:
            continue
        try:
            parsed = parse_upload(f)
            append_transactions(parsed.transactions)
            remember_document(parsed)
            st.session_state.processed_files.add(key)
            st.success(parsed.summary)
        except Exception as exc:
            st.error(f"Could not process {f.name}: {exc}")


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------


def render_dashboard(summary: dict[str, Any], tax_rate: float) -> None:
    """Home page — financial snapshot, uploads, documents, ledger."""

    # ── Headline metrics ──────────────────────────────────────────────────
    _section("Financial overview", "Snapshot")
    _divider_space(0.25)

    pad_l, c_income, c_expense, c_profit, c_tax, pad_r = st.columns(
        [0.04, 1, 1, 1, 1, 0.04], gap="large"
    )
    with c_income:
        _kpi("Income", format_usd(summary["income"]))
    with c_expense:
        _kpi("Expenses", format_usd(summary["expenses"]))
    with c_profit:
        _kpi("Net profit", format_usd(summary["profit"]), summary["profit"])
    with c_tax:
        _kpi("Tax reserve", format_usd(summary["tax_reserve"]))

    _divider_space(0.5)

    # ── Upload ────────────────────────────────────────────────────────────
    _section("Upload documents", "Intake", "Drag in CSV exports or PDF statements. Each file is parsed and remembered for this session.")
    pad_l2, upload_col, pad_r2 = st.columns([0.12, 0.76, 0.12])
    with upload_col:
        render_uploads()
        with st.expander("Ingestion standards", expanded=False):
            st.markdown(
                """
- CSV files should include **date** and **amount** columns.
  Debit/credit splits are also supported.
- PDF files are text-extracted and scanned for dated monetary rows.
- All uploaded data lives in session memory only — nothing is persisted.
                """
            )

    # ── Documents ─────────────────────────────────────────────────────────
    docs = remembered_documents()
    if docs:
        _section("Stored documents", "Memory", "Each upload becomes a separate expandable section.")
        for idx, doc in enumerate(docs, 1):
            source = doc.get("source") or f"Document {idx}"
            rows = doc.get("transactions")
            row_text = "PDF / no transaction rows" if rows in (None, 0) else f"{rows:,} rows"
            uploaded_at = doc.get("uploaded_at") or "This session"
            with st.expander(f"**{source}** — {row_text}", expanded=False):
                st.caption(f"Remembered {uploaded_at}")
                st.write(doc.get("summary", "No summary available."))

    # ── Tax recommendation ─────────────────────────────────────────────────
    _section("Tax savings", "Planning")
    if summary["profit"] > 0:
        pad_l3, tax_col, pad_r3 = st.columns([0.04, 0.5, 0.46])
        with tax_col:
            _kpi("Suggested tax reserve", format_usd(summary["tax_reserve"]))
        st.markdown(
            f"<p class='nd-note'>Set aside <strong>{format_usd(summary['tax_reserve'])}</strong> "
            f"from your current net profit of <strong>{format_usd(summary['profit'])}</strong> "
            f"using your {tax_rate:.0%} reserve rate.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "No positive taxable profit is currently loaded. "
            "Upload income records or add timesheet earnings to calculate a reserve."
        )
    st.caption("Planning guidance only — not formal tax, legal, or accounting advice.")

    # ── Ledger ─────────────────────────────────────────────────────────────
    with st.expander("📋  Full transaction ledger", expanded=False):
        if st.session_state.ledger.empty:
            st.info("Upload a CSV/PDF or add timesheet entries to build your ledger.")
        else:
            st.dataframe(st.session_state.ledger, use_container_width=True, hide_index=True)
            if not summary["top_expenses"].empty:
                st.markdown("**Largest expense categories**")
                st.bar_chart(summary["top_expenses"].set_index("category"))
            csv_bytes = st.session_state.ledger.to_csv(index=False).encode()
            st.download_button(
                "Download ledger CSV",
                csv_bytes,
                file_name=f"phreedom-ledger-{datetime.now().date()}.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# Page: Timesheet
# ---------------------------------------------------------------------------


def render_timesheet(tax_rate: float) -> None:
    """Freelance timesheet — entry form, KPI row, chart, and log."""

    _section("Earnings dashboard", "Monthly target", "Track billable hours and pay against your monthly earnings goal.")

    # ── Settings + entry form side by side ────────────────────────────────
    settings_col, entry_col = st.columns([1, 1.35], gap="large")

    with settings_col:
        st.markdown("##### Target settings")
        monthly_target = st.number_input(
            "Monthly target (USD)",
            min_value=0.0, value=4160.0, step=100.0, format="%.2f",
            help="Your monthly earnings goal.",
        )
        base_rate = st.number_input(
            "Base hourly rate (USD)",
            min_value=0.01, value=26.0, step=1.0, format="%.2f",
            help="Used to derive expected monthly billable hours.",
        )
        as_of = st.date_input("Dashboard as of", value=date.today())

    with entry_col:
        st.markdown("##### Log hours worked")
        with st.form("ts_form", clear_on_submit=True):
            f_date = st.date_input("Date", value=date.today())
            f_proj = st.text_input(
                "Project / client",
                placeholder="Client name, contract, or workstream",
            )
            f_c1, f_c2 = st.columns(2)
            with f_c1:
                f_hours = st.number_input("Hours", min_value=0.0, value=0.0, step=0.25, format="%.2f")
            with f_c2:
                f_rate = st.number_input("Rate (USD/hr)", min_value=0.0, value=base_rate, step=1.0, format="%.2f")
            submitted = st.form_submit_button("Add entry", use_container_width=True)

        if submitted:
            if f_hours <= 0 or f_rate <= 0:
                st.warning("Enter hours > 0 and a rate > 0 before adding an entry.")
                _status("Timesheet entry error: enter positive hours and rate.")
            else:
                add_timesheet_entry(f_date, f_proj, f_hours, f_rate)
                st.success(f"Added {f_hours:.2f} h at ${f_rate:,.2f}/hr = {format_usd(f_hours * f_rate)}.")
                _status("Timesheet entry added successfully.")

    # ── KPI row ───────────────────────────────────────────────────────────
    dash = earnings_dashboard_summary(st.session_state.timesheet, monthly_target, base_rate, as_of)
    _divider_space(0.25)
    st.markdown(
        f"<p class='nd-note'><strong>Period:</strong> "
        f"{dash['month_start'].strftime('%b %d')} – {dash['month_end'].strftime('%b %d, %Y')}"
        f" · Target: <strong>{format_usd(monthly_target)}</strong></p>",
        unsafe_allow_html=True,
    )

    pad_l, k1, k2, k3, k4, pad_r = st.columns([0.03, 1, 1, 1, 1, 0.03], gap="large")
    with k1:
        _kpi("Month Earnings", format_usd(dash["actual_pay"]))
    with k2:
        _kpi("Month Target", format_usd(dash["monthly_target"]))
    with k3:
        _kpi("YTD Earnings", format_usd(dash["ytd_pay"]))
    with k4:
        _kpi("Annual Target", format_usd(dash["annual_target"]))

    _divider_space(0.5)

    pad_l2, s1, s2, s3, s4, pad_r2 = st.columns([0.03, 1, 1, 1, 1, 0.03], gap="large")
    with s1:
        _kpi("Hours Ahead/Behind", f"{dash['hours_gap']:+,.2f}", dash["hours_gap"])
    with s2:
        _kpi("Earnings vs Today", format_usd(dash["earnings_to_date_gap"]), dash["earnings_to_date_gap"])
    with s3:
        _kpi("vs Base Rate", format_usd(dash["earnings_vs_base_rate"]), dash["earnings_vs_base_rate"])
    with s4:
        _kpi("Prev Month Hrs Δ", f"{dash['prev_month_hours_gap']:+,.2f}", dash["prev_month_hours_gap"])

    _divider_space(0.5)

    # ── Chart + tracking table ────────────────────────────────────────────
    with st.expander("📊  Actual vs target chart & tracking variables", expanded=True):
        chart_col, table_col = st.columns([1.3, 1], gap="large")
        chart_data = dash["monthly_chart"]
        month_sort = chart_data["month"].tolist()
        with chart_col:
            bars = (
                alt.Chart(chart_data)
                .mark_bar(color="#8679A4", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("month:N", sort=month_sort, title=None, axis=alt.Axis(labelColor="#C4B8DF")),
                    y=alt.Y("actual:Q", title="USD", axis=alt.Axis(labelColor="#C4B8DF")),
                    tooltip=["month", alt.Tooltip("actual:Q", format="$,.2f", title="Actual")],
                )
            )
            line = (
                alt.Chart(chart_data)
                .mark_line(color="#E0D6F8", strokeWidth=2.5, strokeDash=[6, 3])
                .encode(
                    x=alt.X("month:N", sort=month_sort),
                    y="target:Q",
                    tooltip=["month", alt.Tooltip("target:Q", format="$,.2f", title="Target")],
                )
            )
            st.altair_chart(
                (bars + line)
                .properties(height=240, background="transparent")
                .configure_view(strokeWidth=0)
                .configure_axis(grid=False, domain=False)
                .configure_legend(labelColor="#C4B8DF", titleColor="#8679A4"),
                use_container_width=True,
            )
        with table_col:
            tracking = pd.DataFrame(
                [
                    ["Base Rate", format_usd(dash["base_rate"])],
                    ["Expected hrs (month)", f"{dash['expected_month_hours']:,.2f}"],
                    ["Expected hrs (today)", f"{dash['expected_hours_to_date']:,.2f}"],
                    ["Actual hours", f"{dash['actual_hours']:,.2f}"],
                    ["Avg billable rate", format_usd(dash["avg_rate"])],
                    ["Expected earnings (today)", format_usd(dash["expected_earnings_to_date"])],
                    ["Actual earnings", format_usd(dash["actual_pay"])],
                    ["Earnings gap", format_usd(dash["earnings_to_date_gap"])],
                ],
                columns=["Metric", "Value"],
            )
            st.dataframe(tracking, use_container_width=True, hide_index=True)

    # ── Timesheet log ─────────────────────────────────────────────────────
    _section("Timesheet log", "Entries", "Billable work entries for this month.")

    sum_c1, sum_c2, sum_c3, _ = st.columns([1, 1, 1, 1], gap="large")
    with sum_c1:
        _kpi("Avg billable rate", format_usd(dash["avg_rate"]))
    with sum_c2:
        _kpi("Total hours", f"{dash['actual_hours']:,.2f}")
    with sum_c3:
        _kpi("Total pay", format_usd(dash["actual_pay"]))

    _divider_space(0.5)
    entries = dash["current_month_entries"]
    if entries.empty:
        st.info("No timesheet entries yet. Add hours worked above.")
    else:
        disp = entries.copy()
        disp["date"] = pd.to_datetime(disp["date"]).dt.strftime("%Y-%m-%d")
        disp = disp.rename(columns={
            "date": "Date", "project": "Project",
            "hours": "Hours", "rate": "Rate (USD)", "total_pay": "Total Pay",
        })
        st.dataframe(
            disp.style.format({"Hours": "{:.2f}", "Rate (USD)": "${:,.2f}", "Total Pay": "${:,.2f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download timesheet CSV",
            entries.to_csv(index=False).encode(),
            file_name=f"timesheet-{as_of.strftime('%Y-%m')}.csv",
            mime="text/csv",
        )

    _divider_space(0.25)
    if st.button("Clear timesheet entries", help="Removes all timesheet entries and their ledger rows."):
        _status("Timesheet entries cleared.")
        st.session_state.timesheet = pd.DataFrame(columns=TIMESHEET_COLUMNS)
        st.session_state.ledger = st.session_state.ledger[
            st.session_state.ledger["source"] != "Timesheet"
        ]
        st.rerun()


# ---------------------------------------------------------------------------
# Page: Chat
# ---------------------------------------------------------------------------


def render_chat_page(tax_rate: float, business_type: str, tax_notes: str) -> None:
    """Phreedom bot workspace — full-width minimal chat."""
    _section(
        "Chat with Phreedom",
        "Bot workspace",
        "Ask Phreedom about your uploaded documents, ledger, tax reserve, income, expenses, or cash flow.",
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about expenses, income, taxes, or uploaded files…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Reviewing your financial profile…"):
            answer = generate_agent_response(prompt, tax_rate, business_type, tax_notes)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="N-Deavourservices · Financial Agent",
        page_icon=":moneybag:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_state()
    _inject_styles()

    active_page = render_nav()
    business_type, tax_rate, tax_notes = render_profile_controls()

    summary = financial_summary(st.session_state.ledger, tax_rate)

    if active_page == "Dashboard":
        render_dashboard(summary, tax_rate)
    elif active_page == "Timesheet":
        render_timesheet(tax_rate)
    elif active_page == "Chat":
        render_chat_page(tax_rate, business_type, tax_notes)

    _close_main()


if __name__ == "__main__":
    main()
