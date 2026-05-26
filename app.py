"""N-Deavour Alignment — personal financial agent.

Single-file Streamlit app. Storage is routed through storage_bridge.py (local-first,
cloud-ready). Session state acts as an in-memory cache; every mutation is immediately
flushed to disk so data survives browser refreshes and application restarts.

Architecture
────────────
  storage_bridge.StorageBridge   persistent local vault + manifest
  init_state()                   loads disk → session on cold start
  ingest_to_vault()              hashes, dedupes, vaults, parses, persists
  render_*()                     pure presentation – no I/O except through bridge
"""

from __future__ import annotations

import calendar
import html
import io
import os
import re
from datetime import date, datetime
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pypdf import PdfReader

from storage_bridge import get_bridge


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

from dataclasses import dataclass


@dataclass
class ParsedDocument:
    source: str
    transactions: pd.DataFrame
    summary: str


# ---------------------------------------------------------------------------
# Session state — loaded from disk on cold start
# ---------------------------------------------------------------------------


_DEFAULT_ASSISTANT_MSG = {
    "role": "assistant",
    "content": (
        "Hi — I'm Phreedom, your financial agent. Upload bank exports, "
        "expense reports, invoices, or tax PDFs, then ask me about "
        "income, expenses, cash flow, and tax savings."
    ),
}


def init_state() -> None:
    """Load persisted data from disk into session state (cold-start only)."""
    if "ndeavour_initialized" in st.session_state:
        return

    bridge = get_bridge()
    profile = bridge.fetch_profile()
    chat = bridge.fetch_chat_history()

    st.session_state.ndeavour_initialized = True
    st.session_state.ledger = bridge.fetch_ledger()
    st.session_state.timesheet = bridge.fetch_timesheet()
    st.session_state.messages = chat if chat else [_DEFAULT_ASSISTANT_MSG]
    st.session_state.active_page = "Dashboard"
    # File staging for explicit Submit flow
    st.session_state.ts_uploader_rev = 0
    st.session_state.ts_staged_file = None

    # Profile settings — loaded from manifest, used as widget defaults
    st.session_state.profile_business_type = profile.get("business_type", "")
    st.session_state.profile_tax_rate = float(profile.get("tax_reserve_rate", 0.30))
    st.session_state.profile_tax_notes = profile.get("tax_notes", "")


# ---------------------------------------------------------------------------
# Disk persistence helpers
# ---------------------------------------------------------------------------


def _flush_ledger() -> None:
    get_bridge().save_ledger(st.session_state.ledger)


def _flush_timesheet() -> None:
    get_bridge().save_timesheet(st.session_state.timesheet)


def _flush_chat() -> None:
    get_bridge().save_chat_history(st.session_state.messages)


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
        return -abs(float(text)) if negative else float(text)
    except ValueError:
        return 0.0


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
    kinds = [
        classify_kind(a, df[kind_col].iloc[i] if kind_col else None)
        for i, a in enumerate(amounts)
    ]
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


def detect_timesheet_columns(columns: list[str]) -> dict[str, str | None] | None:
    """Return a column-mapping dict if the CSV looks like a timesheet; otherwise None.

    Detection criteria: must have BOTH a recognisable date column AND a recognisable
    hours/time column. All other columns (project, rate, total) are optional.
    """
    date_col = infer_column(columns, ("date", "work date", "entry date", "day", "log date", "period"))
    hours_col = infer_column(columns, ("hours", "billable hours", "hours worked", "hrs",
                                       "time", "duration", "billable", "logged hours"))
    if not date_col or not hours_col:
        return None
    project_col = infer_column(columns, ("project", "client", "task", "description",
                                          "work", "job", "name", "workstream"))
    rate_col    = infer_column(columns, ("rate", "hourly rate", "rate usd", "rate per hour",
                                          "billing rate", "pay rate", "price"))
    total_col   = infer_column(columns, ("total", "total pay", "total amount", "amount",
                                          "pay", "earnings", "gross"))
    # Avoid mapping total to the hours column
    if total_col == hours_col:
        total_col = None
    return {
        "date_col":    date_col,
        "hours_col":   hours_col,
        "project_col": project_col,
        "rate_col":    rate_col,
        "total_col":   total_col,
    }


def parse_timesheet_csv(file_name: str, data: bytes) -> tuple[pd.DataFrame, str]:
    """Parse a timesheet CSV into a TIMESHEET_COLUMNS DataFrame.

    Returns (dataframe, human-readable summary).
    """
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as exc:
        return pd.DataFrame(columns=TIMESHEET_COLUMNS), f"CSV parse error: {exc}"

    col_map = detect_timesheet_columns(list(df.columns))
    if col_map is None:
        return pd.DataFrame(columns=TIMESHEET_COLUMNS), "No timesheet columns detected."

    dates   = pd.to_datetime(df[col_map["date_col"]], errors="coerce").dt.date
    hours   = pd.to_numeric(df[col_map["hours_col"]], errors="coerce").fillna(0.0)
    projects = (
        df[col_map["project_col"]].fillna("Billable work").astype(str)
        if col_map["project_col"]
        else pd.Series(["Billable work"] * len(df), index=df.index)
    )
    rates = (
        pd.to_numeric(df[col_map["rate_col"]], errors="coerce").fillna(0.0)
        if col_map["rate_col"]
        else pd.Series([0.0] * len(df), index=df.index)
    )
    if col_map["total_col"]:
        totals = pd.to_numeric(df[col_map["total_col"]], errors="coerce")
        totals = totals.where(totals.notna(), hours * rates)
    else:
        totals = (hours * rates)

    ts_df = pd.DataFrame(
        {
            "date":      dates,
            "project":   projects,
            "hours":     hours,
            "rate":      rates,
            "total_pay": totals.round(2),
        },
        columns=TIMESHEET_COLUMNS,
    ).dropna(subset=["date"])
    ts_df = ts_df[ts_df["hours"] > 0].reset_index(drop=True)

    total_hours = float(ts_df["hours"].sum())
    total_pay   = float(ts_df["total_pay"].sum())
    summary = (
        f"Timesheet '{file_name}': {len(ts_df)} entries, "
        f"{total_hours:,.2f} hours, ${total_pay:,.2f} total pay."
    )
    return ts_df, summary


def timesheet_rows_to_ledger(ts_df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Convert TIMESHEET_COLUMNS rows into LEDGER_COLUMNS income rows."""
    if ts_df.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    return pd.DataFrame(
        {
            "date":        ts_df["date"],
            "description": ts_df["project"].apply(lambda p: f"Timesheet: {p}"),
            "amount":      ts_df["total_pay"],
            "kind":        "income",
            "category":    "Billable income",
            "source":      source,
        },
        columns=LEDGER_COLUMNS,
    )



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
        rows.append({
            "date": entry_date, "description": description, "amount": amount,
            "kind": classify_kind(amount), "category": "Uncategorized", "source": file_name,
        })
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def summarize_pdf(file_name: str, text: str, ledger: pd.DataFrame) -> str:
    word_count = len(text.split())
    if ledger.empty:
        return (
            f"PDF '{file_name}' imported ({word_count:,} words). "
            "No dated transaction rows were detected automatically."
        )
    income = float(ledger.loc[ledger["kind"] == "income", "amount"].sum())
    expenses = float(ledger.loc[ledger["kind"] == "expense", "amount"].abs().sum())
    return (
        f"PDF '{file_name}': {len(ledger)} transaction rows "
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


# ---------------------------------------------------------------------------
# Vault-based ingestion pipeline (replaces raw st.file_uploader logic)
# ---------------------------------------------------------------------------


def ingest_to_vault(uploaded_file: Any) -> tuple[bool, str]:
    """Hash, deduplicate, vault, parse, and persist an uploaded file.

    If the CSV is identified as a timesheet (has date + hours columns), its rows
    are written to BOTH st.session_state.timesheet AND the ledger as income entries.
    All other files are routed to the ledger only.

    Returns (was_new: bool, status_message: str).
    """
    bridge = get_bridge()
    name = uploaded_file.name
    content = uploaded_file.getvalue()
    is_csv = name.lower().endswith(".csv")

    # ── Detect timesheet CSV before vaulting so we can tag metadata ──────
    ts_df: pd.DataFrame | None = None
    ts_summary: str = ""
    if is_csv:
        try:
            import io as _io
            import pandas as _pd
            _sample = _pd.read_csv(_io.BytesIO(content), nrows=0)
            if detect_timesheet_columns(list(_sample.columns)) is not None:
                ts_df, ts_summary = parse_timesheet_csv(name, content)
        except Exception:
            ts_df = None

    # ── Determine summary and transaction count for manifest ─────────────
    if ts_df is not None and not ts_df.empty:
        summary = ts_summary
        tx_count = len(ts_df)
        file_type = "timesheet"
    else:
        parsed = parse_upload(uploaded_file)
        summary = parsed.summary
        tx_count = len(parsed.transactions)
        file_type = "ledger"

    # ── Vault with dedup check ────────────────────────────────────────────
    entry = bridge.save_document(
        name,
        content,
        {
            "transaction_count": tx_count,
            "summary": summary,
            "file_type": file_type,
        },
    )
    if entry["duplicate"]:
        return False, f"**{name}** is already in the vault (skipped duplicate)."

    # ── Persist to timesheet + ledger, or ledger only ─────────────────────
    if ts_df is not None and not ts_df.empty:
        # Remove any pre-existing rows from this source to avoid duplicates
        st.session_state.timesheet = st.session_state.timesheet[
            st.session_state.timesheet["date"].astype(str) + st.session_state.timesheet["project"].astype(str)
            != "SENTINEL_NEVER_MATCHES"
        ]
        # Append new timesheet rows
        existing_ts = st.session_state.timesheet
        # Deduplicate by (date, project, hours) against existing entries
        if not existing_ts.empty:
            existing_keys = set(
                zip(existing_ts["date"].astype(str),
                    existing_ts["project"].astype(str),
                    existing_ts["hours"].astype(str))
            )
            ts_df = ts_df[
                ~ts_df.apply(
                    lambda r: (str(r["date"]), str(r["project"]), str(r["hours"])) in existing_keys,
                    axis=1,
                )
            ]
        if not ts_df.empty:
            st.session_state.timesheet = pd.concat(
                [st.session_state.timesheet, ts_df], ignore_index=True
            )
        _flush_timesheet()

        # Mirror as ledger income entries
        ledger_rows = timesheet_rows_to_ledger(ts_df, name)
        append_transactions(ledger_rows)
        _flush_ledger()
    else:
        append_transactions(parsed.transactions)
        _flush_ledger()

    return True, summary


# ---------------------------------------------------------------------------
# Financial calculations
# ---------------------------------------------------------------------------


def financial_summary(ledger: pd.DataFrame, tax_rate: float) -> dict[str, Any]:
    if ledger.empty:
        return {
            "income": 0.0, "expenses": 0.0, "profit": 0.0,
            "tax_reserve": 0.0, "transactions": 0,
            "top_expenses": pd.DataFrame(columns=["category", "amount"]),
        }
    income = float(ledger.loc[ledger["kind"] == "income", "amount"].sum())
    expenses = float(ledger.loc[ledger["kind"] == "expense", "amount"].abs().sum())
    profit = income - expenses
    top_expenses = (
        ledger.loc[ledger["kind"] == "expense"]
        .assign(amount=lambda f: f["amount"].abs())
        .groupby("category", dropna=False)["amount"].sum()
        .sort_values(ascending=False).head(8).reset_index()
    )
    return {
        "income": income, "expenses": expenses, "profit": profit,
        "tax_reserve": max(profit, 0.0) * tax_rate,
        "transactions": len(ledger), "top_expenses": top_expenses,
    }


# ---------------------------------------------------------------------------
# Agent / chat
# ---------------------------------------------------------------------------


def build_context(tax_rate: float, business_type: str, tax_notes: str) -> str:
    summary = financial_summary(st.session_state.ledger, tax_rate)
    top_expense_text = (
        summary["top_expenses"].to_string(index=False, formatters={"amount": "${:,.2f}".format})
        if not summary["top_expenses"].empty else "No expenses loaded."
    )
    docs_list = get_bridge().list_documents()
    docs_text = (
        "\n".join(f"- {d['original_name']}: {d['summary']}" for d in docs_list)
        or "No documents uploaded."
    )
    recent = (
        st.session_state.ledger.tail(12).to_string(index=False)
        if not st.session_state.ledger.empty else "No transactions."
    )
    return (
        f"Business type: {business_type or 'Not specified'}\n"
        f"Tax notes: {tax_notes or 'None'}\n"
        f"Tax reserve rate: {tax_rate:.0%}\n"
        f"Transactions: {summary['transactions']}\n"
        f"Income: ${summary['income']:,.2f}\n"
        f"Expenses: ${summary['expenses']:,.2f}\n"
        f"Net profit: ${summary['profit']:,.2f}\n"
        f"Suggested tax reserve: ${summary['tax_reserve']:,.2f}\n\n"
        f"Top expenses:\n{top_expense_text}\n\n"
        f"Document vault ({len(docs_list)} files):\n{docs_text}\n\n"
        f"Recent transactions:\n{recent}"
    )


def fallback_response(prompt: str, context: str, tax_rate: float) -> str:
    summary = financial_summary(st.session_state.ledger, tax_rate)
    p = prompt.lower()
    if any(w in p for w in ("tax", "reserve", "set aside", "owe")):
        if summary["profit"] > 0:
            return (
                f"Based on your profile, net profit is **${summary['profit']:,.2f}**. "
                f"At {tax_rate:.0%}, I recommend reserving **${summary['tax_reserve']:,.2f}** for taxes. "
                "This is planning guidance — consult a tax professional for formal advice."
            )
        return "No positive profit loaded. Upload income records to calculate a tax reserve."
    if any(w in p for w in ("income", "revenue", "earn")):
        return f"Total income: **${summary['income']:,.2f}** across {summary['transactions']} transactions."
    if any(w in p for w in ("expense", "spend", "cost", "burn")):
        top = summary["top_expenses"]
        if top.empty:
            return "No expense transactions loaded."
        top_text = "\n".join(f"- **{r['category']}**: ${r['amount']:,.2f}" for _, r in top.iterrows())
        return f"Total expenses: **${summary['expenses']:,.2f}**.\n\nLargest categories:\n{top_text}"
    if any(w in p for w in ("profit", "net", "margin")):
        return (
            f"Net profit: **${summary['profit']:,.2f}**. "
            f"Income: ${summary['income']:,.2f} | Expenses: ${summary['expenses']:,.2f}."
        )
    docs = get_bridge().list_documents()
    if any(w in p for w in ("upload", "document", "file", "vault")):
        if not docs:
            return "No documents in the vault. Upload CSV or PDF files from the Dashboard."
        doc_list = "\n".join(f"- **{d['original_name']}**: {d['summary']}" for d in docs)
        return f"Vault contains {len(docs)} file(s):\n\n{doc_list}"
    return (
        f"Profile: **${summary['income']:,.2f}** income, **${summary['expenses']:,.2f}** expenses, "
        f"**${summary['profit']:,.2f}** net profit, **{len(docs)}** vaulted documents. "
        "Ask me about taxes, expenses, income, or uploaded files."
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
        if not conversation or conversation[-1].get("content") != prompt:
            conversation = [*conversation, {"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": (
                    "You are Phreedom, a careful personal financial agent. "
                    "Use only the provided financial context. Be specific with calculations."
                )},
                {"role": "system", "content": f"Financial memory:\n{context}"},
                *conversation,
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or fallback_response(prompt, context, tax_rate)
    except Exception as exc:
        return f"{fallback_response(prompt, context, tax_rate)}\n\n*(OpenAI unavailable: {exc})*"


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
    for col in ("hours", "rate", "total_pay"):
        ts[col] = pd.to_numeric(ts[col], errors="coerce").fillna(0.0)
    return ts.dropna(subset=["date"])


def earnings_dashboard_summary(
    timesheet: pd.DataFrame, monthly_target: float, base_rate: float, as_of: date
) -> dict[str, Any]:
    entries = normalized_timesheet(timesheet)
    month_start, month_end, month_days = month_start_end(as_of)
    elapsed_ratio = min(max(as_of.day, 1), month_days) / month_days
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
    monthly_actual = (
        ytd_entries.assign(month_number=pd.to_datetime(ytd_entries["date"]).dt.month)
        .groupby("month_number")["total_pay"].sum()
        if not ytd_entries.empty else pd.Series(dtype="float64")
    )
    monthly_chart = pd.DataFrame({
        "month_number": range(1, 13),
        "month": [date(as_of.year, m, 1).strftime("%b") for m in range(1, 13)],
    })
    monthly_chart["actual"] = monthly_chart["month_number"].map(monthly_actual).fillna(0.0)
    monthly_chart["target"] = monthly_target
    return {
        "month_start": month_start, "month_end": month_end,
        "elapsed_ratio": elapsed_ratio,
        "monthly_target": monthly_target,
        "annual_target": monthly_target * 12,
        "base_rate": base_rate,
        "expected_month_hours": expected_month_hours,
        "expected_hours_to_date": expected_hours_to_date,
        "actual_hours": actual_hours, "avg_rate": avg_rate, "actual_pay": actual_pay,
        "ytd_pay": float(ytd_entries["total_pay"].sum()),
        "expected_earnings_to_date": expected_earnings_to_date,
        "earnings_to_date_gap": actual_pay - expected_earnings_to_date,
        "earnings_vs_base_rate": actual_pay - (actual_hours * base_rate),
        "hours_gap": actual_hours - expected_hours_to_date,
        "prev_month_hours_gap": float(prev_entries["hours"].sum()) - expected_month_hours,
        "current_month_entries": current_month.sort_values("date"),
        "monthly_chart": monthly_chart,
    }


def add_timesheet_entry(entry_date: date, project: str, hours: float, rate: float) -> None:
    project_name = project.strip() or "Billable work"
    total_pay = round(hours * rate, 2)
    ts_row = pd.DataFrame(
        [{"date": entry_date, "project": project_name, "hours": hours,
          "rate": rate, "total_pay": total_pay}],
        columns=TIMESHEET_COLUMNS,
    )
    st.session_state.timesheet = pd.concat([st.session_state.timesheet, ts_row], ignore_index=True)
    _flush_timesheet()

    ledger_row = pd.DataFrame(
        [{"date": entry_date, "description": f"Timesheet earnings: {project_name}",
          "amount": total_pay, "kind": "income", "category": "Billable income", "source": "Timesheet"}],
        columns=LEDGER_COLUMNS,
    )
    append_transactions(ledger_row)
    _flush_ledger()


def format_usd(amount: float) -> str:
    return f"-${abs(amount):,.2f}" if amount < 0 else f"${amount:,.2f}"


# ---------------------------------------------------------------------------
# ── PRESENTATION LAYER ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

_CSS = """
<style>
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
    --warn:     #FDE68A;
    --radius:   0.85rem;
    --radius-sm:0.55rem;
}
.stApp {
    background:
        radial-gradient(ellipse 80% 40% at 15% -10%, rgba(134,121,164,.20), transparent),
        radial-gradient(ellipse 60% 30% at 90%  5%, rgba(224,214,248,.07), transparent),
        var(--bg);
    color: var(--ink);
}
.block-container { max-width:1180px; padding:2.25rem 2.5rem 4rem; }
h1,h2,h3,h4,h5,h6 { color:var(--ink); letter-spacing:-0.03em; }
p,li,label,span,td,th { color:var(--ink); }
[data-testid="stSidebar"] {
    background:var(--surface);
    border-right:1px solid var(--border);
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span { color:var(--muted); }

/* ── Nav ────────────── */
.nd-nav-brand { align-items:center; display:flex; gap:0.7rem; flex-shrink:0; }
.nd-nav-wordmark { color:var(--accent); font-size:1.15rem; font-weight:800; letter-spacing:-0.04em; line-height:1; }
.nd-nav-tagline  { color:var(--subtle); font-size:0.72rem; font-weight:600; letter-spacing:0.04em; line-height:1; margin-top:0.2rem; text-transform:uppercase; }
.nd-nav-page     { color:var(--subtle); font-size:0.8rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; }

/* ── Section headings ── */
.nd-section { border-top:1px solid var(--border); margin:2.5rem 0 1.25rem; padding-top:1.5rem; }
.nd-section-eyebrow { color:var(--accent); font-size:0.72rem; font-weight:800; letter-spacing:0.16em; margin-bottom:0.35rem; text-transform:uppercase; }
.nd-section-title   { color:var(--ink); font-size:1.45rem; font-weight:700; letter-spacing:-0.035em; line-height:1.1; margin:0; }
.nd-section-body    { color:var(--muted); font-size:0.95rem; line-height:1.65; margin:0.5rem 0 0; max-width:680px; }

/* ── KPI tiles ─────── */
.nd-metric         { background:var(--card); border:1px solid var(--border); border-radius:var(--radius); padding:1.1rem 1.15rem 1.2rem; }
.nd-metric-label   { color:var(--subtle); font-size:0.72rem; font-weight:800; letter-spacing:0.12em; margin-bottom:0.6rem; text-transform:uppercase; }
.nd-metric-value   { color:var(--accent); font-size:clamp(1.35rem,2.2vw,1.9rem); font-weight:750; letter-spacing:-0.04em; line-height:1; }
.nd-metric-value.pos { color:var(--pos); }
.nd-metric-value.neg { color:var(--neg); }

/* ── Registry table row ── */
.nd-registry-row   { background:var(--card); border:1px solid var(--border); border-radius:var(--radius); margin-bottom:0.55rem; padding:0.85rem 1rem; }
.nd-registry-name  { color:var(--ink); font-weight:700; font-size:0.95rem; }
.nd-registry-meta  { color:var(--subtle); font-size:0.78rem; line-height:1.8; }
.nd-badge          { background:rgba(167,243,208,0.14); border:1px solid rgba(167,243,208,0.30); border-radius:999px; color:var(--pos); display:inline-block; font-size:0.68rem; font-weight:800; letter-spacing:0.10em; padding:0.2rem 0.55rem; text-transform:uppercase; vertical-align:middle; }
.nd-badge-warn     { background:rgba(253,230,138,0.12); border-color:rgba(253,230,138,0.28); color:var(--warn); }

/* ── Note ──────────── */
.nd-note { color:var(--muted); font-size:0.92rem; line-height:1.65; margin:0.4rem 0 1rem; }

/* ── Buttons ────────── */
.stButton>button,
.stDownloadButton>button,
.stFormSubmitButton>button {
    background:var(--accent); border:1px solid var(--accent); border-radius:var(--radius-sm);
    color:#170431; font-weight:800; min-height:2.6rem; padding:0 1.1rem;
}
.stButton>button:hover,
.stDownloadButton>button:hover,
.stFormSubmitButton>button:hover {
    background:var(--ink); border-color:var(--ink); color:#170431;
}

/* ── Popover (menu) ── */
div[data-testid="stPopover"] button {
    background:var(--surface) !important; border:1px solid var(--border-hi) !important;
    border-radius:var(--radius-sm) !important; color:var(--accent) !important;
    font-weight:800 !important; min-height:2.6rem; min-width:5.5rem;
}

/* ── Expanders ─────── */
div[data-testid="stExpander"] { background:var(--card); border:1px solid var(--border); border-radius:var(--radius); margin-bottom:0.75rem; }
div[data-testid="stExpander"] details summary p { color:var(--ink); font-weight:700; }

/* ── File uploader ─── */
div[data-testid="stFileUploader"] { background:var(--card); border:1px dashed var(--border-hi); border-radius:var(--radius); padding:0.5rem; }

/* ── Data tables ───── */
div[data-testid="stDataFrame"],
div[data-testid="stDataEditor"] { border:1px solid var(--border); border-radius:var(--radius); overflow:hidden; }

/* ── Chat ──────────── */
.stChatMessage { background:var(--card); border:1px solid var(--border); border-radius:var(--radius); margin-bottom:0.5rem; padding:0.75rem; }

/* ── Inputs ─────────── */
input,textarea,select,
div[data-baseweb="select"],
div[data-baseweb="input"] { min-height:2.5rem; }

/* ── Focus (WCAG 2.2) ─ */
button:focus-visible,a:focus-visible,
input:focus-visible,textarea:focus-visible,
[tabindex]:focus-visible { outline:3px solid var(--accent) !important; outline-offset:3px !important; }

/* ── Alerts ─────────── */
div[data-testid="stAlert"] { background:var(--card); border:1px solid var(--border-hi); border-radius:var(--radius-sm); }

/* ── SR-only ─────────── */
.sr-only { clip:rect(0 0 0 0); clip-path:inset(50%); height:1px; overflow:hidden; position:absolute; white-space:nowrap; width:1px; }

/* ── Skip link ────────── */
a.skip-link { background:var(--accent); border-radius:var(--radius-sm); color:#170431; font-weight:800; left:1rem; padding:0.65rem 1rem; position:absolute; top:-5rem; z-index:9999; }
a.skip-link:focus { top:1rem; }

/* ── Altair ──────────── */
.vega-embed .marks { background:transparent !important; }

/* ── Responsive ──────── */
@media (max-width:640px) {
    .block-container { padding:1rem 1rem 3rem; }
    .nd-nav-wordmark { font-size:1rem; }
}
</style>
"""


def _inject_styles() -> None:
    components.html(
        "<script>window.parent.document.documentElement.setAttribute('lang','en');</script>",
        height=0, width=0,
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
    st.markdown(
        f'<div class="sr-only" role="status" aria-live="polite">{html.escape(message)}</div>',
        unsafe_allow_html=True,
    )


def _section(title: str, eyebrow: str = "", body: str = "") -> None:
    heading_id = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "section"
    parts = [f'<section class="nd-section" aria-labelledby="{heading_id}">']
    if eyebrow:
        parts.append(f'<div class="nd-section-eyebrow">{html.escape(eyebrow)}</div>')
    parts.append(f'<h2 class="nd-section-title" id="{heading_id}">{html.escape(title)}</h2>')
    if body:
        parts.append(f'<p class="nd-section-body">{html.escape(body)}</p>')
    parts.append("</section>")
    st.markdown("\n".join(parts), unsafe_allow_html=True)


def _kpi(label: str, value: str, delta: float | None = None) -> None:
    val_class = "" if delta is None else (" pos" if delta >= 0 else " neg")
    st.markdown(
        f'<div class="nd-metric">'
        f'<div class="nd-metric-label">{html.escape(label)}</div>'
        f'<div class="nd-metric-value{val_class}">{html.escape(value)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _divider_space(rem: float = 1.5) -> None:
    st.markdown(f'<div style="height:{rem}rem"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

_NAV_SVG = (
    '<svg viewBox="0 0 40 40" width="28" height="28" fill="none" aria-hidden="true">'
    '<path d="M20 3L36 12.5V31.5L20 41L4 31.5V12.5Z" stroke="currentColor" stroke-width="3.5" stroke-linejoin="round"/>'
    '<path d="M12 17C14 12 19 11.5 21 16C23 11.5 30 13.5 28 19C30.5 21.5 29 29 22 29C20.5 32.5 15 32.5 13.5 29C6 29.5 4 22 9 18C7 15.5 10 12 12 17Z" stroke="currentColor" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/>'
    '<circle cx="20" cy="28.5" r="2.5" fill="currentColor"/>'
    "</svg>"
)


def render_nav() -> str:
    left_col, _, right_col = st.columns([0.46, 0.34, 0.20], gap="small")
    with left_col:
        brand_col, menu_col = st.columns([0.72, 0.28], gap="small", vertical_alignment="center")
        with brand_col:
            st.markdown(
                f'<div class="nd-nav-brand">{_NAV_SVG}'
                f'<div><div class="nd-nav-wordmark">N-Deavourservices</div>'
                f'<div class="nd-nav-tagline">Excellence through efficiency</div></div></div>',
                unsafe_allow_html=True,
            )
        with menu_col:
            with st.popover("☰  Menu"):
                st.markdown("**Navigate**")
                st.divider()
                for page in APP_PAGES:
                    is_active = page == st.session_state.active_page
                    label = f"→  {page}" if is_active else f"   {page}"
                    if st.button(label, key=f"_nav_{page}", use_container_width=True, disabled=is_active):
                        st.session_state.active_page = page
                        _status(f"Navigated to {page}.")
                        st.rerun()
    with right_col:
        st.markdown(
            f'<p class="nd-nav-page" style="text-align:right;margin:0.4rem 0 0">'
            f'{html.escape(st.session_state.active_page)}</p>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<hr style="border:none;border-top:1px solid var(--border);margin:0 0 1.75rem"/>',
        unsafe_allow_html=True,
    )
    return st.session_state.active_page


# ---------------------------------------------------------------------------
# Profile controls (saves profile to manifest on submit)
# ---------------------------------------------------------------------------


def render_profile_controls() -> tuple[str, float, str]:
    with st.expander("⚙  Profile & tax settings", expanded=False):
        with st.form("profile_form"):
            c1, c2, c3 = st.columns([1.2, 0.9, 1.5], gap="large")
            with c1:
                business_type = st.text_input(
                    "Business or income type",
                    value=st.session_state.profile_business_type,
                    placeholder="Freelancer, LLC, sole proprietor…",
                    help="Used by the chat agent when personalising responses.",
                )
            with c2:
                tax_rate = (
                    st.slider(
                        "Tax reserve rate", 0, 50,
                        int(st.session_state.profile_tax_rate * 100),
                        format="%d%%",
                        help="Percentage of positive net profit to reserve for taxes.",
                    ) / 100
                )
            with c3:
                tax_notes = st.text_area(
                    "Tax notes",
                    value=st.session_state.profile_tax_notes,
                    placeholder="State, filing status, estimated payments…",
                    help="Optional context for Phreedom's tax recommendations.",
                    height=80,
                )
            col_save, col_clear = st.columns([0.35, 0.65])
            with col_save:
                save_clicked = st.form_submit_button("Save profile", use_container_width=True)
            with col_clear:
                clear_clicked = st.form_submit_button(
                    "Clear all financial data",
                    help="Removes uploads, ledger, chat history, timesheet, and vault.",
                )

        if save_clicked:
            bridge = get_bridge()
            bridge.save_profile({
                "business_type": business_type,
                "tax_reserve_rate": tax_rate,
                "tax_notes": tax_notes,
            })
            st.session_state.profile_business_type = business_type
            st.session_state.profile_tax_rate = tax_rate
            st.session_state.profile_tax_notes = tax_notes
            st.success("Profile saved to disk.")
            _status("Profile settings saved.")

        if clear_clicked:
            bridge = get_bridge()
            bridge.purge_all()
            # Re-initialise session state from the freshly wiped storage
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            _status("All financial data cleared from disk and session.")
            st.rerun()

    return (
        st.session_state.profile_business_type,
        st.session_state.profile_tax_rate,
        st.session_state.profile_tax_notes,
    )


# ---------------------------------------------------------------------------
# Permanent Registry (vault browser)
# ---------------------------------------------------------------------------


def render_permanent_registry() -> None:
    """Display and manage all files stored in the secure vault."""
    bridge = get_bridge()

    # ── Handle pending actions (must run before render) ─────────────────
    if "_vault_delete" in st.session_state:
        file_hash = st.session_state.pop("_vault_delete")
        # Find original name to remove matching ledger rows
        docs = bridge.list_documents()
        orig_name = next((d["original_name"] for d in docs if d["hash"] == file_hash), None)
        bridge.delete_document(file_hash)
        if orig_name:
            st.session_state.ledger = st.session_state.ledger[
                st.session_state.ledger["source"] != orig_name
            ]
            _flush_ledger()
        _status(f"Document deleted from vault.")
        st.rerun()

    if "_vault_reanalyze" in st.session_state:
        file_hash = st.session_state.pop("_vault_reanalyze")
        content = bridge.fetch_document(file_hash)
        docs = bridge.list_documents()
        entry = next((d for d in docs if d["hash"] == file_hash), None)
        if content and entry:
            orig_name = entry["original_name"]
            # Remove old rows for this source
            st.session_state.ledger = st.session_state.ledger[
                st.session_state.ledger["source"] != orig_name
            ]
            # Re-parse from vault bytes
            if orig_name.lower().endswith(".csv"):
                parsed = normalize_csv(orig_name, content)
            else:
                text = extract_pdf_text(content)
                tx_df = parse_pdf_transactions(orig_name, text)
                summary = summarize_pdf(orig_name, text, tx_df)
                parsed = ParsedDocument(orig_name, tx_df, summary)
            append_transactions(parsed.transactions)
            _flush_ledger()
            bridge.update_document_metadata(file_hash, {
                "transaction_count": len(parsed.transactions),
                "summary": parsed.summary,
            })
            _status(f"Re-analysed {orig_name}.")
        st.rerun()

    # ── Render the registry ──────────────────────────────────────────────
    docs = bridge.list_documents()
    diag = bridge.diagnostics()

    _section(
        "Permanent document vault",
        "Local vault",
        f"{len(docs)} file(s) stored · backend: {diag['backend'].upper()} · "
        f"profile: {diag['profile_dir']}",
    )

    if not docs:
        st.info(
            "The vault is empty. Upload CSV or PDF files above — "
            "each file is hashed, deduplicated, and persisted to disk permanently."
        )
        return

    for entry in docs:
        name = entry.get("original_name", "Unknown")
        hash_short = entry.get("hash_short", "")
        tx_count = entry.get("transaction_count", 0)
        rows_label = "no transaction rows" if tx_count == 0 else f"{tx_count:,} rows"
        uploaded_at_raw = entry.get("uploaded_at", "")
        try:
            uploaded_at = datetime.fromisoformat(uploaded_at_raw).strftime("%b %d, %Y %H:%M")
        except Exception:
            uploaded_at = uploaded_at_raw
        size_kb = entry.get("file_size_bytes", 0) / 1024
        vault_ok = entry.get("vault_exists", True)
        badge_html = (
            '<span class="nd-badge">Local-Vault</span>'
            if vault_ok
            else '<span class="nd-badge nd-badge-warn">File missing</span>'
        )

        with st.expander(
            f"**{name}** — {rows_label}",
            expanded=False,
        ):
            st.markdown(
                f'<div class="nd-registry-meta">'
                f"{badge_html}&nbsp;&nbsp;"
                f"Hash: <code>{hash_short}…</code>&nbsp;·&nbsp;"
                f"Size: {size_kb:.1f} KB&nbsp;·&nbsp;"
                f"Uploaded: {uploaded_at}"
                f"</div>",
                unsafe_allow_html=True,
            )
            if entry.get("summary"):
                st.markdown(f"> {entry['summary']}")

            act_c1, act_c2, act_c3 = st.columns([0.28, 0.28, 0.44])
            with act_c1:
                if st.button(
                    "Re-analyse",
                    key=f"reanalyze_{entry['hash']}",
                    use_container_width=True,
                    help="Re-parse this file from the vault and refresh ledger rows.",
                ):
                    st.session_state["_vault_reanalyze"] = entry["hash"]
                    st.rerun()
            with act_c2:
                if st.button(
                    "Delete",
                    key=f"delete_{entry['hash']}",
                    use_container_width=True,
                    help="Remove this file from the vault and delete its ledger rows.",
                ):
                    st.session_state["_vault_delete"] = entry["hash"]
                    st.rerun()

    # ── Vault diagnostics in expander ────────────────────────────────────
    with st.expander("Vault diagnostics", expanded=False):
        diag_rows = [
            ["Backend", diag["backend"].upper()],
            ["Profile directory", diag["profile_dir"]],
            ["Vault files on disk", str(diag["vault_file_count"])],
            ["Registry entries", str(diag["registry_count"])],
            ["Ledger on disk", "Yes" if diag["ledger_exists"] else "No"],
            ["Timesheet on disk", "Yes" if diag["timesheet_exists"] else "No"],
            ["Chat history on disk", "Yes" if diag["chat_exists"] else "No"],
            ["Manifest version", diag["manifest_version"]],
            ["Last updated", diag["last_updated"][:19] if diag["last_updated"] else "—"],
        ]
        st.dataframe(
            pd.DataFrame(diag_rows, columns=["Field", "Value"]),
            use_container_width=True, hide_index=True,
        )


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------


def render_dashboard(summary: dict[str, Any], tax_rate: float) -> None:

    # ── Headline metrics ─────────────────────────────────────────────────
    _section("Financial overview", "Snapshot")
    _divider_space(0.25)
    _, c1, c2, c3, c4, _ = st.columns([0.04, 1, 1, 1, 1, 0.04], gap="large")
    with c1:
        _kpi("Income", format_usd(summary["income"]))
    with c2:
        _kpi("Expenses", format_usd(summary["expenses"]))
    with c3:
        _kpi("Net profit", format_usd(summary["profit"]), summary["profit"])
    with c4:
        _kpi("Tax reserve", format_usd(summary["tax_reserve"]))
    _divider_space(0.5)

    # ── Upload & ingestion ───────────────────────────────────────────────
    _section(
        "Upload documents",
        "Intake",
        "Files are hashed, deduplicated, and stored permanently in the local vault. "
        "Each upload survives application restarts.",
    )
    _, upload_col, _ = st.columns([0.12, 0.76, 0.12])
    with upload_col:
        uploaded = st.file_uploader(
            "Drop CSV or PDF files here",
            type=["csv", "pdf"],
            accept_multiple_files=True,
            help="CSV bank exports are parsed into transactions. PDFs are text-extracted.",
        )
        if uploaded:
            for f in uploaded:
                with st.spinner(f"Vaulting {f.name}…"):
                    was_new, msg = ingest_to_vault(f)
                if was_new:
                    st.success(msg)
                else:
                    st.info(msg)

        with st.expander("Ingestion standards", expanded=False):
            st.markdown(
                """
- CSV files need **date** and **amount** columns (debit/credit splits supported).
- PDFs are text-extracted and scanned for dated monetary rows.
- Files are SHA-256 hashed before storage — exact duplicates are rejected.
- All vault files persist in `.ndeavour_profile/secure_vault/` between sessions.
                """
            )

    # ── Permanent registry ───────────────────────────────────────────────
    render_permanent_registry()

    # ── Tax savings ──────────────────────────────────────────────────────
    _section("Tax savings", "Planning")
    if summary["profit"] > 0:
        _, tax_col, _ = st.columns([0.04, 0.5, 0.46])
        with tax_col:
            _kpi("Suggested tax reserve", format_usd(summary["tax_reserve"]))
        st.markdown(
            f"<p class='nd-note'>Set aside <strong>{format_usd(summary['tax_reserve'])}</strong> "
            f"from net profit of <strong>{format_usd(summary['profit'])}</strong> "
            f"at your {tax_rate:.0%} reserve rate.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No positive net profit loaded. Upload income records or add timesheet entries.")
    st.caption("Planning guidance only — not formal tax, legal, or accounting advice.")

    # ── Full ledger (collapsed by default) ──────────────────────────────
    with st.expander("📋  Full transaction ledger", expanded=False):
        if st.session_state.ledger.empty:
            st.info("Upload a CSV/PDF or add timesheet entries to build your ledger.")
        else:
            st.dataframe(st.session_state.ledger, use_container_width=True, hide_index=True)
            if not summary["top_expenses"].empty:
                st.markdown("**Largest expense categories**")
                st.bar_chart(summary["top_expenses"].set_index("category"))
            st.download_button(
                "Download ledger CSV",
                st.session_state.ledger.to_csv(index=False).encode(),
                file_name=f"phreedom-ledger-{datetime.now().date()}.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# Page: Timesheet
# ---------------------------------------------------------------------------


def _render_timesheet_table(df: pd.DataFrame, key: str, download_label: str, download_fname: str) -> None:
    """Render a clean TIMESHEET_COLUMNS DataFrame with formatting and download."""
    if df.empty:
        return
    disp = df.copy()
    disp["date"] = pd.to_datetime(disp["date"]).dt.strftime("%Y-%m-%d")
    disp = disp.rename(columns={
        "date": "Date", "project": "Project",
        "hours": "Hours", "rate": "Rate (USD)", "total_pay": "Total Pay",
    })
    st.dataframe(
        disp.style.format({"Hours": "{:.2f}", "Rate (USD)": "${:,.2f}", "Total Pay": "${:,.2f}"}),
        use_container_width=True,
        hide_index=True,
        key=key,
    )
    st.download_button(
        download_label,
        df.to_csv(index=False).encode(),
        file_name=download_fname,
        mime="text/csv",
        key=f"{key}_dl",
    )


def render_timesheet(tax_rate: float) -> None:
    """Timesheet page: file upload, manual entry, monthly dashboard, full log."""

    # ── Section 1: Add hours ────────────────────────────────────────────────
    _section("Add hours", "Intake",
             "Upload a timesheet CSV or add entries manually. "
             "All data is saved to the permanent vault and survives app restarts.")

    upload_col, pad_col = st.columns([0.55, 0.45], gap="large")
    with upload_col:
        st.markdown("##### Upload timesheet file")

        # File uploader — key increments after submit/cancel to reset the widget
        ts_upload = st.file_uploader(
            "Drop a timesheet CSV here",
            type=["csv"],
            accept_multiple_files=False,
            help=(
                "Supported columns: date, project/client, hours, rate, total_pay. "
                "Column names are detected automatically."
            ),
            key=f"ts_page_uploader_{st.session_state.ts_uploader_rev}",
        )

        # Stage bytes + preview when a new file is attached
        if ts_upload is not None:
            file_key = f"{ts_upload.name}:{ts_upload.size}"
            staged = st.session_state.ts_staged_file
            if staged is None or staged.get("key") != file_key:
                content = ts_upload.getvalue()
                preview_df, preview_summary = parse_timesheet_csv(ts_upload.name, content)
                st.session_state.ts_staged_file = {
                    "key": file_key,
                    "name": ts_upload.name,
                    "content": content,
                    "preview_df": preview_df,
                    "preview_summary": preview_summary,
                }

        # Staged file UI: preview table + Submit / Cancel
        staged = st.session_state.ts_staged_file
        if staged is not None:
            preview_df = staged["preview_df"]
            if not preview_df.empty:
                total_h = float(preview_df["hours"].sum())
                total_p = float(preview_df["total_pay"].sum())
                st.markdown(
                    f'<p class="nd-note"><strong>{staged["name"]}</strong> — '
                    f'{len(preview_df)} rows, {total_h:,.2f} h, {format_usd(total_p)}. '
                    f'Review below then click <strong>Submit to timesheet</strong>.</p>',
                    unsafe_allow_html=True,
                )
                # Compact preview (max 8 rows)
                disp = preview_df.head(8).copy()
                disp["date"] = pd.to_datetime(disp["date"]).dt.strftime("%Y-%m-%d")
                disp = disp.rename(columns={
                    "date": "Date", "project": "Project",
                    "hours": "Hours", "rate": "Rate", "total_pay": "Total Pay",
                })
                st.dataframe(
                    disp.style.format({"Hours": "{:.2f}", "Rate": "${:,.2f}", "Total Pay": "${:,.2f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
                if len(preview_df) > 8:
                    st.caption(f"Showing 8 of {len(preview_df)} rows.")

                _divider_space(0.35)
                submit_c, cancel_c, _ = st.columns([0.32, 0.22, 0.46])
                with submit_c:
                    if st.button(
                        "Submit to timesheet",
                        key="ts_submit_btn",
                        use_container_width=True,
                        help="Save these entries to the permanent vault and timesheet.",
                    ):
                        class _StagedUpload:
                            def __init__(self, n, c):
                                self.name = n
                                self._c = c
                            def getvalue(self):
                                return self._c

                        with st.spinner("Saving to vault…"):
                            was_new, msg = ingest_to_vault(
                                _StagedUpload(staged["name"], staged["content"])
                            )
                        st.session_state.ts_staged_file = None
                        st.session_state.ts_uploader_rev += 1
                        if was_new:
                            st.success(msg)
                            _status(f"Timesheet submitted: {msg}")
                        else:
                            st.info(msg)
                        st.rerun()

                with cancel_c:
                    if st.button(
                        "Cancel",
                        key="ts_cancel_btn",
                        use_container_width=True,
                        help="Discard this file without saving.",
                    ):
                        st.session_state.ts_staged_file = None
                        st.session_state.ts_uploader_rev += 1
                        st.rerun()

            else:
                st.warning(
                    f"**{staged['name']}** — no timesheet columns detected. "
                    "Check that the file has date and hours columns."
                )
                if st.button("Clear", key="ts_clear_bad_btn"):
                    st.session_state.ts_staged_file = None
                    st.session_state.ts_uploader_rev += 1
                    st.rerun()

        with st.expander("Expected CSV format", expanded=False):
            st.markdown(
                """
Column names are matched flexibly (case-insensitive, spaces or underscores).

| Column | Aliases accepted |
|---|---|
| **date** | work date, entry date, day |
| **project** | client, task, description, job |
| **hours** | billable hours, hrs, duration, time |
| **rate** | hourly rate, rate usd, pay rate |
| **total_pay** | total, amount, pay, earnings |

Rows with `hours = 0` are skipped. Rate and total are optional — if both are
absent the row is stored with `$0.00` pay.
                """
            )

    with pad_col:
        st.markdown("##### Manual entry")
        with st.form("ts_form", clear_on_submit=True):
            f_date = st.date_input("Date", value=date.today())
            f_proj = st.text_input("Project / client", placeholder="Client name or workstream")
            fc1, fc2 = st.columns(2)
            with fc1:
                f_hours = st.number_input("Hours", min_value=0.0, value=0.0,
                                          step=0.25, format="%.2f")
            with fc2:
                f_rate = st.number_input("Rate (USD/hr)", min_value=0.0, value=26.0,
                                         step=1.0, format="%.2f")
            submitted = st.form_submit_button("Add entry", use_container_width=True)
        if submitted:
            if f_hours <= 0 or f_rate <= 0:
                st.warning("Enter hours > 0 and a rate > 0.")
                _status("Timesheet entry error: enter positive hours and rate.")
            else:
                add_timesheet_entry(f_date, f_proj, f_hours, f_rate)
                st.success(
                    f"Added {f_hours:.2f} h @ ${f_rate:,.2f}/hr = "
                    f"{format_usd(f_hours * f_rate)} — persisted to disk."
                )
                _status("Timesheet entry added and persisted.")
                st.rerun()

    # ── Section 2: Monthly dashboard ───────────────────────────────────────
    _section("Monthly dashboard", "Target tracking",
             "Progress metrics and pace indicators for the selected period.")

    _, settings_col, pad_mid, as_of_col, _ = st.columns([0.04, 0.4, 0.04, 0.28, 0.24], gap="small")
    with settings_col:
        monthly_target = st.number_input(
            "Monthly target (USD)", min_value=0.0, value=4160.0,
            step=100.0, format="%.2f", help="Your monthly earnings goal.")
        base_rate = st.number_input(
            "Base hourly rate (USD)", min_value=0.01, value=26.0,
            step=1.0, format="%.2f", help="Used to derive expected monthly billable hours.")
    with as_of_col:
        as_of = st.date_input("Dashboard as of", value=date.today())

    dash = earnings_dashboard_summary(st.session_state.timesheet, monthly_target, base_rate, as_of)
    _divider_space(0.25)

    st.markdown(
        f"<p class='nd-note'><strong>Period:</strong> "
        f"{dash['month_start'].strftime('%b %d')} – {dash['month_end'].strftime('%b %d, %Y')}"
        f" &nbsp;·&nbsp; Target: <strong>{format_usd(monthly_target)}</strong>"
        f" &nbsp;·&nbsp; {int(dash['elapsed_ratio'] * 100)}% of month elapsed</p>",
        unsafe_allow_html=True,
    )

    _, k1, k2, k3, k4, _ = st.columns([0.03, 1, 1, 1, 1, 0.03], gap="large")
    with k1: _kpi("Month Earnings",  format_usd(dash["actual_pay"]))
    with k2: _kpi("Month Target",    format_usd(dash["monthly_target"]))
    with k3: _kpi("YTD Earnings",    format_usd(dash["ytd_pay"]))
    with k4: _kpi("Annual Target",   format_usd(dash["annual_target"]))
    _divider_space(0.5)

    _, s1, s2, s3, s4, _ = st.columns([0.03, 1, 1, 1, 1, 0.03], gap="large")
    with s1: _kpi("Hours Ahead/Behind",    f"{dash['hours_gap']:+,.2f}", dash["hours_gap"])
    with s2: _kpi("Earnings vs Pace",      format_usd(dash["earnings_to_date_gap"]), dash["earnings_to_date_gap"])
    with s3: _kpi("vs Base Rate",          format_usd(dash["earnings_vs_base_rate"]), dash["earnings_vs_base_rate"])
    with s4: _kpi("Prev Month Hrs Delta",  f"{dash['prev_month_hours_gap']:+,.2f}", dash["prev_month_hours_gap"])

    _divider_space(0.5)

    with st.expander("Chart & tracking variables", expanded=True):
        chart_col, table_col = st.columns([1.3, 1], gap="large")
        chart_data  = dash["monthly_chart"]
        month_sort  = chart_data["month"].tolist()
        with chart_col:
            bars = (
                alt.Chart(chart_data)
                .mark_bar(color="#8679A4", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("month:N", sort=month_sort, title=None,
                             axis=alt.Axis(labelColor="#C4B8DF")),
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
                .configure_axis(grid=False, domain=False),
                use_container_width=True,
            )
        with table_col:
            st.dataframe(
                pd.DataFrame(
                    [
                        ["Base Rate",               format_usd(dash["base_rate"])],
                        ["Expected hrs (month)",    f"{dash['expected_month_hours']:,.2f}"],
                        ["Expected hrs (today)",    f"{dash['expected_hours_to_date']:,.2f}"],
                        ["Actual hours",            f"{dash['actual_hours']:,.2f}"],
                        ["Avg billable rate",       format_usd(dash["avg_rate"])],
                        ["Expected earnings (today)", format_usd(dash["expected_earnings_to_date"])],
                        ["Actual earnings",         format_usd(dash["actual_pay"])],
                        ["Earnings gap",            format_usd(dash["earnings_to_date_gap"])],
                    ],
                    columns=["Metric", "Value"],
                ),
                use_container_width=True, hide_index=True,
            )

    # ── Section 3: Monthly timesheet view ─────────────────────────────────
    _section("This month", "Entries",
             f"All entries for {dash['month_start'].strftime('%B %Y')}.")

    current_entries = dash["current_month_entries"].copy()

    _, sm1, sm2, sm3, _ = st.columns([0.04, 1, 1, 1, 0.04], gap="large")
    with sm1: _kpi("Total hours",        f"{dash['actual_hours']:,.2f}")
    with sm2: _kpi("Avg billable rate",  format_usd(dash["avg_rate"]))
    with sm3: _kpi("Total pay",          format_usd(dash["actual_pay"]))

    _divider_space(0.5)

    _, tbl_col, _ = st.columns([0.04, 0.92, 0.04])
    with tbl_col:
        if current_entries.empty:
            st.info(
                "No entries for the selected month. "
                "Upload a timesheet CSV above or add entries manually."
            )
        else:
            _render_timesheet_table(
                current_entries,
                key="ts_monthly",
                download_label="Download this month",
                download_fname=f"timesheet-{as_of.strftime('%Y-%m')}.csv",
            )

    # ── Section 4: Full timesheet history (all entries) ─────────────────
    all_entries = normalized_timesheet(st.session_state.timesheet)

    if not all_entries.empty:
        _section("Full timesheet history", "All entries",
                 f"{len(all_entries)} total entries across all periods — persisted to disk.")

        # Month filter
        all_entries_dated = all_entries.copy()
        all_entries_dated["_ym"] = pd.to_datetime(
            all_entries_dated["date"], errors="coerce"
        ).dt.to_period("M")
        months_available = sorted(
            all_entries_dated["_ym"].dropna().unique().tolist(),
            reverse=True,
        )
        months_str = [str(m) for m in months_available]

        _, filter_col, _, pad_right = st.columns([0.04, 0.35, 0.57, 0.04])
        with filter_col:
            month_filter = st.selectbox(
                "Filter by month",
                ["All periods"] + months_str,
                index=0,
                key="ts_history_filter",
                help="Narrow the full log to a specific month.",
            )

        if month_filter != "All periods":
            display_df = all_entries_dated[
                all_entries_dated["_ym"].astype(str) == month_filter
            ].drop(columns="_ym")
        else:
            display_df = all_entries.copy()

        # Project search
        _, search_col, _, _ = st.columns([0.04, 0.45, 0.47, 0.04])
        with search_col:
            search_term = st.text_input(
                "Search project / client",
                placeholder="Type to filter by project name…",
                key="ts_project_search",
            )
        if search_term:
            display_df = display_df[
                display_df["project"].str.contains(search_term, case=False, na=False)
            ]

        _, full_tbl, _ = st.columns([0.04, 0.92, 0.04])
        with full_tbl:
            if display_df.empty:
                st.info("No entries match the current filter.")
            else:
                _render_timesheet_table(
                    display_df.reset_index(drop=True),
                    key="ts_history",
                    download_label="Download filtered view",
                    download_fname=f"timesheet-history-{date.today()}.csv",
                )
                # Summary row
                total_h = float(display_df["hours"].sum())
                total_p = float(display_df["total_pay"].sum())
                avg_r   = total_p / total_h if total_h else 0.0
                st.markdown(
                    f"<p class='nd-note'>"
                    f"<strong>{len(display_df)}</strong> entries shown &nbsp;·&nbsp; "
                    f"<strong>{total_h:,.2f} h</strong> total &nbsp;·&nbsp; "
                    f"<strong>{format_usd(total_p)}</strong> total pay &nbsp;·&nbsp; "
                    f"<strong>{format_usd(avg_r)}/hr</strong> avg rate</p>",
                    unsafe_allow_html=True,
                )

    _divider_space(0.5)
    _, btn_col, _ = st.columns([0.04, 0.35, 0.61])
    with btn_col:
        if st.button("Clear all timesheet entries",
                     help="Removes all timesheet entries and their ledger rows from disk."):
            _status("All timesheet entries cleared from disk.")
            st.session_state.timesheet = pd.DataFrame(columns=TIMESHEET_COLUMNS)
            st.session_state.ledger = st.session_state.ledger[
                ~st.session_state.ledger["source"].isin(["Timesheet"])
            ]
            _flush_timesheet()
            _flush_ledger()
            st.rerun()


# ---------------------------------------------------------------------------
# Page: Chat
# ---------------------------------------------------------------------------


def render_chat_page(tax_rate: float, business_type: str, tax_notes: str) -> None:
    _section("Chat with Phreedom", "Bot workspace",
             "Ask Phreedom about your vaulted documents, ledger, tax reserve, income, or cash flow.")

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
    _flush_chat()


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
