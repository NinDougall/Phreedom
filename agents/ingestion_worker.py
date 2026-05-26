"""IngestionWorker — stateless file-parsing specialist.

Responsibilities
----------------
- Accept raw file bytes and parse CSV / PDF uploads into clean transaction
  DataFrames aligned to the canonical ledger schema.
- SHA-256 hash every file for vault deduplication (delegating the write to
  StorageBridge — this worker never touches disk directly).
- Convert parsed output to a raw, JSON-serialisable result dict so the
  Orchestrator can checkpoint it to the profile manifest.

Design constraints
------------------
- Zero Streamlit imports — this module is UI-framework-agnostic.
- Zero session-state references — all I/O flows through the return value and
  the StorageBridge provided by the Orchestrator.
- All public methods are stateless: given the same bytes they always produce
  the same output.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Column schemas
# ---------------------------------------------------------------------------

LEDGER_COLUMNS: list[str] = ["date", "description", "amount", "kind", "category", "source"]
TIMESHEET_COLUMNS: list[str] = ["date", "project", "hours", "rate", "total_pay"]

CURRENCY_RE = re.compile(r"(?<!\w)[-$]?\$?\s?[\d,]+(?:\.\d{2})?(?!\w)")
DATE_RE = re.compile(
    r"(?P<date>\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b)"
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ParsedDocument:
    """Immutable result produced by the IngestionWorker for a single file."""

    source: str
    transactions: pd.DataFrame
    summary: str
    raw_json: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of this parse result."""
        return {
            "source": self.source,
            "summary": self.summary,
            "transaction_count": len(self.transactions),
            "columns": list(self.transactions.columns),
            "transactions": self.transactions.to_dict(orient="records"),
        }


@dataclass
class IngestionResult:
    """Outcome returned to the Orchestrator after a file ingest attempt."""

    was_new: bool
    message: str
    parsed: ParsedDocument | None
    registry_entry: dict[str, Any] = field(default_factory=dict)
    is_timesheet: bool = False
    timesheet_df: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "was_new": self.was_new,
            "message": self.message,
            "is_timesheet": self.is_timesheet,
            "parsed": self.parsed.to_dict() if self.parsed else None,
            "registry_entry": self.registry_entry,
        }


# ---------------------------------------------------------------------------
# Pure parsing helpers (no I/O)
# ---------------------------------------------------------------------------


def infer_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """Return the first column name that fuzzy-matches one of the candidates."""
    normalized = {c.lower().strip().replace("_", " "): c for c in columns}
    for candidate in candidates:
        candidate = candidate.lower()
        for norm, orig in normalized.items():
            if candidate == norm or candidate in norm:
                return orig
    return None


def coerce_money(value: Any) -> float:
    """Parse a raw cell value (string, int, float, or NaN) into a float."""
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
    """Classify a transaction as 'income' or 'expense'."""
    if explicit_value is not None:
        v = str(explicit_value).lower()
        if any(k in v for k in ("income", "credit", "revenue", "deposit")):
            return "income"
        if any(k in v for k in ("expense", "debit", "withdrawal", "charge")):
            return "expense"
    return "income" if amount > 0 else "expense"


def normalize_csv(file_name: str, data: bytes) -> ParsedDocument:
    """Parse a CSV byte blob into a canonical ledger DataFrame."""
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as exc:
        return ParsedDocument(
            file_name,
            pd.DataFrame(columns=LEDGER_COLUMNS),
            f"CSV parse error: {exc}",
        )
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
            return ParsedDocument(
                file_name, pd.DataFrame(columns=LEDGER_COLUMNS), "No numeric column found."
            )
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


def extract_pdf_text(data: bytes) -> str:
    """Extract all text from a PDF byte blob."""
    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def parse_pdf_transactions(file_name: str, text: str) -> pd.DataFrame:
    """Scan PDF text for dated monetary lines and return a ledger DataFrame."""
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
            "date": entry_date,
            "description": description,
            "amount": amount,
            "kind": classify_kind(amount),
            "category": "Uncategorized",
            "source": file_name,
        })
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def detect_timesheet_columns(columns: list[str]) -> dict[str, str | None] | None:
    """Return a column-mapping dict when the CSV looks like a timesheet, else ``None``.

    Detection requires BOTH a recognisable date column AND an hours/time column.
    All other columns (project, rate, total) are optional.
    """
    date_col = infer_column(
        columns, ("date", "work date", "entry date", "day", "log date", "period")
    )
    hours_col = infer_column(
        columns,
        ("hours", "billable hours", "hours worked", "hrs",
         "time", "duration", "billable", "logged hours"),
    )
    if not date_col or not hours_col:
        return None
    project_col = infer_column(
        columns, ("project", "client", "task", "description", "work", "job", "name", "workstream")
    )
    rate_col = infer_column(
        columns, ("rate", "hourly rate", "rate usd", "rate per hour", "billing rate", "pay rate", "price")
    )
    total_col = infer_column(
        columns, ("total", "total pay", "total amount", "amount", "pay", "earnings", "gross")
    )
    if total_col == hours_col:
        total_col = None
    return {
        "date_col": date_col,
        "hours_col": hours_col,
        "project_col": project_col,
        "rate_col": rate_col,
        "total_col": total_col,
    }


def parse_timesheet_csv(file_name: str, data: bytes) -> tuple[pd.DataFrame, str]:
    """Parse a timesheet CSV into a ``TIMESHEET_COLUMNS`` DataFrame.

    Returns ``(dataframe, human-readable summary)``.
    """
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as exc:
        return pd.DataFrame(columns=TIMESHEET_COLUMNS), f"CSV parse error: {exc}"

    col_map = detect_timesheet_columns(list(df.columns))
    if col_map is None:
        return pd.DataFrame(columns=TIMESHEET_COLUMNS), "No timesheet columns detected."

    dates = pd.to_datetime(df[col_map["date_col"]], errors="coerce").dt.date
    hours = pd.to_numeric(df[col_map["hours_col"]], errors="coerce").fillna(0.0)
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
        totals = hours * rates

    ts_df = pd.DataFrame(
        {
            "date": dates,
            "project": projects,
            "hours": hours,
            "rate": rates,
            "total_pay": totals.round(2),
        },
        columns=TIMESHEET_COLUMNS,
    ).dropna(subset=["date"])
    ts_df = ts_df[ts_df["hours"] > 0].reset_index(drop=True)

    total_hours = float(ts_df["hours"].sum())
    total_pay = float(ts_df["total_pay"].sum())
    summary = (
        f"Timesheet '{file_name}': {len(ts_df)} entries, "
        f"{total_hours:,.2f} hours, ${total_pay:,.2f} total pay."
    )
    return ts_df, summary


def timesheet_rows_to_ledger(ts_df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Convert ``TIMESHEET_COLUMNS`` rows into ``LEDGER_COLUMNS`` income rows."""
    if ts_df.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    return pd.DataFrame(
        {
            "date": ts_df["date"],
            "description": ts_df["project"].apply(lambda p: f"Timesheet: {p}"),
            "amount": ts_df["total_pay"],
            "kind": "income",
            "category": "Billable income",
            "source": source,
        },
        columns=LEDGER_COLUMNS,
    )


def summarize_pdf(file_name: str, text: str, ledger: pd.DataFrame) -> str:
    """Build a human-readable summary string for a parsed PDF."""
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


# ---------------------------------------------------------------------------
# IngestionWorker
# ---------------------------------------------------------------------------


class IngestionWorker:
    """Stateless specialist for file parsing and vault deduplication.

    The worker does not hold any mutable state — it is safe to instantiate
    once and reuse across multiple ``ingest()`` calls.  All vault writes and
    registry updates are delegated to the ``StorageBridge`` instance supplied
    by the Orchestrator.

    Parameters
    ----------
    bridge:
        A ``StorageBridge`` instance used exclusively for vault deduplication
        and persistence.  The worker never reads or writes the ledger or chat
        history — those responsibilities belong to the Orchestrator.
    """

    def __init__(self, bridge: Any) -> None:
        self._bridge = bridge

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, file_name: str, content: bytes) -> IngestionResult:
        """Parse, deduplicate, and vault a single file.

        Parameters
        ----------
        file_name:
            Original filename (e.g. ``"bank_export.csv"``).
        content:
            Raw file bytes.

        Returns
        -------
        IngestionResult
            ``was_new=False`` when the vault already holds a file with the
            same SHA-256 hash; otherwise ``was_new=True`` with parsed
            transactions and a fresh registry entry.  When the file is
            detected as a timesheet CSV, ``is_timesheet=True`` and
            ``timesheet_df`` contains the parsed timesheet rows.
        """
        is_ts, ts_df, ts_summary = self._try_parse_timesheet(file_name, content)

        if is_ts:
            summary = ts_summary
            tx_count = len(ts_df)
            file_type = "timesheet"
        else:
            parsed = self._parse(file_name, content)
            summary = parsed.summary
            tx_count = len(parsed.transactions)
            file_type = "ledger"

        entry = self._bridge.save_document(
            file_name,
            content,
            {
                "transaction_count": tx_count,
                "summary": summary,
                "file_type": file_type,
            },
        )

        if entry.get("duplicate"):
            return IngestionResult(
                was_new=False,
                message=f"**{file_name}** is already in the vault (skipped duplicate).",
                parsed=None,
                registry_entry=entry,
                is_timesheet=is_ts,
                timesheet_df=None,
            )

        if is_ts:
            ledger_rows = timesheet_rows_to_ledger(ts_df, file_name)
            ts_parsed = ParsedDocument(file_name, ledger_rows, summary)
            ts_parsed.raw_json = ts_parsed.to_dict()
            return IngestionResult(
                was_new=True,
                message=summary,
                parsed=ts_parsed,
                registry_entry=entry,
                is_timesheet=True,
                timesheet_df=ts_df,
            )

        return IngestionResult(
            was_new=True,
            message=parsed.summary,  # type: ignore[possibly-undefined]
            parsed=parsed,           # type: ignore[possibly-undefined]
            registry_entry=entry,
            is_timesheet=False,
            timesheet_df=None,
        )

    def reparse(self, file_name: str, content: bytes) -> ParsedDocument:
        """Re-parse a file that is already vaulted (e.g. for re-analysis).

        Unlike ``ingest()``, this never writes to the vault — it only returns
        the parsed result so the Orchestrator can rebuild ledger rows.
        """
        return self._parse(file_name, content)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_parse_timesheet(
        file_name: str, content: bytes
    ) -> tuple[bool, pd.DataFrame, str]:
        """Attempt to parse content as a timesheet CSV.

        Returns ``(is_timesheet, ts_df, summary)``.  ``is_timesheet`` is
        ``False`` for non-CSV files or when no timesheet columns are found.
        """
        if not file_name.lower().endswith(".csv"):
            return False, pd.DataFrame(columns=TIMESHEET_COLUMNS), ""
        try:
            sample = pd.read_csv(io.BytesIO(content), nrows=0)
            if detect_timesheet_columns(list(sample.columns)) is None:
                return False, pd.DataFrame(columns=TIMESHEET_COLUMNS), ""
            ts_df, summary = parse_timesheet_csv(file_name, content)
            if ts_df.empty:
                return False, pd.DataFrame(columns=TIMESHEET_COLUMNS), ""
            return True, ts_df, summary
        except Exception:
            return False, pd.DataFrame(columns=TIMESHEET_COLUMNS), ""

    def _parse(self, file_name: str, content: bytes) -> ParsedDocument:
        """Route to the correct parser based on file extension."""
        if file_name.lower().endswith(".csv"):
            doc = normalize_csv(file_name, content)
        else:
            text = extract_pdf_text(content)
            tx_df = parse_pdf_transactions(file_name, text)
            summary = summarize_pdf(file_name, text, tx_df)
            doc = ParsedDocument(file_name, tx_df, summary)

        # Attach JSON snapshot for manifest checkpointing
        doc.raw_json = doc.to_dict()
        return doc
