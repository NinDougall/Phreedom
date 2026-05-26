"""N-Deavour Alignment: premium private financial agent interface.

A clean Streamlit front-end for sensitive expense tracking, tax modeling, and
bookkeeping workflows. The app is intentionally modular and mock-backed so the
interface is immediately interactive while backend services evolve.
"""

from __future__ import annotations

import io
from datetime import date, datetime
from typing import Any

import pandas as pd
import streamlit as st
from pypdf import PdfReader


APP_SECTIONS = ["Dashboard", "Document Ingestion", "Ledger Management", "Agent Workspace"]
LEDGER_COLUMNS = [
    "date",
    "vendor",
    "description",
    "category",
    "subcategory",
    "amount",
    "type",
    "tax_status",
    "source",
]
CATEGORY_OPTIONS = [
    "",
    "Revenue",
    "Software",
    "Contractors",
    "Travel",
    "Meals",
    "Office",
    "Bank Fees",
    "Taxes",
    "Owner Draw",
    "Uncategorized",
]
SUBCATEGORY_OPTIONS = {
    "": [""],
    "Revenue": ["", "Client Services", "Retainer", "Product Sales", "Interest"],
    "Software": ["", "AI Tools", "Cloud Hosting", "Subscriptions", "Security"],
    "Contractors": ["", "Engineering", "Design", "Operations", "Accounting"],
    "Travel": ["", "Airfare", "Ground Transport", "Hotel", "Mileage"],
    "Meals": ["", "Client Meals", "Team Meals", "Coffee"],
    "Office": ["", "Equipment", "Supplies", "Coworking"],
    "Bank Fees": ["", "Processing", "Wire Fees", "Monthly Fees"],
    "Taxes": ["", "Estimated Tax", "Payroll Tax", "Sales Tax"],
    "Owner Draw": ["", "Distribution", "Reimbursement"],
    "Uncategorized": ["", "Needs Review"],
}
TAX_STATUS_OPTIONS = ["", "Deductible", "Partially deductible", "Non-deductible", "Needs review"]
TYPE_OPTIONS = ["", "Income", "Expense", "Transfer"]


# -----------------------------------------------------------------------------
# State and mock data
# -----------------------------------------------------------------------------


def seed_mock_ledger() -> pd.DataFrame:
    """Create a realistic, immediately interactive mock ledger."""

    rows = [
        ["2026-05-01", "Northstar Studio", "Client retainer", "Revenue", "Retainer", 5200.00, "Income", "Needs review", "Mock ledger"],
        ["2026-05-03", "OpenAI", "AI workflow tooling", "Software", "AI Tools", -240.00, "Expense", "Deductible", "Mock ledger"],
        ["2026-05-04", "Linear", "Project management", "Software", "Subscriptions", -39.00, "Expense", "Deductible", "Mock ledger"],
        ["2026-05-07", "Mercury", "Wire transfer fee", "Bank Fees", "Wire Fees", -18.00, "Expense", "Deductible", "Mock ledger"],
        ["2026-05-09", "Figma", "Design systems", "Software", "Subscriptions", -45.00, "Expense", "Deductible", "Mock ledger"],
        ["2026-05-12", "Atlas Contractors", "Backend automation support", "Contractors", "Engineering", -1320.00, "Expense", "Deductible", "Mock ledger"],
        ["2026-05-16", "Delta", "Client onsite travel", "Travel", "Airfare", -410.00, "Expense", "Partially deductible", "Mock ledger"],
        ["2026-05-17", "Blue Bottle", "Client meeting", "Meals", "Client Meals", -64.80, "Expense", "Partially deductible", "Mock ledger"],
        ["2026-05-20", "Northstar Studio", "Implementation milestone", "Revenue", "Client Services", 3800.00, "Income", "Needs review", "Mock ledger"],
        ["2026-05-22", "IRS EFTPS", "Federal estimated tax", "Taxes", "Estimated Tax", -1250.00, "Expense", "Non-deductible", "Mock ledger"],
    ]
    ledger = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    ledger["date"] = pd.to_datetime(ledger["date"]).dt.date
    return ledger


def init_state() -> None:
    """Initialize all user interaction state in Streamlit session memory."""

    if "active_section" not in st.session_state:
        st.session_state.active_section = "Dashboard"
    if "ledger" not in st.session_state:
        st.session_state.ledger = seed_mock_ledger()
    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = []
    if "processed_upload_keys" not in st.session_state:
        st.session_state.processed_upload_keys = set()
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "N-Deavour is ready. Ask about burn, deductions, tax runway, or ledger anomalies.",
            }
        ]
    if "tax_reserve_balance" not in st.session_state:
        st.session_state.tax_reserve_balance = 14800.00
    if "target_tax_rate" not in st.session_state:
        st.session_state.target_tax_rate = 0.30


# -----------------------------------------------------------------------------
# Styling and layout primitives
# -----------------------------------------------------------------------------


def apply_design_system() -> None:
    """Apply N-Deavour design tokens and minimal structural styling."""

    st.markdown(
        """
        <style>
            :root {
                --accent: #0D7A87;
                --slate: #1E293B;
                --bg: #F8FAFC;
                --card: #FFFFFF;
                --border: #E2E8F0;
                --text: #0F172A;
                --muted: #64748B;
            }
            .stApp {
                background: var(--bg);
                color: var(--text);
            }
            .block-container {
                max-width: 1220px;
                padding: 3.2rem 3.5rem 5rem;
            }
            [data-testid="stSidebar"] {
                background: #FFFFFF;
                border-right: 1px solid var(--border);
            }
            [data-testid="stSidebar"] * {
                color: var(--slate);
            }
            h1, h2, h3 {
                color: var(--text);
                letter-spacing: -0.035em;
            }
            .ndeavour-kicker {
                color: var(--accent);
                font-size: 0.76rem;
                font-weight: 700;
                letter-spacing: 0.16em;
                margin-bottom: 0.9rem;
                text-transform: uppercase;
            }
            .ndeavour-title {
                color: var(--text);
                font-size: clamp(2.2rem, 5vw, 4.4rem);
                font-weight: 520;
                line-height: 1.02;
                margin: 0;
            }
            .ndeavour-subtitle {
                color: var(--muted);
                font-size: 1.02rem;
                line-height: 1.75;
                margin-top: 1.1rem;
                max-width: 680px;
            }
            .section-header {
                margin: 3.4rem 0 1.3rem;
            }
            .section-header h2 {
                font-size: 1.55rem;
                font-weight: 630;
                margin: 0.15rem 0 0;
            }
            .section-header p {
                color: var(--muted);
                font-size: 0.98rem;
                line-height: 1.7;
                margin: 0.55rem 0 0;
                max-width: 760px;
            }
            .card {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 1.1rem;
                padding: 1.35rem 1.4rem;
            }
            .metric-tile {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 1.15rem;
                min-height: 8.75rem;
                padding: 1.3rem 1.4rem;
            }
            .metric-label {
                color: var(--muted);
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.10em;
                text-transform: uppercase;
            }
            .metric-value {
                color: var(--text);
                font-size: clamp(1.7rem, 3vw, 2.4rem);
                font-weight: 620;
                letter-spacing: -0.045em;
                margin-top: 1rem;
            }
            .metric-note {
                color: var(--muted);
                font-size: 0.88rem;
                line-height: 1.55;
                margin-top: 0.8rem;
            }
            .accent-badge {
                background: rgba(13, 122, 135, 0.10);
                border: 1px solid rgba(13, 122, 135, 0.24);
                border-radius: 999px;
                color: var(--accent);
                display: inline-flex;
                font-size: 0.76rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                padding: 0.34rem 0.65rem;
                text-transform: uppercase;
            }
            .sidebar-brand {
                border-bottom: 1px solid var(--border);
                margin-bottom: 1.4rem;
                padding-bottom: 1.4rem;
            }
            .sidebar-brand-title {
                color: var(--slate);
                font-size: 1.05rem;
                font-weight: 720;
                letter-spacing: -0.02em;
            }
            .sidebar-brand-meta {
                color: var(--muted);
                font-size: 0.82rem;
                line-height: 1.55;
                margin-top: 0.35rem;
            }
            div[data-testid="stExpander"] {
                background: #FFFFFF;
                border: 1px solid var(--border);
                border-radius: 1rem;
                margin-bottom: 1rem;
            }
            div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
                border: 1px solid var(--border);
                border-radius: 1rem;
                overflow: hidden;
            }
            .stChatMessage {
                background: #FFFFFF;
                border: 1px solid var(--border);
                border-radius: 1rem;
                padding: 0.65rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(kicker: str, title: str, body: str) -> None:
    """Render a high-clarity section heading."""

    st.markdown(
        f"""
        <div class="section-header">
            <div class="ndeavour-kicker">{kicker}</div>
            <h2>{title}</h2>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_tile(label: str, value: str, note: str) -> None:
    """Render a minimal accessible dashboard metric tile."""

    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_usd(amount: float) -> str:
    """Format currency with a leading negative sign."""

    return f"-${abs(amount):,.2f}" if amount < 0 else f"${amount:,.2f}"


# -----------------------------------------------------------------------------
# Data operations
# -----------------------------------------------------------------------------


def normalize_uploaded_table(name: str, raw: bytes, separator: str) -> pd.DataFrame:
    """Parse uploaded CSV/TSV files into the ledger schema where possible."""

    table = pd.read_csv(io.BytesIO(raw), sep=separator)
    if table.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    normalized_columns = {column.lower().strip().replace("_", " "): column for column in table.columns}

    def pick(*candidates: str) -> str | None:
        for candidate in candidates:
            for normalized, original in normalized_columns.items():
                if candidate in normalized:
                    return original
        return None

    date_col = pick("date", "posted", "transaction")
    vendor_col = pick("vendor", "merchant", "payee", "name")
    description_col = pick("description", "memo", "details", "note")
    category_col = pick("category", "class")
    amount_col = pick("amount", "total", "net")
    debit_col = pick("debit", "withdrawal", "charge")
    credit_col = pick("credit", "deposit", "income")

    if amount_col:
        amounts = table[amount_col].map(parse_amount)
    elif debit_col or credit_col:
        debits = table[debit_col].map(parse_amount) if debit_col else 0
        credits = table[credit_col].map(parse_amount) if credit_col else 0
        amounts = pd.Series(credits, index=table.index).fillna(0) - pd.Series(debits, index=table.index).fillna(0)
    else:
        numeric_columns = table.select_dtypes(include="number").columns.tolist()
        if not numeric_columns:
            return pd.DataFrame(columns=LEDGER_COLUMNS)
        amounts = table[numeric_columns[0]].map(parse_amount)

    dates = pd.to_datetime(table[date_col], errors="coerce").dt.date if date_col else date.today()
    vendors = table[vendor_col].fillna("Imported vendor") if vendor_col else "Imported vendor"
    descriptions = table[description_col].fillna("Imported transaction") if description_col else "Imported transaction"
    categories = table[category_col].fillna("Uncategorized") if category_col else "Uncategorized"

    ledger = pd.DataFrame(
        {
            "date": dates,
            "vendor": vendors,
            "description": descriptions,
            "category": categories,
            "subcategory": "",
            "amount": amounts,
            "type": ["Income" if amount >= 0 else "Expense" for amount in amounts],
            "tax_status": "Needs review",
            "source": name,
        },
        columns=LEDGER_COLUMNS,
    )
    valid_categories = set(CATEGORY_OPTIONS[1:])
    ledger["category"] = ledger["category"].where(ledger["category"].isin(valid_categories), "Uncategorized")
    return ledger.dropna(subset=["date"])


def parse_amount(value: Any) -> float:
    """Parse common accounting currency values."""

    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    negative = text.startswith("(") and text.endswith(")")
    text = text.replace("(", "").replace(")", "")
    cleaned = "".join(char for char in text if char.isdigit() or char in ".-")
    if cleaned in {"", "-", "."}:
        return 0.0
    amount = float(cleaned)
    return -abs(amount) if negative else amount


def extract_pdf_summary(raw: bytes) -> tuple[str, int]:
    """Extract lightweight PDF text metadata without requiring backend OCR."""

    reader = PdfReader(io.BytesIO(raw))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        return "PDF received. No extractable text was found; route this file to OCR or manual review.", len(reader.pages)
    summary = " ".join(text.split())[:420]
    return summary, len(reader.pages)


def current_month_ledger() -> pd.DataFrame:
    """Return rows for the current month, falling back to latest ledger month."""

    ledger = st.session_state.ledger.copy()
    ledger["date"] = pd.to_datetime(ledger["date"], errors="coerce")
    today = pd.Timestamp(date.today())
    current = ledger[(ledger["date"].dt.year == today.year) & (ledger["date"].dt.month == today.month)]
    if not current.empty or ledger["date"].dropna().empty:
        return current

    latest = ledger["date"].max()
    return ledger[(ledger["date"].dt.year == latest.year) & (ledger["date"].dt.month == latest.month)]


def dashboard_metrics() -> dict[str, float]:
    """Calculate high-level metrics for the dashboard."""

    month = current_month_ledger()
    expenses = month.loc[month["type"] == "Expense", "amount"].sum()
    income = month.loc[month["type"] == "Income", "amount"].sum()
    monthly_burn = abs(float(expenses))
    tax_payments = abs(float(month.loc[month["category"] == "Taxes", "amount"].sum()))
    net_operational_expenses = max(monthly_burn - tax_payments, 0.0)
    net_profit = float(income) - monthly_burn
    monthly_tax_target = max(net_profit, 0.0) * st.session_state.target_tax_rate
    runway = st.session_state.tax_reserve_balance / monthly_tax_target if monthly_tax_target else 0.0
    return {
        "monthly_burn": monthly_burn,
        "net_operational_expenses": net_operational_expenses,
        "tax_savings_runway": runway,
        "income": float(income),
        "net_profit": net_profit,
        "monthly_tax_target": monthly_tax_target,
    }


def filtered_ledger(period: str, transaction_type: str, categories: list[str], needs_review_only: bool) -> pd.DataFrame:
    """Apply Ledger Management filters."""

    ledger = st.session_state.ledger.copy()
    ledger["date"] = pd.to_datetime(ledger["date"], errors="coerce")

    if period == "Current month":
        today = pd.Timestamp(date.today())
        ledger = ledger[(ledger["date"].dt.year == today.year) & (ledger["date"].dt.month == today.month)]
    elif period == "Year to date":
        today = pd.Timestamp(date.today())
        ledger = ledger[(ledger["date"].dt.year == today.year) & (ledger["date"] <= today)]

    if transaction_type:
        ledger = ledger[ledger["type"] == transaction_type]
    if categories:
        ledger = ledger[ledger["category"].isin(categories)]
    if needs_review_only:
        ledger = ledger[ledger["tax_status"] == "Needs review"]

    ledger["date"] = ledger["date"].dt.date
    return ledger


# -----------------------------------------------------------------------------
# Sidebar and pages
# -----------------------------------------------------------------------------


def render_sidebar() -> str:
    """Render structural navigation and return the active section."""

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-title">N-Deavour Alignment</div>
                <div class="sidebar-brand-meta">Private automated finance workspace.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected = st.selectbox(
            "Section",
            APP_SECTIONS,
            index=None,
            placeholder="Select workspace section",
            help="Switch between the four financial workspace sections.",
        )
        if selected:
            st.session_state.active_section = selected

        st.divider()
        st.markdown("**Tax model placeholder**")
        st.session_state.target_tax_rate = st.slider(
            "Reserve rate",
            min_value=0,
            max_value=50,
            value=int(st.session_state.target_tax_rate * 100),
            format="%d%%",
        ) / 100
        st.session_state.tax_reserve_balance = st.number_input(
            "Tax reserve balance",
            min_value=0.0,
            value=float(st.session_state.tax_reserve_balance),
            step=250.0,
            format="%.2f",
        )

    return selected


def render_app_header() -> None:
    """Render the persistent product header."""

    left, center, right = st.columns([0.08, 0.84, 0.08])
    with center:
        st.markdown(
            """
            <div class="ndeavour-kicker">Private financial alignment</div>
            <h1 class="ndeavour-title">N-Deavour Alignment</h1>
            <p class="ndeavour-subtitle">
                A calm command surface for expense classification, tax runway monitoring,
                document ingestion, and natural-language bookkeeping review.
            </p>
            """,
            unsafe_allow_html=True,
        )


def render_dashboard() -> None:
    """Render the high-level overview page with minimalist metrics."""

    section_header(
        "Dashboard",
        "Operational overview",
        "A sparse, high-signal view of this month's cash activity and tax readiness.",
    )
    metrics = dashboard_metrics()

    left_pad, col_a, col_b, col_c, right_pad = st.columns([0.05, 1, 1, 1, 0.05], gap="large")
    with col_a:
        metric_tile(
            "Total monthly burn",
            format_usd(metrics["monthly_burn"]),
            "All current-month outflows, including tax payments and operating costs.",
        )
    with col_b:
        metric_tile(
            "Net operational expenses",
            format_usd(metrics["net_operational_expenses"]),
            "Current-month burn excluding tax payments and owner-level transfers.",
        )
    with col_c:
        metric_tile(
            "Tax Savings Runway",
            f"{metrics['tax_savings_runway']:.1f} mo",
            f"Based on {format_usd(metrics['monthly_tax_target'])} monthly target reserve.",
        )

    section_header(
        "Signals",
        "Model assumptions",
        "These values are mock-backed until live bookkeeping and tax services are connected.",
    )
    signal_cols = st.columns([1, 1, 1], gap="large")
    with signal_cols[0]:
        st.markdown('<div class="card"><span class="accent-badge">Income</span><br><br>' + format_usd(metrics["income"]) + '</div>', unsafe_allow_html=True)
    with signal_cols[1]:
        st.markdown('<div class="card"><span class="accent-badge">Net profit</span><br><br>' + format_usd(metrics["net_profit"]) + '</div>', unsafe_allow_html=True)
    with signal_cols[2]:
        st.markdown('<div class="card"><span class="accent-badge">Reserve balance</span><br><br>' + format_usd(st.session_state.tax_reserve_balance) + '</div>', unsafe_allow_html=True)


def render_document_ingestion() -> None:
    """Render the focused upload workflow for CSV, TSV, and PDF files."""

    section_header(
        "Document Ingestion",
        "Upload financial source files",
        "Drag in normalized exports or statements. Files remain session-scoped in this mock front end.",
    )

    left, main, right = st.columns([0.12, 0.76, 0.12])
    with main:
        uploads = st.file_uploader(
            "Financial source files",
            type=["csv", "tsv", "pdf"],
            accept_multiple_files=True,
            help="Supported formats: CSV, TSV, and PDF.",
        )

        with st.expander("**Ingestion standards**", expanded=False):
            st.markdown(
                """
                - CSV/TSV files should include date and amount fields whenever possible.
                - Recommended columns: `date`, `vendor`, `description`, `category`, and `amount`.
                - Debits may be represented as negative amounts or split into debit/credit columns.
                - PDF files are stored as document memory and summarized from extractable text.
                - Sensitive files should be reviewed before connecting any external processing service.
                """
            )

        if uploads:
            ingest_uploads(uploads)

        if st.session_state.uploaded_documents:
            section_header("Stored", "Uploaded document memory", "Click into each upload for source details and extraction notes.")
            for document in st.session_state.uploaded_documents:
                with st.expander(f"**{document['name']}** - {document['kind']}", expanded=False):
                    st.write(document["summary"])
                    meta = pd.DataFrame(
                        [
                            ["Uploaded", document["uploaded_at"]],
                            ["Rows added", document["rows_added"]],
                            ["Source type", document["kind"]],
                        ],
                        columns=["Field", "Value"],
                    )
                    st.dataframe(meta, use_container_width=True, hide_index=True)


def ingest_uploads(uploads: list[Any]) -> None:
    """Process new upload streams and update ledger/document state."""

    for upload in uploads:
        key = f"{upload.name}:{upload.size}"
        if key in st.session_state.processed_upload_keys:
            continue

        raw = upload.getvalue()
        extension = upload.name.rsplit(".", 1)[-1].lower()
        rows_added = 0
        if extension in {"csv", "tsv"}:
            separator = "\t" if extension == "tsv" else ","
            parsed = normalize_uploaded_table(upload.name, raw, separator)
            rows_added = len(parsed)
            if rows_added:
                st.session_state.ledger = pd.concat([st.session_state.ledger, parsed], ignore_index=True)
            summary = f"Parsed {rows_added:,} ledger rows from tabular upload."
        else:
            summary, page_count = extract_pdf_summary(raw)
            summary = f"PDF pages: {page_count}. Extracted summary: {summary}"

        st.session_state.uploaded_documents.append(
            {
                "name": upload.name,
                "kind": extension.upper(),
                "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "rows_added": rows_added,
                "summary": summary,
            }
        )
        st.session_state.processed_upload_keys.add(key)
        st.success(f"Ingested {upload.name}")


def render_ledger_management() -> None:
    """Render interactive ledger auditing controls and expandable tables."""

    section_header(
        "Ledger Management",
        "Audit and classify transactions",
        "Review sensitive financial records with low-density filters and explicit manual categorization controls.",
    )

    with st.expander("**Filters**", expanded=True):
        period_col, type_col, category_col, review_col = st.columns([1, 1, 1.4, 1], gap="large")
        with period_col:
            period = st.selectbox("Period", ["", "Current month", "Year to date", "All records"], index=0, placeholder="Select period")
            period = "All records" if period == "" else period
        with type_col:
            transaction_type = st.selectbox("Transaction type", TYPE_OPTIONS, index=0, placeholder="Select transaction type")
        with category_col:
            categories = st.multiselect("Categories", CATEGORY_OPTIONS[1:], default=[], placeholder="Select categories")
        with review_col:
            needs_review_only = st.toggle("Needs review only", value=False)

    visible = filtered_ledger(period, transaction_type, categories, needs_review_only)

    with st.expander("**Ledger table**", expanded=True):
        edited = st.data_editor(
            visible,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "category": st.column_config.SelectboxColumn("Category", options=CATEGORY_OPTIONS, required=False),
                "subcategory": st.column_config.SelectboxColumn(
                    "Subcategory",
                    options=sorted({item for options in SUBCATEGORY_OPTIONS.values() for item in options}),
                    required=False,
                ),
                "type": st.column_config.SelectboxColumn("Type", options=TYPE_OPTIONS, required=False),
                "tax_status": st.column_config.SelectboxColumn("Tax status", options=TAX_STATUS_OPTIONS, required=False),
                "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            },
            key="ledger_editor",
        )
        if st.button("Apply visible table edits"):
            apply_ledger_edits(edited)
            st.success("Visible ledger edits applied.")
            st.rerun()

    with st.expander("**Manual audit controls**", expanded=False):
        candidate_labels = [""] + [
            f"{idx} | {row['date']} | {row['vendor']} | {format_usd(float(row['amount']))}"
            for idx, row in visible.iterrows()
        ]
        selected_label = st.selectbox("Transaction", candidate_labels, index=0, placeholder="Select transaction")
        selected_index = int(selected_label.split(" | ", 1)[0]) if selected_label else None

        audit_cols = st.columns([1, 1, 1], gap="large")
        with audit_cols[0]:
            new_category = st.selectbox("Category", CATEGORY_OPTIONS, index=0, key="manual_category", placeholder="Select category")
        subcategory_options = SUBCATEGORY_OPTIONS.get(new_category, [""])
        with audit_cols[1]:
            new_subcategory = st.selectbox("Subcategory", subcategory_options, index=0, key="manual_subcategory", placeholder="Select subcategory")
        with audit_cols[2]:
            new_tax_status = st.selectbox("Tax status", TAX_STATUS_OPTIONS, index=0, key="manual_tax_status", placeholder="Select tax status")

        if st.button("Apply manual classification"):
            if selected_index is None:
                st.warning("Select a transaction before applying manual classification.")
            else:
                if new_category:
                    st.session_state.ledger.loc[selected_index, "category"] = new_category
                if new_subcategory:
                    st.session_state.ledger.loc[selected_index, "subcategory"] = new_subcategory
                if new_tax_status:
                    st.session_state.ledger.loc[selected_index, "tax_status"] = new_tax_status
                st.success("Manual classification applied.")
                st.rerun()


def apply_ledger_edits(edited: pd.DataFrame) -> None:
    """Apply edits from the visible data editor back to session ledger."""

    for index, row in edited.iterrows():
        if index in st.session_state.ledger.index:
            for column in LEDGER_COLUMNS:
                if column in edited.columns:
                    st.session_state.ledger.loc[index, column] = row[column]


def render_agent_workspace() -> None:
    """Render a full-width minimal chat interface for financial prompting."""

    section_header(
        "Agent Workspace",
        "Prompt against your financial model",
        "A quiet natural-language workspace for reviewing burn, runway, categorization gaps, and tax posture.",
    )

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about runway, deductions, burn, or ledger cleanup")
    if not prompt:
        return

    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = generate_mock_agent_response(prompt)
    st.session_state.chat_messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)


def generate_mock_agent_response(prompt: str) -> str:
    """Generate a deterministic mock response from local financial state."""

    metrics = dashboard_metrics()
    needs_review = int((st.session_state.ledger["tax_status"] == "Needs review").sum())
    prompt_lower = prompt.lower()

    if "tax" in prompt_lower or "runway" in prompt_lower:
        return (
            f"Tax reserve balance is {format_usd(st.session_state.tax_reserve_balance)}. "
            f"At the current modeled monthly reserve need of {format_usd(metrics['monthly_tax_target'])}, "
            f"runway is approximately {metrics['tax_savings_runway']:.1f} months."
        )
    if "burn" in prompt_lower or "expense" in prompt_lower:
        return (
            f"Current monthly burn is {format_usd(metrics['monthly_burn'])}. "
            f"Net operational expenses are {format_usd(metrics['net_operational_expenses'])} after excluding tax payments."
        )
    if "review" in prompt_lower or "categor" in prompt_lower:
        return f"There are {needs_review} transactions marked as needing review. Start in Ledger Management, then filter by Needs review only."
    return (
        "I can help inspect monthly burn, tax runway, deductible classifications, and ledger cleanup. "
        f"Current modeled net profit is {format_usd(metrics['net_profit'])}."
    )


# -----------------------------------------------------------------------------
# App entrypoint
# -----------------------------------------------------------------------------


def main() -> None:
    """Run the Streamlit application."""

    st.set_page_config(page_title="N-Deavour Alignment", page_icon=":bar_chart:", layout="wide")
    init_state()
    apply_design_system()
    active_section = render_sidebar()
    render_app_header()

    if active_section == "Dashboard":
        render_dashboard()
    elif active_section == "Document Ingestion":
        render_document_ingestion()
    elif active_section == "Ledger Management":
        render_ledger_management()
    elif active_section == "Agent Workspace":
        render_agent_workspace()


if __name__ == "__main__":
    main()
