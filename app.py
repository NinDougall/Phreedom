"""N-Deavour Alignment — personal financial agent (multi-agent edition).

Architecture
────────────
  agents.OrchestratorAgent   central manager — delegates to workers
    ├── IngestionWorker       parse / vault / dedup (file uploads)
    ├── TaxEngine             financial metrics, tax reserve, context
    └── AdvisorUI             Phreedom chat persona, OpenAI routing

  storage_bridge.StorageBridge   persistent local vault + manifest
  init_state()                   loads disk → session on cold start
  render_*()                     pure presentation – no I/O except through orchestrator
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

from agents import OrchestratorAgent
from storage_bridge import get_bridge


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEDGER_COLUMNS = ["date", "description", "amount", "kind", "category", "source"]
TIMESHEET_COLUMNS = ["date", "project", "hours", "rate", "total_pay"]
APP_PAGES = ["Dashboard", "Timesheet", "Chat"]


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


def _get_orchestrator() -> OrchestratorAgent:
    """Return the process-level OrchestratorAgent singleton."""
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = OrchestratorAgent(get_bridge())
    return st.session_state.orchestrator


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

    # File staging for explicit Submit flow on the Timesheet page
    st.session_state.ts_uploader_rev = 0
    st.session_state.ts_staged_file = None

    st.session_state.profile_business_type = profile.get("business_type", "")
    st.session_state.profile_tax_rate = float(profile.get("tax_reserve_rate", 0.30))
    st.session_state.profile_tax_notes = profile.get("tax_notes", "")


# ---------------------------------------------------------------------------
# Disk persistence helpers (thin wrappers — real I/O is in the bridge)
# ---------------------------------------------------------------------------


def _flush_ledger() -> None:
    get_bridge().save_ledger(st.session_state.ledger)


def _flush_timesheet() -> None:
    get_bridge().save_timesheet(st.session_state.timesheet)


def _flush_chat() -> None:
    get_bridge().save_chat_history(st.session_state.messages)


# ---------------------------------------------------------------------------
# Timesheet calculations (kept in app.py — UI-only, no agent needed)
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
    st.session_state.timesheet = pd.concat(
        [st.session_state.timesheet, ts_row], ignore_index=True
    )
    _flush_timesheet()

    ledger_row = pd.DataFrame(
        [{"date": entry_date, "description": f"Timesheet earnings: {project_name}",
          "amount": total_pay, "kind": "income", "category": "Billable income",
          "source": "Timesheet"}],
        columns=LEDGER_COLUMNS,
    )
    st.session_state.ledger = pd.concat(
        [st.session_state.ledger, ledger_row[LEDGER_COLUMNS]], ignore_index=True
    )
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
# Profile controls
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
    orchestrator = _get_orchestrator()

    # ── Handle pending actions (must run before render) ─────────────────
    if "_vault_delete" in st.session_state:
        file_hash = st.session_state.pop("_vault_delete")
        docs = bridge.list_documents()
        orig_name = next((d["original_name"] for d in docs if d["hash"] == file_hash), None)
        bridge.delete_document(file_hash)
        if orig_name:
            st.session_state.ledger = st.session_state.ledger[
                st.session_state.ledger["source"] != orig_name
            ]
            _flush_ledger()
        _status("Document deleted from vault.")
        st.rerun()

    if "_vault_reanalyze" in st.session_state:
        file_hash = st.session_state.pop("_vault_reanalyze")
        content = bridge.fetch_document(file_hash)
        docs = bridge.list_documents()
        entry = next((d for d in docs if d["hash"] == file_hash), None)
        if content and entry:
            orig_name = entry["original_name"]
            result = orchestrator.handle_reparse(
                orig_name, content, st.session_state.ledger
            )
            st.session_state.ledger = result["ledger"]
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

        with st.expander(f"**{name}** — {rows_label}", expanded=False):
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


def render_dashboard(tax_rate: float) -> None:
    from agents.tax_engine import TaxEngine

    summary = TaxEngine().compute_summary(st.session_state.ledger, tax_rate)

    _section("Financial overview", "Snapshot")
    _divider_space(0.25)
    _, c1, c2, c3, c4, _ = st.columns([0.04, 1, 1, 1, 1, 0.04], gap="large")
    with c1: _kpi("Income", format_usd(summary.income))
    with c2: _kpi("Expenses", format_usd(summary.expenses))
    with c3: _kpi("Net profit", format_usd(summary.profit), summary.profit)
    with c4: _kpi("Tax reserve", format_usd(summary.tax_reserve))
    _divider_space(0.5)

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
            orchestrator = _get_orchestrator()
            for f in uploaded:
                with st.spinner(f"Vaulting {f.name}…"):
                    result = orchestrator.handle_upload(
                        f.name, f.getvalue(), st.session_state.ledger
                    )
                if result["was_new"]:
                    st.session_state.ledger = result["ledger"]
                    st.success(result["message"])
                else:
                    st.info(result["message"])

        with st.expander("Ingestion standards", expanded=False):
            st.markdown(
                """
- CSV files need **date** and **amount** columns (debit/credit splits supported).
- PDFs are text-extracted and scanned for dated monetary rows.
- Files are SHA-256 hashed before storage — exact duplicates are rejected.
- All vault files persist in `.ndeavour_profile/secure_vault/` between sessions.
                """
            )

    render_permanent_registry()

    _section("Tax savings", "Planning")
    if summary.profit > 0:
        _, tax_col, _ = st.columns([0.04, 0.5, 0.46])
        with tax_col:
            _kpi("Suggested tax reserve", format_usd(summary.tax_reserve))
        st.markdown(
            f"<p class='nd-note'>Set aside <strong>{format_usd(summary.tax_reserve)}</strong> "
            f"from net profit of <strong>{format_usd(summary.profit)}</strong> "
            f"at your {tax_rate:.0%} reserve rate.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No positive net profit loaded. Upload income records or add timesheet entries.")
    st.caption("Planning guidance only — not formal tax, legal, or accounting advice.")

    with st.expander("📋  Full transaction ledger", expanded=False):
        if st.session_state.ledger.empty:
            st.info("Upload a CSV/PDF or add timesheet entries to build your ledger.")
        else:
            st.dataframe(st.session_state.ledger, use_container_width=True, hide_index=True)
            if not summary.top_expenses.empty:
                st.markdown("**Largest expense categories**")
                st.bar_chart(summary.top_expenses.set_index("category"))
            st.download_button(
                "Download ledger CSV",
                st.session_state.ledger.to_csv(index=False).encode(),
                file_name=f"phreedom-ledger-{datetime.now().date()}.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# Page: Timesheet
# ---------------------------------------------------------------------------


def _render_timesheet_table(
    df: pd.DataFrame, key: str, download_label: str, download_fname: str
) -> None:
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
            from agents.ingestion_worker import parse_timesheet_csv as _parse_ts_csv
            file_key = f"{ts_upload.name}:{ts_upload.size}"
            staged = st.session_state.ts_staged_file
            if staged is None or staged.get("key") != file_key:
                content = ts_upload.getvalue()
                preview_df, preview_summary = _parse_ts_csv(ts_upload.name, content)
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
                        orchestrator = _get_orchestrator()
                        with st.spinner("Saving to vault…"):
                            result = orchestrator.handle_upload(
                                staged["name"], staged["content"], st.session_state.ledger
                            )
                        st.session_state.ts_staged_file = None
                        st.session_state.ts_uploader_rev += 1
                        if result["was_new"]:
                            st.session_state.ledger = result["ledger"]
                            if result.get("is_timesheet") and result.get("timesheet") is not None:
                                new_ts = result["timesheet"]
                                existing_ts = st.session_state.timesheet
                                if not existing_ts.empty:
                                    existing_keys = set(
                                        zip(
                                            existing_ts["date"].astype(str),
                                            existing_ts["project"].astype(str),
                                            existing_ts["hours"].astype(str),
                                        )
                                    )
                                    new_ts = new_ts[
                                        ~new_ts.apply(
                                            lambda r: (
                                                str(r["date"]), str(r["project"]), str(r["hours"])
                                            ) in existing_keys,
                                            axis=1,
                                        )
                                    ]
                                if not new_ts.empty:
                                    st.session_state.timesheet = pd.concat(
                                        [st.session_state.timesheet, new_ts], ignore_index=True
                                    )
                                    _flush_timesheet()
                            st.success(result["message"])
                            _status(f"Timesheet submitted: {result['message']}")
                        else:
                            st.info(result["message"])
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
    with s1: _kpi("Hours Ahead/Behind",   f"{dash['hours_gap']:+,.2f}", dash["hours_gap"])
    with s2: _kpi("Earnings vs Pace",     format_usd(dash["earnings_to_date_gap"]), dash["earnings_to_date_gap"])
    with s3: _kpi("vs Base Rate",         format_usd(dash["earnings_vs_base_rate"]), dash["earnings_vs_base_rate"])
    with s4: _kpi("Prev Month Hrs Delta", f"{dash['prev_month_hours_gap']:+,.2f}", dash["prev_month_hours_gap"])

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
    with sm1: _kpi("Total hours",       f"{dash['actual_hours']:,.2f}")
    with sm2: _kpi("Avg billable rate", format_usd(dash["avg_rate"]))
    with sm3: _kpi("Total pay",         format_usd(dash["actual_pay"]))

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
            orchestrator = _get_orchestrator()
            answer = orchestrator.handle_chat(
                prompt=prompt,
                conversation=st.session_state.messages[:-1],
                profile={
                    "tax_rate": tax_rate,
                    "business_type": business_type,
                    "tax_notes": tax_notes,
                },
                ledger=st.session_state.ledger,
            )
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

    if active_page == "Dashboard":
        render_dashboard(tax_rate)
    elif active_page == "Timesheet":
        render_timesheet(tax_rate)
    elif active_page == "Chat":
        render_chat_page(tax_rate, business_type, tax_notes)

    _close_main()


if __name__ == "__main__":
    main()
