"""N-Deavour corporate values framework — Integrity & Curiosity alignment engines."""

from __future__ import annotations

from typing import Any

import streamlit as st

WHY = (
    "We exist to remove administrative friction so local operators can protect "
    "human sovereignty and reinvest attention in work that genuinely serves community well-being."
)

VISION = (
    "A financial operating layer where every contract, ledger entry, and insight "
    "reinforces ethical commerce — quietly, affordably, and at human scale."
)

MISSION = (
    "Deliver precise bookkeeping, tax visibility, and decision support that keeps "
    "integrity non-negotiable and curiosity longer than convenience."
)

VALUES_INTEGRITY = "Integrity"
VALUES_CURIOSITY = "Curiosity"

ALIGNMENT_CHECKLIST_LABEL = (
    "Does this entity's underlying intent align with the collective well-being of "
    "the community it serves? If it exploits or compromises human sovereignty, "
    "reject the contract."
)

NON_ALIGNED_LABEL = "NON-ALIGNED PROCESS"
CURIOSITY_PROMPT = (
    "Stay curious longer. Before changing structure or baselines, document what "
    "you observed — context, constraints, and what the variance might mean."
)

SLATE_HIGHLIGHT = "#64748B"
ND_VOID = "#0a0514"
ND_LAVENDER = "#d8b4fe"
ND_GLASS = "rgba(15, 8, 29, 0.72)"


def render_alignment_blueprint() -> None:
    """Collapsible high-whitespace brand banner for sidebar or header."""
    with st.expander("Alignment Blueprint", expanded=False):
        st.markdown(
            f"""
<div style="background:{ND_GLASS};border:1px solid rgba(216,180,254,0.3);border-radius:0.85rem;
padding:1.5rem 1.75rem;margin:0;color:#ffffff;line-height:1.7;backdrop-filter:blur(8px);">
<p style="font-size:0.65rem;letter-spacing:0.35em;text-transform:uppercase;
color:{ND_LAVENDER};margin:0 0 1rem;font-weight:300;">N-Deavour Services</p>
<p style="margin:0 0 1.25rem;"><strong style="color:#ffffff;font-weight:300;">Why</strong><br>{WHY}</p>
<p style="margin:0 0 1.25rem;"><strong style="color:#ffffff;font-weight:300;">Vision</strong><br>{VISION}</p>
<p style="margin:0 0 1.25rem;"><strong style="color:#ffffff;font-weight:300;">Mission</strong><br>{MISSION}</p>
<p style="margin:0;font-size:0.8rem;color:{SLATE_HIGHLIGHT};">
<span style="color:{ND_LAVENDER};font-weight:400;">{VALUES_INTEGRITY}</span>
&nbsp;·&nbsp;
<span style="color:{ND_LAVENDER};font-weight:400;">{VALUES_CURIOSITY}</span>
</p>
</div>
            """,
            unsafe_allow_html=True,
        )


def render_go_nogo_checklist(key_prefix: str, *, label: str | None = None) -> bool:
    """Mandatory integrity gateway. Returns True when the operator affirms alignment."""
    st.markdown(
        f'<p style="font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;'
        f'color:{ND_LAVENDER};font-weight:300;margin:0 0 0.5rem;">Go / No-Go · Integrity</p>',
        unsafe_allow_html=True,
    )
    checked = st.checkbox(
        label or ALIGNMENT_CHECKLIST_LABEL,
        key=f"nd_alignment_{key_prefix}",
        value=st.session_state.get(f"nd_alignment_{key_prefix}", False),
    )
    st.session_state[f"nd_alignment_{key_prefix}"] = checked
    if not checked:
        st.markdown(
            f'<div style="background:{SLATE_HIGHLIGHT};color:{ND_VOID};padding:0.65rem 1rem;'
            f'border-radius:0.55rem;font-size:0.8rem;font-weight:600;letter-spacing:0.08em;'
            f'text-transform:uppercase;margin-top:0.5rem;">{NON_ALIGNED_LABEL}</div>',
            unsafe_allow_html=True,
        )
    return checked


def flag_non_aligned_process(key: str, name: str) -> None:
    registry: dict[str, Any] = st.session_state.setdefault("nd_non_aligned_registry", {})
    registry[key] = {"name": name, "status": NON_ALIGNED_LABEL}


def is_non_aligned(key: str) -> bool:
    return key in st.session_state.get("nd_non_aligned_registry", {})


def detect_expense_variance(
    current_expenses: float,
    previous_expenses: float | None,
    *,
    threshold_ratio: float = 0.12,
) -> bool:
    if previous_expenses is None or previous_expenses <= 0:
        return False
    delta = abs(current_expenses - previous_expenses) / previous_expenses
    return delta >= threshold_ratio


def activate_curiosity_gate(reason: str) -> None:
    st.session_state.nd_curiosity_required = True
    st.session_state.nd_curiosity_reason = reason


def curiosity_is_required() -> bool:
    return bool(st.session_state.get("nd_curiosity_required"))


def render_curiosity_gate(commit_key: str, *, min_chars: int = 40) -> bool:
    if not curiosity_is_required():
        return True

    reason = st.session_state.get("nd_curiosity_reason", "Unexpected variance detected.")
    with st.expander("Stay Curious Longer — required context", expanded=True):
        st.markdown(
            f'<p style="color:#ffffff;margin:0 0 0.75rem;">{CURIOSITY_PROMPT}</p>'
            f'<p style="color:{SLATE_HIGHLIGHT};font-size:0.9rem;margin:0 0 1rem;">{reason}</p>',
            unsafe_allow_html=True,
        )
        observations = st.text_area(
            "Observations before baseline change",
            key=f"nd_curiosity_obs_{commit_key}",
            placeholder="What changed? What did you verify? What remains uncertain?",
            height=120,
        )
        if len((observations or "").strip()) < min_chars:
            st.caption(f"Enter at least {min_chars} characters to unlock structural updates.")
            return False
        st.session_state.nd_curiosity_observations = observations.strip()
        if st.button("Commit observations & release gate", key=f"nd_curiosity_release_{commit_key}"):
            st.session_state.nd_curiosity_required = False
            st.session_state.nd_curiosity_reason = ""
            return True
    return False
