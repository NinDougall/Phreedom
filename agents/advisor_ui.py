"""AdvisorUI — frontend messaging loop for the Phreedom chat agent.

Responsibilities
----------------
- Maintain the Phreedom persona and the minimalistic N-Deavour tone system.
- Route user prompts to OpenAI (when a key is available) or the deterministic
  keyword-based fallback.
- Post-process LLM responses to enforce brand voice constraints (concise,
  data-grounded, no fluff).
- Persist the full conversation to disk after every exchange via the bridge
  supplied by the Orchestrator.

Tone system (N-Deavour minimalism)
-----------------------------------
- Short, declarative sentences.
- Numbers before narrative.
- No filler phrases ("certainly", "great question", "of course").
- Bold the key figure in every reply.
- End with exactly one actionable suggestion when relevant.

Design constraints
------------------
- Zero Streamlit imports — this module is UI-framework-agnostic.
- The ``generate_response()`` method is the only entry point the Orchestrator
  needs; all internal routing is transparent to callers.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Tone enforcement
# ---------------------------------------------------------------------------

# Phrases that contradict the N-Deavour minimalist voice.  Any occurrence in
# an LLM response is stripped or replaced with an empty string.
_FILLER_PHRASES: tuple[str, ...] = (
    "certainly!", "certainly,", "certainly ",
    "great question!", "great question,",
    "of course!", "of course,",
    "absolutely!", "absolutely,",
    "sure thing!", "sure,",
    "i'd be happy to", "i would be happy to",
    "feel free to", "don't hesitate to",
    "as an ai", "as a language model",
)


def _enforce_tone(text: str) -> str:
    """Strip filler phrases that violate the N-Deavour minimalist tone."""
    lowered = text.lower()
    for phrase in _FILLER_PHRASES:
        start = lowered.find(phrase)
        if start != -1:
            text = text[:start] + text[start + len(phrase):]
            lowered = text.lower()
    return text.strip()


# ---------------------------------------------------------------------------
# AdvisorUI
# ---------------------------------------------------------------------------


class AdvisorUI:
    """Manages the frontend messaging loop with the Phreedom persona.

    Parameters
    ----------
    bridge:
        Optional ``StorageBridge`` for persisting chat history after each
        exchange.  Pass ``None`` to disable persistence (e.g. in tests).
    """

    SYSTEM_PROMPT = (
        "You are Phreedom, a careful personal financial agent for N-Deavour Services. "
        "Rules: (1) Use only the provided financial context — never invent numbers. "
        "(2) Be concise: lead with the key figure, follow with one sentence of context. "
        "(3) Bold the most important number in every response. "
        "(4) End with one short actionable suggestion when it adds value. "
        "(5) No filler phrases (no 'certainly', 'great question', 'of course'). "
        "(6) This is planning guidance — always append a brief disclaimer when giving "
        "tax estimates."
    )

    def __init__(self, bridge: Any = None) -> None:
        self._bridge = bridge

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_response(
        self,
        prompt: str,
        context: str,
        conversation: list[dict[str, str]],
        tax_rate: float,
        ledger: pd.DataFrame,
    ) -> str:
        """Generate a Phreedom reply for the given user prompt.

        Routing logic
        -------------
        1. If an OpenAI API key is available, forward to ``gpt-4o-mini``
           (or ``OPENAI_MODEL`` env var) with the financial context injected
           as a system message.
        2. Otherwise, fall back to the deterministic keyword-based responder.

        In both cases the response is post-processed by ``_enforce_tone()``.

        Parameters
        ----------
        prompt:
            The user's latest message.
        context:
            Financial context string produced by ``TaxEngine.build_context()``.
        conversation:
            Recent message history (list of ``{"role": ..., "content": ...}`` dicts).
            The last 10 messages are sent to the LLM to bound token usage.
        tax_rate:
            Current tax reserve rate (decimal fraction).
        ledger:
            Full ledger DataFrame — passed to the fallback responder.

        Returns
        -------
        str
            Phreedom's response, tone-checked and ready for display.
        """
        api_key = self._get_openai_key()
        if not api_key:
            return _enforce_tone(self._fallback_response(prompt, context, tax_rate, ledger))

        try:
            from openai import OpenAI  # lazy import — optional dependency

            client = OpenAI(api_key=api_key)
            recent = list(conversation[-10:])
            if not recent or recent[-1].get("content") != prompt:
                recent = [*recent, {"role": "user", "content": prompt}]

            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "system", "content": f"Financial memory:\n{context}"},
                    *recent,
                ],
                temperature=0.2,
            )
            raw = response.choices[0].message.content or ""
            return _enforce_tone(raw) or _enforce_tone(
                self._fallback_response(prompt, context, tax_rate, ledger)
            )
        except Exception as exc:
            fallback = _enforce_tone(self._fallback_response(prompt, context, tax_rate, ledger))
            return f"{fallback}\n\n*(OpenAI unavailable: {exc})*"

    def persist_conversation(self, messages: list[dict[str, str]]) -> None:
        """Write the current conversation to disk via the bridge.

        Called by the Orchestrator after every chat exchange checkpoint.
        """
        if self._bridge is not None:
            self._bridge.save_chat_history(messages)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_openai_key() -> str | None:
        """Read the OpenAI API key from Streamlit secrets or the environment."""
        key: str | None = None
        try:
            import streamlit as st
            key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[union-attr]
        except Exception:
            pass
        return key or os.getenv("OPENAI_API_KEY")

    @staticmethod
    def _fallback_response(
        prompt: str,
        context: str,  # noqa: ARG004 — reserved for future keyword injection
        tax_rate: float,
        ledger: pd.DataFrame,
    ) -> str:
        """Keyword-based deterministic responder (no external API required)."""
        from agents.tax_engine import TaxEngine

        engine = TaxEngine()
        summary = engine.compute_summary(ledger, tax_rate)
        p = prompt.lower()

        if any(w in p for w in ("tax", "reserve", "set aside", "owe")):
            if summary.profit > 0:
                return (
                    f"Net profit is **${summary.profit:,.2f}**. "
                    f"At {tax_rate:.0%}, reserve **${summary.tax_reserve:,.2f}** for taxes. "
                    "This is planning guidance — consult a tax professional for formal advice."
                )
            return "No positive profit loaded. Upload income records to calculate a tax reserve."

        if any(w in p for w in ("income", "revenue", "earn")):
            return (
                f"Total income: **${summary.income:,.2f}** "
                f"across {summary.transactions} transactions."
            )

        if any(w in p for w in ("expense", "spend", "cost", "burn")):
            if summary.top_expenses.empty:
                return "No expense transactions loaded."
            top_text = "\n".join(
                f"- **{r['category']}**: ${r['amount']:,.2f}"
                for _, r in summary.top_expenses.iterrows()
            )
            return f"Total expenses: **${summary.expenses:,.2f}**.\n\nLargest categories:\n{top_text}"

        if any(w in p for w in ("profit", "net", "margin")):
            return (
                f"Net profit: **${summary.profit:,.2f}**. "
                f"Income: ${summary.income:,.2f} | Expenses: ${summary.expenses:,.2f}."
            )

        if any(w in p for w in ("upload", "document", "file", "vault")):
            return (
                "Ask about income, expenses, taxes, or net profit. "
                "Upload CSV or PDF files from the Dashboard to load transactions."
            )

        return (
            f"**${summary.income:,.2f}** income · **${summary.expenses:,.2f}** expenses · "
            f"**${summary.profit:,.2f}** net profit. "
            "Ask about taxes, expenses, income, or uploaded files."
        )
