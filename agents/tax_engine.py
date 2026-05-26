"""TaxEngine — high-precision tax logic and financial metrics worker.

Responsibilities
----------------
- Compute income / expense / profit / tax-reserve summaries from the ledger.
- Validate the mathematical integrity of every computed metric (self-audit).
- Build structured financial context strings injected into chat prompts.
- Expose tax-category metadata from the profile manifest.
- Persist computed snapshots to the profile manifest so every run is auditable.

Design constraints
------------------
- Zero Streamlit imports — this module is UI-framework-agnostic.
- All public methods accept explicit arguments (no global state reads).
- ``validate_metrics()`` raises ``TaxValidationError`` when arithmetic is
  inconsistent, making bugs detectable before they reach the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TaxValidationError(ValueError):
    """Raised when computed financial metrics fail internal consistency checks."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FinancialSummary:
    """Immutable snapshot of computed financial metrics for a given ledger."""

    income: float
    expenses: float
    profit: float
    tax_reserve: float
    tax_rate: float
    transactions: int
    top_expenses: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot (no DataFrames)."""
        return {
            "income": self.income,
            "expenses": self.expenses,
            "profit": self.profit,
            "tax_reserve": self.tax_reserve,
            "tax_rate": self.tax_rate,
            "transactions": self.transactions,
            "top_expenses": (
                self.top_expenses.to_dict(orient="records")
                if not self.top_expenses.empty
                else []
            ),
        }


# ---------------------------------------------------------------------------
# TaxEngine
# ---------------------------------------------------------------------------


class TaxEngine:
    """High-precision worker dedicated to tax logic and metric validation.

    The engine is stateless with respect to the ledger — it receives data
    as arguments and returns structured results.  Persistence (writing the
    computed snapshot to the manifest) is handled by the Orchestrator via the
    ``record_snapshot`` method, which requires a ``StorageBridge`` instance.

    Parameters
    ----------
    bridge:
        Optional ``StorageBridge`` used by ``record_snapshot()``.  Pass
        ``None`` to use the engine in pure-compute mode (e.g. unit tests).
    """

    def __init__(self, bridge: Any = None) -> None:
        self._bridge = bridge

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_summary(
        self,
        ledger: pd.DataFrame,
        tax_rate: float,
    ) -> FinancialSummary:
        """Compute and validate the full financial summary for a ledger.

        Parameters
        ----------
        ledger:
            DataFrame with columns ``[date, description, amount, kind,
            category, source]``.
        tax_rate:
            Decimal fraction (e.g. ``0.30`` for 30 %).

        Returns
        -------
        FinancialSummary
            Validated snapshot.  Raises ``TaxValidationError`` if metrics
            are arithmetically inconsistent.
        """
        if ledger is None or ledger.empty:
            return FinancialSummary(
                income=0.0,
                expenses=0.0,
                profit=0.0,
                tax_reserve=0.0,
                tax_rate=tax_rate,
                transactions=0,
                top_expenses=pd.DataFrame(columns=["category", "amount"]),
            )

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

        summary = FinancialSummary(
            income=income,
            expenses=expenses,
            profit=profit,
            tax_reserve=tax_reserve,
            tax_rate=tax_rate,
            transactions=len(ledger),
            top_expenses=top_expenses,
        )

        self.validate_metrics(summary)
        return summary

    def validate_metrics(self, summary: FinancialSummary) -> None:
        """Assert arithmetic consistency across all metrics.

        Raises ``TaxValidationError`` on any discrepancy exceeding the
        floating-point epsilon (1e-6).

        Checks performed
        ----------------
        1. profit == income − expenses
        2. tax_reserve == max(profit, 0) × tax_rate
        3. No metric is NaN or infinite
        """
        epsilon = 1e-6

        for name, val in [
            ("income", summary.income),
            ("expenses", summary.expenses),
            ("profit", summary.profit),
            ("tax_reserve", summary.tax_reserve),
        ]:
            import math
            if math.isnan(val) or math.isinf(val):
                raise TaxValidationError(f"Metric '{name}' is {val} — data integrity error.")

        expected_profit = summary.income - summary.expenses
        if abs(summary.profit - expected_profit) > epsilon:
            raise TaxValidationError(
                f"Profit mismatch: stored={summary.profit:.6f}, "
                f"computed={expected_profit:.6f} (income={summary.income}, expenses={summary.expenses})."
            )

        expected_reserve = max(summary.profit, 0.0) * summary.tax_rate
        if abs(summary.tax_reserve - expected_reserve) > epsilon:
            raise TaxValidationError(
                f"Tax reserve mismatch: stored={summary.tax_reserve:.6f}, "
                f"computed={expected_reserve:.6f} (profit={summary.profit}, rate={summary.tax_rate})."
            )

    def build_context(
        self,
        summary: FinancialSummary,
        docs_list: list[dict[str, Any]],
        ledger: pd.DataFrame,
        business_type: str = "",
        tax_notes: str = "",
    ) -> str:
        """Assemble a financial context string for injection into chat prompts.

        Parameters
        ----------
        summary:
            Pre-computed ``FinancialSummary`` (from ``compute_summary()``).
        docs_list:
            List of vault registry entries from ``StorageBridge.list_documents()``.
        ledger:
            Full ledger DataFrame — the last 12 rows are included.
        business_type:
            User's business or income type label.
        tax_notes:
            Optional free-form tax notes from the user's profile.

        Returns
        -------
        str
            Multi-line context block ready for insertion into a system prompt.
        """
        top_expense_text = (
            summary.top_expenses.to_string(
                index=False,
                formatters={"amount": "${:,.2f}".format},
            )
            if not summary.top_expenses.empty
            else "No expenses loaded."
        )

        docs_text = (
            "\n".join(f"- {d['original_name']}: {d['summary']}" for d in docs_list)
            or "No documents uploaded."
        )

        recent = (
            ledger.tail(12).to_string(index=False)
            if ledger is not None and not ledger.empty
            else "No transactions."
        )

        return (
            f"Business type: {business_type or 'Not specified'}\n"
            f"Tax notes: {tax_notes or 'None'}\n"
            f"Tax reserve rate: {summary.tax_rate:.0%}\n"
            f"Transactions: {summary.transactions}\n"
            f"Income: ${summary.income:,.2f}\n"
            f"Expenses: ${summary.expenses:,.2f}\n"
            f"Net profit: ${summary.profit:,.2f}\n"
            f"Suggested tax reserve: ${summary.tax_reserve:,.2f}\n\n"
            f"Top expenses:\n{top_expense_text}\n\n"
            f"Document vault ({len(docs_list)} files):\n{docs_text}\n\n"
            f"Recent transactions:\n{recent}"
        )

    def tax_categories(self) -> list[str]:
        """Return the tax category list from the manifest (or the default set)."""
        if self._bridge is None:
            return [
                "Revenue", "Software", "Contractors", "Travel", "Meals",
                "Office", "Bank Fees", "Taxes", "Owner Draw", "Uncategorized",
            ]
        return self._bridge.fetch_tax_categories()

    def record_snapshot(self, summary: FinancialSummary) -> None:
        """Persist the computed summary into the profile manifest.

        This method is called by the Orchestrator after the compute node
        completes, ensuring every run is durably recorded.  It is a no-op
        when ``bridge`` is ``None``.
        """
        if self._bridge is None:
            return
        manifest = self._bridge._read_manifest()
        manifest["last_tax_snapshot"] = summary.to_dict()
        self._bridge._write_manifest(manifest)
