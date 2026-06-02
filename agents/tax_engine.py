"""TaxEngine — high-precision tax logic and financial metrics worker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


class TaxValidationError(ValueError):
    """Raised when computed financial metrics fail internal consistency checks."""


@dataclass
class FinancialSummary:
    """Immutable snapshot of computed financial metrics for a given ledger."""

    income: float
    expenses: float
    personal_expenses: float
    profit: float
    tax_reserve: float
    tax_rate: float
    transactions: int
    top_expenses: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict[str, Any]:
        return {
            "income": self.income,
            "expenses": self.expenses,
            "personal_expenses": self.personal_expenses,
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


def _business_expense_rows(ledger: pd.DataFrame) -> pd.DataFrame:
    """Expense rows included in professional ledgers and tax calculations."""
    if ledger is None or ledger.empty:
        return pd.DataFrame()
    expenses = ledger.loc[ledger["kind"] == "expense"].copy()
    if expenses.empty:
        return expenses
    if "expense_type" in expenses.columns:
        return expenses.loc[expenses["expense_type"].fillna("Personal") == "Business"]
    return expenses


class TaxEngine:
    """High-precision worker dedicated to tax logic and metric validation."""

    def __init__(self, bridge: Any = None) -> None:
        self._bridge = bridge

    def compute_summary(self, ledger: pd.DataFrame, tax_rate: float) -> FinancialSummary:
        if ledger is None or ledger.empty:
            return FinancialSummary(
                income=0.0,
                expenses=0.0,
                personal_expenses=0.0,
                profit=0.0,
                tax_reserve=0.0,
                tax_rate=tax_rate,
                transactions=0,
                top_expenses=pd.DataFrame(columns=["category", "amount"]),
            )

        income = float(ledger.loc[ledger["kind"] == "income", "amount"].sum())
        all_expenses = ledger.loc[ledger["kind"] == "expense"]
        personal_expenses = 0.0
        if not all_expenses.empty and "expense_type" in all_expenses.columns:
            personal_expenses = float(
                all_expenses.loc[
                    all_expenses["expense_type"].fillna("Personal") == "Personal", "amount"
                ].abs().sum()
            )
        business_expenses = _business_expense_rows(ledger)
        expenses = (
            float(business_expenses["amount"].abs().sum()) if not business_expenses.empty else 0.0
        )
        profit = income - expenses
        tax_reserve = max(profit, 0.0) * tax_rate

        top_expenses = (
            business_expenses.assign(amount=lambda f: f["amount"].abs())
            .groupby("category", dropna=False)["amount"]
            .sum()
            .sort_values(ascending=False)
            .head(8)
            .reset_index()
            if not business_expenses.empty
            else pd.DataFrame(columns=["category", "amount"])
        )

        summary = FinancialSummary(
            income=income,
            expenses=expenses,
            personal_expenses=personal_expenses,
            profit=profit,
            tax_reserve=tax_reserve,
            tax_rate=tax_rate,
            transactions=len(ledger),
            top_expenses=top_expenses,
        )
        self.validate_metrics(summary)
        return summary

    def validate_metrics(self, summary: FinancialSummary) -> None:
        epsilon = 1e-6
        import math

        for name, val in [
            ("income", summary.income),
            ("expenses", summary.expenses),
            ("personal_expenses", summary.personal_expenses),
            ("profit", summary.profit),
            ("tax_reserve", summary.tax_reserve),
        ]:
            if math.isnan(val) or math.isinf(val):
                raise TaxValidationError(f"Metric '{name}' is {val} — data integrity error.")

        expected_profit = summary.income - summary.expenses
        if abs(summary.profit - expected_profit) > epsilon:
            raise TaxValidationError(
                f"Profit mismatch: stored={summary.profit:.6f}, "
                f"computed={expected_profit:.6f}."
            )

        expected_reserve = max(summary.profit, 0.0) * summary.tax_rate
        if abs(summary.tax_reserve - expected_reserve) > epsilon:
            raise TaxValidationError(
                f"Tax reserve mismatch: stored={summary.tax_reserve:.6f}, "
                f"computed={expected_reserve:.6f}."
            )

    def build_context(
        self,
        summary: FinancialSummary,
        docs_list: list[dict[str, Any]],
        ledger: pd.DataFrame,
        business_type: str = "",
        tax_notes: str = "",
    ) -> str:
        top_expense_text = (
            summary.top_expenses.to_string(
                index=False,
                formatters={"amount": "${:,.2f}".format},
            )
            if not summary.top_expenses.empty
            else "No business expenses loaded."
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
            f"Business expenses (tax ledger): ${summary.expenses:,.2f}\n"
            f"Personal expenses (excluded): ${summary.personal_expenses:,.2f}\n"
            f"Net profit: ${summary.profit:,.2f}\n"
            f"Suggested tax reserve: ${summary.tax_reserve:,.2f}\n\n"
            f"Top business expenses:\n{top_expense_text}\n\n"
            f"Document vault ({len(docs_list)} files):\n{docs_text}\n\n"
            f"Recent transactions:\n{recent}"
        )

    def tax_categories(self) -> list[str]:
        if self._bridge is None:
            return [
                "Revenue", "Software", "Contractors", "Travel", "Meals",
                "Office", "Bank Fees", "Taxes", "Owner Draw", "Uncategorized",
            ]
        return self._bridge.fetch_tax_categories()

    def record_snapshot(self, summary: FinancialSummary) -> None:
        if self._bridge is None:
            return
        manifest = self._bridge._read_manifest()
        manifest["last_tax_snapshot"] = summary.to_dict()
        self._bridge._write_manifest(manifest)
