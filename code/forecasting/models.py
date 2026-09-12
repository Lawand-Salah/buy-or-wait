"""Typed domain objects for the forecasting engine.

These mirror the columns of dataset/financial_profiles.csv and
dataset/financial_events.csv (AGENTS.md section 6.1 / problem_statement.md
"Files provided"). Kept free of any CSV/IO or LLM concerns so the rest of
this package stays unit-testable with plain Python objects instead of
fixture files.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Tuple

# Event types that represent a lifecycle marker rather than a recurring
# cash commitment. Recurrence detection never projects these forward --
# doing so would mean fabricating future investment activity or future
# refunds, which problem_statement.md rules out ("do not invent
# unsupported income, expenses, payment options, or financial
# information") and which the brief explicitly says is out of scope
# ("The task does not require predicting asset prices ...").
NON_RECURRING_EVENT_TYPES = frozenset(
    {"refund", "investment_purchase", "investment_sale", "investment_valuation"}
)

# Statuses that represent a live, still-to-clear cash commitment that must
# be reserved/included even though it hasn't settled yet.
RESERVED_STATUSES = frozenset({"pending", "scheduled"})


@dataclass(frozen=True)
class Profile:
    """One row of financial_profiles.csv."""

    user_id: str
    home_currency: str
    current_available_balance: float
    minimum_balance_to_keep: float
    financial_priorities: Tuple[str, ...]
    expense_categories_to_protect: Tuple[str, ...]
    expense_categories_user_is_willing_to_reduce: Tuple[str, ...]
    expense_categories_user_is_willing_to_stop: Tuple[str, ...]
    payment_methods_user_will_consider: Tuple[str, ...]
    max_installment_months: Optional[int]


@dataclass(frozen=True)
class Event:
    """One row of financial_events.csv.

    ``amount`` is ``None`` when the CSV cell is blank -- per
    problem_statement.md this means the real amount only exists in a
    linked image and must never be treated as zero. This module never
    invents that number itself; callers may supply it via
    ``amount_overrides`` (keyed by ``event_id``) once the vision/LLM layer
    has resolved it. Until then, such an event is excluded from both the
    ledger and recurrence detection, and a warning is surfaced.
    """

    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str  # "debit" | "credit" | "non_cash"
    amount: Optional[float]
    currency: str
    event_date: date
    settlement_date: Optional[date]
    status: str
    linked_event_id: Optional[str]
    flexibility: str
    minimum_allowed_amount: Optional[float]

    @property
    def effective_date(self) -> date:
        """The date cash actually moves: settlement_date, falling back to event_date."""
        return self.settlement_date or self.event_date

    def resolved_amount(self, overrides: "dict[str, float]") -> Optional[float]:
        if self.amount is not None:
            return self.amount
        return overrides.get(self.event_id)
