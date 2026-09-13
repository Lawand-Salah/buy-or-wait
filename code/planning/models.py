"""Typed objects for plan selection: a supplied payment option
(request_payment_options.csv) and a candidate/chosen Plan.

Kept free of any CSV/IO or LLM concerns, same principle as
forecasting/models.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Tuple


def format_amount(value: float) -> str:
    """Whole numbers render without a decimal point (25256), fractional
    ones keep only as many decimal places as needed (17229139.2) --
    matching the style used in sample_requests.csv.
    """
    rounded = round(value, 2)
    if rounded == int(rounded):
        return str(int(rounded))
    text = f"{rounded:.2f}".rstrip("0").rstrip(".")
    return text


@dataclass(frozen=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str  # "full_payment" | "installments"
    payment_amount: float
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: Optional[int]
    financing_fee: float
    total_payable_amount: float

    @property
    def sort_key(self) -> int:
        """Numeric part of payment_option_id, for tie-break 6 ("lowest
        payment_option_id") -- comparing the strings directly would sort
        "payment_option_10" before "payment_option_9".
        """
        digits = "".join(ch for ch in self.payment_option_id if ch.isdigit())
        return int(digits) if digits else 0

    def build_schedule(self) -> Tuple[Tuple[date, float], ...]:
        """The full, literal payment schedule this option describes:
        number_of_payments equal payments of payment_amount, spaced
        payment_frequency_days apart starting at first_payment_date.
        Works for a single-payment full_payment option too (one iteration).
        """
        schedule = []
        current = self.first_payment_date
        step = self.payment_frequency_days or 0
        for _ in range(self.number_of_payments):
            schedule.append((current, self.payment_amount))
            current = current.fromordinal(current.toordinal() + step)
        return tuple(schedule)


@dataclass(frozen=True)
class Plan:
    """One fully-specified candidate (or chosen) plan."""

    method: str  # "full_payment" | "partial_payment" | "installments" | "wait" | "not_recommended"
    payments: Tuple[Tuple[date, float], ...]  # chronological (date, amount)
    option_sort_key: int  # tie-break 6; 0 for plans not tied to a supplied option (partial_payment)
    spending_changes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def total_cost(self) -> float:
        return sum(amount for _, amount in self.payments)

    @property
    def start_date(self) -> Optional[date]:
        return self.payments[0][0] if self.payments else None

    @property
    def end_date(self) -> Optional[date]:
        return self.payments[-1][0] if self.payments else None

    @property
    def num_payments(self) -> int:
        return len(self.payments)

    def completes_by(self, deadline: date) -> bool:
        return self.end_date is not None and self.end_date <= deadline

    @property
    def affordability_status(self) -> str:
        if self.method == "not_recommended":
            return "not_affordable"
        if self.spending_changes:
            return "affordable_with_plan"
        if self.method == "full_payment":
            return "affordable_now"
        if self.method == "wait":
            return "affordable_later"
        return "affordable_with_plan"  # partial_payment, installments

    def payment_plan_string(self) -> str:
        if not self.payments:
            return "none"
        return "|".join(f"{d.isoformat()}:{format_amount(a)}" for d, a in self.payments)

    def spending_changes_string(self) -> str:
        return "|".join(self.spending_changes) if self.spending_changes else "none"


NOT_RECOMMENDED = Plan(method="not_recommended", payments=(), option_sort_key=0)
