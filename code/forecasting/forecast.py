"""Top-level orchestration: profile + events + rates -> the two capacity
numbers (amount_safe_to_pay, earliest_date_for_full_payment) plus the
ledger/trajectory/warnings that produced them.

This module deliberately stops here. It does not know about payment
options, payment-method preferences, spending-changes, or plan ranking --
those are a separate layer built on top of ``ForecastResult``, and
``amount_safe_to_pay`` / ``earliest_date_for_full_payment`` are defined by
the spec as the numbers *before* any of that (problem_statement.md:
"the amount safe on request_date before optional spending changes").
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from .currency import RateTable
from .data_loader import index_events_by_user, load_events, load_profiles, load_rate_table
from .ledger import LedgerEntry, build_ledger
from .models import Event, Profile
from .recurrence import RecurringPattern
from .simulate import Trajectory, compute_amount_safe_to_pay, compute_earliest_full_payment_date, simulate_trajectory

DEFAULT_HORIZON_DAYS = 90


@dataclass(frozen=True)
class ForecastResult:
    user_id: str
    request_date: date
    horizon_end: date
    requested_amount: float
    starting_balance: float
    minimum_balance_to_keep: float
    amount_safe_to_pay: float
    earliest_date_for_full_payment: Optional[date]
    trajectory: Trajectory
    ledger_entries: List[LedgerEntry]
    recurring_patterns: List[RecurringPattern]
    warnings: List[str]


def run_forecast(
    *,
    profile: Profile,
    user_events: List[Event],
    rate_table: RateTable,
    request_date: date,
    requested_amount: float,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    amount_overrides: Optional[Dict[str, float]] = None,
) -> ForecastResult:
    """Pure function version -- no file I/O, easy to unit test with
    hand-built ``Profile``/``Event`` objects.
    """
    ledger_result = build_ledger(
        profile=profile,
        user_events=user_events,
        rate_table=rate_table,
        request_date=request_date,
        horizon_days=horizon_days,
        amount_overrides=amount_overrides,
    )

    trajectory = simulate_trajectory(
        starting_balance=profile.current_available_balance,
        ledger_entries=ledger_result.entries,
        request_date=request_date,
        horizon_days=horizon_days,
    )

    amount_safe_to_pay = compute_amount_safe_to_pay(
        trajectory,
        minimum_balance_to_keep=profile.minimum_balance_to_keep,
        requested_amount=requested_amount,
    )
    earliest_date = compute_earliest_full_payment_date(
        trajectory,
        minimum_balance_to_keep=profile.minimum_balance_to_keep,
        requested_amount=requested_amount,
    )

    return ForecastResult(
        user_id=profile.user_id,
        request_date=request_date,
        horizon_end=trajectory.horizon_end,
        requested_amount=requested_amount,
        starting_balance=profile.current_available_balance,
        minimum_balance_to_keep=profile.minimum_balance_to_keep,
        amount_safe_to_pay=amount_safe_to_pay,
        earliest_date_for_full_payment=earliest_date,
        trajectory=trajectory,
        ledger_entries=ledger_result.entries,
        recurring_patterns=ledger_result.patterns,
        warnings=ledger_result.warnings,
    )


class Forecaster:
    """Convenience wrapper that loads dataset/*.csv once and exposes a
    single ``forecast(...)`` call per request.
    """

    def __init__(self, dataset_dir: "str | Path"):
        dataset_dir = Path(dataset_dir)
        self.profiles: Dict[str, Profile] = load_profiles(dataset_dir / "financial_profiles.csv")
        events = load_events(dataset_dir / "financial_events.csv")
        self.events_by_user: Dict[str, List[Event]] = index_events_by_user(events)
        self.rate_table: RateTable = load_rate_table(dataset_dir / "exchange_rates.csv")

    def forecast(
        self,
        *,
        user_id: str,
        request_date: date,
        requested_amount: float,
        horizon_days: int = DEFAULT_HORIZON_DAYS,
        amount_overrides: Optional[Dict[str, float]] = None,
    ) -> ForecastResult:
        profile = self.profiles[user_id]
        user_events = self.events_by_user.get(user_id, [])
        return run_forecast(
            profile=profile,
            user_events=user_events,
            rate_table=self.rate_table,
            request_date=request_date,
            requested_amount=requested_amount,
            horizon_days=horizon_days,
            amount_overrides=amount_overrides,
        )
