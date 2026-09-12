"""Day-by-day balance simulation and the two capacity numbers derived from it.

problem_statement.md, "90-Day Safety Check":

    Forecast the user's balance for the next 90 days ... A plan is safe
    only if the balance never falls below minimum_balance_to_keep.

    - amount_safe_to_pay: the most the user can pay today before optional
      spending changes without breaking the 90-day safety check, capped
      at requested_amount.
    - earliest_date_for_full_payment: the first date the full amount
      passes the safety check without optional spending changes.

The key subtlety this module exists to get right: a payment made on
``request_date`` stays subtracted for every later day too (it doesn't
reverse), so the binding constraint on how much is safe to pay *today* is
the *lowest* point the balance reaches anywhere in the 90-day window --
not today's balance in isolation. A bill due in three weeks can be the
real limit even when today looks comfortable.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional

from .ledger import LedgerEntry


@dataclass(frozen=True)
class Trajectory:
    """Balance at the end of each day, from ``request_date`` through the
    horizon, as if the request being evaluated were never paid.
    """

    request_date: date
    horizon_end: date
    daily_balance: Dict[date, float]  # end-of-day balance, one entry per day

    @property
    def dates(self) -> List[date]:
        return sorted(self.daily_balance)

    @property
    def minimum(self) -> float:
        return min(self.daily_balance.values())


def simulate_trajectory(
    *,
    starting_balance: float,
    ledger_entries: List[LedgerEntry],
    request_date: date,
    horizon_days: int,
) -> Trajectory:
    """Walks the balance forward day by day applying every ledger entry on
    its date. Entries dated before ``request_date`` (shouldn't normally
    happen -- ledger.py already clamps to it) or after the horizon are
    ignored; the former can't be un-spent and the latter is outside what
    we're asked to guarantee.
    """
    horizon_end = request_date + timedelta(days=horizon_days)

    daily_delta: Dict[date, float] = {}
    for entry in ledger_entries:
        if entry.date < request_date or entry.date > horizon_end:
            continue
        daily_delta[entry.date] = daily_delta.get(entry.date, 0.0) + entry.amount

    daily_balance: Dict[date, float] = {}
    balance = starting_balance
    d = request_date
    while d <= horizon_end:
        balance += daily_delta.get(d, 0.0)
        daily_balance[d] = balance
        d += timedelta(days=1)

    return Trajectory(request_date=request_date, horizon_end=horizon_end, daily_balance=daily_balance)


def _suffix_minimum(trajectory: Trajectory) -> Dict[date, float]:
    """min(daily_balance[d'] for d' >= d), computed in one backward pass."""
    dates = trajectory.dates
    suffix: Dict[date, float] = {}
    running = float("inf")
    for d in reversed(dates):
        running = min(running, trajectory.daily_balance[d])
        suffix[d] = running
    return suffix


def compute_amount_safe_to_pay(
    trajectory: Trajectory, *, minimum_balance_to_keep: float, requested_amount: float
) -> float:
    """The most that can be paid on ``trajectory.request_date`` without the
    balance dropping below ``minimum_balance_to_keep`` on any day of the
    window, capped at ``requested_amount`` and floored at 0.
    """
    worst = trajectory.minimum
    safe = worst - minimum_balance_to_keep
    return max(0.0, min(safe, requested_amount))


def compute_earliest_full_payment_date(
    trajectory: Trajectory, *, minimum_balance_to_keep: float, requested_amount: float
) -> Optional[date]:
    """The earliest day on which paying the *entire* ``requested_amount``
    keeps every subsequent day (through the horizon) at or above
    ``minimum_balance_to_keep``. ``None`` if no day in the window qualifies.
    """
    suffix_min = _suffix_minimum(trajectory)
    for d in trajectory.dates:  # trajectory.dates is already sorted ascending
        if suffix_min[d] - requested_amount >= minimum_balance_to_keep:
            return d
    return None
