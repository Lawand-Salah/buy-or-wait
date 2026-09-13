"""Generalized "is this payment schedule safe" check -- works for a single
lump payment, a two-payment partial_payment split, or a many-payment
installment schedule alike, without any change to forecasting/simulate.py.

problem_statement.md: "A plan is safe only if the balance never falls
below minimum_balance_to_keep ... throughout the 90-day forecast." Each
payment in the schedule permanently subtracts from every day on/after its
date (same principle forecasting.simulate already uses for a single
payment) -- this just generalizes that to an arbitrary list of (date,
amount) debits.

One deliberate, documented limitation: a payment dated beyond the 90-day
horizon (a long installment plan can run to ~2 years) simply cannot be
checked against a forecast that only covers 90 days -- the spec frames
safety strictly in terms of that window, so this only verifies the
in-window portion.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, Sequence, Tuple

from forecasting.simulate import Trajectory


def is_schedule_safe(
    trajectory: Trajectory,
    *,
    minimum_balance_to_keep: float,
    schedule: Sequence[Tuple[date, float]],
) -> bool:
    if not schedule:
        return True  # the null plan can never make anything worse

    deltas: Dict[date, float] = {}
    for d, amount in schedule:
        if d > trajectory.horizon_end:
            continue  # outside what a 90-day forecast can verify
        deltas[d] = deltas.get(d, 0.0) + amount
    if not deltas:
        return True

    running_extra = 0.0
    started = False
    for d in trajectory.dates:  # ascending
        if d in deltas:
            running_extra += deltas[d]
            started = True
        if not started:
            continue  # a day before the first payment isn't this schedule's doing
        if trajectory.daily_balance[d] - running_extra < minimum_balance_to_keep:
            return False
    return True
