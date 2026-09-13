"""Chooses the final Plan for one request: builds every eligible, safe
candidate (full_payment, wait, partial_payment, one per eligible
installment option), ranks them by the six tie-breakers in
problem_statement.md's "Choosing Between Safe Plans", and -- only if
nothing is safe without one -- retries with spending changes applied.
"""
from __future__ import annotations

from datetime import date
from typing import List, Optional, Sequence

from forecasting.forecast import ForecastResult
from forecasting.ledger import build_ledger
from forecasting.models import Event, Profile
from forecasting.simulate import (
    Trajectory,
    compute_amount_safe_to_pay,
    compute_earliest_full_payment_date,
    simulate_trajectory,
)

from .models import NOT_RECOMMENDED, PaymentOption, Plan
from .schedule_safety import is_schedule_safe
from .spending_changes import apply_candidates_to_ledger, find_spending_change_candidates


class RequestInfo:
    """The subset of a requests.csv row select_plan() needs -- kept
    decoupled from however code/main.py chooses to parse the CSV row.
    """

    def __init__(
        self,
        *,
        request_date: date,
        requested_amount: float,
        desired_completion_date: date,
        allows_partial_payment: bool,
    ) -> None:
        self.request_date = request_date
        self.requested_amount = requested_amount
        self.desired_completion_date = desired_completion_date
        self.allows_partial_payment = allows_partial_payment


def _build_candidates(
    *,
    trajectory: Trajectory,
    amount_safe_to_pay: float,
    earliest_date_for_full_payment: Optional[date],
    profile: Profile,
    request: RequestInfo,
    payment_options: Sequence[PaymentOption],
) -> List[Plan]:
    accepted = set(profile.payment_methods_user_will_consider)
    candidates: List[Plan] = []
    minimum_balance = profile.minimum_balance_to_keep

    full_option = next((o for o in payment_options if o.payment_method == "full_payment"), None)
    full_option_key = full_option.sort_key if full_option else 0

    if "full_payment" in accepted and earliest_date_for_full_payment is not None:
        full_schedule = ((earliest_date_for_full_payment, request.requested_amount),)
        if is_schedule_safe(trajectory, minimum_balance_to_keep=minimum_balance, schedule=full_schedule):
            method = "full_payment" if earliest_date_for_full_payment == request.request_date else "wait"
            candidates.append(Plan(method=method, payments=full_schedule, option_sort_key=full_option_key))

    if (
        request.allows_partial_payment
        and "partial_payment" in accepted
        and 0 < amount_safe_to_pay < request.requested_amount
        and earliest_date_for_full_payment is not None
        and earliest_date_for_full_payment <= request.desired_completion_date
    ):
        remainder = request.requested_amount - amount_safe_to_pay
        schedule = ((request.request_date, amount_safe_to_pay), (earliest_date_for_full_payment, remainder))
        if is_schedule_safe(trajectory, minimum_balance_to_keep=minimum_balance, schedule=schedule):
            candidates.append(Plan(method="partial_payment", payments=schedule, option_sort_key=0))

    if "installments" in accepted and profile.max_installment_months is not None:
        for option in payment_options:
            if option.payment_method != "installments":
                continue
            if option.number_of_payments > profile.max_installment_months:
                continue
            schedule = option.build_schedule()
            if is_schedule_safe(trajectory, minimum_balance_to_keep=minimum_balance, schedule=schedule):
                candidates.append(Plan(method="installments", payments=schedule, option_sort_key=option.sort_key))

    return candidates


def _rank(candidates: List[Plan], desired_completion_date: date) -> List[Plan]:
    def key(plan: Plan):
        return (
            0 if plan.completes_by(desired_completion_date) else 1,  # 1. complete by deadline
            0 if not plan.spending_changes else 1,  # 2. no spending changes
            plan.total_cost,  # 3. minimize total paid
            plan.start_date,  # 4. start earlier
            plan.num_payments,  # 5. fewer payments
            plan.option_sort_key,  # 6. lowest payment_option_id
        )

    return sorted(candidates, key=key)


def select_plan(
    *,
    forecast_result: ForecastResult,
    profile: Profile,
    user_events: List[Event],
    request: RequestInfo,
    payment_options: Sequence[PaymentOption],
) -> Plan:
    candidates = _build_candidates(
        trajectory=forecast_result.trajectory,
        amount_safe_to_pay=forecast_result.amount_safe_to_pay,
        earliest_date_for_full_payment=forecast_result.earliest_date_for_full_payment,
        profile=profile,
        request=request,
        payment_options=payment_options,
    )
    ranked = _rank(candidates, request.desired_completion_date)
    if ranked:
        return ranked[0]

    # Nothing safe without a spending change -- try unlocking one, trying
    # the smallest number of changes first (highest-impact candidates
    # first, per find_spending_change_candidates' own ordering).
    all_change_candidates = find_spending_change_candidates(
        user_events=user_events,
        profile=profile,
        window_start=request.request_date,
        window_end=forecast_result.horizon_end,
    )
    for n in range(1, len(all_change_candidates) + 1):
        applied = all_change_candidates[:n]
        modified_entries = apply_candidates_to_ledger(forecast_result.ledger_entries, applied)
        trajectory = simulate_trajectory(
            starting_balance=forecast_result.starting_balance,
            ledger_entries=modified_entries,
            request_date=request.request_date,
            horizon_days=90,
        )
        amount_safe = compute_amount_safe_to_pay(
            trajectory, minimum_balance_to_keep=profile.minimum_balance_to_keep,
            requested_amount=request.requested_amount,
        )
        earliest = compute_earliest_full_payment_date(
            trajectory, minimum_balance_to_keep=profile.minimum_balance_to_keep,
            requested_amount=request.requested_amount,
        )
        retry_candidates = _build_candidates(
            trajectory=trajectory, amount_safe_to_pay=amount_safe, earliest_date_for_full_payment=earliest,
            profile=profile, request=request, payment_options=payment_options,
        )
        retry_ranked = _rank(retry_candidates, request.desired_completion_date)
        if retry_ranked:
            change_strings = tuple(c.action_string() for c in applied)
            winner = retry_ranked[0]
            return Plan(
                method=winner.method, payments=winner.payments,
                option_sort_key=winner.option_sort_key, spending_changes=change_strings,
            )

    return NOT_RECOMMENDED
