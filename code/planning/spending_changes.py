"""Candidate stop/reduce actions on a user's own flexible, dated events,
and applying them to a ledger for re-simulation.

problem_statement.md: "spending_changes_needed may contain up to three
changes ... Only recurring expenses marked as flexible may be changed"
and "Stopping and reducing the same financial event are mutually
exclusive." AGENTS.md 6.2 likewise: "Only non-protected, flexible events
in a category the user permits may be changed."

Only *dated, real* events qualify -- a purely-projected recurrence
occurrence has no event_id to name, so it can never become a
spending_changes_needed entry. That means only a user's own pending/
scheduled rows within the 90-day window are ever candidates here, not the
recurring pattern itself.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import List, Optional

from forecasting.ledger import LedgerEntry
from forecasting.models import Event, Profile

from .models import format_amount

MAX_SPENDING_CHANGES = 3


@dataclass(frozen=True)
class SpendingChangeCandidate:
    action: str  # "stop" | "reduce_to"
    event_id: str
    amount_freed: float
    new_amount: Optional[float] = None  # only for "reduce_to"

    def action_string(self) -> str:
        if self.action == "stop":
            return f"stop:{self.event_id}"
        return f"reduce_to:{self.event_id}:{format_amount(self.new_amount)}"


def find_spending_change_candidates(
    *,
    user_events: List[Event],
    profile: Profile,
    window_start: date,
    window_end: date,
) -> List[SpendingChangeCandidate]:
    """Every eligible stop/reduce action, sorted by cash freed (descending)
    and capped at MAX_SPENDING_CHANGES -- the caller applies a prefix of
    this list, smallest first, so at most 3 are ever used and the
    highest-impact ones are tried first.
    """
    protect = set(profile.expense_categories_to_protect)
    stoppable_categories = set(profile.expense_categories_user_is_willing_to_stop)
    reducible_categories = set(profile.expense_categories_user_is_willing_to_reduce)

    candidates: List[SpendingChangeCandidate] = []
    for e in user_events:
        if e.status not in ("pending", "scheduled"):
            continue
        if e.direction != "debit":
            continue
        if e.amount is None:
            continue
        if e.category in protect:
            continue
        if not (window_start <= e.effective_date <= window_end):
            continue

        # A category the user permits both ways: prefer stopping (frees
        # the full amount) over reducing when both are on the table for
        # this one event -- mutually exclusive per event, never both.
        if e.flexibility in ("stoppable", "reducible_or_stoppable") and e.category in stoppable_categories:
            candidates.append(SpendingChangeCandidate(action="stop", event_id=e.event_id, amount_freed=e.amount))
        elif e.flexibility in ("reducible", "reducible_or_stoppable") and e.category in reducible_categories:
            if e.minimum_allowed_amount is not None and e.minimum_allowed_amount < e.amount:
                freed = e.amount - e.minimum_allowed_amount
                candidates.append(
                    SpendingChangeCandidate(
                        action="reduce_to", event_id=e.event_id, amount_freed=freed,
                        new_amount=e.minimum_allowed_amount,
                    )
                )

    candidates.sort(key=lambda c: -c.amount_freed)
    return candidates[:MAX_SPENDING_CHANGES]


def apply_candidates_to_ledger(
    entries: List[LedgerEntry], candidates: List[SpendingChangeCandidate]
) -> List[LedgerEntry]:
    """A new ledger-entry list with each candidate's action applied: a
    "stop" entry is dropped; a "reduce_to" entry has its amount rewritten
    to the new (smaller) magnitude, preserving its debit sign.
    """
    stops = {c.event_id for c in candidates if c.action == "stop"}
    reduces = {c.event_id: c.new_amount for c in candidates if c.action == "reduce_to"}

    modified: List[LedgerEntry] = []
    for entry in entries:
        if entry.event_id in stops:
            continue
        if entry.event_id in reduces:
            sign = -1.0 if entry.amount < 0 else 1.0
            modified.append(replace(entry, amount=sign * reduces[entry.event_id]))
        else:
            modified.append(entry)
    return modified
