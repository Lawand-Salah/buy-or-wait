"""Builds the unified, dated 90-day cash-flow ledger for one user.

Combines two sources, per AGENTS.md 6.3 and the problem_statement.md
"90-Day Safety Check":

* Explicit rows from financial_events.csv that are still live cash
  commitments: ``pending`` debits (reserved even though not yet cleared)
  and ``scheduled`` rows (both the "next confirmed salary" credit and any
  bills already scheduled). ``cancelled``, ``failed``, and ``unrealized``
  rows are excluded outright -- they never happened, or never became
  cash. Unsettled *credits* (a pending refund, for instance) are also
  excluded: "Reserve pending debits. Do not count pending credits ...
  until they settle."
* Recurring patterns mined from ``settled`` history (recurrence.py),
  projected forward -- but only for occurrences *after* the latest
  explicit row already covering that same (category, direction), so a
  known "next confirmed salary" row is never double-counted with a
  pattern-projected one.

Every entry is converted into the user's home currency at the date the
cash actually moves.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from .currency import RateTable
from .models import RESERVED_STATUSES, Event, Profile
from .recurrence import RecurringPattern, detect_recurrence, project_occurrences


@dataclass(frozen=True)
class LedgerEntry:
    date: date
    amount: float  # signed, in the user's home currency (+credit / -debit)
    category: str
    description: str
    source: str  # "pending" | "scheduled" | "projected_recurrence"
    event_id: Optional[str]  # None for a projected (not-yet-real) occurrence


@dataclass(frozen=True)
class LedgerResult:
    entries: List[LedgerEntry]
    patterns: List[RecurringPattern]
    warnings: List[str]


def build_ledger(
    *,
    profile: Profile,
    user_events: List[Event],
    rate_table: RateTable,
    request_date: date,
    horizon_days: int,
    amount_overrides: Optional[Dict[str, float]] = None,
) -> LedgerResult:
    """Builds the sorted cash-flow ledger for ``profile.user_id`` over
    ``[request_date, request_date + horizon_days]``.
    """

    amount_overrides = amount_overrides or {}
    warnings: List[str] = []
    horizon_end = request_date + timedelta(days=horizon_days)

    entries: List[LedgerEntry] = []
    explicit_anchor: Dict[Tuple[str, str], date] = {}

    for e in user_events:
        if e.status not in RESERVED_STATUSES:
            continue
        if e.direction not in ("debit", "credit"):
            continue  # non_cash (e.g. an unrealized valuation) never moves cash
        if e.direction == "credit" and e.status == "pending":
            # A pending credit (refund, bonus, etc.) is not counted until
            # it actually settles.
            continue

        amount = e.resolved_amount(amount_overrides)
        if amount is None:
            warnings.append(f"{e.event_id}: blank amount with no override supplied; excluded from ledger")
            continue

        signed = amount if e.direction == "credit" else -amount
        signed_home = rate_table.convert(signed, e.currency, profile.home_currency, e.effective_date)

        # An overdue-but-still-pending item still lands in the forecast;
        # it just can't be dated before the day the forecast starts.
        entry_date = max(e.effective_date, request_date)
        entries.append(
            LedgerEntry(
                date=entry_date,
                amount=signed_home,
                category=e.category,
                description=e.description,
                source=e.status,
                event_id=e.event_id,
            )
        )

        key = (e.category, e.direction)
        if key not in explicit_anchor or e.effective_date > explicit_anchor[key]:
            explicit_anchor[key] = e.effective_date

    # detect_recurrence does its own status filtering (settled history plus
    # any scheduled/pending anchor); it just needs the full event list.
    patterns = detect_recurrence(user_events)

    for pattern in patterns:
        # If the pattern's own next expected occurrence was already due
        # *before* request_date and nothing (settled, scheduled, or
        # pending) ever showed up for it, treat the pattern as
        # discontinued rather than assume it silently continues -- e.g. a
        # second household income that ran for four months and then just
        # stops appearing, with no cancellation event because there was
        # never a specific event to cancel. This is checked against the
        # pattern's own cadence, not the (possibly more recent, and
        # possibly a different sub-pattern's) explicit_anchor below.
        next_expected = pattern.last_known_date + timedelta(days=pattern.interval_days)
        if next_expected < request_date:
            continue

        # Step from the latest explicit (scheduled/pending) row for this
        # category+direction when one exists and is fresher than the
        # pattern's own history -- e.g. a "next confirmed salary" row a
        # couple of days off the plain monthly-interval grid -- so the
        # projection continues from the real confirmed date instead of
        # landing a near-duplicate entry right next to it. request_date is
        # only ever a floor (nothing gets projected on/before it), never a
        # re-phasing of the cadence.
        phase_start = explicit_anchor.get(pattern.anchor_key, pattern.last_known_date)
        for occ_date in project_occurrences(
            pattern, phase_start=phase_start, floor=request_date, through=horizon_end
        ):
            signed = pattern.projected_amount if pattern.direction == "credit" else -pattern.projected_amount
            signed_home = rate_table.convert(signed, pattern.currency, profile.home_currency, occ_date)
            entries.append(
                LedgerEntry(
                    date=occ_date,  # already guaranteed > request_date by the anchor clamp above
                    amount=signed_home,
                    category=pattern.category,
                    description=f"{pattern.description} (projected recurrence)",
                    source="projected_recurrence",
                    event_id=None,
                )
            )

    entries.sort(key=lambda entry: entry.date)
    return LedgerResult(entries=entries, patterns=patterns, warnings=warnings)
