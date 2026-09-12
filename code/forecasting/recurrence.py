"""Recurrence detection over *settled* financial_events.csv history.

AGENTS.md 6.3: "Detect recurrence only when history supports it. Forecast
essential variable spending conservatively." This module never projects a
group with too few settled occurrences or an inconsistent cadence -- a
category with irregular, one-off history is left alone rather than
extrapolated.

Two-pass grouping, category first
----------------------------------
The dataset uses two different description styles for recurring cash
flow, and a single grouping strategy misses one of them:

* Rotating-description categories (groceries, transport, dining, and
  freelance/gig-style income) use a *different* description every single
  occurrence ("Local taxi", "Rail pass", "Fuel refill", ...; "Website
  project payment", "Content contract payment", ...) but the underlying
  cadence is genuinely regular across the whole category (transport
  every ~5 days, say). Grouping by (category, direction) -- ignoring
  description -- finds these.
* Fixed-description categories (rent, utilities, subscriptions) reuse the
  exact same description every occurrence, and sometimes a category also
  mixes one real recurring signal with unrelated one-off noise (a
  purchase that was authorized, cancelled, and retried under a
  *different* description, sitting alongside a monthly "Clothing and
  household items" charge under the same "shopping" category, for
  example). Grouping by (category, description) finds the clean
  sub-signal in that case.

The whole-category pass runs *first*: it's the one that recovers the
full rotating-description signal, and it fails safely (an inconsistent
cadence) when a category is really a clean signal plus noise, which is
exactly when the per-description fallback pass is needed. Running
per-description first would do the opposite -- it can accidentally
"claim" a coincidentally-regular slice of a rotating category (e.g. just
the "Parking and tolls" occurrences out of all transport spending, which
happen to recur every ~20 days purely because that description comes up
roughly every fourth rotation), quietly starving the whole-category pass
of the full signal and undercounting real spending.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from datetime import date, timedelta
from statistics import mean, median, pstdev
from typing import Dict, List, Tuple

from .models import NON_RECURRING_EVENT_TYPES, Event

# A description on the most recent occurrence that explicitly says the
# commitment ended (e.g. "Final employer payroll") is direct textual
# evidence that history does *not* support projecting it further -- the
# same principle as detecting recurrence in the first place, just arguing
# the other way. Without this, a perfectly regular cadence would
# otherwise sail straight through the ordinary consistency checks.
_TERMINATION_RE = re.compile(r"\b(final|last|closing|terminat\w*|ended?|closed)\b", re.IGNORECASE)

# Minimum occurrences before a pattern is trusted at all.
MIN_OCCURRENCES = 3
# Income is relaxed to 2: a brand-new employee's history is exactly
# {settled "Prorated first salary", scheduled "Next confirmed salary"} --
# two points, one interval -- and salary is the one category
# problem_statement.md singles out as a confirmed, ongoing commitment
# rather than something to treat with the same "prove it three times"
# caution as an arbitrary expense category.
MIN_OCCURRENCES_INCOME = 2
# Two consecutive intervals must agree within this fraction of the median
# interval, or the group is treated as irregular (no recurrence claimed).
INTERVAL_TOLERANCE = 0.35
# Coefficient of variation below this is "fixed amount" (use the latest
# observed amount); at/above it the projection uses a conservative
# estimate instead of the mean -- the highest observed amount for
# expenses (don't under-reserve), the lowest observed amount for income
# (don't over-count money that may not show up).
AMOUNT_COV_THRESHOLD = 0.03


@dataclass(frozen=True)
class RecurringPattern:
    user_id: str
    category: str
    direction: str  # "debit" | "credit"
    description: str  # representative label only, for explanations/logs
    currency: str
    flexibility: str
    interval_days: int
    projected_amount: float
    is_variable_amount: bool
    last_known_date: date
    occurrences_used: int

    @property
    def anchor_key(self) -> Tuple[str, str]:
        """Key used to line this pattern up against explicit ledger rows
        (pending/scheduled events of the same category+direction) so a
        known next occurrence is never double-counted with a projected one.
        """
        return (self.category, self.direction)


def _build_pattern(user_id: str, category: str, group: List[Event]) -> "RecurringPattern | None":
    group = sorted(group, key=lambda e: e.effective_date)
    if _TERMINATION_RE.search(group[-1].description):
        return None  # e.g. "Final employer payroll" -- don't project past it
    direction = group[-1].direction
    min_occurrences = MIN_OCCURRENCES_INCOME if direction == "credit" else MIN_OCCURRENCES
    if len(group) < min_occurrences:
        return None

    dates = [e.effective_date for e in group]
    deltas = [(b - a).days for a, b in zip(dates, dates[1:])]
    med = median(deltas)
    if med <= 0:
        return None
    if any(abs(d - med) > INTERVAL_TOLERANCE * med for d in deltas):
        return None  # cadence isn't consistent enough to trust

    amounts = [e.amount for e in group]  # type: ignore[misc]  # amount is guaranteed non-None by caller
    avg = mean(amounts)
    cov = (pstdev(amounts) / avg) if avg else 0.0
    is_variable = cov >= AMOUNT_COV_THRESHOLD

    last_event = group[-1]
    if not is_variable:
        projected_amount = last_event.amount
    elif last_event.status in ("scheduled", "pending"):
        # The most recent data point is a confirmed future figure (e.g.
        # "Next confirmed salary"), not an inferred one -- trust it
        # directly for occurrences beyond it rather than blending it with
        # an earlier value that may be anomalous for a known reason (a
        # prorated first paycheck, say).
        projected_amount = last_event.amount
    elif direction == "credit":
        projected_amount = min(amounts)  # conservative: don't over-count income
    else:
        projected_amount = max(amounts)  # conservative: don't under-reserve an expense

    return RecurringPattern(
        user_id=user_id,
        category=category,
        direction=direction,
        description=group[-1].description,
        currency=group[-1].currency,
        flexibility=group[-1].flexibility,
        interval_days=round(med),
        projected_amount=projected_amount,
        is_variable_amount=is_variable,
        last_known_date=dates[-1],
        occurrences_used=len(group),
    )


def detect_recurrence(events: List[Event]) -> List[RecurringPattern]:
    """Detect recurring patterns from one user's event history.

    The mining pool is ``settled`` history plus any ``scheduled`` /
    ``pending`` rows (``cancelled``, ``failed``, and ``unrealized`` never
    enter it): a scheduled "next confirmed salary" or a scheduled bill is
    just as informative for establishing cadence as a settled one, and is
    what lets a thin-history category (see ``MIN_OCCURRENCES_INCOME``)
    still be trusted.
    """

    eligible = [
        e
        for e in events
        if e.status in ("settled", "scheduled", "pending")
        and e.direction in ("debit", "credit")
        and e.amount is not None
        and e.event_type not in NON_RECURRING_EVENT_TYPES
    ]
    if not eligible:
        return []
    user_id = eligible[0].user_id

    patterns: List[RecurringPattern] = []
    claimed_ids: set = set()

    # Pass 1: whole-category match -- (category, direction), ignoring
    # description. This is what recovers a rotating-description category's
    # true full cadence.
    by_cat_dir: Dict[Tuple[str, str], List[Event]] = {}
    for e in eligible:
        by_cat_dir.setdefault((e.category, e.direction), []).append(e)

    for (category, _direction), group in by_cat_dir.items():
        pattern = _build_pattern(user_id, category, group)
        if pattern is not None:
            patterns.append(pattern)
            claimed_ids.update(e.event_id for e in group)

    # Pass 2: exact (category, description) match over whatever pass 1
    # left unclaimed -- salvages a clean fixed-description sub-signal from
    # a category whose *aggregate* cadence was too inconsistent to trust
    # (a real monthly charge sitting alongside unrelated one-off noise).
    leftover = [e for e in eligible if e.event_id not in claimed_ids]
    by_cat_desc: Dict[Tuple[str, str], List[Event]] = {}
    for e in leftover:
        by_cat_desc.setdefault((e.category, e.description), []).append(e)

    for (category, _description), group in by_cat_desc.items():
        pattern = _build_pattern(user_id, category, group)
        if pattern is not None:
            patterns.append(pattern)

    return patterns


def project_occurrences(
    pattern: RecurringPattern, *, phase_start: date, floor: date, through: date
) -> List[date]:
    """Future occurrence dates, spaced ``pattern.interval_days`` apart
    starting from ``phase_start``, that fall strictly after ``floor`` and
    up to ``through`` inclusive.

    ``phase_start`` should be the latest date already covered by an
    explicit event for this pattern (a scheduled/pending row, or the
    pattern's own last settled date if there is no such row) -- stepping
    from it, rather than always from the pattern's original historical
    date, keeps the projection lined up with a fresher confirmed date
    (e.g. "next confirmed salary") instead of drifting a couple of days
    off and landing a near-duplicate entry right next to it.

    ``floor`` keeps the caller from projecting anything before the
    forecast's own start date. It is deliberately a separate argument
    from ``phase_start`` rather than folded into one "after" cutoff: using
    ``floor`` to shift *where the stepping begins* would re-phase an
    ordinary pattern (no fresher explicit anchor than its own history) to
    the request date instead of its true recurring day, which is wrong
    whenever request_date isn't itself on that cadence.
    """
    if pattern.interval_days <= 0:
        return []
    d = phase_start
    lower_bound = max(phase_start, floor)
    while d <= lower_bound:
        d += timedelta(days=pattern.interval_days)
    out: List[date] = []
    while d <= through:
        out.append(d)
        d += timedelta(days=pattern.interval_days)
    return out
