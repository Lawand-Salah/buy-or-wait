"""CSV parsing for dataset/financial_profiles.csv, financial_events.csv,
and exchange_rates.csv into the typed objects in models.py / currency.py.

Kept separate from models.py so the domain objects stay pure and the
parsing quirks (blank cells, pipe-delimited lists, date strings) live in
exactly one place.
"""
from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from .currency import RateTable
from .models import Event, Profile


def _parse_date(value: str) -> date:
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def _parse_optional_date(value: str) -> Optional[date]:
    value = (value or "").strip()
    return _parse_date(value) if value else None


def _parse_optional_float(value: str) -> Optional[float]:
    value = (value or "").strip()
    return float(value) if value else None


def _parse_optional_int(value: str) -> Optional[int]:
    value = (value or "").strip()
    return int(float(value)) if value else None


def _parse_optional_str(value: str) -> Optional[str]:
    value = (value or "").strip()
    return value if value else None


def _parse_pipe_list(value: str) -> tuple:
    value = (value or "").strip()
    if not value:
        return ()
    return tuple(part.strip() for part in value.split("|") if part.strip())


def load_profiles(path: "str | Path") -> Dict[str, Profile]:
    """Returns {user_id: Profile}."""
    profiles: Dict[str, Profile] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            profile = Profile(
                user_id=row["user_id"],
                home_currency=row["home_currency"],
                current_available_balance=float(row["current_available_balance"]),
                minimum_balance_to_keep=float(row["minimum_balance_to_keep"]),
                financial_priorities=_parse_pipe_list(row["financial_priorities"]),
                expense_categories_to_protect=_parse_pipe_list(row["expense_categories_to_protect"]),
                expense_categories_user_is_willing_to_reduce=_parse_pipe_list(
                    row["expense_categories_user_is_willing_to_reduce"]
                ),
                expense_categories_user_is_willing_to_stop=_parse_pipe_list(
                    row["expense_categories_user_is_willing_to_stop"]
                ),
                payment_methods_user_will_consider=_parse_pipe_list(
                    row["payment_methods_user_will_consider"]
                ),
                max_installment_months=_parse_optional_int(row["max_installment_months"]),
            )
            profiles[profile.user_id] = profile
    return profiles


def load_events(path: "str | Path") -> List[Event]:
    events: List[Event] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            events.append(
                Event(
                    event_id=row["event_id"],
                    user_id=row["user_id"],
                    event_type=row["event_type"],
                    description=row["description"],
                    category=row["category"],
                    direction=row["direction"],
                    amount=_parse_optional_float(row["amount"]),
                    currency=row["currency"],
                    event_date=_parse_date(row["event_date"]),
                    settlement_date=_parse_optional_date(row["settlement_date"]),
                    status=row["status"],
                    linked_event_id=_parse_optional_str(row["linked_event_id"]),
                    flexibility=row["flexibility"],
                    minimum_allowed_amount=_parse_optional_float(row["minimum_allowed_amount"]),
                )
            )
    return events


def index_events_by_user(events: List[Event]) -> Dict[str, List[Event]]:
    by_user: Dict[str, List[Event]] = {}
    for e in events:
        by_user.setdefault(e.user_id, []).append(e)
    return by_user


def load_rate_table(path: "str | Path") -> RateTable:
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "rate_date": _parse_date(row["rate_date"]),
                    "from_currency": row["from_currency"],
                    "to_currency": row["to_currency"],
                    "rate": row["rate"],
                }
            )
    return RateTable(rows)
