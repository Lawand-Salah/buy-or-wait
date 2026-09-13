"""Loads dataset/request_payment_options.csv into PaymentOption objects,
indexed by request_id (a request has 2-4 supplied options).
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .models import PaymentOption


def _parse_optional_int(value: str) -> Optional[int]:
    value = (value or "").strip()
    return int(float(value)) if value else None


def load_payment_options_by_request(path: "str | Path") -> Dict[str, List[PaymentOption]]:
    by_request: Dict[str, List[PaymentOption]] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            option = PaymentOption(
                payment_option_id=row["payment_option_id"],
                request_id=row["request_id"],
                payment_method=row["payment_method"],
                payment_amount=float(row["payment_amount"]),
                number_of_payments=int(float(row["number_of_payments"])),
                first_payment_date=datetime.strptime(row["first_payment_date"], "%Y-%m-%d").date(),
                payment_frequency_days=_parse_optional_int(row["payment_frequency_days"]),
                financing_fee=float(row["financing_fee"]),
                total_payable_amount=float(row["total_payable_amount"]),
            )
            by_request.setdefault(option.request_id, []).append(option)
    return by_request
