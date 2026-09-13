#!/usr/bin/env python3
"""Buy or Wait? -- full pipeline entry point.

For every row in dataset/requests.csv:
  1. Resolve evidence (blank amounts via vision, messages via
     classification) -- code/evidence/.
  2. Run the 90-day forecast on the evidence-augmented event list --
     code/forecasting/.
  3. Select the best safe plan against the supplied payment options,
     falling back to spending changes only if nothing else is safe --
     code/planning/.
  4. Generate decision_explanation via one more LLM call.

Writes dataset/output.csv and code/evaluation/usage_report.md.

Setup:
    pip install -r code/requirements.txt
    # ANTHROPIC_API_KEY in a .env file at the repo root, or exported

Run (from anywhere; paths below are resolved relative to this file):
    python code/main.py
    python code/main.py --limit 5              # quick smoke test
    python code/main.py --request-id request_1  # a single row, for debugging
"""
from __future__ import annotations

import argparse
import csv
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from evidence.llm_client import LLMClient  # noqa: E402
from evidence.resolver import EvidenceResolver  # noqa: E402
from evidence.usage_tracker import UsageTracker  # noqa: E402
from forecasting.forecast import Forecaster, run_forecast  # noqa: E402
from planning.explanation import (  # noqa: E402
    ExplanationError,
    build_fallback_explanation,
    generate_decision_explanation,
)
from planning.models import format_amount  # noqa: E402
from planning.options import load_payment_options_by_request  # noqa: E402
from planning.selector import RequestInfo, select_plan  # noqa: E402

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


def _parse_date(value: str):
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def _parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def load_requests(path: Path) -> list:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def process_one_request(
    *,
    row: dict,
    forecaster: Forecaster,
    resolver: EvidenceResolver,
    payment_options_by_request: dict,
    llm_client: LLMClient,
) -> dict:
    """Runs the full pipeline for one requests.csv row and returns the
    output.csv row as a dict. Never raises -- any failure produces a safe
    fallback row instead, so one bad row never loses the whole run.
    """
    request_id = row["request_id"]
    try:
        user_id = row["user_id"]
        request_date = _parse_date(row["request_date"])
        requested_amount = float(row["requested_amount"])
        desired_completion_date = _parse_date(row["desired_completion_date"])
        allows_partial_payment = _parse_bool(row["allows_partial_payment"])

        profile = forecaster.profiles[user_id]
        user_events = forecaster.events_by_user.get(user_id, [])

        evidence = resolver.resolve_for_user(
            user_id=user_id, request_id=request_id, profile=profile, user_events=user_events,
        )

        # Not forecaster.forecast(): that looks up the user's *raw* CSV
        # events internally. We need run_forecast() directly, over the
        # evidence-augmented event list (resolved amounts, corrected/
        # synthetic events) instead.
        forecast = run_forecast(
            profile=profile, user_events=evidence.events, rate_table=forecaster.rate_table,
            request_date=request_date, requested_amount=requested_amount,
            amount_overrides=evidence.amount_overrides,
        )

        options = payment_options_by_request.get(request_id, [])
        request_info = RequestInfo(
            request_date=request_date, requested_amount=requested_amount,
            desired_completion_date=desired_completion_date, allows_partial_payment=allows_partial_payment,
        )
        plan = select_plan(
            forecast_result=forecast, profile=profile, user_events=evidence.events,
            request=request_info, payment_options=options,
        )

        amount_safe_to_pay = forecast.amount_safe_to_pay
        affordability_status = plan.affordability_status
        recommended_payment_method = plan.method
        payment_plan_str = plan.payment_plan_string()
        earliest_date = forecast.earliest_date_for_full_payment
        spending_changes_str = plan.spending_changes_string()

        try:
            explanation = generate_decision_explanation(
                request_id=request_id, request_text=row["request_text"], requested_amount=requested_amount,
                currency=profile.home_currency, affordability_status=affordability_status,
                recommended_payment_method=recommended_payment_method, payment_plan=payment_plan_str,
                earliest_date_for_full_payment=earliest_date, spending_changes_needed=spending_changes_str,
                minimum_balance_to_keep=profile.minimum_balance_to_keep, llm_client=llm_client,
            )
        except ExplanationError as exc:
            print(f"[{request_id}] explanation LLM call failed, using fallback text: {exc}", file=sys.stderr)
            explanation = build_fallback_explanation(
                affordability_status=affordability_status, recommended_payment_method=recommended_payment_method,
                payment_plan=payment_plan_str, earliest_date_for_full_payment=earliest_date,
                spending_changes_needed=spending_changes_str, currency=profile.home_currency,
                minimum_balance_to_keep=profile.minimum_balance_to_keep,
            )

        for warning in evidence.warnings:
            print(f"[{request_id}] {warning}", file=sys.stderr)

        return {
            "request_id": request_id,
            "amount_safe_to_pay": format_amount(amount_safe_to_pay),
            "affordability_status": affordability_status,
            "recommended_payment_method": recommended_payment_method,
            "payment_plan": payment_plan_str,
            "earliest_date_for_full_payment": earliest_date.isoformat() if earliest_date else "",
            "spending_changes_needed": spending_changes_str,
            "decision_explanation": explanation,
        }
    except Exception:  # noqa: BLE001 -- a single bad row must never abort the whole run
        print(f"[{request_id}] FAILED, writing not_recommended fallback row:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {
            "request_id": request_id,
            "amount_safe_to_pay": "0",
            "affordability_status": "not_affordable",
            "recommended_payment_method": "not_recommended",
            "payment_plan": "none",
            "earliest_date_for_full_payment": "",
            "spending_changes_needed": "none",
            "decision_explanation": "Unable to evaluate this request due to a processing error.",
        }


def run(
    *,
    dataset_dir: Path,
    output_path: Path,
    usage_report_path: Path,
    limit: Optional[int] = None,
    only_request_id: Optional[str] = None,
) -> None:
    requests_rows = load_requests(dataset_dir / "requests.csv")
    if only_request_id:
        requests_rows = [r for r in requests_rows if r["request_id"] == only_request_id]
    if limit is not None:
        requests_rows = requests_rows[:limit]

    print(f"Loading dataset from {dataset_dir} ...")
    forecaster = Forecaster(dataset_dir)
    usage_tracker = UsageTracker()
    llm_client = LLMClient(usage_tracker=usage_tracker)
    resolver = EvidenceResolver(dataset_dir, llm_client)
    payment_options_by_request = load_payment_options_by_request(dataset_dir / "request_payment_options.csv")

    output_rows = []
    total = len(requests_rows)
    for i, row in enumerate(requests_rows, start=1):
        output_rows.append(
            process_one_request(
                row=row, forecaster=forecaster, resolver=resolver,
                payment_options_by_request=payment_options_by_request, llm_client=llm_client,
            )
        )
        if i % 10 == 0 or i == total:
            print(f"  processed {i}/{total} requests")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows to {output_path}")

    usage_report_path.parent.mkdir(parents=True, exist_ok=True)
    usage_tracker.write_report(usage_report_path, total_requests_in_dataset=total)
    print(f"Wrote usage report to {usage_report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default=str(REPO_ROOT / "dataset"))
    parser.add_argument("--output", default=str(REPO_ROOT / "dataset" / "output.csv"))
    parser.add_argument("--usage-report", default=str(CODE_DIR / "evaluation" / "usage_report.md"))
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N requests (for a quick smoke test).")
    parser.add_argument("--request-id", default=None, help="Process only this one request_id (for debugging).")
    args = parser.parse_args()

    run(
        dataset_dir=Path(args.dataset_dir),
        output_path=Path(args.output),
        usage_report_path=Path(args.usage_report),
        limit=args.limit,
        only_request_id=args.request_id,
    )


if __name__ == "__main__":
    main()
