"""Generates decision_explanation: one more LLM call per request, given
only the facts the deterministic pipeline already decided. This call
explains -- it never decides anything and is never allowed to change a
number, date, or status it was given.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from evidence.llm_client import LLMCallError, LLMClient, LLMResponseError

EXPLANATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {"decision_explanation": {"type": "string"}},
    "required": ["decision_explanation"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """\
You write a short, concrete explanation for a financial affordability \
decision that a deterministic system has already made. You do not decide \
anything and must never invent, change, or second-guess any number, date, \
status, or amount given to you -- only explain the given facts in plain \
language, grounded strictly in what's provided. State the recommended \
action and amount, then the key constraint it respects (the minimum \
balance, the deadline, or a listed spending change). Keep it to 1-3 \
concise sentences, in the style of: "Pay ZAR 25,256 today. This leaves at \
least ZAR 18,000 available over the next 90 days." Respond only with the \
requested JSON field."""


class ExplanationError(Exception):
    """The LLM call for decision_explanation failed or returned nothing usable."""


def generate_decision_explanation(
    *,
    request_id: str,
    request_text: str,
    requested_amount: float,
    currency: str,
    affordability_status: str,
    recommended_payment_method: str,
    payment_plan: str,
    earliest_date_for_full_payment: Optional[date],
    spending_changes_needed: str,
    minimum_balance_to_keep: float,
    llm_client: LLMClient,
) -> str:
    user_text = (
        f'Request: "{request_text}"\n'
        f"Requested amount: {requested_amount} {currency}\n"
        "Decision facts (already determined -- explain them, do not change them):\n"
        f"- affordability_status: {affordability_status}\n"
        f"- recommended_payment_method: {recommended_payment_method}\n"
        f"- payment_plan: {payment_plan}\n"
        f"- earliest_date_for_full_payment: "
        f"{earliest_date_for_full_payment.isoformat() if earliest_date_for_full_payment else 'not safe within 90 days'}\n"
        f"- spending_changes_needed: {spending_changes_needed}\n"
        f"- minimum_balance_to_keep: {minimum_balance_to_keep} {currency}\n\n"
        "Write the decision_explanation."
    )

    try:
        raw = llm_client.call_json(
            system=_SYSTEM_PROMPT,
            user_content=user_text,
            json_schema=EXPLANATION_JSON_SCHEMA,
            call_type="decision_explanation",
            request_id=request_id,
            detail=request_id,
            max_tokens=200,
        )
    except (LLMCallError, LLMResponseError) as exc:
        raise ExplanationError(f"LLM call failed for {request_id}: {exc}") from exc

    text = str(raw.get("decision_explanation", "")).strip()
    if not text:
        raise ExplanationError(f"Empty decision_explanation for {request_id}")
    return text


def build_fallback_explanation(
    *,
    affordability_status: str,
    recommended_payment_method: str,
    payment_plan: str,
    earliest_date_for_full_payment: Optional[date],
    spending_changes_needed: str,
    currency: str,
    minimum_balance_to_keep: float,
) -> str:
    """A deterministic, no-LLM explanation used only if the LLM call
    itself fails -- so a row is never left without one.
    """
    if recommended_payment_method == "not_recommended":
        return (
            f"This request is not recommended: no safe payment plan keeps the "
            f"{minimum_balance_to_keep} {currency} minimum balance protected within the forecast period."
        )
    parts = [f"Recommended: {recommended_payment_method} ({payment_plan})."]
    if earliest_date_for_full_payment:
        parts.append(f"Full payment is safe by {earliest_date_for_full_payment.isoformat()}.")
    if spending_changes_needed and spending_changes_needed != "none":
        parts.append(f"Requires spending changes: {spending_changes_needed}.")
    parts.append(f"Keeps at least {minimum_balance_to_keep} {currency} available throughout the forecast period.")
    return " ".join(parts)
