"""Resolve one blank-amount financial_events.csv row from its linked image.

problem_statement.md: "When a financial event has a blank amount, use its
event_id to find the matching related_event_id in images.csv, then extract
the amount from that image. Do not treat a blank amount as zero."

This module never fabricates a number. If the model can't confidently
read one final amount, ``resolve_amount_from_image`` raises
``VisionResolutionError`` rather than guessing -- the caller (resolver.py)
then leaves that event unresolved, exactly like the "no override supplied"
path forecasting.ledger already handles.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .llm_client import LLMClient

VISION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "amount": {"type": "number"},
        "currency": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "notes": {"type": "string"},
    },
    "required": ["amount", "currency", "confidence", "notes"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "You extract a single financial amount from a receipt, payslip, or invoice "
    "image for an automated bookkeeping pipeline. Respond only with the "
    "requested JSON fields. Read the ONE final amount the document confirms "
    "(e.g. net pay, total paid, final invoice/receipt total) -- not a "
    "subtotal, a tax line, or an unrelated figure elsewhere on the page. If "
    "you cannot confidently identify that figure, still return your best "
    "reading but set confidence to \"low\" -- never invent a number you "
    "can't find in the image."
)


class VisionResolutionError(Exception):
    """The image couldn't be confidently resolved to a single trusted amount."""


@dataclass(frozen=True)
class VisionResult:
    amount: float
    currency: str
    confidence: str  # "high" | "medium" | "low"
    notes: str


def resolve_amount_from_image(
    *,
    image_path: "str | Path",
    event_description: str,
    expected_currency: str,
    llm_client: LLMClient,
    request_id: Optional[str] = None,
    detail: str = "",
) -> VisionResult:
    path = Path(image_path)
    if not path.is_file():
        raise VisionResolutionError(f"Image file not found: {path}")

    image_b64 = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    user_content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
        {
            "type": "text",
            "text": (
                f'This image documents: "{event_description}" (expected currency: {expected_currency}). '
                "Extract the single final amount this document confirms."
            ),
        },
    ]

    raw = llm_client.call_json(
        system=_SYSTEM_PROMPT,
        user_content=user_content,
        json_schema=VISION_JSON_SCHEMA,
        call_type="vision",
        request_id=request_id,
        detail=detail or path.name,
        max_tokens=300,
    )

    try:
        amount = float(raw["amount"])
        currency = str(raw["currency"])
        confidence = str(raw["confidence"])
        notes = str(raw["notes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VisionResolutionError(f"Malformed vision result for {detail}: {raw!r}") from exc

    if confidence not in ("high", "medium", "low"):
        raise VisionResolutionError(f"Unexpected confidence value for {detail}: {confidence!r}")
    if amount <= 0:
        raise VisionResolutionError(f"Non-positive amount extracted for {detail}: {amount!r}")
    if confidence == "low":
        # Per problem_statement.md, don't treat a blank amount as zero --
        # but don't trust a low-confidence guess either. Leaving it
        # unresolved (via this exception) is the safer of the two.
        raise VisionResolutionError(f"Low-confidence extraction for {detail}, not trusted: {notes}")

    return VisionResult(amount=amount, currency=currency, confidence=confidence, notes=notes)
