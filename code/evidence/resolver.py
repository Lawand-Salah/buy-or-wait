"""Orchestrates vision.py + messages.py into what forecasting.forecast
already knows how to consume: an augmented event list plus an
amount_overrides dict.

resolve_for_user() is the one entry point. It never touches
code/forecasting/ -- everything it does is producing a different *input*
to run_forecast()/Forecaster.forecast(), not a different code path inside
it:

* A resolved blank amount becomes one entry in ``amount_overrides``
  (exactly the parameter run_forecast() already accepts).
* A "status_amendment" message becomes a corrected *copy* of that one
  Event, substituted in place -- forecasting.ledger reads status/amount
  generically regardless of where the Event came from.
* An "income_or_expense_amendment" message becomes a new synthetic
  ``Event(status="scheduled", ...)`` appended to the list --
  forecasting.recurrence/ledger already treat any scheduled row as an
  explicit anchor, so it naturally keeps the recurrence engine from
  double-counting the corresponding pattern past that date.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional

from forecasting.models import Event, Profile

from .llm_client import LLMClient
from .messages import MessageClassification, MessageClassificationError, classify_message
from .vision import VisionResolutionError, resolve_amount_from_image


@dataclass(frozen=True)
class MessageRecord:
    message_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    sent_at: str
    source_type: str
    message_text: str


@dataclass(frozen=True)
class EvidenceResult:
    events: List[Event]  # originals, with any status_amendment applied and synthetic rows appended
    amount_overrides: Dict[str, float]
    warnings: List[str]
    message_classifications: List[MessageClassification] = field(default_factory=list)


def _load_images_by_event_id(path: Path) -> Dict[str, str]:
    """{related_event_id: image_id} from images.csv."""
    mapping: Dict[str, str] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            event_id = row.get("related_event_id", "").strip()
            if event_id:
                mapping[event_id] = row["image_id"]
    return mapping


def _load_messages_by_user_id(path: Path) -> Dict[str, List[MessageRecord]]:
    by_user: Dict[str, List[MessageRecord]] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            record = MessageRecord(
                message_id=row["message_id"],
                user_id=row["user_id"],
                request_id=row["request_id"].strip() or None,
                related_event_id=row["related_event_id"].strip() or None,
                sent_at=row["sent_at"],
                source_type=row["source_type"],
                message_text=row["message_text"],
            )
            by_user.setdefault(record.user_id, []).append(record)
    return by_user


class EvidenceResolver:
    def __init__(self, dataset_dir: "str | Path", llm_client: LLMClient) -> None:
        dataset_dir = Path(dataset_dir)
        self.images_by_event_id = _load_images_by_event_id(dataset_dir / "images.csv")
        self.messages_by_user_id = _load_messages_by_user_id(dataset_dir / "messages.csv")
        self.media_dir = dataset_dir / "media" / "images"
        self.llm_client = llm_client

    @classmethod
    def from_mappings(
        cls,
        *,
        images_by_event_id: Dict[str, str],
        messages_by_user_id: Dict[str, List[MessageRecord]],
        media_dir: "str | Path",
        llm_client: LLMClient,
    ) -> "EvidenceResolver":
        """Alternate constructor that skips CSV loading -- lets tests build
        a resolver directly from in-memory mappings instead of real files.
        """
        self = cls.__new__(cls)
        self.images_by_event_id = images_by_event_id
        self.messages_by_user_id = messages_by_user_id
        self.media_dir = Path(media_dir)
        self.llm_client = llm_client
        return self

    def resolve_for_user(
        self,
        *,
        user_id: str,
        request_id: Optional[str],
        profile: Profile,
        user_events: List[Event],
    ) -> EvidenceResult:
        events = list(user_events)
        events_by_id = {e.event_id: e for e in events}
        amount_overrides: Dict[str, float] = {}
        warnings: List[str] = []
        classifications: List[MessageClassification] = []

        for e in user_events:
            if e.amount is not None:
                continue
            image_id = self.images_by_event_id.get(e.event_id)
            if image_id is None:
                warnings.append(f"{e.event_id}: blank amount, no image available; excluded")
                continue

            image_path = self.media_dir / f"{image_id}.png"
            try:
                result = resolve_amount_from_image(
                    image_path=image_path,
                    event_description=e.description,
                    expected_currency=e.currency,
                    llm_client=self.llm_client,
                    request_id=request_id,
                    detail=e.event_id,
                )
            except VisionResolutionError as exc:
                warnings.append(f"{e.event_id}: vision resolution failed ({exc}); excluded")
                continue

            if result.currency != e.currency:
                # The event's own currency (a CSV fact) is authoritative for
                # how the ledger will convert this amount; a mismatch here
                # is worth surfacing but doesn't change which currency we use.
                warnings.append(
                    f"{e.event_id}: image read currency {result.currency!r}, "
                    f"differs from the event's recorded currency {e.currency!r}; using {e.currency!r}"
                )
            amount_overrides[e.event_id] = result.amount

        for m in self.messages_by_user_id.get(user_id, []):
            related_event = events_by_id.get(m.related_event_id) if m.related_event_id else None
            try:
                classification = classify_message(
                    message_id=m.message_id,
                    message_text=m.message_text,
                    source_type=m.source_type,
                    sent_at=m.sent_at,
                    related_event=related_event,
                    home_currency=profile.home_currency,
                    llm_client=self.llm_client,
                    request_id=request_id,
                )
            except MessageClassificationError as exc:
                warnings.append(f"{m.message_id}: message classification failed ({exc}); ignored")
                continue

            classifications.append(classification)

            if classification.category in ("confirms_existing", "narrative_only"):
                continue

            if classification.category == "status_amendment":
                target = events_by_id.get(classification.target_event_id or "")
                if target is None:
                    warnings.append(
                        f"{m.message_id}: status_amendment target "
                        f"{classification.target_event_id!r} not found; ignored"
                    )
                    continue
                corrected = replace(
                    target,
                    status=classification.new_status or target.status,
                    amount=classification.new_amount if classification.new_amount is not None else target.amount,
                    currency=classification.new_currency or target.currency,
                )
                # Look up by event_id, not list.index(target) -- Event is a
                # plain dataclass, so index() would match by field-value
                # equality rather than identity, and could pick the wrong
                # row if two events ever had identical field values.
                idx = next(i for i, e in enumerate(events) if e.event_id == target.event_id)
                events[idx] = corrected
                events_by_id[corrected.event_id] = corrected

            elif classification.category == "income_or_expense_amendment":
                synthetic = Event(
                    event_id=f"msg_{classification.message_id}_synthetic",
                    user_id=user_id,
                    event_type="income" if classification.amendment_direction == "credit" else "expense",
                    description=f"Message-confirmed update ({classification.message_id})",
                    category=classification.amendment_category or "other",
                    direction=classification.amendment_direction or "debit",
                    amount=classification.new_amount,
                    currency=classification.new_currency or profile.home_currency,
                    event_date=classification.effective_date,
                    settlement_date=classification.effective_date,
                    status="scheduled",
                    linked_event_id=None,
                    flexibility="fixed",
                    minimum_allowed_amount=None,
                )
                events.append(synthetic)
                events_by_id[synthetic.event_id] = synthetic

        return EvidenceResult(
            events=events,
            amount_overrides=amount_overrides,
            warnings=warnings,
            message_classifications=classifications,
        )
