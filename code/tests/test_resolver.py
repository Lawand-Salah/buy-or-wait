import unittest
from datetime import date
from pathlib import Path

import _pathfix  # noqa: F401
from _factories import make_event, make_profile
from _fake_llm import FakeAnthropicClient, FakeResponse
from evidence.llm_client import LLMClient
from evidence.resolver import EvidenceResolver, MessageRecord
from evidence.usage_tracker import UsageTracker
from forecasting.currency import RateTable
from forecasting.forecast import run_forecast

_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
    "de000000017352474200aece1ce90000000467414d410000b18f0bfc61050000"
    "000774494d4507e50101010009fbf1980000000c49444154789c636000010000"
    "0500010d0a2db40000000049454e44ae426082"
)

_NULL_MSG_FIELDS = {
    "target_event_id": None,
    "new_status": None,
    "amendment_category": None,
    "amendment_direction": None,
    "new_amount": None,
    "new_currency": None,
    "effective_date": None,
}


def _make_llm(*responses):
    return LLMClient(usage_tracker=UsageTracker(), client=FakeAnthropicClient(list(responses)))


class TestVisionResolutionViaResolver(unittest.TestCase):
    def setUp(self):
        self.media_dir = Path(self.id().split(".")[-1] + "_media")
        self.media_dir.mkdir(exist_ok=True)
        (self.media_dir / "image_01.png").write_bytes(_TINY_PNG)
        self.addCleanup(lambda: [(self.media_dir / "image_01.png").unlink(), self.media_dir.rmdir()])

    def test_blank_amount_resolved_becomes_an_override(self):
        blank_event = make_event(
            "event_253", status="settled", direction="credit", amount=None,
            category="salary", description="August 2019 net salary", currency="IDR",
        )
        llm = _make_llm(FakeResponse(data={
            "amount": 4365000, "currency": "IDR", "confidence": "high", "notes": "net pay"
        }))
        resolver = EvidenceResolver.from_mappings(
            images_by_event_id={"event_253": "image_01"},
            messages_by_user_id={},
            media_dir=self.media_dir,
            llm_client=llm,
        )
        result = resolver.resolve_for_user(
            user_id="user_03", request_id="request_03", profile=make_profile(home_currency="IDR"),
            user_events=[blank_event],
        )
        self.assertEqual(result.amount_overrides, {"event_253": 4365000})
        self.assertEqual(result.warnings, [])
        # the original event is untouched -- resolution flows through overrides, not mutation
        self.assertIsNone(result.events[0].amount)

    def test_blank_amount_with_no_image_produces_a_warning_and_no_llm_call(self):
        blank_event = make_event("event_x", status="settled", amount=None)
        llm = _make_llm()  # no responses queued -- a call would raise AssertionError
        resolver = EvidenceResolver.from_mappings(
            images_by_event_id={}, messages_by_user_id={}, media_dir=self.media_dir, llm_client=llm,
        )
        result = resolver.resolve_for_user(
            user_id="user_x", request_id="request_x", profile=make_profile(), user_events=[blank_event],
        )
        self.assertEqual(result.amount_overrides, {})
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("event_x", result.warnings[0])


class TestMessageResolutionViaResolver(unittest.TestCase):
    def test_confirms_existing_produces_no_change(self):
        event = make_event("event_9", status="pending", direction="credit")
        llm = _make_llm(FakeResponse(data={
            **_NULL_MSG_FIELDS, "category": "confirms_existing", "reasoning": "matches"
        }))
        resolver = EvidenceResolver.from_mappings(
            images_by_event_id={}, media_dir=Path("."), llm_client=llm,
            messages_by_user_id={"user_9": [MessageRecord(
                message_id="m1", user_id="user_9", request_id=None, related_event_id="event_9",
                sent_at="2024-01-01T00:00:00Z", source_type="merchant", message_text="refund initiated",
            )]},
        )
        result = resolver.resolve_for_user(
            user_id="user_9", request_id="request_9", profile=make_profile(), user_events=[event],
        )
        self.assertEqual(result.events, [event])
        self.assertEqual(result.amount_overrides, {})

    def test_status_amendment_replaces_the_event_in_place(self):
        event = make_event("event_9", status="pending", direction="credit", amount=8640.0)
        llm = _make_llm(FakeResponse(data={
            **_NULL_MSG_FIELDS, "category": "status_amendment",
            "target_event_id": "event_9", "new_status": "settled", "reasoning": "confirmed settled",
        }))
        resolver = EvidenceResolver.from_mappings(
            images_by_event_id={}, media_dir=Path("."), llm_client=llm,
            messages_by_user_id={"user_9": [MessageRecord(
                message_id="m2", user_id="user_9", request_id=None, related_event_id="event_9",
                sent_at="2024-01-01T00:00:00Z", source_type="merchant", message_text="it settled",
            )]},
        )
        result = resolver.resolve_for_user(
            user_id="user_9", request_id="request_9", profile=make_profile(), user_events=[event],
        )
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].event_id, "event_9")
        self.assertEqual(result.events[0].status, "settled")
        self.assertEqual(result.events[0].amount, 8640.0)  # unchanged, only status was amended

    def test_income_amendment_appends_a_synthetic_scheduled_event(self):
        llm = _make_llm(FakeResponse(data={
            **_NULL_MSG_FIELDS, "category": "income_or_expense_amendment",
            "amendment_category": "salary", "amendment_direction": "credit",
            "new_amount": 42750000, "new_currency": "IDR", "effective_date": "2025-08-15",
            "reasoning": "confirmed raise",
        }))
        resolver = EvidenceResolver.from_mappings(
            images_by_event_id={}, media_dir=Path("."), llm_client=llm,
            messages_by_user_id={"user_2": [MessageRecord(
                message_id="m3", user_id="user_2", request_id=None, related_event_id=None,
                sent_at="2024-01-01T00:00:00Z", source_type="employer", message_text="salary raised",
            )]},
        )
        result = resolver.resolve_for_user(
            user_id="user_2", request_id="request_2", profile=make_profile(home_currency="IDR"), user_events=[],
        )
        self.assertEqual(len(result.events), 1)
        synthetic = result.events[0]
        self.assertEqual(synthetic.status, "scheduled")
        self.assertEqual(synthetic.direction, "credit")
        self.assertEqual(synthetic.category, "salary")
        self.assertEqual(synthetic.amount, 42750000)
        self.assertEqual(synthetic.event_date, date(2025, 8, 15))

    def test_narrative_only_message_produces_no_ledger_change(self):
        llm = _make_llm(FakeResponse(data={
            **_NULL_MSG_FIELDS, "category": "narrative_only", "reasoning": "still pending approval"
        }))
        resolver = EvidenceResolver.from_mappings(
            images_by_event_id={}, media_dir=Path("."), llm_client=llm,
            messages_by_user_id={"user_4": [MessageRecord(
                message_id="m4", user_id="user_4", request_id="request_4", related_event_id=None,
                sent_at="2024-01-01T00:00:00Z", source_type="employer", message_text="bonus expected, pending review",
            )]},
        )
        result = resolver.resolve_for_user(
            user_id="user_4", request_id="request_4", profile=make_profile(), user_events=[],
        )
        self.assertEqual(result.events, [])
        self.assertEqual(len(result.message_classifications), 1)
        self.assertEqual(result.message_classifications[0].category, "narrative_only")


class TestEvidenceFlowsUnchangedIntoForecasting(unittest.TestCase):
    def test_synthetic_event_and_override_reach_run_forecast_untouched(self):
        # A blank-amount scheduled salary event, plus a message confirming
        # a later (higher) salary -- both should show up as ledger entries
        # produced by the *unmodified* forecasting engine.
        blank_salary = make_event(
            "event_1", status="scheduled", direction="credit", category="salary",
            description="July salary", amount=None, event_date=date(2024, 8, 5),
        )
        llm = _make_llm(
            FakeResponse(data={"amount": 3000.0, "currency": "USD", "confidence": "high", "notes": "net pay"}),
            FakeResponse(data={
                **_NULL_MSG_FIELDS, "category": "income_or_expense_amendment",
                "amendment_category": "salary", "amendment_direction": "credit",
                "new_amount": 3500.0, "new_currency": "USD", "effective_date": "2024-08-15",
                "reasoning": "confirmed raise",
            }),
        )
        resolver = EvidenceResolver.from_mappings(
            images_by_event_id={"event_1": "image_01"},
            media_dir=Path(__file__).parent,  # unused: LLM is faked, image bytes are never read
            llm_client=llm,
            messages_by_user_id={"user_1": [MessageRecord(
                message_id="m5", user_id="user_1", request_id="request_1", related_event_id=None,
                sent_at="2024-08-01T00:00:00Z", source_type="employer", message_text="raise confirmed",
            )]},
        )
        # Vision needs a real file on disk; point it at the tiny PNG fixture.
        image_path = Path(self.id().split(".")[-1] + "_img.png")
        image_path.write_bytes(_TINY_PNG)
        self.addCleanup(image_path.unlink)
        resolver.media_dir = image_path.parent
        resolver.images_by_event_id = {"event_1": image_path.stem}

        profile = make_profile(
            home_currency="USD", current_available_balance=5000.0, minimum_balance_to_keep=1000.0,
        )
        result = resolver.resolve_for_user(
            user_id="user_1", request_id="request_1", profile=profile, user_events=[blank_salary],
        )

        forecast = run_forecast(
            profile=profile, user_events=result.events, rate_table=RateTable([]),
            request_date=date(2024, 8, 1), requested_amount=100.0,
            horizon_days=90, amount_overrides=result.amount_overrides,
        )

        self.assertEqual(forecast.warnings, [])
        salary_dates = {e.date: e.amount for e in forecast.ledger_entries if e.category == "salary"}
        # The resolved image override reaches the ledger via the original
        # (now-resolved) scheduled event ...
        self.assertIn(date(2024, 8, 5), salary_dates)
        self.assertEqual(salary_dates[date(2024, 8, 5)], 3000.0)
        # ... and the message-confirmed raise reaches it via the appended
        # synthetic scheduled event -- both without any forecasting.* change.
        self.assertIn(date(2024, 8, 15), salary_dates)
        self.assertEqual(salary_dates[date(2024, 8, 15)], 3500.0)


if __name__ == "__main__":
    unittest.main()
