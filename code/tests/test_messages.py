import unittest
from datetime import date

import _pathfix  # noqa: F401
from _factories import make_event
from _fake_llm import FakeAnthropicClient, FakeResponse
from evidence.llm_client import LLMClient
from evidence.messages import MessageClassificationError, classify_message
from evidence.usage_tracker import UsageTracker

_NULL_FIELDS = {
    "target_event_id": None,
    "new_status": None,
    "amendment_category": None,
    "amendment_direction": None,
    "new_amount": None,
    "new_currency": None,
    "effective_date": None,
}


def _make_llm(data):
    return LLMClient(usage_tracker=UsageTracker(), client=FakeAnthropicClient([FakeResponse(data=data)]))


class TestClassifyMessage(unittest.TestCase):
    def test_confirms_existing(self):
        llm = _make_llm({**_NULL_FIELDS, "category": "confirms_existing", "reasoning": "matches CSV"})
        result = classify_message(
            message_id="m1", message_text="your refund is on its way", source_type="merchant",
            sent_at="2024-01-01T00:00:00Z", related_event=None, home_currency="USD", llm_client=llm,
        )
        self.assertEqual(result.category, "confirms_existing")

    def test_narrative_only_for_speculative_language(self):
        llm = _make_llm({**_NULL_FIELDS, "category": "narrative_only", "reasoning": "still pending approval"})
        result = classify_message(
            message_id="m2", message_text="your bonus is expected, pending final review", source_type="employer",
            sent_at="2024-01-01T00:00:00Z", related_event=None, home_currency="USD", llm_client=llm,
        )
        self.assertEqual(result.category, "narrative_only")

    def test_status_amendment_requires_target_and_status(self):
        llm = _make_llm({**_NULL_FIELDS, "category": "status_amendment", "reasoning": "r"})  # no target/status
        with self.assertRaises(MessageClassificationError):
            classify_message(
                message_id="m3", message_text="it settled", source_type="merchant",
                sent_at="2024-01-01T00:00:00Z", related_event=None, home_currency="USD", llm_client=llm,
            )

    def test_status_amendment_valid(self):
        event = make_event("event_9", status="pending", direction="credit")
        llm = _make_llm({
            **_NULL_FIELDS, "category": "status_amendment",
            "target_event_id": "event_9", "new_status": "settled", "reasoning": "confirmed settled",
        })
        result = classify_message(
            message_id="m4", message_text="your refund has settled", source_type="merchant",
            sent_at="2024-01-01T00:00:00Z", related_event=event, home_currency="USD", llm_client=llm,
        )
        self.assertEqual(result.category, "status_amendment")
        self.assertEqual(result.target_event_id, "event_9")
        self.assertEqual(result.new_status, "settled")

    def test_status_amendment_target_mismatch_rejected(self):
        event = make_event("event_9", status="pending", direction="credit")
        llm = _make_llm({
            **_NULL_FIELDS, "category": "status_amendment",
            "target_event_id": "event_OTHER", "new_status": "settled", "reasoning": "r",
        })
        with self.assertRaises(MessageClassificationError):
            classify_message(
                message_id="m5", message_text="x", source_type="merchant", sent_at="2024-01-01T00:00:00Z",
                related_event=event, home_currency="USD", llm_client=llm,
            )

    def test_income_amendment_requires_all_fields(self):
        llm = _make_llm({
            **_NULL_FIELDS, "category": "income_or_expense_amendment",
            "amendment_category": "salary", "reasoning": "raise",
            # amendment_direction / new_amount / effective_date left null -> should fail
        })
        with self.assertRaises(MessageClassificationError):
            classify_message(
                message_id="m6", message_text="your salary is now X", source_type="employer",
                sent_at="2024-01-01T00:00:00Z", related_event=None, home_currency="IDR", llm_client=llm,
            )

    def test_income_amendment_valid(self):
        llm = _make_llm({
            **_NULL_FIELDS, "category": "income_or_expense_amendment",
            "amendment_category": "salary", "amendment_direction": "credit",
            "new_amount": 42750000, "new_currency": "IDR", "effective_date": "2025-08-15",
            "reasoning": "confirmed raise",
        })
        result = classify_message(
            message_id="m7", message_text="salary raised", source_type="employer",
            sent_at="2024-01-01T00:00:00Z", related_event=None, home_currency="IDR", llm_client=llm,
        )
        self.assertEqual(result.category, "income_or_expense_amendment")
        self.assertEqual(result.new_amount, 42750000)
        self.assertEqual(result.effective_date, date(2025, 8, 15))

    def test_invalid_category_rejected(self):
        llm = _make_llm({**_NULL_FIELDS, "category": "not_a_real_category", "reasoning": "r"})
        with self.assertRaises(MessageClassificationError):
            classify_message(
                message_id="m8", message_text="x", source_type="merchant", sent_at="2024-01-01T00:00:00Z",
                related_event=None, home_currency="USD", llm_client=llm,
            )

    def test_invalid_effective_date_format_rejected(self):
        llm = _make_llm({
            **_NULL_FIELDS, "category": "income_or_expense_amendment",
            "amendment_category": "salary", "amendment_direction": "credit",
            "new_amount": 100, "new_currency": "USD", "effective_date": "15 Aug 2025",
            "reasoning": "r",
        })
        with self.assertRaises(MessageClassificationError):
            classify_message(
                message_id="m9", message_text="x", source_type="employer", sent_at="2024-01-01T00:00:00Z",
                related_event=None, home_currency="USD", llm_client=llm,
            )


if __name__ == "__main__":
    unittest.main()
