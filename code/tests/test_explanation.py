import unittest
from datetime import date

import _pathfix  # noqa: F401
from _fake_llm import FakeAnthropicClient, FakeResponse
from evidence.llm_client import LLMClient
from evidence.usage_tracker import UsageTracker
from planning.explanation import ExplanationError, build_fallback_explanation, generate_decision_explanation


def _make_llm(*responses):
    return LLMClient(usage_tracker=UsageTracker(), client=FakeAnthropicClient(list(responses)))


class TestGenerateDecisionExplanation(unittest.TestCase):
    def test_returns_the_text_and_records_usage(self):
        tracker = UsageTracker()
        client = FakeAnthropicClient([FakeResponse(data={"decision_explanation": "Pay 100 today."})])
        llm = LLMClient(usage_tracker=tracker, client=client)

        text = generate_decision_explanation(
            request_id="request_1", request_text="Can I afford this?", requested_amount=100.0, currency="USD",
            affordability_status="affordable_now", recommended_payment_method="full_payment",
            payment_plan="2024-01-01:100", earliest_date_for_full_payment=date(2024, 1, 1),
            spending_changes_needed="none", minimum_balance_to_keep=500.0, llm_client=llm,
        )
        self.assertEqual(text, "Pay 100 today.")
        self.assertEqual(len(tracker.records), 1)
        self.assertEqual(tracker.records[0].request_id, "request_1")
        self.assertEqual(tracker.records[0].call_type, "decision_explanation")

    def test_empty_text_raises_explanation_error(self):
        llm = _make_llm(FakeResponse(data={"decision_explanation": "   "}))
        with self.assertRaises(ExplanationError):
            generate_decision_explanation(
                request_id="request_2", request_text="x", requested_amount=1.0, currency="USD",
                affordability_status="not_affordable", recommended_payment_method="not_recommended",
                payment_plan="none", earliest_date_for_full_payment=None,
                spending_changes_needed="none", minimum_balance_to_keep=1.0, llm_client=llm,
            )

    def test_api_failure_raises_explanation_error(self):
        llm = _make_llm(RuntimeError("boom"))
        with self.assertRaises(ExplanationError):
            generate_decision_explanation(
                request_id="request_3", request_text="x", requested_amount=1.0, currency="USD",
                affordability_status="not_affordable", recommended_payment_method="not_recommended",
                payment_plan="none", earliest_date_for_full_payment=None,
                spending_changes_needed="none", minimum_balance_to_keep=1.0, llm_client=llm,
            )


class TestFallbackExplanation(unittest.TestCase):
    def test_not_recommended_case(self):
        text = build_fallback_explanation(
            affordability_status="not_affordable", recommended_payment_method="not_recommended",
            payment_plan="none", earliest_date_for_full_payment=None, spending_changes_needed="none",
            currency="USD", minimum_balance_to_keep=500.0,
        )
        self.assertIn("not recommended", text)
        self.assertIn("500.0", text)

    def test_mentions_spending_changes_when_present(self):
        text = build_fallback_explanation(
            affordability_status="affordable_with_plan", recommended_payment_method="full_payment",
            payment_plan="2024-01-01:100", earliest_date_for_full_payment=date(2024, 1, 1),
            spending_changes_needed="stop:event_1", currency="USD", minimum_balance_to_keep=500.0,
        )
        self.assertIn("stop:event_1", text)


if __name__ == "__main__":
    unittest.main()
