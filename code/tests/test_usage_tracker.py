import unittest

import _pathfix  # noqa: F401
from evidence.usage_tracker import UsageTracker


class TestUsageTracker(unittest.TestCase):
    def test_totals_by_model_aggregates_across_calls(self):
        tracker = UsageTracker()
        tracker.record(model="claude-haiku-4-5", call_type="vision", input_tokens=100, output_tokens=20)
        tracker.record(model="claude-haiku-4-5", call_type="message_classification", input_tokens=200, output_tokens=30)

        totals = tracker.totals_by_model()
        self.assertEqual(totals["claude-haiku-4-5"]["calls"], 2)
        self.assertEqual(totals["claude-haiku-4-5"]["input_tokens"], 300)
        self.assertEqual(totals["claude-haiku-4-5"]["output_tokens"], 50)

    def test_cost_computed_from_the_pricing_table(self):
        tracker = UsageTracker()
        tracker.record(model="claude-haiku-4-5", call_type="vision", input_tokens=1_000_000, output_tokens=1_000_000)
        totals = tracker.totals_by_model()
        # $1.00 input + $5.00 output per 1M tokens
        self.assertAlmostEqual(totals["claude-haiku-4-5"]["cost"], 6.00)

    def test_unknown_model_reports_cost_as_unknown_not_zero(self):
        tracker = UsageTracker()
        tracker.record(model="some-future-model", call_type="vision", input_tokens=1000, output_tokens=1000)
        totals = tracker.totals_by_model()
        self.assertFalse(totals["some-future-model"]["cost_known"])
        overall = tracker.overall_totals()
        self.assertFalse(overall["cost_known"])

    def test_distinct_request_count(self):
        tracker = UsageTracker()
        tracker.record(model="claude-haiku-4-5", call_type="vision", input_tokens=1, output_tokens=1, request_id="r1")
        tracker.record(model="claude-haiku-4-5", call_type="vision", input_tokens=1, output_tokens=1, request_id="r1")
        tracker.record(model="claude-haiku-4-5", call_type="vision", input_tokens=1, output_tokens=1, request_id="r2")
        self.assertEqual(tracker.distinct_request_count(), 2)

    def test_render_report_contains_required_sections(self):
        tracker = UsageTracker()
        tracker.record(
            model="claude-haiku-4-5", call_type="vision", input_tokens=500, output_tokens=100, request_id="r1"
        )
        report = tracker.render_report(total_requests_in_dataset=250)

        self.assertIn("claude-haiku-4-5", report)
        self.assertIn("Total model calls: 1", report)
        self.assertIn("Total input tokens: 500", report)
        self.assertIn("Total output tokens: 100", report)
        self.assertIn("Requests this run covered: 250", report)
        self.assertIn("Estimated total cost", report)
        self.assertIn("Estimated average cost per request", report)

    def test_empty_tracker_renders_without_dividing_by_zero(self):
        tracker = UsageTracker()
        report = tracker.render_report()  # no records, no total_requests_in_dataset
        self.assertIn("Total model calls: 0", report)


if __name__ == "__main__":
    unittest.main()
