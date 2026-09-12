import unittest
from datetime import date

import _pathfix  # noqa: F401
from _factories import make_event, make_profile
from forecasting.currency import RateTable
from forecasting.ledger import build_ledger


class TestLedgerStatusHandling(unittest.TestCase):
    def setUp(self):
        self.profile = make_profile()
        self.rate_table = RateTable([])
        self.request_date = date(2024, 6, 1)

    def _ledger(self, events):
        return build_ledger(
            profile=self.profile,
            user_events=events,
            rate_table=self.rate_table,
            request_date=self.request_date,
            horizon_days=90,
        )

    def test_pending_debit_is_reserved(self):
        events = [make_event("e1", status="pending", direction="debit", amount=200.0,
                              event_date=date(2024, 6, 3), settlement_date=date(2024, 6, 5))]
        result = self._ledger(events)
        self.assertEqual(len(result.entries), 1)
        self.assertEqual(result.entries[0].amount, -200.0)
        self.assertEqual(result.entries[0].date, date(2024, 6, 5))

    def test_pending_credit_refund_is_excluded(self):
        events = [make_event("e1", status="pending", direction="credit", event_type="refund",
                              amount=200.0, event_date=date(2024, 6, 3))]
        result = self._ledger(events)
        self.assertEqual(result.entries, [])

    def test_scheduled_next_salary_is_included(self):
        events = [make_event("e1", status="scheduled", direction="credit", event_type="income",
                              category="salary", amount=3000.0, event_date=date(2024, 6, 15))]
        result = self._ledger(events)
        self.assertEqual(len(result.entries), 1)
        self.assertEqual(result.entries[0].amount, 3000.0)

    def test_cancelled_and_failed_are_excluded(self):
        events = [
            make_event("e1", status="cancelled", direction="debit", amount=500.0, event_date=date(2024, 6, 3)),
            make_event("e2", status="failed", direction="debit", amount=500.0, event_date=date(2024, 6, 4)),
        ]
        result = self._ledger(events)
        self.assertEqual(result.entries, [])

    def test_unrealized_investment_valuation_is_excluded(self):
        events = [make_event("e1", status="unrealized", direction="non_cash", category="investment",
                              event_type="investment_valuation", amount=1000.0, event_date=date(2024, 6, 3))]
        result = self._ledger(events)
        self.assertEqual(result.entries, [])

    def test_blank_amount_without_override_is_excluded_and_warned(self):
        events = [make_event("e1", status="pending", direction="debit", amount=None, event_date=date(2024, 6, 3))]
        result = self._ledger(events)
        self.assertEqual(result.entries, [])
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("e1", result.warnings[0])

    def test_blank_amount_with_override_is_included(self):
        events = [make_event("e1", status="pending", direction="debit", amount=None, event_date=date(2024, 6, 3))]
        result = build_ledger(
            profile=self.profile,
            user_events=events,
            rate_table=self.rate_table,
            request_date=self.request_date,
            horizon_days=90,
            amount_overrides={"e1": 400.0},
        )
        self.assertEqual(len(result.entries), 1)
        self.assertEqual(result.entries[0].amount, -400.0)


class TestLedgerRecurrenceMerge(unittest.TestCase):
    def test_projected_recurrence_does_not_duplicate_an_explicit_scheduled_row(self):
        profile = make_profile()
        settled_rent = [
            make_event(f"r{i}", status="settled", category="rent", description="Apartment rent transfer",
                       amount=1000.0, event_date=date(2024, m, 3))
            for i, m in enumerate([1, 2, 3, 4], start=1)
        ]
        # The next rent payment is already explicitly scheduled ...
        scheduled_next = make_event("r5", status="scheduled", category="rent",
                                     description="Apartment rent transfer", amount=1000.0,
                                     event_date=date(2024, 5, 3))
        request_date = date(2024, 4, 20)
        result = build_ledger(
            profile=profile,
            user_events=settled_rent + [scheduled_next],
            rate_table=RateTable([]),
            request_date=request_date,
            horizon_days=90,
        )
        rent_dates = sorted(e.date for e in result.entries if e.category == "rent")
        # No duplicate entry for 2024-05-03, and the pattern should only
        # add occurrences strictly after it.
        self.assertEqual(rent_dates.count(date(2024, 5, 3)), 1)
        self.assertTrue(all(d > date(2024, 5, 3) or d == date(2024, 5, 3) for d in rent_dates))
        for d in rent_dates:
            if d != date(2024, 5, 3):
                self.assertGreater(d, date(2024, 5, 3))


if __name__ == "__main__":
    unittest.main()
