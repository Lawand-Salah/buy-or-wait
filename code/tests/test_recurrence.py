import unittest
from datetime import date

import _pathfix  # noqa: F401
from _factories import make_event
from forecasting.recurrence import detect_recurrence, project_occurrences


class TestFixedDescriptionRecurrence(unittest.TestCase):
    def test_monthly_rent_same_description_is_detected(self):
        events = [
            make_event(f"e{i}", category="rent", description="Apartment rent transfer",
                       amount=5000.0, event_date=date(2024, m, 3))
            for i, m in enumerate([1, 2, 3, 4, 5], start=1)
        ]
        patterns = detect_recurrence(events)
        self.assertEqual(len(patterns), 1)
        p = patterns[0]
        self.assertEqual(p.category, "rent")
        self.assertFalse(p.is_variable_amount)
        self.assertEqual(p.projected_amount, 5000.0)
        self.assertIn(p.interval_days, range(27, 32))  # ~monthly

    def test_two_occurrences_is_not_enough(self):
        events = [
            make_event("e1", description="Apartment rent transfer", event_date=date(2024, 1, 3)),
            make_event("e2", description="Apartment rent transfer", event_date=date(2024, 2, 3)),
        ]
        self.assertEqual(detect_recurrence(events), [])

    def test_irregular_dates_are_not_flagged_as_recurring(self):
        events = [
            make_event("e1", description="One-off gift", event_date=date(2024, 1, 3)),
            make_event("e2", description="One-off gift", event_date=date(2024, 1, 20)),
            make_event("e3", description="One-off gift", event_date=date(2024, 5, 1)),
        ]
        self.assertEqual(detect_recurrence(events), [])


class TestRotatingDescriptionFallback(unittest.TestCase):
    def test_groceries_with_rotating_descriptions_falls_back_to_category_direction(self):
        descriptions = ["Fresh food shop", "Local market purchase", "Bulk pantry shop", "Grocery delivery"]
        events = [
            make_event(f"g{i}", category="groceries", description=d, amount=1000.0 + i,
                       event_date=date(2024, 1, 1 + i * 10))
            for i, d in enumerate(descriptions)
        ]
        patterns = detect_recurrence(events)
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].category, "groceries")
        self.assertIn(patterns[0].interval_days, range(8, 13))

    def test_per_description_fallback_isolates_noise_from_the_clean_signal(self):
        # A clean monthly signal under one description ...
        rent_like = [
            make_event(f"s{i}", category="shopping", description="Clothing and household items",
                       amount=1400.0, event_date=date(2024, m, 13))
            for i, m in enumerate([3, 4, 5, 6, 7], start=1)
        ]
        # ... plus two unrelated one-off noise events sharing the same category.
        noise = [
            make_event("noise1", category="shopping", description="Card charge later reversed",
                       amount=769500.0, event_date=date(2024, 6, 26)),
            make_event("noise2", category="shopping", description="Settled card purchase",
                       amount=1077300.0, event_date=date(2024, 7, 18)),
        ]
        patterns = detect_recurrence(rent_like + noise)
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].description, "Clothing and household items")
        self.assertEqual(patterns[0].occurrences_used, 5)


class TestConservativeAmountSelection(unittest.TestCase):
    def test_variable_expense_uses_the_highest_observed_amount(self):
        amounts = [900.0, 1000.0, 1100.0, 1500.0]
        events = [
            make_event(f"e{i}", category="utilities", description="Household utility payment",
                       direction="debit", amount=amt, event_date=date(2024, i, 7))
            for i, amt in enumerate(amounts, start=1)
        ]
        patterns = detect_recurrence(events)
        self.assertEqual(len(patterns), 1)
        self.assertTrue(patterns[0].is_variable_amount)
        self.assertEqual(patterns[0].projected_amount, 1500.0)

    def test_variable_income_uses_the_lowest_observed_amount(self):
        amounts = [9000.0, 12000.0, 15000.0, 17000.0]
        events = [
            make_event(f"i{i}", category="salary", description="Freelance milestone payment",
                       direction="credit", amount=amt, event_date=date(2024, i, 8))
            for i, amt in enumerate(amounts, start=1)
        ]
        patterns = detect_recurrence(events)
        self.assertEqual(len(patterns), 1)
        self.assertTrue(patterns[0].is_variable_amount)
        self.assertEqual(patterns[0].projected_amount, 9000.0)


class TestExclusions(unittest.TestCase):
    def test_investment_lifecycle_events_are_never_projected(self):
        events = [
            make_event(f"inv{i}", category="investment", description="Investment contribution",
                       event_type="investment_purchase", amount=500.0, event_date=date(2024, i, 1))
            for i in range(1, 6)
        ]
        self.assertEqual(detect_recurrence(events), [])

    def test_refunds_are_never_projected(self):
        events = [
            make_event(f"r{i}", category="shopping", description="Refund issued",
                       event_type="refund", direction="credit", amount=200.0, event_date=date(2024, i, 1))
            for i in range(1, 6)
        ]
        self.assertEqual(detect_recurrence(events), [])


class TestProjectOccurrences(unittest.TestCase):
    def test_projects_within_window_and_skips_past_explicit_anchor(self):
        events = [
            make_event(f"e{i}", category="rent", description="Apartment rent transfer",
                       amount=5000.0, event_date=date(2024, m, 3))
            for i, m in enumerate([1, 2, 3, 4, 5], start=1)
        ]
        pattern = detect_recurrence(events)[0]
        # last settled occurrence is 2024-05-03; an explicit scheduled row
        # already covers 2024-06-03, so projection should start after that
        # (phase_start), while floor is just today's forecast start date.
        occs = project_occurrences(
            pattern, phase_start=date(2024, 6, 3), floor=date(2024, 5, 20), through=date(2024, 8, 15)
        )
        self.assertGreaterEqual(len(occs), 1)
        self.assertEqual(occs, sorted(occs))
        self.assertTrue(all(d > date(2024, 6, 3) for d in occs))
        self.assertTrue(all(d <= date(2024, 8, 15) for d in occs))

    def test_floor_does_not_rephase_an_ordinary_pattern(self):
        events = [
            make_event(f"e{i}", category="rent", description="Apartment rent transfer",
                       amount=5000.0, event_date=date(2024, m, 2))
            for i, m in enumerate([1, 2, 3, 4, 5], start=1)
        ]
        pattern = detect_recurrence(events)[0]  # last known: 2024-05-02, no explicit anchor
        # request_date (floor, 2024-05-20) is later than the last known
        # date but not itself on the pattern's cadence grid; occurrences
        # must stay spaced from phase_start (the pattern's own history) by
        # whole intervals, not get re-phased to start counting from floor.
        occs = project_occurrences(
            pattern, phase_start=pattern.last_known_date, floor=date(2024, 5, 20), through=date(2024, 8, 1)
        )
        self.assertGreaterEqual(len(occs), 1)
        for d in occs:
            days_from_phase_start = (d - pattern.last_known_date).days
            self.assertEqual(days_from_phase_start % pattern.interval_days, 0)
        self.assertTrue(all(d > date(2024, 5, 20) for d in occs))

    def test_phase_start_occurrence_itself_is_never_reemitted(self):
        events = [
            make_event(f"e{i}", category="rent", description="Apartment rent transfer",
                       amount=5000.0, event_date=date(2024, m, 3))
            for i, m in enumerate([1, 2, 3, 4, 5], start=1)
        ]
        pattern = detect_recurrence(events)[0]
        occs = project_occurrences(
            pattern, phase_start=date(2024, 6, 3), floor=date(2024, 5, 20), through=date(2024, 8, 15)
        )
        self.assertNotIn(date(2024, 6, 3), occs)


if __name__ == "__main__":
    unittest.main()
