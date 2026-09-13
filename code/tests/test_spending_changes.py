import unittest
from datetime import date

import _pathfix  # noqa: F401
from _factories import make_event, make_profile
from forecasting.ledger import LedgerEntry
from planning.spending_changes import apply_candidates_to_ledger, find_spending_change_candidates

WINDOW_START = date(2024, 1, 1)
WINDOW_END = date(2024, 3, 31)


def _profile(**kwargs):
    return make_profile(**kwargs)


class TestFindSpendingChangeCandidates(unittest.TestCase):
    def test_stoppable_event_in_a_willing_to_stop_category(self):
        profile = _profile(expense_categories_user_is_willing_to_stop=("streaming",))
        event = make_event(
            "e1", status="scheduled", category="streaming", direction="debit",
            amount=20.0, flexibility="stoppable", event_date=date(2024, 2, 1),
        )
        candidates = find_spending_change_candidates(
            user_events=[event], profile=profile, window_start=WINDOW_START, window_end=WINDOW_END,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].action, "stop")
        self.assertEqual(candidates[0].event_id, "e1")
        self.assertEqual(candidates[0].amount_freed, 20.0)
        self.assertEqual(candidates[0].action_string(), "stop:e1")

    def test_reducible_event_uses_minimum_allowed_amount(self):
        profile = _profile(expense_categories_user_is_willing_to_reduce=("groceries",))
        event = make_event(
            "e2", status="pending", category="groceries", direction="debit",
            amount=500.0, flexibility="reducible", minimum_allowed_amount=200.0, event_date=date(2024, 2, 1),
        )
        candidates = find_spending_change_candidates(
            user_events=[event], profile=profile, window_start=WINDOW_START, window_end=WINDOW_END,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].action, "reduce_to")
        self.assertEqual(candidates[0].new_amount, 200.0)
        self.assertEqual(candidates[0].amount_freed, 300.0)
        self.assertEqual(candidates[0].action_string(), "reduce_to:e2:200")

    def test_reducible_event_without_a_minimum_allowed_amount_is_skipped(self):
        profile = _profile(expense_categories_user_is_willing_to_reduce=("groceries",))
        event = make_event(
            "e3", status="pending", category="groceries", direction="debit",
            amount=500.0, flexibility="reducible", minimum_allowed_amount=None, event_date=date(2024, 2, 1),
        )
        candidates = find_spending_change_candidates(
            user_events=[event], profile=profile, window_start=WINDOW_START, window_end=WINDOW_END,
        )
        self.assertEqual(candidates, [])

    def test_protected_category_is_never_a_candidate_even_if_flexible(self):
        profile = _profile(
            expense_categories_to_protect=("rent",),
            expense_categories_user_is_willing_to_stop=("rent",),  # contradictory on purpose
        )
        event = make_event(
            "e4", status="scheduled", category="rent", direction="debit",
            amount=1000.0, flexibility="stoppable", event_date=date(2024, 2, 1),
        )
        candidates = find_spending_change_candidates(
            user_events=[event], profile=profile, window_start=WINDOW_START, window_end=WINDOW_END,
        )
        self.assertEqual(candidates, [])

    def test_fixed_flexibility_is_never_a_candidate(self):
        profile = _profile(expense_categories_user_is_willing_to_stop=("streaming",))
        event = make_event(
            "e5", status="scheduled", category="streaming", direction="debit",
            amount=20.0, flexibility="fixed", event_date=date(2024, 2, 1),
        )
        candidates = find_spending_change_candidates(
            user_events=[event], profile=profile, window_start=WINDOW_START, window_end=WINDOW_END,
        )
        self.assertEqual(candidates, [])

    def test_settled_events_are_never_candidates(self):
        # Only dated, still-pending/scheduled commitments can be changed --
        # a settled (already happened) event can't be un-spent.
        profile = _profile(expense_categories_user_is_willing_to_stop=("streaming",))
        event = make_event(
            "e6", status="settled", category="streaming", direction="debit",
            amount=20.0, flexibility="stoppable", event_date=date(2024, 2, 1),
        )
        candidates = find_spending_change_candidates(
            user_events=[event], profile=profile, window_start=WINDOW_START, window_end=WINDOW_END,
        )
        self.assertEqual(candidates, [])

    def test_outside_the_window_is_excluded(self):
        profile = _profile(expense_categories_user_is_willing_to_stop=("streaming",))
        event = make_event(
            "e7", status="scheduled", category="streaming", direction="debit",
            amount=20.0, flexibility="stoppable", event_date=date(2024, 6, 1),  # after WINDOW_END
        )
        candidates = find_spending_change_candidates(
            user_events=[event], profile=profile, window_start=WINDOW_START, window_end=WINDOW_END,
        )
        self.assertEqual(candidates, [])

    def test_sorted_by_amount_freed_descending_and_capped_at_three(self):
        profile = _profile(expense_categories_user_is_willing_to_stop=("a", "b", "c", "d"))
        events = [
            make_event(f"e{i}", status="scheduled", category=cat, direction="debit",
                       amount=amt, flexibility="stoppable", event_date=date(2024, 2, 1))
            for i, (cat, amt) in enumerate([("a", 10.0), ("b", 50.0), ("c", 30.0), ("d", 20.0)])
        ]
        candidates = find_spending_change_candidates(
            user_events=events, profile=profile, window_start=WINDOW_START, window_end=WINDOW_END,
        )
        self.assertEqual(len(candidates), 3)
        self.assertEqual([c.amount_freed for c in candidates], [50.0, 30.0, 20.0])


class TestApplyCandidatesToLedger(unittest.TestCase):
    def test_stop_removes_the_entry(self):
        entries = [LedgerEntry(date(2024, 2, 1), -20.0, "streaming", "x", "scheduled", "e1")]
        from planning.spending_changes import SpendingChangeCandidate
        result = apply_candidates_to_ledger(entries, [SpendingChangeCandidate(action="stop", event_id="e1", amount_freed=20.0)])
        self.assertEqual(result, [])

    def test_reduce_to_rewrites_the_amount_preserving_sign(self):
        entries = [LedgerEntry(date(2024, 2, 1), -500.0, "groceries", "x", "pending", "e2")]
        from planning.spending_changes import SpendingChangeCandidate
        result = apply_candidates_to_ledger(
            entries, [SpendingChangeCandidate(action="reduce_to", event_id="e2", amount_freed=300.0, new_amount=200.0)]
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].amount, -200.0)

    def test_unaffected_entries_pass_through_unchanged(self):
        entries = [LedgerEntry(date(2024, 2, 1), -50.0, "rent", "x", "scheduled", "e3")]
        result = apply_candidates_to_ledger(entries, [])
        self.assertEqual(result, entries)


if __name__ == "__main__":
    unittest.main()
