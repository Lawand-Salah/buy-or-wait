import unittest
from datetime import date

import _pathfix  # noqa: F401
from forecasting.simulate import (
    compute_amount_safe_to_pay,
    compute_earliest_full_payment_date,
    simulate_trajectory,
)
from planning.schedule_safety import is_schedule_safe

REQUEST_DATE = date(2024, 6, 1)


class TestIsScheduleSafe(unittest.TestCase):
    def test_empty_schedule_is_always_safe(self):
        trajectory = simulate_trajectory(starting_balance=0.0, ledger_entries=[], request_date=REQUEST_DATE, horizon_days=90)
        self.assertTrue(is_schedule_safe(trajectory, minimum_balance_to_keep=1000.0, schedule=[]))

    def test_single_payment_matches_compute_amount_safe_to_pay(self):
        # A single-payment schedule must agree exactly with the existing
        # capacity function it generalizes -- same underlying rule.
        trajectory = simulate_trajectory(starting_balance=5000.0, ledger_entries=[], request_date=REQUEST_DATE, horizon_days=90)
        safe_amount = compute_amount_safe_to_pay(trajectory, minimum_balance_to_keep=1000.0, requested_amount=4000.0)
        self.assertTrue(is_schedule_safe(trajectory, minimum_balance_to_keep=1000.0, schedule=[(REQUEST_DATE, safe_amount)]))
        self.assertFalse(
            is_schedule_safe(trajectory, minimum_balance_to_keep=1000.0, schedule=[(REQUEST_DATE, safe_amount + 1)])
        )

    def test_single_payment_matches_earliest_full_payment_date(self):
        from forecasting.ledger import LedgerEntry
        entries = [LedgerEntry(date(2024, 6, 15), 5000.0, "salary", "salary", "scheduled", "e1")]
        trajectory = simulate_trajectory(starting_balance=1000.0, ledger_entries=entries, request_date=REQUEST_DATE, horizon_days=90)
        earliest = compute_earliest_full_payment_date(trajectory, minimum_balance_to_keep=500.0, requested_amount=4000.0)
        self.assertIsNotNone(earliest)
        self.assertTrue(is_schedule_safe(trajectory, minimum_balance_to_keep=500.0, schedule=[(earliest, 4000.0)]))
        # One day earlier must not be safe (that's the whole point of "earliest").
        from datetime import timedelta
        self.assertFalse(
            is_schedule_safe(trajectory, minimum_balance_to_keep=500.0, schedule=[(earliest - timedelta(days=1), 4000.0)])
        )

    def test_multi_payment_schedule_each_payment_compounds(self):
        trajectory = simulate_trajectory(starting_balance=1000.0, ledger_entries=[], request_date=REQUEST_DATE, horizon_days=90)
        # Two payments of 400 each leave 200 -- safe at a 200 minimum, unsafe at 300.
        schedule = [(REQUEST_DATE, 400.0), (date(2024, 6, 10), 400.0)]
        self.assertTrue(is_schedule_safe(trajectory, minimum_balance_to_keep=200.0, schedule=schedule))
        self.assertFalse(is_schedule_safe(trajectory, minimum_balance_to_keep=300.0, schedule=schedule))

    def test_a_transient_dip_that_recovers_before_our_payment_is_not_our_fault(self):
        # A temporary dip below the minimum, fully recovered by the time
        # our schedule's payment lands, must not fail the schedule.
        from forecasting.ledger import LedgerEntry
        entries = [
            LedgerEntry(date(2024, 6, 5), -900.0, "rent", "rent", "scheduled", "e1"),   # 1000 -> 100 (transient dip)
            LedgerEntry(date(2024, 6, 10), 2000.0, "salary", "salary", "scheduled", "e2"),  # 100 -> 2100 (recovered)
        ]
        trajectory = simulate_trajectory(starting_balance=1000.0, ledger_entries=entries, request_date=REQUEST_DATE, horizon_days=90)
        schedule = [(date(2024, 6, 20), 50.0)]  # lands after the recovery: 2100 - 50 = 2050, well above 500
        self.assertTrue(is_schedule_safe(trajectory, minimum_balance_to_keep=500.0, schedule=schedule))

    def test_a_persistent_pre_existing_shortfall_still_fails(self):
        # Unlike a transient dip, a shortfall that never recovers is a real
        # constraint at the time of our payment too -- adding to it is
        # correctly still unsafe.
        from forecasting.ledger import LedgerEntry
        entries = [LedgerEntry(date(2024, 6, 5), -900.0, "rent", "rent", "scheduled", "e1")]  # 1000 -> 100, stays there
        trajectory = simulate_trajectory(starting_balance=1000.0, ledger_entries=entries, request_date=REQUEST_DATE, horizon_days=90)
        schedule = [(date(2024, 6, 20), 50.0)]  # 100 - 50 = 50, below the 500 minimum
        self.assertFalse(is_schedule_safe(trajectory, minimum_balance_to_keep=500.0, schedule=schedule))

    def test_payment_beyond_the_horizon_is_not_checked(self):
        trajectory = simulate_trajectory(starting_balance=100.0, ledger_entries=[], request_date=REQUEST_DATE, horizon_days=90)
        far_future = trajectory.horizon_end.replace(year=trajectory.horizon_end.year + 5)
        self.assertTrue(
            is_schedule_safe(trajectory, minimum_balance_to_keep=50.0, schedule=[(far_future, 1_000_000.0)])
        )


if __name__ == "__main__":
    unittest.main()
