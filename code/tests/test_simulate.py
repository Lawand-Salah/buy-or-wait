import unittest
from datetime import date

import _pathfix  # noqa: F401
from forecasting.ledger import LedgerEntry
from forecasting.simulate import (
    compute_amount_safe_to_pay,
    compute_earliest_full_payment_date,
    simulate_trajectory,
)

REQUEST_DATE = date(2024, 6, 1)


class TestSimulateTrajectory(unittest.TestCase):
    def test_flat_balance_with_no_ledger_entries(self):
        trajectory = simulate_trajectory(
            starting_balance=1000.0, ledger_entries=[], request_date=REQUEST_DATE, horizon_days=90
        )
        self.assertEqual(len(trajectory.dates), 91)  # inclusive of both endpoints
        self.assertEqual(trajectory.minimum, 1000.0)
        self.assertEqual(trajectory.daily_balance[REQUEST_DATE], 1000.0)

    def test_entries_apply_cumulatively_on_their_date(self):
        entries = [
            LedgerEntry(date(2024, 6, 5), -200.0, "rent", "rent", "pending", "e1"),
            LedgerEntry(date(2024, 6, 10), 500.0, "salary", "salary", "scheduled", "e2"),
        ]
        trajectory = simulate_trajectory(
            starting_balance=1000.0, ledger_entries=entries, request_date=REQUEST_DATE, horizon_days=90
        )
        self.assertEqual(trajectory.daily_balance[date(2024, 6, 4)], 1000.0)
        self.assertEqual(trajectory.daily_balance[date(2024, 6, 5)], 800.0)
        self.assertEqual(trajectory.daily_balance[date(2024, 6, 9)], 800.0)
        self.assertEqual(trajectory.daily_balance[date(2024, 6, 10)], 1300.0)

    def test_entries_outside_the_window_are_ignored(self):
        entries = [LedgerEntry(date(2024, 12, 1), -999.0, "x", "x", "scheduled", "e1")]
        trajectory = simulate_trajectory(
            starting_balance=1000.0, ledger_entries=entries, request_date=REQUEST_DATE, horizon_days=90
        )
        self.assertEqual(trajectory.minimum, 1000.0)


class TestAmountSafeToPay(unittest.TestCase):
    def test_capped_at_requested_amount(self):
        trajectory = simulate_trajectory(
            starting_balance=100000.0, ledger_entries=[], request_date=REQUEST_DATE, horizon_days=90
        )
        safe = compute_amount_safe_to_pay(trajectory, minimum_balance_to_keep=1000.0, requested_amount=500.0)
        self.assertEqual(safe, 500.0)

    def test_floored_at_zero_when_already_below_minimum(self):
        trajectory = simulate_trajectory(
            starting_balance=500.0, ledger_entries=[], request_date=REQUEST_DATE, horizon_days=90
        )
        safe = compute_amount_safe_to_pay(trajectory, minimum_balance_to_keep=1000.0, requested_amount=500.0)
        self.assertEqual(safe, 0.0)

    def test_a_future_dip_is_the_binding_constraint_not_todays_balance(self):
        # Today's balance looks generous, but a big bill lands in three weeks.
        entries = [LedgerEntry(date(2024, 6, 21), -9000.0, "rent", "rent", "scheduled", "e1")]
        trajectory = simulate_trajectory(
            starting_balance=10000.0, ledger_entries=entries, request_date=REQUEST_DATE, horizon_days=90
        )
        # Worst day balance = 10000 - 9000 = 1000; minimum_balance_to_keep=500
        # => only 500 is safe to pay today, even though 10000-500=9500 looks safe in isolation.
        safe = compute_amount_safe_to_pay(trajectory, minimum_balance_to_keep=500.0, requested_amount=9999.0)
        self.assertEqual(safe, 500.0)


class TestEarliestFullPaymentDate(unittest.TestCase):
    def test_affordable_now_returns_request_date(self):
        trajectory = simulate_trajectory(
            starting_balance=10000.0, ledger_entries=[], request_date=REQUEST_DATE, horizon_days=90
        )
        earliest = compute_earliest_full_payment_date(
            trajectory, minimum_balance_to_keep=500.0, requested_amount=100.0
        )
        self.assertEqual(earliest, REQUEST_DATE)

    def test_waits_for_incoming_salary_when_not_safe_yet(self):
        entries = [LedgerEntry(date(2024, 6, 15), 5000.0, "salary", "salary", "scheduled", "e1")]
        trajectory = simulate_trajectory(
            starting_balance=1000.0, ledger_entries=entries, request_date=REQUEST_DATE, horizon_days=90
        )
        earliest = compute_earliest_full_payment_date(
            trajectory, minimum_balance_to_keep=500.0, requested_amount=4000.0
        )
        self.assertEqual(earliest, date(2024, 6, 15))

    def test_none_when_never_safe_within_the_horizon(self):
        trajectory = simulate_trajectory(
            starting_balance=1000.0, ledger_entries=[], request_date=REQUEST_DATE, horizon_days=90
        )
        earliest = compute_earliest_full_payment_date(
            trajectory, minimum_balance_to_keep=500.0, requested_amount=1_000_000.0
        )
        self.assertIsNone(earliest)

    def test_a_later_dip_constrains_every_earlier_day_too(self):
        # A large debit ten days out means paying the full amount on ANY
        # day up to and including that dip must still clear it, since the
        # payment stays subtracted going forward.
        entries = [LedgerEntry(date(2024, 6, 11), -3000.0, "rent", "rent", "scheduled", "e1")]
        trajectory = simulate_trajectory(
            starting_balance=5000.0, ledger_entries=entries, request_date=REQUEST_DATE, horizon_days=90
        )
        # Paying 1500 on request_date leaves 3500; the 2024-06-11 rent then
        # drops it to exactly 500, the minimum -- still safe.
        earliest = compute_earliest_full_payment_date(
            trajectory, minimum_balance_to_keep=500.0, requested_amount=1500.0
        )
        self.assertEqual(earliest, REQUEST_DATE)

        # One unit more breaks that same floor on every day of the window
        # (post-dip balance is a flat 2000 for the rest of the horizon),
        # so no day ever qualifies.
        earliest_larger = compute_earliest_full_payment_date(
            trajectory, minimum_balance_to_keep=500.0, requested_amount=1501.0
        )
        self.assertIsNone(earliest_larger)


if __name__ == "__main__":
    unittest.main()
