import unittest
from datetime import date

import _pathfix  # noqa: F401
from planning.models import Plan, PaymentOption, format_amount


class TestFormatAmount(unittest.TestCase):
    def test_whole_number_has_no_decimal_point(self):
        self.assertEqual(format_amount(25256.0), "25256")

    def test_fractional_keeps_only_needed_decimals(self):
        self.assertEqual(format_amount(17229139.2), "17229139.2")

    def test_rounds_to_two_decimal_places(self):
        self.assertEqual(format_amount(1852.111), "1852.11")


class TestPaymentOptionScheduleAndSortKey(unittest.TestCase):
    def test_sort_key_is_numeric_not_lexicographic(self):
        opt9 = PaymentOption("payment_option_9", "r1", "installments", 1.0, 1, date(2024, 1, 1), 30, 0.0, 1.0)
        opt10 = PaymentOption("payment_option_10", "r1", "installments", 1.0, 1, date(2024, 1, 1), 30, 0.0, 1.0)
        self.assertLess(opt9.sort_key, opt10.sort_key)

    def test_build_schedule_spaces_payments_by_frequency(self):
        opt = PaymentOption("payment_option_1", "r1", "installments", 100.0, 3, date(2024, 1, 1), 30, 0.0, 300.0)
        schedule = opt.build_schedule()
        self.assertEqual(schedule, ((date(2024, 1, 1), 100.0), (date(2024, 1, 31), 100.0), (date(2024, 3, 1), 100.0)))

    def test_full_payment_option_builds_a_single_payment_schedule(self):
        opt = PaymentOption("payment_option_1", "r1", "full_payment", 500.0, 1, date(2024, 3, 3), None, 0.0, 500.0)
        self.assertEqual(opt.build_schedule(), ((date(2024, 3, 3), 500.0),))


class TestPlan(unittest.TestCase):
    def test_affordability_status_full_payment_no_changes_is_affordable_now(self):
        plan = Plan(method="full_payment", payments=((date(2024, 1, 1), 100.0),), option_sort_key=1)
        self.assertEqual(plan.affordability_status, "affordable_now")

    def test_affordability_status_wait_is_affordable_later(self):
        plan = Plan(method="wait", payments=((date(2024, 2, 1), 100.0),), option_sort_key=1)
        self.assertEqual(plan.affordability_status, "affordable_later")

    def test_affordability_status_with_spending_changes_is_affordable_with_plan_regardless_of_method(self):
        plan = Plan(
            method="full_payment", payments=((date(2024, 1, 1), 100.0),), option_sort_key=1,
            spending_changes=("stop:event_1",),
        )
        self.assertEqual(plan.affordability_status, "affordable_with_plan")

    def test_affordability_status_not_recommended(self):
        from planning.models import NOT_RECOMMENDED
        self.assertEqual(NOT_RECOMMENDED.affordability_status, "not_affordable")

    def test_payment_plan_string_chronological(self):
        plan = Plan(
            method="partial_payment",
            payments=((date(2024, 1, 1), 300.0), (date(2024, 2, 15), 641.6)),
            option_sort_key=0,
        )
        self.assertEqual(plan.payment_plan_string(), "2024-01-01:300|2024-02-15:641.6")

    def test_no_payments_renders_none(self):
        from planning.models import NOT_RECOMMENDED
        self.assertEqual(NOT_RECOMMENDED.payment_plan_string(), "none")

    def test_spending_changes_string(self):
        plan = Plan(method="full_payment", payments=(), option_sort_key=0, spending_changes=("stop:e1", "reduce_to:e2:100"))
        self.assertEqual(plan.spending_changes_string(), "stop:e1|reduce_to:e2:100")

    def test_no_spending_changes_renders_none(self):
        plan = Plan(method="full_payment", payments=(), option_sort_key=0)
        self.assertEqual(plan.spending_changes_string(), "none")

    def test_completes_by_deadline(self):
        plan = Plan(method="installments", payments=((date(2024, 1, 1), 50.0), (date(2024, 3, 1), 50.0)), option_sort_key=1)
        self.assertTrue(plan.completes_by(date(2024, 3, 1)))
        self.assertFalse(plan.completes_by(date(2024, 2, 1)))


if __name__ == "__main__":
    unittest.main()
