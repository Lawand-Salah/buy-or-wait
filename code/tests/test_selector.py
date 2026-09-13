import unittest
from datetime import date

import _pathfix  # noqa: F401
from _factories import make_event, make_profile
from forecasting.currency import RateTable
from forecasting.forecast import run_forecast
from planning.models import PaymentOption
from planning.selector import RequestInfo, select_plan

REQUEST_DATE = date(2024, 6, 1)


def _forecast(*, profile, events, requested_amount):
    return run_forecast(
        profile=profile, user_events=events, rate_table=RateTable([]),
        request_date=REQUEST_DATE, requested_amount=requested_amount, horizon_days=90,
    )


def _full_option(amount, option_id="payment_option_1"):
    return PaymentOption(option_id, "r1", "full_payment", amount, 1, REQUEST_DATE, None, 0.0, amount)


def _installment_option(*, option_id, number_of_payments, payment_amount, first_payment_date=REQUEST_DATE, frequency=30):
    total = payment_amount * number_of_payments
    return PaymentOption(option_id, "r1", "installments", payment_amount, number_of_payments, first_payment_date, frequency, 0.0, total)


class TestFullPaymentAndWait(unittest.TestCase):
    def test_full_payment_now_when_safe_and_accepted(self):
        profile = make_profile(current_available_balance=10000.0, minimum_balance_to_keep=1000.0,
                                payment_methods_user_will_consider=("full_payment",))
        forecast = _forecast(profile=profile, events=[], requested_amount=500.0)
        request = RequestInfo(request_date=REQUEST_DATE, requested_amount=500.0,
                               desired_completion_date=date(2024, 7, 1), allows_partial_payment=False)
        plan = select_plan(forecast_result=forecast, profile=profile, user_events=[], request=request,
                            payment_options=[_full_option(500.0)])
        self.assertEqual(plan.method, "full_payment")
        self.assertEqual(plan.affordability_status, "affordable_now")
        self.assertEqual(plan.payments, ((REQUEST_DATE, 500.0),))

    def test_wait_when_not_safe_now_but_safe_after_incoming_salary(self):
        profile = make_profile(current_available_balance=100.0, minimum_balance_to_keep=50.0,
                                payment_methods_user_will_consider=("full_payment",))
        salary = make_event("e1", status="scheduled", direction="credit", category="salary",
                            amount=5000.0, event_date=date(2024, 6, 15))
        forecast = _forecast(profile=profile, events=[salary], requested_amount=1000.0)
        request = RequestInfo(request_date=REQUEST_DATE, requested_amount=1000.0,
                               desired_completion_date=date(2024, 8, 1), allows_partial_payment=False)
        plan = select_plan(forecast_result=forecast, profile=profile, user_events=[salary], request=request,
                            payment_options=[_full_option(1000.0)])
        self.assertEqual(plan.method, "wait")
        self.assertEqual(plan.affordability_status, "affordable_later")
        self.assertEqual(plan.start_date, date(2024, 6, 15))

    def test_not_recommended_when_full_payment_not_accepted_and_nothing_else_offered(self):
        profile = make_profile(current_available_balance=10000.0, minimum_balance_to_keep=1000.0,
                                payment_methods_user_will_consider=())  # accepts nothing
        forecast = _forecast(profile=profile, events=[], requested_amount=500.0)
        request = RequestInfo(request_date=REQUEST_DATE, requested_amount=500.0,
                               desired_completion_date=date(2024, 7, 1), allows_partial_payment=False)
        plan = select_plan(forecast_result=forecast, profile=profile, user_events=[], request=request,
                            payment_options=[_full_option(500.0)])
        self.assertEqual(plan.method, "not_recommended")
        self.assertEqual(plan.affordability_status, "not_affordable")


class TestPartialPayment(unittest.TestCase):
    def test_partial_payment_chosen_when_eligible_and_full_payment_not_safe(self):
        profile = make_profile(current_available_balance=1300.0, minimum_balance_to_keep=1000.0,
                                payment_methods_user_will_consider=("partial_payment", "full_payment"))
        salary = make_event("e1", status="scheduled", direction="credit", category="salary",
                            amount=5000.0, event_date=date(2024, 6, 20))
        forecast = _forecast(profile=profile, events=[salary], requested_amount=1000.0)
        request = RequestInfo(request_date=REQUEST_DATE, requested_amount=1000.0,
                               desired_completion_date=date(2024, 8, 1), allows_partial_payment=True)
        plan = select_plan(forecast_result=forecast, profile=profile, user_events=[salary], request=request,
                            payment_options=[_full_option(1000.0)])
        self.assertEqual(plan.method, "partial_payment")
        self.assertEqual(plan.affordability_status, "affordable_with_plan")
        self.assertEqual(len(plan.payments), 2)
        self.assertAlmostEqual(sum(a for _, a in plan.payments), 1000.0)

    def test_partial_payment_not_offered_when_request_disallows_it(self):
        profile = make_profile(current_available_balance=1300.0, minimum_balance_to_keep=1000.0,
                                payment_methods_user_will_consider=("partial_payment", "full_payment"))
        salary = make_event("e1", status="scheduled", direction="credit", category="salary",
                            amount=5000.0, event_date=date(2024, 6, 20))
        forecast = _forecast(profile=profile, events=[salary], requested_amount=1000.0)
        request = RequestInfo(request_date=REQUEST_DATE, requested_amount=1000.0,
                               desired_completion_date=date(2024, 8, 1), allows_partial_payment=False)
        plan = select_plan(forecast_result=forecast, profile=profile, user_events=[salary], request=request,
                            payment_options=[_full_option(1000.0)])
        self.assertNotEqual(plan.method, "partial_payment")
        self.assertEqual(plan.method, "wait")


class TestInstallments(unittest.TestCase):
    def test_option_exceeding_max_installment_months_is_excluded(self):
        profile = make_profile(current_available_balance=10000.0, minimum_balance_to_keep=1000.0,
                                payment_methods_user_will_consider=("installments",), max_installment_months=6)
        forecast = _forecast(profile=profile, events=[], requested_amount=1200.0)
        request = RequestInfo(request_date=REQUEST_DATE, requested_amount=1200.0,
                               desired_completion_date=date(2025, 6, 1), allows_partial_payment=False)
        too_long = _installment_option(option_id="payment_option_9", number_of_payments=24, payment_amount=50.0)
        plan = select_plan(forecast_result=forecast, profile=profile, user_events=[], request=request,
                            payment_options=[too_long])
        self.assertEqual(plan.method, "not_recommended")

    def test_installments_chosen_when_within_limit_and_safe(self):
        profile = make_profile(current_available_balance=10000.0, minimum_balance_to_keep=1000.0,
                                payment_methods_user_will_consider=("installments",), max_installment_months=12)
        forecast = _forecast(profile=profile, events=[], requested_amount=1200.0)
        request = RequestInfo(request_date=REQUEST_DATE, requested_amount=1200.0,
                               desired_completion_date=date(2024, 9, 1), allows_partial_payment=False)
        option = _installment_option(option_id="payment_option_5", number_of_payments=6, payment_amount=200.0)
        plan = select_plan(forecast_result=forecast, profile=profile, user_events=[], request=request,
                            payment_options=[option])
        self.assertEqual(plan.method, "installments")
        self.assertEqual(plan.num_payments, 6)
        self.assertEqual(plan.option_sort_key, 5)

    def test_no_installments_offered_when_max_installment_months_is_none(self):
        profile = make_profile(current_available_balance=10000.0, minimum_balance_to_keep=1000.0,
                                payment_methods_user_will_consider=("installments",), max_installment_months=None)
        forecast = _forecast(profile=profile, events=[], requested_amount=1200.0)
        request = RequestInfo(request_date=REQUEST_DATE, requested_amount=1200.0,
                               desired_completion_date=date(2024, 9, 1), allows_partial_payment=False)
        option = _installment_option(option_id="payment_option_5", number_of_payments=6, payment_amount=200.0)
        plan = select_plan(forecast_result=forecast, profile=profile, user_events=[], request=request,
                            payment_options=[option])
        self.assertEqual(plan.method, "not_recommended")


class TestRankingTieBreaks(unittest.TestCase):
    def test_prefers_completing_by_deadline_over_lower_cost(self):
        # Balance/minimum are deliberately loose so both options are
        # trivially safe regardless of timing -- isolating the ranking
        # logic itself from schedule-safety mechanics. Two installment
        # options for the same request: a cheap one that finishes well
        # past the deadline, and a pricier one that finishes on time.
        # Tie-break #1 (completes by deadline) must dominate #3 (cost).
        profile = make_profile(current_available_balance=100000.0, minimum_balance_to_keep=0.0,
                                payment_methods_user_will_consider=("installments",), max_installment_months=24)
        forecast = _forecast(profile=profile, events=[], requested_amount=300.0)
        from datetime import timedelta
        request = RequestInfo(request_date=REQUEST_DATE, requested_amount=300.0,
                               desired_completion_date=REQUEST_DATE + timedelta(days=40),
                               allows_partial_payment=False)

        cheap_but_late = _installment_option(
            option_id="payment_option_1", number_of_payments=24, payment_amount=10.0,
        )  # total 240, last payment ~690 days out
        pricier_but_on_time = _installment_option(
            option_id="payment_option_2", number_of_payments=2, payment_amount=200.0,
        )  # total 400, last payment 30 days out -- within the deadline

        plan = select_plan(forecast_result=forecast, profile=profile, user_events=[], request=request,
                            payment_options=[cheap_but_late, pricier_but_on_time])
        self.assertEqual(plan.option_sort_key, 2)
        self.assertTrue(plan.completes_by(request.desired_completion_date))
        self.assertEqual(plan.total_cost, 400.0)

    def test_prefers_lower_cost_when_both_complete_by_deadline(self):
        profile = make_profile(current_available_balance=10000.0, minimum_balance_to_keep=1000.0,
                                payment_methods_user_will_consider=("full_payment", "installments"),
                                max_installment_months=12)
        forecast = _forecast(profile=profile, events=[], requested_amount=1200.0)
        request = RequestInfo(request_date=REQUEST_DATE, requested_amount=1200.0,
                               desired_completion_date=date(2025, 1, 1), allows_partial_payment=False)
        installment = _installment_option(option_id="payment_option_2", number_of_payments=6, payment_amount=250.0)
        plan = select_plan(forecast_result=forecast, profile=profile, user_events=[], request=request,
                            payment_options=[_full_option(1200.0), installment])
        # full_payment costs 1200 total; installments cost 1500 -- full_payment wins on cost.
        self.assertEqual(plan.method, "full_payment")


class TestSpendingChangesFallback(unittest.TestCase):
    def test_unlocks_full_payment_when_stopping_a_flexible_event_frees_enough_cash(self):
        profile = make_profile(
            current_available_balance=1000.0, minimum_balance_to_keep=900.0,
            payment_methods_user_will_consider=("full_payment",),
            expense_categories_user_is_willing_to_stop=("streaming",),
        )
        streaming = make_event("e1", status="scheduled", direction="debit", category="streaming",
                                amount=150.0, flexibility="stoppable", event_date=date(2024, 6, 10))
        forecast = _forecast(profile=profile, events=[streaming], requested_amount=100.0)
        request = RequestInfo(request_date=REQUEST_DATE, requested_amount=100.0,
                               desired_completion_date=date(2024, 7, 1), allows_partial_payment=False)
        plan = select_plan(forecast_result=forecast, profile=profile, user_events=[streaming], request=request,
                            payment_options=[_full_option(100.0)])
        self.assertEqual(plan.method, "full_payment")
        self.assertEqual(plan.affordability_status, "affordable_with_plan")
        self.assertEqual(plan.spending_changes, ("stop:e1",))

    def test_not_recommended_when_spending_changes_are_insufficient(self):
        profile = make_profile(
            current_available_balance=100.0, minimum_balance_to_keep=90.0,
            payment_methods_user_will_consider=("full_payment",),
            expense_categories_user_is_willing_to_stop=("streaming",),
        )
        streaming = make_event("e1", status="scheduled", direction="debit", category="streaming",
                                amount=5.0, flexibility="stoppable", event_date=date(2024, 6, 10))
        forecast = _forecast(profile=profile, events=[streaming], requested_amount=1000.0)
        request = RequestInfo(request_date=REQUEST_DATE, requested_amount=1000.0,
                               desired_completion_date=date(2024, 7, 1), allows_partial_payment=False)
        plan = select_plan(forecast_result=forecast, profile=profile, user_events=[streaming], request=request,
                            payment_options=[_full_option(1000.0)])
        self.assertEqual(plan.method, "not_recommended")


if __name__ == "__main__":
    unittest.main()
