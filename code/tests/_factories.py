"""Shared test-data builders so every test file constructs Event/Profile
objects the same way instead of repeating boilerplate.
"""
from datetime import date

from forecasting.models import Event, Profile


def make_event(
    event_id,
    *,
    user_id="user_x",
    event_type="expense",
    description="desc",
    category="rent",
    direction="debit",
    amount=100.0,
    currency="USD",
    event_date=date(2024, 1, 1),
    settlement_date=None,
    status="settled",
    linked_event_id=None,
    flexibility="fixed",
    minimum_allowed_amount=None,
):
    return Event(
        event_id=event_id,
        user_id=user_id,
        event_type=event_type,
        description=description,
        category=category,
        direction=direction,
        amount=amount,
        currency=currency,
        event_date=event_date,
        settlement_date=settlement_date or event_date,
        status=status,
        linked_event_id=linked_event_id,
        flexibility=flexibility,
        minimum_allowed_amount=minimum_allowed_amount,
    )


def make_profile(
    *,
    user_id="user_x",
    home_currency="USD",
    current_available_balance=10000.0,
    minimum_balance_to_keep=1000.0,
    financial_priorities=(),
    expense_categories_to_protect=(),
    expense_categories_user_is_willing_to_reduce=(),
    expense_categories_user_is_willing_to_stop=(),
    payment_methods_user_will_consider=("full_payment",),
    max_installment_months=None,
):
    return Profile(
        user_id=user_id,
        home_currency=home_currency,
        current_available_balance=current_available_balance,
        minimum_balance_to_keep=minimum_balance_to_keep,
        financial_priorities=financial_priorities,
        expense_categories_to_protect=expense_categories_to_protect,
        expense_categories_user_is_willing_to_reduce=expense_categories_user_is_willing_to_reduce,
        expense_categories_user_is_willing_to_stop=expense_categories_user_is_willing_to_stop,
        payment_methods_user_will_consider=payment_methods_user_will_consider,
        max_installment_months=max_installment_months,
    )
