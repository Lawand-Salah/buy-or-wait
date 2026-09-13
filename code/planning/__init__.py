"""Plan selection: turns a ForecastResult (from code/forecasting) plus the
evidence-augmented event list (from code/evidence) and the supplied
request_payment_options.csv rows into the final recommended Plan --
ranked per problem_statement.md's "Choosing Between Safe Plans" tie-break
order, falling back to spending changes only when no plan is otherwise
safe.

Like code/evidence/, this package sits in front of code/forecasting/ and
never modifies it -- every safety check here is built from
forecasting.simulate's already-public Trajectory/compute_* functions.
"""
