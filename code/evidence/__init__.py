"""LLM-backed evidence resolution: turns images.csv/messages.csv into facts
the forecasting engine can consume, without changing that engine at all.

This package sits in front of code/forecasting/, never inside it. Its one
job is to produce, for a given user, an augmented event list and an
amount_overrides dict that forecasting.forecast.run_forecast() already
knows how to accept -- see resolver.py.
"""
