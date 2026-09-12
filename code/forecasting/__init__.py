"""Buy-or-Wait 90-day balance forecasting engine.

This package is the deterministic financial core of the solution: given a
user's profile and financial-event history, it answers "how much can this
user safely pay today, and when (if ever) does the full amount become
safe?" -- independent of any LLM/vision layer, payment-method preferences,
or plan ranking. Those live on top of this module, not inside it.

See problem_statement.md ("90-Day Safety Check", "Choosing Between Safe
Plans") and AGENTS.md section 6.3 for the rules this module implements.
"""
