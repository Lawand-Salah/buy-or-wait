# Buy or Wait? — Solution

Python 3. Standard library only for the deterministic core; one
third-party dependency (`anthropic`) for the LLM-backed evidence layer.

## Setup

```bash
pip install -r requirements.txt
```

Put your Anthropic API key in a `.env` file at the **repository root**
(one directory above `code/` — the same directory as `AGENTS.md`):

```
ANTHROPIC_API_KEY=sk-ant-...
```

`evidence/env.py` reads this file directly, so there's no need to
`export` the key in your shell first.

## Run

```bash
python3 code/main.py                        # full run: all 250 requests
python3 code/main.py --limit 5              # quick smoke test, first 5 rows
python3 code/main.py --request-id request_1 # a single row, for debugging
```

Writes the final `output.csv` at the repository root and
`code/evaluation/usage_report.md`. Other overrides: `--dataset-dir`,
`--output`, `--usage-report`.

## Approach

The solution is two layers on top of each other, kept deliberately
separate:

**1. A deterministic forecasting engine (`forecasting/`) — no LLM calls,
fully offline, unit-tested.** Given a user's profile and financial-event
history, it reconstructs a dated cash-flow ledger (recurring
income/expenses detected from settled history, pending items reserved,
scheduled items included, currency converted at the right rate) and
simulates the balance day by day across the 90-day window to produce
`amount_safe_to_pay` and `earliest_date_for_full_payment`. This is pure
arithmetic on structured CSV data — it should be exact and repeatable
every time, so it has no business going through an LLM. `planning/` sits
on top of it: given those two numbers plus the supplied payment options,
it builds every eligible candidate plan (full payment, wait, partial
payment, each installment option), checks each one against the same
day-by-day balance simulation, and ranks the safe ones by the six
tie-breakers in the problem statement (completes by deadline → no
spending changes → lowest cost → earliest start → fewest payments →
lowest option id) — falling back to stopping/reducing a flexible expense
only when nothing else is safe.

**2. An LLM-backed evidence layer (`evidence/`) for exactly the two
things in this dataset that are *not* structured data: reading a number
out of an image, and interpreting what a free-text message actually
means.** `vision.py` resolves a blank `financial_events.csv` amount from
its linked receipt/payslip image. `messages.py` classifies each message
into one of four fixed categories (confirms what's already known / a
confirmed status change / a confirmed recurring amount change / just
narrative color) — never left to free-form judgment, and re-validated in
Python after parsing, since message content is untrusted input. Both
produce plain data (a resolved amount, a corrected or synthetic event)
that flows straight into the *same* forecasting engine above, unchanged
— the LLM never touches the arithmetic, it only fills in facts the CSVs
alone don't have. `explanation.py` makes one further LLM call per request
to write `decision_explanation` in plain language from the
already-decided facts — again, explaining a decision, not making one.

The split matters: everything that can be computed exactly, is —
deterministically and testably. The LLM is reserved for the parts that
inherently require reading an image or interpreting free text, where no
deterministic alternative exists, and even there its output is
constrained (structured JSON schemas, fixed taxonomies, post-hoc
validation) rather than trusted freely.

`code/main.py` is the entry point tying it together: for each request it
resolves evidence, runs the forecast, selects a plan, generates the
explanation, and writes one `output.csv` row — falling back to a safe
`not_recommended` row on any per-request failure so one bad row never
loses the whole run.

## Tests

```bash
cd tests
python3 -m unittest discover -s . -p "test_*.py"
```

Fast, offline, mocked tests for every module. The handful of real
(non-mocked) tests in `test_evidence_integration.py`/
`test_forecast_integration.py` skip automatically without a real API key
or the `dataset/` folder.

## Model

All LLM calls (vision extraction, message classification, decision
explanation) use `claude-haiku-4-5` — small, structured
extraction/summarization tasks that don't need a larger model.
