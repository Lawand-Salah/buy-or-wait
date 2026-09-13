# Buy or Wait? — solution code

Python 3, standard library only for the deterministic core; one third-party
dependency (`anthropic`) for the LLM-backed evidence and explanation layer.

## Packages

- **`forecasting/`** — deterministic 90-day balance forecasting engine.
  Loads `financial_profiles.csv`/`financial_events.csv`/`exchange_rates.csv`,
  builds a unified dated cash-flow ledger (recurrence detection, pending/
  scheduled reservation, currency conversion), and simulates the balance
  day by day to produce `amount_safe_to_pay` and
  `earliest_date_for_full_payment` — before any spending changes or
  payment-method preference is applied. No LLM calls; fully offline and
  unit-tested.
- **`evidence/`** — LLM-backed evidence resolution, sitting in front of
  `forecasting/` without modifying it. Resolves blank `financial_events.csv`
  amounts from their linked image (`vision.py`), classifies each user's
  message into a fixed taxonomy — `confirms_existing` /
  `status_amendment` / `income_or_expense_amendment` / `narrative_only`
  (`messages.py`) — and produces the augmented event list +
  `amount_overrides` that flow into `forecasting.forecast.run_forecast()`
  unchanged (`resolver.py`). `llm_client.py` is the single call-site every
  LLM request goes through; `usage_tracker.py` feeds
  `evaluation/usage_report.md`.
- **`planning/`** — turns the forecast's capacity numbers plus
  `request_payment_options.csv` into the final recommendation: builds every
  eligible, safe candidate plan (full payment now, wait, partial payment,
  one per eligible installment option), ranks them by the six tie-breakers
  in the problem statement's "Choosing Between Safe Plans" section, and
  falls back to spending changes only when nothing else is safe.
  `explanation.py` makes one more LLM call per request for
  `decision_explanation`, grounded only in the already-decided facts.
- **`main.py`** — the entry point. Loads everything once, loops over
  `dataset/requests.csv`, writes the final `output.csv` at the repository
  root (`dataset/output.csv` is only the blank template) and
  `evaluation/usage_report.md`. A per-request `try/except` means one bad
  row can never lose the whole run — it falls back to a safe
  `not_recommended` row instead.
- **`tests/`** — `unittest`-based (no pytest needed). Fast, offline, mocked
  tests for every module, plus a small number of real (non-mocked, opt-in)
  integration tests against the live API and real dataset files.
- **`evaluation/usage_report.md`** — token usage and estimated cost for the
  most recent full run.

## Setup

```bash
pip install -r requirements.txt
```

Put your Anthropic API key in a `.env` file at the **repository root**
(one directory above this `code/` folder — the same directory as
`AGENTS.md`), one line:

```
ANTHROPIC_API_KEY=sk-ant-...
```

`.env` is already git-ignored. There's no need to `export` the key in your
shell — `evidence/env.py` reads the file directly.

## Run

From anywhere (paths are resolved relative to `main.py`'s own location):

```bash
python main.py                       # full run: all 250 requests
python main.py --limit 5             # quick smoke test on the first 5 rows
python main.py --request-id request_1  # a single row, for debugging
```

Optional overrides: `--dataset-dir`, `--output`, `--usage-report`.

## Tests

```bash
cd tests
python -m unittest discover -s . -p "test_*.py"
```

The real-API tests in `test_evidence_integration.py` skip automatically
unless a real-looking `ANTHROPIC_API_KEY` is available; everything else
runs offline in a few seconds.

## Model

All LLM calls (vision extraction, message classification, decision
explanation) use `claude-haiku-4-5` — these are small, structured
extraction/summarization tasks, not tasks that need a larger model.
