# Buy or Wait?

An AI-assisted financial agent that decides whether a user can safely afford a requested expense, built for the HackerRank Orchestrate hackathon (September 2026).

For each request it decides whether to pay in full, pay partially, use installments, wait, or not proceed, and it explains why.

## How it works

- **Deterministic core:** a 90-day day-by-day balance forecast in plain Python. It handles recurring items, pending payments and currency conversion, and it picks a payment plan that never drops below the user's minimum balance.
- **LLM layer (Claude Haiku 4.5):** used only to read amounts from receipt images, interpret free-text messages, and write the plain-language explanation. It never does the arithmetic.

## Run it

```bash
pip install -r code/requirements.txt
# put ANTHROPIC_API_KEY=... in a .env file at the repo root
python3 code/main.py
```

This writes `output.csv` in the repo root and a token and cost report to `code/evaluation/usage_report.md`.

## More detail

- Full setup, approach and tests: [code/README.md](code/README.md)
- Task specification: [problem_statement.md](problem_statement.md)
- Predictions for the 250 requests: [output.csv](output.csv)
