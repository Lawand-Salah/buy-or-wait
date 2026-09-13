"""Token usage and cost tracking for AGENTS.md 6.5 / problem_statement.md
"Token Usage and Cost Analysis": every model call this package makes must
end up summarized in evaluation/usage_report.md (model providers and
names, model calls, input/output tokens, total and average tokens per
request, estimated total and per-request cost).

Every LLM call in this package -- vision or message classification --
goes through exactly one place: LLMClient.call_json() in llm_client.py,
which calls UsageTracker.record() once per call. Nothing else in this
package touches token bookkeeping, so the count in a written report is
never at risk of missing a call site.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Anthropic first-party API rates, $ per 1M tokens (input, output). Source:
# the claude-api skill's cached pricing table (cached 2026-06-24) -- verify
# against https://www.anthropic.com/pricing before a final submission if
# rates may have changed since.
PRICING_PER_MILLION_TOKENS: Dict[str, Tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
}


@dataclass(frozen=True)
class UsageRecord:
    timestamp: str  # ISO-8601 UTC
    model: str
    call_type: str  # "vision" | "message_classification"
    request_id: Optional[str]
    detail: str  # e.g. an event_id or message_id, for audit
    input_tokens: int
    output_tokens: int


@dataclass
class UsageTracker:
    records: List[UsageRecord] = field(default_factory=list)

    def record(
        self,
        *,
        model: str,
        call_type: str,
        input_tokens: int,
        output_tokens: int,
        request_id: Optional[str] = None,
        detail: str = "",
    ) -> None:
        self.records.append(
            UsageRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                model=model,
                call_type=call_type,
                request_id=request_id,
                detail=detail,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )

    def _cost(self, model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
        rates = PRICING_PER_MILLION_TOKENS.get(model)
        if rates is None:
            return None
        input_rate, output_rate = rates
        return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate

    def totals_by_model(self) -> Dict[str, dict]:
        """Per-model {calls, input_tokens, output_tokens, cost}."""
        totals: Dict[str, dict] = {}
        for r in self.records:
            bucket = totals.setdefault(
                r.model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0, "cost_known": True}
            )
            bucket["calls"] += 1
            bucket["input_tokens"] += r.input_tokens
            bucket["output_tokens"] += r.output_tokens
            cost = self._cost(r.model, r.input_tokens, r.output_tokens)
            if cost is None:
                bucket["cost_known"] = False
            else:
                bucket["cost"] += cost
        return totals

    def overall_totals(self) -> dict:
        by_model = self.totals_by_model()
        return {
            "calls": sum(b["calls"] for b in by_model.values()),
            "input_tokens": sum(b["input_tokens"] for b in by_model.values()),
            "output_tokens": sum(b["output_tokens"] for b in by_model.values()),
            "cost": sum(b["cost"] for b in by_model.values()),
            "cost_known": all(b["cost_known"] for b in by_model.values()) if by_model else True,
        }

    def distinct_request_count(self) -> int:
        return len({r.request_id for r in self.records if r.request_id is not None})

    def render_report(self, *, total_requests_in_dataset: Optional[int] = None) -> str:
        """Builds the evaluation/usage_report.md content as a string (kept
        separate from write_report() so tests can check the content
        without touching the filesystem).
        """
        by_model = self.totals_by_model()
        overall = self.overall_totals()
        total_tokens = overall["input_tokens"] + overall["output_tokens"]
        denom = total_requests_in_dataset or self.distinct_request_count() or 1

        lines = [
            "# Token Usage and Cost Report",
            "",
            "Model provider: Anthropic.",
            "",
            "## Per-model breakdown",
            "",
            "| Model | Calls | Input tokens | Output tokens | Estimated cost (USD) |",
            "|---|---|---|---|---|",
        ]
        for model, b in sorted(by_model.items()):
            cost_str = f"${b['cost']:.4f}" if b["cost_known"] else "unknown (no listed rate)"
            lines.append(f"| {model} | {b['calls']} | {b['input_tokens']} | {b['output_tokens']} | {cost_str} |")

        lines += [
            "",
            "## Overall",
            "",
            f"- Total model calls: {overall['calls']}",
            f"- Total input tokens: {overall['input_tokens']}",
            f"- Total output tokens: {overall['output_tokens']}",
            f"- Total tokens: {total_tokens}",
            f"- Requests this run covered: {denom}",
            f"- Average tokens per request: {total_tokens / denom:.1f}",
            (
                f"- Estimated total cost: ${overall['cost']:.4f}"
                if overall["cost_known"]
                else "- Estimated total cost: unknown (a model with no listed rate was used)"
            ),
            (
                f"- Estimated average cost per request: ${overall['cost'] / denom:.6f}"
                if overall["cost_known"]
                else "- Estimated average cost per request: unknown"
            ),
        ]
        return "\n".join(lines) + "\n"

    def write_report(self, path: "str | Path", *, total_requests_in_dataset: Optional[int] = None) -> None:
        Path(path).write_text(self.render_report(total_requests_in_dataset=total_requests_in_dataset), encoding="utf-8")
