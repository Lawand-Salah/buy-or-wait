"""Thin wrapper around one Claude Messages API call, returning validated JSON.

Every LLM call this whole package makes -- vision extraction, message
classification -- goes through ``LLMClient.call_json()``. That's
deliberate: it's the single place usage gets recorded (so
evaluation/usage_report.md can never miss a call), the single place a
malformed response is turned into a typed error instead of an uncaught
exception deep in a caller, and the single place a test needs to fake to
exercise every other module without a network call.

Model: per the user's instruction, vision and message-classification are
small structured-extraction tasks, not complex reasoning, so this defaults
to the fastest/cheapest current model (Claude Haiku 4.5) rather than a
larger one.

Structured output: uses ``output_config: {"format": {"type": "json_schema",
...}}`` (the "Raw Schema" pattern), which guarantees the response's first
content block is text containing JSON that validates against the given
schema -- no free-form parsing of natural-language output.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Protocol

from .env import load_dotenv
from .usage_tracker import UsageTracker

DEFAULT_MODEL = "claude-haiku-4-5"


class LLMError(Exception):
    """Base class for every error this module raises."""


class LLMCallError(LLMError):
    """The underlying API call itself failed (auth, rate limit, network, ...)."""


class LLMResponseError(LLMError):
    """The call succeeded but the response didn't contain valid, schema-shaped JSON."""


class _AnthropicLikeClient(Protocol):
    """Structural type for whatever ``client`` is passed in -- either a real
    ``anthropic.Anthropic()`` or a test double. Only the surface this
    module actually uses needs to match.
    """

    messages: Any


class LLMClient:
    def __init__(
        self,
        *,
        usage_tracker: UsageTracker,
        model: str = DEFAULT_MODEL,
        client: Optional[_AnthropicLikeClient] = None,
    ) -> None:
        self.model = model
        self.usage_tracker = usage_tracker
        self._client = client  # lazily created on first real use if None

    def _get_client(self) -> _AnthropicLikeClient:
        if self._client is not None:
            return self._client
        load_dotenv()  # fills os.environ["ANTHROPIC_API_KEY"] from .env if not already set
        import anthropic  # deferred: unit tests that inject a fake client never need this installed

        self._client = anthropic.Anthropic()
        return self._client

    def call_json(
        self,
        *,
        system: str,
        user_content: "str | List[Dict[str, Any]]",
        json_schema: Dict[str, Any],
        call_type: str,
        request_id: Optional[str] = None,
        detail: str = "",
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """Makes one Messages API call constrained to ``json_schema``,
        records its token usage, and returns the parsed JSON dict.

        Raises ``LLMCallError`` if the API call itself fails, or
        ``LLMResponseError`` if the call succeeds but the response can't be
        read as the expected JSON (should be rare given json_schema, but a
        caller must never treat an unparseable response as "no answer" --
        that would risk silently treating a real signal as absent).
        """
        client = self._get_client()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_content}],
                output_config={"format": {"type": "json_schema", "schema": json_schema}},
            )
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: wrap any SDK failure uniformly
            raise LLMCallError(f"LLM call failed ({call_type}, {detail}): {exc}") from exc

        self.usage_tracker.record(
            model=getattr(response, "model", self.model),
            call_type=call_type,
            request_id=request_id,
            detail=detail,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        text = next((block.text for block in response.content if block.type == "text"), None)
        if text is None:
            raise LLMResponseError(f"No text content block in response ({call_type}, {detail})")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"Response text was not valid JSON ({call_type}, {detail}): {exc}") from exc
