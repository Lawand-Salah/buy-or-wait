"""A fake Anthropic client double for testing evidence/llm_client.py and
everything built on top of it, without any real network call.

Usage: ``FakeAnthropicClient([FakeResponse(text=json.dumps({...})), ...])``
-- each call to ``.messages.create()`` pops the next queued response (or
raises it, if a queued item is an ``Exception``) and records the kwargs it
was called with, so tests can assert on the prompt/schema sent.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeTextBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class FakeResponse:
    def __init__(
        self,
        *,
        data: "dict | None" = None,
        text: "str | None" = None,
        input_tokens: int = 100,
        output_tokens: int = 20,
        model: str = "claude-haiku-4-5",
    ):
        if data is not None:
            text = json.dumps(data)
        assert text is not None, "FakeResponse needs either data= or text="
        self.content = [_FakeTextBlock(text)]
        self.usage = _FakeUsage(input_tokens, output_tokens)
        self.model = model


class _FakeMessagesAPI:
    def __init__(self, responses: List[Any]):
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeAnthropicClient ran out of queued responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeAnthropicClient:
    def __init__(self, responses: List[Any]):
        self.messages = _FakeMessagesAPI(responses)
