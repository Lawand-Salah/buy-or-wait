import unittest

import _pathfix  # noqa: F401
from _fake_llm import FakeAnthropicClient, FakeResponse
from evidence.llm_client import LLMCallError, LLMClient, LLMResponseError
from evidence.usage_tracker import UsageTracker


class TestLLMClientCallJson(unittest.TestCase):
    def test_returns_parsed_dict_and_records_usage(self):
        tracker = UsageTracker()
        client = FakeAnthropicClient([FakeResponse(data={"answer": 42}, input_tokens=123, output_tokens=45)])
        llm = LLMClient(usage_tracker=tracker, client=client)

        result = llm.call_json(
            system="sys", user_content="hi", json_schema={"type": "object"},
            call_type="vision", request_id="request_01", detail="event_1",
        )

        self.assertEqual(result, {"answer": 42})
        self.assertEqual(len(tracker.records), 1)
        record = tracker.records[0]
        self.assertEqual(record.input_tokens, 123)
        self.assertEqual(record.output_tokens, 45)
        self.assertEqual(record.call_type, "vision")
        self.assertEqual(record.request_id, "request_01")
        self.assertEqual(record.detail, "event_1")

    def test_sends_the_json_schema_in_output_config(self):
        tracker = UsageTracker()
        schema = {"type": "object", "properties": {"x": {"type": "number"}}}
        client = FakeAnthropicClient([FakeResponse(data={"x": 1})])
        llm = LLMClient(usage_tracker=tracker, client=client)

        llm.call_json(system="s", user_content="u", json_schema=schema, call_type="vision")

        sent = client.messages.calls[0]
        self.assertEqual(sent["output_config"]["format"]["schema"], schema)
        self.assertEqual(sent["output_config"]["format"]["type"], "json_schema")

    def test_uses_the_configured_model(self):
        tracker = UsageTracker()
        client = FakeAnthropicClient([FakeResponse(data={"x": 1})])
        llm = LLMClient(usage_tracker=tracker, model="claude-haiku-4-5", client=client)

        llm.call_json(system="s", user_content="u", json_schema={}, call_type="vision")

        self.assertEqual(client.messages.calls[0]["model"], "claude-haiku-4-5")

    def test_api_failure_raises_llm_call_error(self):
        tracker = UsageTracker()
        client = FakeAnthropicClient([RuntimeError("boom")])
        llm = LLMClient(usage_tracker=tracker, client=client)

        with self.assertRaises(LLMCallError):
            llm.call_json(system="s", user_content="u", json_schema={}, call_type="vision")
        self.assertEqual(tracker.records, [])  # a failed call records no usage

    def test_missing_text_block_raises_llm_response_error(self):
        tracker = UsageTracker()
        response = FakeResponse(data={"x": 1})
        response.content = []  # simulate a response with no text block at all
        client = FakeAnthropicClient([response])
        llm = LLMClient(usage_tracker=tracker, client=client)

        with self.assertRaises(LLMResponseError):
            llm.call_json(system="s", user_content="u", json_schema={}, call_type="vision")

    def test_non_json_text_raises_llm_response_error(self):
        tracker = UsageTracker()
        client = FakeAnthropicClient([FakeResponse(text="not json at all")])
        llm = LLMClient(usage_tracker=tracker, client=client)

        with self.assertRaises(LLMResponseError):
            llm.call_json(system="s", user_content="u", json_schema={}, call_type="vision")


if __name__ == "__main__":
    unittest.main()
