import unittest
from pathlib import Path

import _pathfix  # noqa: F401
from _fake_llm import FakeAnthropicClient, FakeResponse
from evidence.llm_client import LLMClient
from evidence.usage_tracker import UsageTracker
from evidence.vision import VisionResolutionError, resolve_amount_from_image

# A 1x1 PNG, just to have a real file on disk for the "image exists" path.
_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
    "de000000017352474200aece1ce90000000467414d410000b18f0bfc61050000"
    "000774494d4507e50101010009fbf1980000000c49444154789c636000010000"
    "0500010d0a2db40000000049454e44ae426082"
)


def _make_llm(response):
    return LLMClient(usage_tracker=UsageTracker(), client=FakeAnthropicClient([response]))


class TestResolveAmountFromImage(unittest.TestCase):
    def setUp(self):
        self.tmp_path = Path(self.id().split(".")[-1] + "_tmp.png")
        self.tmp_path.write_bytes(_TINY_PNG)
        self.addCleanup(self.tmp_path.unlink)

    def test_high_confidence_result_parses(self):
        llm = _make_llm(FakeResponse(data={
            "amount": 4365000, "currency": "IDR", "confidence": "high", "notes": "Net pay line"
        }))
        result = resolve_amount_from_image(
            image_path=self.tmp_path, event_description="August 2019 net salary",
            expected_currency="IDR", llm_client=llm, detail="event_253",
        )
        self.assertEqual(result.amount, 4365000)
        self.assertEqual(result.currency, "IDR")
        self.assertEqual(result.confidence, "high")

    def test_low_confidence_is_not_trusted(self):
        llm = _make_llm(FakeResponse(data={
            "amount": 100, "currency": "USD", "confidence": "low", "notes": "hard to read"
        }))
        with self.assertRaises(VisionResolutionError):
            resolve_amount_from_image(
                image_path=self.tmp_path, event_description="x", expected_currency="USD",
                llm_client=llm, detail="event_x",
            )

    def test_non_positive_amount_rejected(self):
        llm = _make_llm(FakeResponse(data={
            "amount": 0, "currency": "USD", "confidence": "high", "notes": "n/a"
        }))
        with self.assertRaises(VisionResolutionError):
            resolve_amount_from_image(
                image_path=self.tmp_path, event_description="x", expected_currency="USD",
                llm_client=llm, detail="event_x",
            )

    def test_malformed_response_missing_field_raises(self):
        llm = _make_llm(FakeResponse(data={"amount": 100, "currency": "USD"}))  # missing confidence/notes
        with self.assertRaises(VisionResolutionError):
            resolve_amount_from_image(
                image_path=self.tmp_path, event_description="x", expected_currency="USD",
                llm_client=llm, detail="event_x",
            )

    def test_missing_image_file_raises_without_calling_the_llm(self):
        llm = LLMClient(usage_tracker=UsageTracker(), client=FakeAnthropicClient([]))
        with self.assertRaises(VisionResolutionError):
            resolve_amount_from_image(
                image_path=Path("does_not_exist_12345.png"), event_description="x",
                expected_currency="USD", llm_client=llm, detail="event_x",
            )


if __name__ == "__main__":
    unittest.main()
