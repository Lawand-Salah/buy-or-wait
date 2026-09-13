"""Small, opt-in integration checks against the real Anthropic API and the
real dataset files -- not mocked. Skipped automatically unless a
real-looking ANTHROPIC_API_KEY is available (via .env or the shell
environment), so the main suite stays fast, offline, and deterministic;
this file is the one exception, run manually or in CI with a real key.
"""
import os
import unittest
from datetime import date
from pathlib import Path

import _pathfix  # noqa: F401
from evidence.env import load_dotenv
from evidence.llm_client import LLMClient
from evidence.messages import classify_message
from evidence.usage_tracker import UsageTracker
from evidence.vision import resolve_amount_from_image
from forecasting.models import Event

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"

load_dotenv()
_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
_HAS_REAL_KEY = _api_key.startswith("sk-ant-") and len(_api_key) > 20


def _make_event(**overrides):
    defaults = dict(
        event_id="x", user_id="x", event_type="income", description="x", category="x",
        direction="credit", amount=None, currency="USD", event_date=date(2024, 1, 1),
        settlement_date=date(2024, 1, 1), status="settled", linked_event_id=None,
        flexibility="fixed", minimum_allowed_amount=None,
    )
    defaults.update(overrides)
    return Event(**defaults)


@unittest.skipUnless(_HAS_REAL_KEY, "no real-looking ANTHROPIC_API_KEY available")
@unittest.skipUnless(DATASET_DIR.exists(), "dataset/ not found")
class TestRealVisionCall(unittest.TestCase):
    def test_image_01_resolves_to_the_known_payslip_amount(self):
        # image_01.png is a payslip whose Net Pay line reads IDR 4,365,000
        # (confirmed by eye while designing this module).
        llm = LLMClient(usage_tracker=UsageTracker())
        result = resolve_amount_from_image(
            image_path=DATASET_DIR / "media" / "images" / "image_01.png",
            event_description="August 2019 net salary",
            expected_currency="IDR",
            llm_client=llm,
            detail="event_253",
        )
        self.assertEqual(result.amount, 4365000)
        self.assertEqual(result.currency, "IDR")
        self.assertIn(result.confidence, ("high", "medium"))


@unittest.skipUnless(_HAS_REAL_KEY, "no real-looking ANTHROPIC_API_KEY available")
class TestRealMessageClassification(unittest.TestCase):
    def test_speculative_pending_invoice_is_narrative_only(self):
        # message_18 (user_26): "expected settlement", other invoices
        # "still pending approval" -- no confirmed fact, not linked to any
        # existing event row. Must never become a ledger entry.
        llm = LLMClient(usage_tracker=UsageTracker())
        result = classify_message(
            message_id="message_18",
            message_text=(
                "Halo, ini InvoiceFlow. Klien menyetujui pembayaran faktur sebesar IDR 30780000. "
                "Penyelesaian diperkirakan pada 2025-08-15; faktur lain yang diajukan masih menunggu "
                "persetujuan. Hanya faktur yang sudah dikonfirmasi yang dapat dimasukkan dalam "
                "pembayaran berikutnya. Ref kasus SER-0018."
            ),
            source_type="service_provider",
            sent_at="2025-07-27T09:30:00Z",
            related_event=None,
            home_currency="IDR",
            llm_client=llm,
        )
        self.assertEqual(result.category, "narrative_only")

    def test_settled_prize_confirmation_is_confirms_existing(self):
        # message_17 (user_24) narrates event_2165, which is already
        # status=settled with its real amount -- the message adds nothing
        # actionable that isn't already reflected.
        related_event = _make_event(
            event_id="event_2165", category="windfall", direction="credit",
            amount=33550.0, currency="INR", status="settled",
            event_date=date(2025, 12, 27), settlement_date=date(2025, 12, 28),
        )
        llm = LLMClient(usage_tracker=UsageTracker())
        result = classify_message(
            message_id="message_17",
            message_text=(
                "Here’s the latest account information from PrizeTrack. The prize proceeds have "
                "reached your account after withholding. The claim is now closed and there are no "
                "further scheduled payments. There won’t be another payment unless a separate "
                "prize is confirmed. Account ref FIN-0017."
            ),
            source_type="financial_service",
            sent_at="2025-12-29T09:30:00Z",
            related_event=related_event,
            home_currency="INR",
            llm_client=llm,
        )
        self.assertEqual(result.category, "confirms_existing")


if __name__ == "__main__":
    unittest.main()
