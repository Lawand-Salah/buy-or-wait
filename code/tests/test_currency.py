import unittest
from datetime import date

import _pathfix  # noqa: F401  (adds code/ to sys.path)
from forecasting.currency import RateNotFoundError, RateTable


def make_table():
    return RateTable(
        [
            {"rate_date": date(2023, 10, 15), "from_currency": "EUR", "to_currency": "ZAR", "rate": "20"},
            {"rate_date": date(2023, 10, 15), "from_currency": "USD", "to_currency": "EUR", "rate": "0.92"},
            {"rate_date": date(2023, 10, 15), "from_currency": "USD", "to_currency": "IDR", "rate": "15833.33"},
        ]
    )


class TestRateTable(unittest.TestCase):
    def test_same_currency_is_a_no_op(self):
        table = make_table()
        self.assertEqual(table.convert(100.0, "ZAR", "ZAR", date(2023, 10, 15)), 100.0)

    def test_direct_rate(self):
        table = make_table()
        self.assertAlmostEqual(table.convert(10.0, "EUR", "ZAR", date(2023, 10, 15)), 200.0)

    def test_inverse_of_a_listed_rate(self):
        table = make_table()
        # Only EUR,ZAR is listed; ZAR->EUR must be the inverse (1/20).
        self.assertAlmostEqual(table.convert(200.0, "ZAR", "EUR", date(2023, 10, 15)), 10.0)

    def test_two_hop_chain_through_a_shared_hub(self):
        table = make_table()
        # No ZAR<->IDR row at all; must chain ZAR->EUR (inverse) ->? no direct EUR->IDR either,
        # so chain via USD isn't reachable from ZAR directly -- this checks the EUR/USD hub instead.
        # USD -> EUR -> ZAR: 1 USD = 0.92 EUR = 18.4 ZAR
        self.assertAlmostEqual(table.convert(1.0, "USD", "ZAR", date(2023, 10, 15)), 0.92 * 20)

    def test_missing_date_falls_back_to_nearest(self):
        table = make_table()
        # No rows at all for this date; nearest (and only) date is 2023-10-15.
        self.assertAlmostEqual(table.convert(10.0, "EUR", "ZAR", date(2023, 11, 1)), 200.0)

    def test_unresolvable_pair_raises(self):
        table = RateTable(
            [{"rate_date": date(2023, 10, 15), "from_currency": "EUR", "to_currency": "ZAR", "rate": "20"}]
        )
        with self.assertRaises(RateNotFoundError):
            table.convert(1.0, "INR", "USD", date(2023, 10, 15))


if __name__ == "__main__":
    unittest.main()
