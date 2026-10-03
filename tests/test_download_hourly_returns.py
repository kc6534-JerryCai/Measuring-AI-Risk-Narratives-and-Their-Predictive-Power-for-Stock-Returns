import math
import unittest

import pandas as pd

from download_hourly_returns import (
    FetchResult,
    WINDOW_MODE_BRIDGE_07_10,
    WINDOW_MODE_CLOCK_10_16,
    align_provider_bars,
    build_expected_windows,
    compute_returns,
)


class WindowTests(unittest.TestCase):
    def test_normal_and_early_close_windows(self):
        schedule = pd.DataFrame(
            {
                "market_open": [
                    pd.Timestamp("2026-09-08 13:30:00Z"),
                    pd.Timestamp("2026-11-27 14:30:00Z"),
                ],
                "market_close": [
                    pd.Timestamp("2026-09-08 20:00:00Z"),
                    pd.Timestamp("2026-11-27 18:00:00Z"),
                ],
            },
            index=pd.DatetimeIndex(["2026-09-08", "2026-11-27"]),
        )
        windows = build_expected_windows(schedule)
        normal = windows[windows["trading_date"] == "2026-09-08"]
        early = windows[windows["trading_date"] == "2026-11-27"]
        self.assertEqual(normal["duration_minutes"].tolist(), [60, 60, 60, 60, 60, 60, 30])
        self.assertEqual(early["duration_minutes"].tolist(), [60, 60, 60, 30])
        self.assertFalse(normal["is_early_close"].any())
        self.assertTrue(early["is_early_close"].all())

    def test_clock_10_16_windows_are_six_full_new_york_hours(self):
        schedule = pd.DataFrame(
            {
                "market_open": [
                    pd.Timestamp("2026-09-08 13:30:00Z"),
                    pd.Timestamp("2026-11-27 14:30:00Z"),
                ],
                "market_close": [
                    pd.Timestamp("2026-09-08 20:00:00Z"),
                    pd.Timestamp("2026-11-27 18:00:00Z"),
                ],
            },
            index=pd.DatetimeIndex(["2026-09-08", "2026-11-27"]),
        )
        windows = build_expected_windows(schedule, window_mode=WINDOW_MODE_CLOCK_10_16)
        normal = windows[windows["trading_date"] == "2026-09-08"]
        early = windows[windows["trading_date"] == "2026-11-27"]
        self.assertEqual(normal["duration_minutes"].tolist(), [60] * 6)
        self.assertEqual(
            normal["bar_start_utc"].tolist(),
            [pd.Timestamp(f"2026-09-08 {hour:02d}:00:00Z") for hour in range(14, 20)],
        )
        self.assertEqual(early["duration_minutes"].tolist(), [60] * 3)
        self.assertEqual(
            early["bar_start_utc"].tolist(),
            [pd.Timestamp(f"2026-11-27 {hour:02d}:00:00Z") for hour in range(15, 18)],
        )
        self.assertFalse(normal["short_window"].any())
        self.assertTrue(early["is_early_close"].all())

    def test_bridge_07_10_contains_three_lag_only_windows(self):
        schedule = pd.DataFrame(
            {
                "market_open": [
                    pd.Timestamp("2026-09-08 13:30:00Z"),
                    pd.Timestamp("2026-11-27 14:30:00Z"),
                ],
                "market_close": [
                    pd.Timestamp("2026-09-08 20:00:00Z"),
                    pd.Timestamp("2026-11-27 18:00:00Z"),
                ],
            },
            index=pd.DatetimeIndex(["2026-09-08", "2026-11-27"]),
        )
        windows = build_expected_windows(schedule, window_mode=WINDOW_MODE_BRIDGE_07_10)
        normal = windows[windows["trading_date"] == "2026-09-08"]
        early = windows[windows["trading_date"] == "2026-11-27"]
        self.assertEqual(
            normal["bar_start_utc"].tolist(),
            [
                pd.Timestamp("2026-09-08 11:00:00Z"),
                pd.Timestamp("2026-09-08 12:00:00Z"),
                pd.Timestamp("2026-09-08 13:00:00Z"),
            ],
        )
        self.assertEqual(
            early["bar_start_utc"].tolist(),
            [
                pd.Timestamp("2026-11-27 12:00:00Z"),
                pd.Timestamp("2026-11-27 13:00:00Z"),
                pd.Timestamp("2026-11-27 14:00:00Z"),
            ],
        )
        self.assertEqual(normal["premarket_minutes"].tolist(), [60, 60, 30])
        self.assertEqual(normal["regular_minutes"].tolist(), [0, 0, 30])
        self.assertEqual(set(windows["window_role"]), {"lag_only_bridge"})
        self.assertTrue(windows["duration_minutes"].eq(60).all())

    def test_five_minute_aggregation_uses_real_subbars(self):
        schedule = pd.DataFrame(
            {
                "market_open": [pd.Timestamp("2026-09-08 13:30:00Z")],
                "market_close": [pd.Timestamp("2026-09-08 20:00:00Z")],
            },
            index=pd.DatetimeIndex(["2026-09-08"]),
        )
        expected = build_expected_windows(schedule).iloc[:1]
        times = pd.date_range("2026-09-08 13:30:00Z", periods=12, freq="5min")
        bars = pd.DataFrame(
            {
                "source_timestamp_utc": times,
                "open": range(100, 112),
                "high": range(101, 113),
                "low": range(99, 111),
                "close": [value + 0.5 for value in range(100, 112)],
                "volume": [10] * 12,
                "split_factor": [0] * 12,
            }
        )
        fetch = FetchResult(
            bars=bars,
            source="test",
            feed="test",
            adjustment="raw",
            interval="5m",
            timestamp_semantics="bar_start",
            cache_hit=False,
        )
        windows, stats = align_provider_bars(fetch, expected)
        row = windows.iloc[0]
        self.assertTrue(row["window_complete"])
        self.assertEqual(row["expected_subbars"], 12)
        self.assertEqual(row["observed_subbars"], 12)
        self.assertEqual(row["open"], 100)
        self.assertEqual(row["high"], 112)
        self.assertEqual(row["low"], 99)
        self.assertEqual(row["close"], 111.5)
        self.assertEqual(row["volume"], 120)
        self.assertTrue(stats["quality_pass"])

    def test_clock_mode_ignores_intentionally_excluded_opening_half_hour(self):
        schedule = pd.DataFrame(
            {
                "market_open": [pd.Timestamp("2026-09-08 13:30:00Z")],
                "market_close": [pd.Timestamp("2026-09-08 20:00:00Z")],
            },
            index=pd.DatetimeIndex(["2026-09-08"]),
        )
        expected = build_expected_windows(schedule, window_mode=WINDOW_MODE_CLOCK_10_16)
        times = pd.date_range("2026-09-08 13:30:00Z", periods=78, freq="5min")
        bars = pd.DataFrame(
            {
                "source_timestamp_utc": times,
                "open": [100.0] * 78,
                "high": [101.0] * 78,
                "low": [99.0] * 78,
                "close": [100.5] * 78,
                "volume": [10] * 78,
                "split_factor": [0] * 78,
            }
        )
        fetch = FetchResult(
            bars=bars,
            source="test",
            feed="test",
            adjustment="raw",
            interval="5m",
            timestamp_semantics="bar_start",
            cache_hit=False,
        )
        windows, stats = align_provider_bars(fetch, expected)
        self.assertEqual(len(windows), 6)
        self.assertTrue(windows["window_complete"].all())
        self.assertEqual(stats["source_rows_inside_session_outside_selected_windows"], 6)
        self.assertEqual(stats["misaligned_source_rows_inside_session"], 0)
        self.assertTrue(stats["quality_pass"])

    def test_bridge_mode_requires_all_extended_hours_subbars(self):
        schedule = pd.DataFrame(
            {
                "market_open": [pd.Timestamp("2026-09-08 13:30:00Z")],
                "market_close": [pd.Timestamp("2026-09-08 20:00:00Z")],
            },
            index=pd.DatetimeIndex(["2026-09-08"]),
        )
        expected = build_expected_windows(schedule, window_mode=WINDOW_MODE_BRIDGE_07_10)
        times = pd.date_range("2026-09-08 11:00:00Z", periods=36, freq="5min")
        bars = pd.DataFrame(
            {
                "source_timestamp_utc": times,
                "open": [100.0] * 36,
                "high": [101.0] * 36,
                "low": [99.0] * 36,
                "close": [100.5] * 36,
                "volume": [10] * 36,
                "split_factor": [0] * 36,
            }
        )
        fetch = FetchResult(
            bars=bars,
            source="test",
            feed="extended",
            adjustment="raw",
            interval="5m",
            timestamp_semantics="bar_start",
            cache_hit=False,
        )
        complete, complete_stats = align_provider_bars(fetch, expected)
        self.assertTrue(complete["window_complete"].all())
        self.assertTrue(complete_stats["quality_pass"])
        self.assertEqual(complete_stats["source_rows_outside_regular_session"], 30)

        missing_fetch = FetchResult(
            bars=bars.iloc[1:].copy(),
            source="test",
            feed="extended",
            adjustment="raw",
            interval="5m",
            timestamp_semantics="bar_start",
            cache_hit=False,
        )
        incomplete, incomplete_stats = align_provider_bars(missing_fetch, expected)
        self.assertFalse(incomplete.iloc[0]["window_complete"])
        self.assertEqual(incomplete.iloc[0]["missing_subbars"], 1)
        self.assertFalse(incomplete_stats["quality_pass"])


class ReturnTests(unittest.TestCase):
    @staticmethod
    def row(day, start, end, open_value, close_value, complete=True, duration=60):
        return {
            "ticker": "TEST",
            "trading_date": day,
            "session_open_utc": "2026-09-08T13:30:00Z" if day == "2026-09-08" else "2026-09-09T13:30:00Z",
            "bar_start_utc": start,
            "bar_end_utc": end,
            "session_close_utc": "2026-09-08T20:00:00Z" if day == "2026-09-08" else "2026-09-09T20:00:00Z",
            "window_complete": complete,
            "duration_minutes": duration,
            "open": open_value,
            "high": max(open_value, close_value),
            "low": min(open_value, close_value),
            "close": close_value,
            "volume": 100,
            "split_event": False,
            "adjustment": "raw",
        }

    def test_first_bar_overnight_and_gap_are_separate(self):
        prices = pd.DataFrame(
            [
                self.row("2026-09-08", "2026-09-08T13:30:00Z", "2026-09-08T19:00:00Z", 100, 101),
                self.row("2026-09-08", "2026-09-08T19:00:00Z", "2026-09-08T20:00:00Z", 101, 102),
                self.row("2026-09-09", "2026-09-09T13:30:00Z", "2026-09-09T14:30:00Z", 103, 104),
                self.row("2026-09-09", "2026-09-09T14:30:00Z", "2026-09-09T15:30:00Z", 104, 105, complete=False),
                self.row("2026-09-09", "2026-09-09T15:30:00Z", "2026-09-09T16:30:00Z", 105, 106),
            ]
        )
        returns = compute_returns(prices, jump_threshold=0.20)
        self.assertTrue(math.isnan(returns.loc[0, "ret_cc"]))
        self.assertAlmostEqual(returns.loc[1, "ret_cc"], 102 / 101 - 1)
        self.assertTrue(math.isnan(returns.loc[2, "ret_cc"]))
        self.assertAlmostEqual(returns.loc[2, "ret_overnight_open_prev_close"], 103 / 102 - 1)
        self.assertTrue(math.isnan(returns.loc[4, "ret_cc"]))
        self.assertEqual(returns.loc[4, "cc_gap_reason"], "previous_window_incomplete")

    def test_no_overnight_when_prior_close_window_missing(self):
        prices = pd.DataFrame(
            [
                self.row("2026-09-08", "2026-09-08T19:30:00Z", "2026-09-08T20:00:00Z", 100, 101, complete=False, duration=30),
                self.row("2026-09-09", "2026-09-09T13:30:00Z", "2026-09-09T14:30:00Z", 102, 103),
            ]
        )
        returns = compute_returns(prices, jump_threshold=0.20)
        self.assertTrue(math.isnan(returns.loc[1, "ret_overnight_open_prev_close"]))

    def test_no_overnight_return_when_first_included_window_starts_at_ten(self):
        prices = pd.DataFrame(
            [
                self.row("2026-09-08", "2026-09-08T19:00:00Z", "2026-09-08T20:00:00Z", 100, 101),
                self.row("2026-09-09", "2026-09-09T14:00:00Z", "2026-09-09T15:00:00Z", 103, 104),
            ]
        )
        returns = compute_returns(prices, jump_threshold=0.20)
        self.assertTrue(math.isnan(returns.loc[1, "ret_overnight_open_prev_close"]))
        self.assertEqual(returns.loc[1, "cc_gap_reason"], "first_window_of_trading_day")


if __name__ == "__main__":
    unittest.main()
