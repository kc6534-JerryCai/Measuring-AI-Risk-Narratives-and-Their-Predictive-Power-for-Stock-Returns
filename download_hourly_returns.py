#!/usr/bin/env python3
"""Download US intraday bars and construct exchange-aligned hourly returns.

The program keeps provider responses/cache separate from processed outputs.  It
uses the NYSE calendar, treats provider timestamps as bar starts, and constructs
regular-session windows anchored either at the session open or at New York
clock hours from 10:00 through 16:00.  It never shifts a provider's bars to
manufacture alignment: when native 1-hour bars do not align, it downloads
5-minute bars and aggregates them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency error is reported in main
    load_dotenv = None

try:
    import pandas_market_calendars as mcal
except ImportError:  # pragma: no cover - dependency error is reported in main
    mcal = None


NY_TZ = "America/New_York"
UTC = timezone.utc
SCHEMA_VERSION = "hourly_returns_v2"
WINDOW_MODE_SESSION_OPEN = "session_open"
WINDOW_MODE_CLOCK_10_16 = "clock_10_16"
WINDOW_MODE_BRIDGE_07_10 = "bridge_07_10"
WINDOW_MODES = (
    WINDOW_MODE_SESSION_OPEN,
    WINDOW_MODE_CLOCK_10_16,
    WINDOW_MODE_BRIDGE_07_10,
)

DEFAULT_TICKERS = [
    "MSFT", "GOOGL", "META", "AMZN", "TSLA",
    "ORCL", "CRM", "NOW", "ADBE", "WDAY",
    "INTU", "PATH", "AAPL", "PLTR", "UBER",
    "SNOW", "SHOP", "RBLX", "SNAP", "PINS",
]

SOURCE_DOCS = {
    "yahoo": {
        "documentation": "https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html",
        "notes": (
            "Intraday history is limited to the most recent 60 days. start is inclusive, "
            "end is exclusive. prepost is explicit for each window mode and auto_adjust=False."
        ),
    },
    "alpaca": {
        "documentation": "https://docs.alpaca.markets/us/reference/stockbars",
        "feed_documentation": "https://docs.alpaca.markets/us/docs/market-data-faq",
        "notes": (
            "Bars are timestamped at the left edge. SIP consolidates all US exchanges; "
            "IEX is a single-exchange feed. Pagination follows next_page_token."
        ),
    },
    "twelve_data": {
        "documentation": "https://twelvedata.com/docs#time-series",
        "history_documentation": "https://support.twelvedata.com/en/articles/5214728-getting-historical-data",
        "notes": "time_series accepts start_date/end_date and returns at most 5,000 points per request.",
    },
    "fmp": {
        "documentation": "https://site.financialmodelingprep.com/developer/docs/stable/intraday-1-hour",
        "notes": "Stable historical-chart endpoints are used for 1hour and 5min data.",
    },
    "calendar": {
        "documentation": "https://pandas-market-calendars.readthedocs.io/en/stable/",
        "notes": "NYSE market_open/market_close schedule supplies holidays, DST and early closes.",
    },
}


class SourceError(RuntimeError):
    """A provider error with a stable classification for fallback logic."""

    def __init__(self, message: str, *, kind: str = "provider_error", status_code: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code


@dataclass
class FetchResult:
    bars: pd.DataFrame
    source: str
    feed: str
    adjustment: str
    interval: str
    timestamp_semantics: str
    cache_hit: bool
    raw_cache_files: list[str] = field(default_factory=list)
    request_ids: list[str] = field(default_factory=list)
    provider_notes: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    windows: pd.DataFrame
    fetch: FetchResult
    stats: dict[str, Any]
    attempt_id: str
    source_rank: int
    interval_rank: int


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def iso_utc(value: pd.Timestamp | datetime | None) -> str:
    if value is None or pd.isna(value):
        return ""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def iso_ny(value: pd.Timestamp | datetime | None) -> str:
    if value is None or pd.isna(value):
        return ""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY_TZ)
    else:
        ts = ts.tz_convert(NY_TZ)
    return ts.isoformat()


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def first_env(*names: str) -> tuple[str | None, str | None]:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip(), name
    return None, None


def configured_credentials() -> dict[str, Any]:
    alpaca_key, alpaca_key_name = first_env(
        "APCA_API_KEY_ID", "ALPACA_API_KEY_ID", "ALPACA_API_KEY"
    )
    alpaca_secret, alpaca_secret_name = first_env(
        "APCA_API_SECRET_KEY", "ALPACA_API_SECRET_KEY", "ALPACA_SECRET_KEY"
    )
    twelve_key, twelve_name = first_env("TWELVE_DATA_API_KEY", "TWELVEDATA_API_KEY")
    fmp_key, fmp_name = first_env("FMP_API_KEY")
    return {
        "alpaca": {
            "configured": bool(alpaca_key and alpaca_secret),
            "key": alpaca_key,
            "secret": alpaca_secret,
            "key_variable": alpaca_key_name,
            "secret_variable": alpaca_secret_name,
        },
        "twelve_data": {
            "configured": bool(twelve_key),
            "key": twelve_key,
            "key_variable": twelve_name,
        },
        "fmp": {
            "configured": bool(fmp_key),
            "key": fmp_key,
            "key_variable": fmp_name,
        },
    }


def redact(text: Any, secrets: Iterable[str | None] = ()) -> str:
    message = str(text or "")
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    message = re.sub(r"(?i)(apikey|api_key|secret|token)=([^&\s]+)", r"\1=[REDACTED]", message)
    message = re.sub(r"(?i)(APCA-API-(?:KEY-ID|SECRET-KEY)):\s*\S+", r"\1: [REDACTED]", message)
    return message[:1500]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", na_rep="")
    temporary.replace(path)


def cache_tag(ticker: str, start_date: date, end_date: date, interval: str) -> str:
    return f"{ticker.upper()}_{start_date.isoformat()}_{end_date.isoformat()}_{interval}"


def sleep_for_retry(attempt: int, retry_after: str | None = None) -> None:
    if retry_after:
        try:
            seconds = min(60.0, max(0.0, float(retry_after)))
        except ValueError:
            seconds = 0.0
    else:
        seconds = min(8.0, 2.0 ** attempt) + random.uniform(0.0, 0.25)
    time.sleep(seconds)


def request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    params: dict[str, Any],
    headers: dict[str, str] | None,
    timeout: float,
    max_retries: int,
    secrets: Iterable[str | None],
) -> tuple[Any, dict[str, str]]:
    """Request JSON with bounded retry; never returns or logs a credential-bearing URL."""
    for attempt in range(max_retries + 1):
        try:
            response = session.request(
                method,
                url,
                params=params,
                headers=headers,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise SourceError(redact(exc, secrets), kind="transport") from exc
            sleep_for_retry(attempt)
            continue

        status = response.status_code
        if status == 429 or 500 <= status <= 599:
            if attempt >= max_retries:
                raise SourceError(
                    f"HTTP {status} after {max_retries} retries",
                    kind="rate_limit" if status == 429 else "server_error",
                    status_code=status,
                )
            sleep_for_retry(attempt, response.headers.get("Retry-After"))
            continue

        try:
            payload = response.json()
        except ValueError as exc:
            body = redact(response.text[:500], secrets)
            raise SourceError(f"HTTP {status} returned non-JSON: {body}", kind="parse", status_code=status) from exc

        if status >= 400:
            kind = "auth" if status == 401 else "permission" if status in {403, 422} else "request"
            detail = ""
            if isinstance(payload, dict):
                detail = str(payload.get("message") or payload.get("error") or payload.get("detail") or "")
            raise SourceError(
                redact(f"HTTP {status}: {detail or 'provider rejected request'}", secrets),
                kind=kind,
                status_code=status,
            )
        return payload, dict(response.headers)
    raise AssertionError("unreachable")


def normalise_bar_frame(frame: pd.DataFrame, *, assumed_timezone: str) -> pd.DataFrame:
    required = ["source_timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SourceError(f"provider response missing columns: {missing}", kind="schema")
    result = frame.copy()
    timestamps = pd.to_datetime(result["source_timestamp"], errors="coerce")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(assumed_timezone, ambiguous="infer", nonexistent="shift_forward")
    timestamps = timestamps.dt.tz_convert("UTC")
    result["source_timestamp_utc"] = timestamps
    for column in ["open", "high", "low", "close", "volume"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if "split_factor" not in result.columns:
        result["split_factor"] = 0.0
    result["split_factor"] = pd.to_numeric(result["split_factor"], errors="coerce").fillna(0.0)
    result = result.dropna(subset=["source_timestamp_utc"]).sort_values("source_timestamp_utc")
    result = result[["source_timestamp_utc", "open", "high", "low", "close", "volume", "split_factor"]]
    return result.reset_index(drop=True)


class BaseAdapter:
    source = "base"

    def __init__(self, out_dir: Path, *, timeout: float, max_retries: int, refresh: bool):
        self.out_dir = out_dir
        self.raw_dir = out_dir / "raw" / self.source
        self.timeout = timeout
        self.max_retries = max_retries
        self.refresh = refresh
        self.session = requests.Session()

    def availability(self) -> tuple[bool, str]:
        return True, "available"

    def fetch(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        start_utc: pd.Timestamp,
        end_utc: pd.Timestamp,
        interval: str,
    ) -> FetchResult:
        raise NotImplementedError


class YahooAdapter(BaseAdapter):
    source = "yahoo"

    def __init__(self, *args: Any, include_prepost: bool = False, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.include_prepost = include_prepost

    def availability(self) -> tuple[bool, str]:
        try:
            import yfinance  # noqa: F401
        except ImportError:
            return False, "yfinance is not installed"
        return True, "no API key required"

    @staticmethod
    def _flatten_download(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if not isinstance(frame.columns, pd.MultiIndex):
            return frame
        symbol = ticker.upper()
        for level in range(frame.columns.nlevels):
            values = {str(value).upper() for value in frame.columns.get_level_values(level)}
            if symbol in values:
                return frame.xs(symbol, axis=1, level=level, drop_level=True)
        flattened = frame.copy()
        flattened.columns = ["_".join(str(part) for part in item if str(part)) for item in flattened.columns]
        return flattened

    def fetch(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        start_utc: pd.Timestamp,
        end_utc: pd.Timestamp,
        interval: str,
    ) -> FetchResult:
        import yfinance as yf

        yf_interval = "1h" if interval == "1h" else "5m"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        tag = cache_tag(ticker, start_date, end_date, interval)
        if self.include_prepost:
            tag = f"{tag}_prepost"
        provider_file = self.raw_dir / f"{tag}_provider_response.csv"
        cache_file = self.raw_dir / f"{tag}_normalised.csv"
        legacy_cache_file = self.raw_dir / f"{tag}_response.csv"
        meta_file = self.raw_dir / f"{tag}_metadata.json"
        readable_cache_file = cache_file if cache_file.exists() else legacy_cache_file
        cache_hit = readable_cache_file.exists() and not self.refresh

        if cache_hit:
            cached = pd.read_csv(readable_cache_file)
            bars = normalise_bar_frame(cached, assumed_timezone=NY_TZ)
        else:
            try:
                yf.set_tz_cache_location(str(self.raw_dir / "_yfinance_cache"))
            except Exception:
                pass
            if hasattr(yf, "config"):
                try:
                    yf.config.network.retries = 0
                    yf.config.debug.hide_exceptions = False
                except Exception:
                    pass

            last_error: Exception | None = None
            downloaded: pd.DataFrame | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    downloaded = yf.download(
                        ticker,
                        start=start_date.isoformat(),
                        end=(end_date + timedelta(days=1)).isoformat(),
                        interval=yf_interval,
                        prepost=self.include_prepost,
                        auto_adjust=False,
                        back_adjust=False,
                        repair=False,
                        actions=True,
                        threads=False,
                        progress=False,
                        keepna=True,
                        ignore_tz=False,
                        timeout=self.timeout,
                        group_by="column",
                        multi_level_index=True,
                    )
                    downloaded = self._flatten_download(downloaded, ticker)
                    if downloaded is not None and not downloaded.empty:
                        break
                    last_error = SourceError("empty yfinance response", kind="empty")
                except Exception as exc:  # yfinance wraps HTTP errors in several exception classes
                    last_error = exc
                if attempt < self.max_retries:
                    sleep_for_retry(attempt)
            if downloaded is None or downloaded.empty:
                raise SourceError(redact(last_error or "empty yfinance response"), kind="empty")

            raw = downloaded.copy()
            raw.index.name = "source_timestamp"
            raw = raw.reset_index()
            write_csv_atomic(raw, provider_file)
            rename = {str(column).strip().casefold(): column for column in raw.columns}

            def find_column(*names: str) -> str | None:
                for name in names:
                    if name.casefold() in rename:
                        return str(rename[name.casefold()])
                return None

            timestamp_column = find_column("source_timestamp", "datetime", "date")
            mapping = {
                "source_timestamp": timestamp_column,
                "open": find_column("open"),
                "high": find_column("high"),
                "low": find_column("low"),
                "close": find_column("close"),
                "volume": find_column("volume"),
                "split_factor": find_column("stock splits", "stock_splits"),
            }
            missing = [name for name in ["source_timestamp", "open", "high", "low", "close", "volume"] if not mapping[name]]
            if missing:
                raise SourceError(f"yfinance schema missing {missing}; columns={list(raw.columns)}", kind="schema")
            prepared = pd.DataFrame({
                name: raw[column] if column else 0.0
                for name, column in mapping.items()
            })
            bars = normalise_bar_frame(prepared, assumed_timezone=NY_TZ)
            cache_payload = bars.rename(columns={"source_timestamp_utc": "source_timestamp"}).copy()
            cache_payload["source_timestamp"] = cache_payload["source_timestamp"].map(iso_utc)
            write_csv_atomic(cache_payload, cache_file)
            write_json(meta_file, {
                "schema_version": SCHEMA_VERSION,
                "source": "Yahoo Finance via yfinance",
                "ticker": ticker,
                "interval": yf_interval,
                "start_inclusive": start_date.isoformat(),
                "end_exclusive": (end_date + timedelta(days=1)).isoformat(),
                "prepost": self.include_prepost,
                "auto_adjust": False,
                "back_adjust": False,
                "repair": False,
                "timestamp_semantics": "bar_start",
                "retrieved_at": utc_now_iso(),
                "yfinance_version": getattr(yf, "__version__", "unknown"),
            })

        bars = bars[(bars["source_timestamp_utc"] >= start_utc) & (bars["source_timestamp_utc"] < end_utc)].copy()
        if bars.empty:
            raise SourceError("Yahoo returned no bars inside the requested complete-session range", kind="empty")
        return FetchResult(
            bars=bars,
            source=self.source,
            feed=(
                "yahoo_consolidated_extended_hours"
                if self.include_prepost
                else "yahoo_consolidated"
            ),
            adjustment="raw_unadjusted_auto_adjust_false",
            interval=interval,
            timestamp_semantics="bar_start",
            cache_hit=cache_hit,
            raw_cache_files=[
                str(path.relative_to(self.out_dir))
                for path in [provider_file, readable_cache_file if cache_hit else cache_file]
                if path.exists()
            ],
            provider_notes=[
                "Yahoo/yfinance end date is exclusive; API index is treated as bar start after schedule alignment.",
                f"Extended-hours inclusion requested with prepost={self.include_prepost}.",
            ],
        )


class AlpacaAdapter(BaseAdapter):
    source = "alpaca"

    def __init__(self, *args: Any, feed: str, credentials: dict[str, Any], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.feed = feed
        self.credentials = credentials
        self.raw_dir = self.raw_dir / feed

    def availability(self) -> tuple[bool, str]:
        if not self.credentials.get("configured"):
            return False, "missing APCA_API_KEY_ID/APCA_API_SECRET_KEY (or compatible ALPACA aliases)"
        return True, f"credentials configured; explicit feed={self.feed}"

    def fetch(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        start_utc: pd.Timestamp,
        end_utc: pd.Timestamp,
        interval: str,
    ) -> FetchResult:
        api_interval = "1Hour" if interval == "1h" else "5Min"
        tag = cache_tag(ticker, start_date, end_date, interval)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        combined_file = self.raw_dir / f"{tag}_normalised.csv"
        cache_hit = combined_file.exists() and not self.refresh
        request_ids: list[str] = []
        raw_files: list[str] = []

        if cache_hit:
            bars = normalise_bar_frame(pd.read_csv(combined_file), assumed_timezone="UTC")
            raw_files.append(str(combined_file.relative_to(self.out_dir)))
        else:
            headers = {
                "APCA-API-KEY-ID": self.credentials["key"],
                "APCA-API-SECRET-KEY": self.credentials["secret"],
            }
            page_token: str | None = None
            page_index = 0
            rows: list[dict[str, Any]] = []
            while True:
                page_index += 1
                page_file = self.raw_dir / f"{tag}_page_{page_index:03d}.json"
                if page_file.exists() and not self.refresh:
                    payload = json.loads(page_file.read_text(encoding="utf-8"))
                else:
                    params: dict[str, Any] = {
                        "symbols": ticker,
                        "timeframe": api_interval,
                        "start": iso_utc(start_utc),
                        "end": iso_utc(end_utc),
                        "limit": 10000,
                        "adjustment": "raw",
                        "feed": self.feed,
                        "sort": "asc",
                    }
                    if page_token:
                        params["page_token"] = page_token
                    payload, response_headers = request_json(
                        self.session,
                        "GET",
                        "https://data.alpaca.markets/v2/stocks/bars",
                        params=params,
                        headers=headers,
                        timeout=self.timeout,
                        max_retries=self.max_retries,
                        secrets=[self.credentials.get("key"), self.credentials.get("secret")],
                    )
                    request_id = response_headers.get("X-Request-ID") or response_headers.get("x-request-id")
                    if request_id:
                        request_ids.append(request_id)
                    write_json(page_file, payload)
                raw_files.append(str(page_file.relative_to(self.out_dir)))
                if not isinstance(payload, dict):
                    raise SourceError("Alpaca response is not an object", kind="schema")
                provider_bars = payload.get("bars", {})
                if isinstance(provider_bars, dict):
                    rows.extend(provider_bars.get(ticker, []) or provider_bars.get(ticker.upper(), []) or [])
                page_token = payload.get("next_page_token")
                if not page_token:
                    break
                if page_index > 1000:
                    raise SourceError("Alpaca pagination exceeded 1,000 pages", kind="pagination")

            prepared = pd.DataFrame({
                "source_timestamp": [row.get("t") for row in rows],
                "open": [row.get("o") for row in rows],
                "high": [row.get("h") for row in rows],
                "low": [row.get("l") for row in rows],
                "close": [row.get("c") for row in rows],
                "volume": [row.get("v") for row in rows],
                "split_factor": 0.0,
            })
            if prepared.empty:
                raise SourceError(f"Alpaca {self.feed} returned no bars", kind="empty")
            bars = normalise_bar_frame(prepared, assumed_timezone="UTC")
            saved = bars.rename(columns={"source_timestamp_utc": "source_timestamp"}).copy()
            saved["source_timestamp"] = saved["source_timestamp"].map(iso_utc)
            write_csv_atomic(saved, combined_file)
            raw_files.append(str(combined_file.relative_to(self.out_dir)))
            write_json(self.raw_dir / f"{tag}_metadata.json", {
                "schema_version": SCHEMA_VERSION,
                "source": "Alpaca Market Data",
                "ticker": ticker,
                "feed": self.feed,
                "interval": api_interval,
                "adjustment": "raw",
                "timestamp_semantics": "bar_start",
                "start": iso_utc(start_utc),
                "end": iso_utc(end_utc),
                "retrieved_at": utc_now_iso(),
                "request_ids": request_ids,
            })

        bars = bars[(bars["source_timestamp_utc"] >= start_utc) & (bars["source_timestamp_utc"] < end_utc)].copy()
        if bars.empty:
            raise SourceError(f"Alpaca {self.feed} has no bars inside requested range", kind="empty")
        notes = [
            "Alpaca timestamps are bar starts.",
            "IEX contains only Investors Exchange trades, not consolidated US market coverage."
            if self.feed == "iex" else
            "SIP consolidates eligible trades from all US exchanges.",
        ]
        return FetchResult(
            bars=bars,
            source=self.source,
            feed=self.feed,
            adjustment="raw",
            interval=interval,
            timestamp_semantics="bar_start",
            cache_hit=cache_hit,
            raw_cache_files=raw_files,
            request_ids=request_ids,
            provider_notes=notes,
        )


class TwelveDataAdapter(BaseAdapter):
    source = "twelve_data"

    def __init__(self, *args: Any, credentials: dict[str, Any], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.credentials = credentials

    def availability(self) -> tuple[bool, str]:
        if not self.credentials.get("configured"):
            return False, "missing TWELVE_DATA_API_KEY or TWELVEDATA_API_KEY"
        return True, "credentials configured"

    def fetch(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        start_utc: pd.Timestamp,
        end_utc: pd.Timestamp,
        interval: str,
    ) -> FetchResult:
        api_interval = "1h" if interval == "1h" else "5min"
        tag = cache_tag(ticker, start_date, end_date, interval)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        combined_file = self.raw_dir / f"{tag}_normalised.csv"
        cache_hit = combined_file.exists() and not self.refresh
        raw_files: list[str] = []

        if cache_hit:
            bars = normalise_bar_frame(pd.read_csv(combined_file), assumed_timezone="UTC")
            raw_files.append(str(combined_file.relative_to(self.out_dir)))
        else:
            rows: list[dict[str, Any]] = []
            chunk_days = 20 if interval == "5m" else 90
            chunk_start = pd.Timestamp(start_utc).tz_convert("UTC")
            final_end = pd.Timestamp(end_utc).tz_convert("UTC")
            chunk_index = 0
            while chunk_start < final_end:
                chunk_end = min(chunk_start + pd.Timedelta(days=chunk_days), final_end)
                chunk_index += 1
                chunk_file = self.raw_dir / f"{tag}_chunk_{chunk_index:03d}.json"
                if chunk_file.exists() and not self.refresh:
                    payload = json.loads(chunk_file.read_text(encoding="utf-8"))
                else:
                    params = {
                        "symbol": ticker,
                        "interval": api_interval,
                        "start_date": chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
                        "end_date": chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
                        "timezone": "UTC",
                        "order": "ASC",
                        "prepost": "false",
                        "adjust": "none",
                        "apikey": self.credentials["key"],
                    }
                    payload, _ = request_json(
                        self.session,
                        "GET",
                        "https://api.twelvedata.com/time_series",
                        params=params,
                        headers=None,
                        timeout=self.timeout,
                        max_retries=self.max_retries,
                        secrets=[self.credentials.get("key")],
                    )
                    write_json(chunk_file, payload)
                raw_files.append(str(chunk_file.relative_to(self.out_dir)))
                if not isinstance(payload, dict):
                    raise SourceError("Twelve Data response is not an object", kind="schema")
                if payload.get("status") == "error" or "values" not in payload:
                    code = payload.get("code")
                    message = payload.get("message") or "missing values"
                    kind = "auth" if code in {401, 403} else "provider_error"
                    raise SourceError(redact(f"Twelve Data error {code}: {message}", [self.credentials.get("key")]), kind=kind)
                rows.extend(payload.get("values") or [])
                chunk_start = chunk_end

            prepared = pd.DataFrame({
                "source_timestamp": [row.get("datetime") for row in rows],
                "open": [row.get("open") for row in rows],
                "high": [row.get("high") for row in rows],
                "low": [row.get("low") for row in rows],
                "close": [row.get("close") for row in rows],
                "volume": [row.get("volume", np.nan) for row in rows],
                "split_factor": 0.0,
            })
            if prepared.empty:
                raise SourceError("Twelve Data returned no bars", kind="empty")
            bars = normalise_bar_frame(prepared, assumed_timezone="UTC")
            saved = bars.rename(columns={"source_timestamp_utc": "source_timestamp"}).copy()
            saved["source_timestamp"] = saved["source_timestamp"].map(iso_utc)
            write_csv_atomic(saved, combined_file)
            raw_files.append(str(combined_file.relative_to(self.out_dir)))
            write_json(self.raw_dir / f"{tag}_metadata.json", {
                "schema_version": SCHEMA_VERSION,
                "source": "Twelve Data",
                "ticker": ticker,
                "interval": api_interval,
                "adjust": "none",
                "timezone": "UTC",
                "timestamp_semantics": "bar_start",
                "retrieved_at": utc_now_iso(),
            })

        bars = bars[(bars["source_timestamp_utc"] >= start_utc) & (bars["source_timestamp_utc"] < end_utc)].copy()
        if bars.empty:
            raise SourceError("Twelve Data has no bars inside requested range", kind="empty")
        return FetchResult(
            bars=bars,
            source=self.source,
            feed="twelve_data_us_equities",
            adjustment="none_requested",
            interval=interval,
            timestamp_semantics="bar_start",
            cache_hit=cache_hit,
            raw_cache_files=raw_files,
            provider_notes=["Twelve Data datetime is interpreted in the explicitly requested UTC timezone."],
        )


class FMPAdapter(BaseAdapter):
    source = "fmp"

    def __init__(self, *args: Any, credentials: dict[str, Any], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.credentials = credentials

    def availability(self) -> tuple[bool, str]:
        if not self.credentials.get("configured"):
            return False, "missing FMP_API_KEY"
        return True, "credentials configured"

    def fetch(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        start_utc: pd.Timestamp,
        end_utc: pd.Timestamp,
        interval: str,
    ) -> FetchResult:
        api_interval = "1hour" if interval == "1h" else "5min"
        tag = cache_tag(ticker, start_date, end_date, interval)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        combined_file = self.raw_dir / f"{tag}_normalised.csv"
        cache_hit = combined_file.exists() and not self.refresh
        raw_files: list[Path] = []

        if cache_hit:
            bars = normalise_bar_frame(pd.read_csv(combined_file), assumed_timezone=NY_TZ)
        else:
            rows: list[dict[str, Any]] = []
            chunk_days = 30 if interval == "5m" else 365
            chunk_start = start_date
            chunk_index = 0
            while chunk_start <= end_date:
                chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), end_date)
                chunk_index += 1
                response_file = self.raw_dir / f"{tag}_chunk_{chunk_index:03d}.json"
                raw_files.append(response_file)
                if response_file.exists() and not self.refresh:
                    payload = json.loads(response_file.read_text(encoding="utf-8"))
                else:
                    params = {
                        "symbol": ticker,
                        "from": chunk_start.isoformat(),
                        "to": chunk_end.isoformat(),
                        "nonadjusted": "true",
                        "extended": "false",
                        "apikey": self.credentials["key"],
                    }
                    payload, _ = request_json(
                        self.session,
                        "GET",
                        f"https://financialmodelingprep.com/stable/historical-chart/{api_interval}",
                        params=params,
                        headers=None,
                        timeout=self.timeout,
                        max_retries=self.max_retries,
                        secrets=[self.credentials.get("key")],
                    )
                    write_json(response_file, payload)
                if isinstance(payload, dict):
                    message = payload.get("Error Message") or payload.get("error") or payload.get("message")
                    raise SourceError(redact(f"FMP returned error object: {message or payload}", [self.credentials.get("key")]), kind="provider_error")
                if not isinstance(payload, list):
                    raise SourceError("FMP response is not an OHLCV array", kind="schema")
                rows.extend(payload)
                chunk_start = chunk_end + timedelta(days=1)
            if not rows:
                raise SourceError("FMP returned no bars", kind="empty")
            prepared = pd.DataFrame({
                "source_timestamp": [row.get("date") for row in rows],
                "open": [row.get("open") for row in rows],
                "high": [row.get("high") for row in rows],
                "low": [row.get("low") for row in rows],
                "close": [row.get("close") for row in rows],
                "volume": [row.get("volume", np.nan) for row in rows],
                "split_factor": 0.0,
            })
            bars = normalise_bar_frame(prepared, assumed_timezone=NY_TZ)
            saved = bars.rename(columns={"source_timestamp_utc": "source_timestamp"}).copy()
            saved["source_timestamp"] = saved["source_timestamp"].map(iso_utc)
            write_csv_atomic(saved, combined_file)
            write_json(self.raw_dir / f"{tag}_metadata.json", {
                "schema_version": SCHEMA_VERSION,
                "source": "Financial Modeling Prep",
                "ticker": ticker,
                "interval": api_interval,
                "from": start_date.isoformat(),
                "to": end_date.isoformat(),
                "timestamp_semantics": "bar_start_assumed_and_schedule_validated",
                "adjustment": "raw_unadjusted_nonadjusted_true",
                "extended": False,
                "chunk_days": chunk_days,
                "chunk_count": chunk_index,
                "retrieved_at": utc_now_iso(),
            })

        bars = bars[(bars["source_timestamp_utc"] >= start_utc) & (bars["source_timestamp_utc"] < end_utc)].copy()
        if bars.empty:
            raise SourceError("FMP has no bars inside requested range", kind="empty")
        return FetchResult(
            bars=bars,
            source=self.source,
            feed="fmp_us_equities",
            adjustment="raw_unadjusted_nonadjusted_true",
            interval=interval,
            timestamp_semantics="bar_start_assumed_and_schedule_validated",
            cache_hit=cache_hit,
            raw_cache_files=[
                str(path.relative_to(self.out_dir))
                for path in [*raw_files, combined_file]
                if path.exists()
            ],
            provider_notes=["FMP intraday timestamps are treated as bar starts only after exact exchange-window alignment."],
        )


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}; expected YYYY-MM-DD") from exc


def resolve_schedule(
    *,
    months: int,
    requested_start: date | None,
    requested_end: date | None,
    now_ny: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if mcal is None:
        raise RuntimeError("pandas-market-calendars is required")
    now = now_ny if now_ny is not None else pd.Timestamp.now(tz=NY_TZ)
    if now.tzinfo is None:
        now = now.tz_localize(NY_TZ)
    else:
        now = now.tz_convert(NY_TZ)

    end_requested = requested_end or now.date()
    start_requested = requested_start or (now.date() - relativedelta(months=months))
    if start_requested > end_requested:
        raise ValueError("start date must not be after end date")

    calendar = mcal.get_calendar("NYSE")
    search_start = start_requested - timedelta(days=10)
    search_end = min(end_requested, now.date())
    candidate = calendar.schedule(start_date=search_start, end_date=search_end)
    if candidate.empty:
        raise ValueError("NYSE calendar has no sessions in the requested range")

    now_utc = now.tz_convert("UTC")
    candidate = candidate[candidate["market_close"] <= now_utc]
    candidate = candidate[candidate.index.date <= end_requested]
    schedule = candidate[candidate.index.date >= start_requested].copy()
    if schedule.empty:
        raise ValueError("no fully completed NYSE session exists in the requested range")

    actual_start = schedule.index[0].date()
    actual_end = schedule.index[-1].date()
    metadata = {
        "runtime_new_york": now.isoformat(),
        "requested_start_date": start_requested.isoformat(),
        "requested_end_date": end_requested.isoformat(),
        "actual_start_trading_date": actual_start.isoformat(),
        "actual_end_trading_date": actual_end.isoformat(),
        "actual_start_market_open_utc": iso_utc(schedule.iloc[0]["market_open"]),
        "actual_end_market_close_utc": iso_utc(schedule.iloc[-1]["market_close"]),
        "calendar": "NYSE",
        "calendar_timezone": NY_TZ,
        "sessions": int(len(schedule)),
    }
    return schedule, metadata


def build_expected_windows(
    schedule: pd.DataFrame,
    *,
    window_mode: str = WINDOW_MODE_SESSION_OPEN,
) -> pd.DataFrame:
    if window_mode not in WINDOW_MODES:
        raise ValueError(f"unsupported window mode: {window_mode}")

    rows: list[dict[str, Any]] = []
    for session_label, session in schedule.iterrows():
        market_open = pd.Timestamp(session["market_open"]).tz_convert("UTC")
        market_close = pd.Timestamp(session["market_close"]).tz_convert("UTC")
        total_minutes = int((market_close - market_open).total_seconds() // 60)
        is_early_close = total_minutes < 390
        window_number = 1

        if window_mode == WINDOW_MODE_SESSION_OPEN:
            candidate_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
            start = market_open
            while start < market_close:
                end = min(start + pd.Timedelta(minutes=60), market_close)
                candidate_windows.append((start, end))
                start = end
        elif window_mode == WINDOW_MODE_CLOCK_10_16:
            trading_date = session_label.date().isoformat()
            candidate_windows = []
            for hour in range(10, 16):
                start_ny = pd.Timestamp(f"{trading_date} {hour:02d}:00:00", tz=NY_TZ)
                end_ny = start_ny + pd.Timedelta(hours=1)
                start = start_ny.tz_convert("UTC")
                end = end_ny.tz_convert("UTC")
                if start >= market_open and end <= market_close:
                    candidate_windows.append((start, end))
        else:
            trading_date = session_label.date().isoformat()
            candidate_windows = []
            for hour in range(7, 10):
                start_ny = pd.Timestamp(f"{trading_date} {hour:02d}:00:00", tz=NY_TZ)
                end_ny = start_ny + pd.Timedelta(hours=1)
                candidate_windows.append((start_ny.tz_convert("UTC"), end_ny.tz_convert("UTC")))

        for start, end in candidate_windows:
            duration = int((end - start).total_seconds() // 60)
            premarket_end = min(end, market_open)
            premarket_minutes = max(0, int((premarket_end - start).total_seconds() // 60))
            regular_start = max(start, market_open)
            regular_end = min(end, market_close)
            regular_minutes = max(0, int((regular_end - regular_start).total_seconds() // 60))
            rows.append({
                "trading_date": session_label.date().isoformat(),
                "session_open_utc": market_open,
                "session_close_utc": market_close,
                "bar_start_utc": start,
                "bar_end_utc": end,
                "window_mode": window_mode,
                "window_role": (
                    "lag_only_bridge"
                    if window_mode == WINDOW_MODE_BRIDGE_07_10
                    else "analysis_window"
                ),
                "premarket_minutes": premarket_minutes,
                "regular_minutes": regular_minutes,
                "duration_minutes": duration,
                "window_number": window_number,
                "short_window": duration != 60,
                "is_early_close": is_early_close,
            })
            window_number += 1
    expected = pd.DataFrame(rows)
    if expected.empty:
        raise ValueError("calendar produced no expected hourly windows")
    return expected


def valid_ohlcv(values: dict[str, Any]) -> tuple[bool, bool, bool]:
    prices = [values.get(column) for column in ["open", "high", "low", "close"]]
    numeric_prices = [float(value) for value in prices if pd.notna(value)]
    nonpositive = len(numeric_prices) == 4 and any(value <= 0 for value in numeric_prices)
    if len(numeric_prices) != 4:
        return False, nonpositive, False
    open_value, high_value, low_value, close_value = numeric_prices
    logical = (
        high_value >= max(open_value, low_value, close_value)
        and low_value <= min(open_value, high_value, close_value)
    )
    volume = values.get("volume")
    volume_valid = pd.notna(volume) and float(volume) >= 0
    return bool(logical and not nonpositive), nonpositive, bool(volume_valid)


def align_provider_bars(fetch: FetchResult, expected: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    bars = fetch.bars.copy().sort_values("source_timestamp_utc")
    duplicate_mask = bars.duplicated("source_timestamp_utc", keep=False)
    duplicate_timestamps = int(duplicate_mask.sum())
    bars = bars.drop_duplicates("source_timestamp_utc", keep="last").set_index("source_timestamp_utc", drop=False)

    expected_starts = set(expected["bar_start_utc"].tolist())
    expected_five_minute: set[pd.Timestamp] = set()
    for row in expected.itertuples(index=False):
        if fetch.interval == "5m":
            expected_five_minute.update(
                pd.date_range(row.bar_start_utc, row.bar_end_utc, freq="5min", inclusive="left").tolist()
            )
    expected_source_times = expected_starts if fetch.interval == "1h" else expected_five_minute

    session_ranges = [
        (pd.Timestamp(row.session_open_utc), pd.Timestamp(row.session_close_utc))
        for row in expected.drop_duplicates("trading_date").itertuples(index=False)
    ]
    selected_ranges = [
        (
            pd.Timestamp(group["bar_start_utc"].min()),
            pd.Timestamp(group["bar_end_utc"].max()),
        )
        for _, group in expected.groupby("trading_date", sort=False)
    ]

    def is_in_session(timestamp: pd.Timestamp) -> bool:
        return any(start <= timestamp < end for start, end in session_ranges)

    def is_in_selected_range(timestamp: pd.Timestamp) -> bool:
        return any(start <= timestamp < end for start, end in selected_ranges)

    in_session_times = [timestamp for timestamp in bars.index if is_in_session(timestamp)]
    selected_range_times = [timestamp for timestamp in in_session_times if is_in_selected_range(timestamp)]
    excluded_session_times = len(in_session_times) - len(selected_range_times)
    misaligned_in_session = sum(timestamp not in expected_source_times for timestamp in selected_range_times)
    outside_rth = int(len(bars) - len(in_session_times))

    output_rows: list[dict[str, Any]] = []
    for expected_row in expected.itertuples(index=False):
        if fetch.interval == "1h":
            source_times = [expected_row.bar_start_utc]
        else:
            source_times = pd.date_range(
                expected_row.bar_start_utc,
                expected_row.bar_end_utc,
                freq="5min",
                inclusive="left",
            ).tolist()
        observed = bars.loc[bars.index.intersection(source_times)].sort_index()
        expected_subbars = len(source_times)
        observed_subbars = int(len(observed))
        missing_times = [timestamp for timestamp in source_times if timestamp not in bars.index]

        if observed.empty:
            values = {column: np.nan for column in ["open", "high", "low", "close", "volume"]}
            split_factor = 0.0
            first_source_timestamp: pd.Timestamp | None = None
        elif fetch.interval == "1h":
            source_row = observed.iloc[-1]
            values = {column: source_row[column] for column in ["open", "high", "low", "close", "volume"]}
            split_factor = float(source_row.get("split_factor", 0.0) or 0.0)
            first_source_timestamp = pd.Timestamp(source_row["source_timestamp_utc"])
        else:
            values = {
                "open": observed.iloc[0]["open"],
                "high": observed["high"].max(skipna=True),
                "low": observed["low"].min(skipna=True),
                "close": observed.iloc[-1]["close"],
                "volume": observed["volume"].sum(min_count=observed_subbars),
            }
            nonzero_splits = observed.loc[observed["split_factor"] > 0, "split_factor"]
            split_factor = float(nonzero_splits.max()) if not nonzero_splits.empty else 0.0
            first_source_timestamp = pd.Timestamp(observed.iloc[0]["source_timestamp_utc"])

        ohlc_valid, nonpositive_price, volume_valid = valid_ohlcv(values)
        window_complete = (
            observed_subbars == expected_subbars
            and ohlc_valid
            and volume_valid
            and not missing_times
        )
        if observed_subbars == 0:
            quality_status = "missing_window"
        elif observed_subbars < expected_subbars:
            quality_status = "incomplete_subbars"
        elif not ohlc_valid:
            quality_status = "invalid_ohlc"
        elif not volume_valid:
            quality_status = "invalid_volume"
        else:
            quality_status = "ok"

        output_rows.append({
            "trading_date": expected_row.trading_date,
            "session_open_ny": iso_ny(expected_row.session_open_utc),
            "session_close_ny": iso_ny(expected_row.session_close_utc),
            "session_open_utc": iso_utc(expected_row.session_open_utc),
            "session_close_utc": iso_utc(expected_row.session_close_utc),
            "bar_start_ny": iso_ny(expected_row.bar_start_utc),
            "bar_end_ny": iso_ny(expected_row.bar_end_utc),
            "bar_start_utc": iso_utc(expected_row.bar_start_utc),
            "bar_end_utc": iso_utc(expected_row.bar_end_utc),
            "window_mode": expected_row.window_mode,
            "window_role": expected_row.window_role,
            "premarket_minutes": expected_row.premarket_minutes,
            "regular_minutes": expected_row.regular_minutes,
            "window_number": expected_row.window_number,
            "duration_minutes": expected_row.duration_minutes,
            "short_window": bool(expected_row.short_window),
            "is_early_close": bool(expected_row.is_early_close),
            "open": values["open"],
            "high": values["high"],
            "low": values["low"],
            "close": values["close"],
            "volume": values["volume"],
            "source": fetch.source,
            "feed": fetch.feed,
            "adjustment": fetch.adjustment,
            "source_interval": fetch.interval,
            "bar_construction": "native_1h" if fetch.interval == "1h" else "aggregated_from_5m",
            "source_timestamp_semantics": fetch.timestamp_semantics,
            "first_source_timestamp_utc": iso_utc(first_source_timestamp),
            "expected_subbars": expected_subbars,
            "observed_subbars": observed_subbars,
            "missing_subbars": expected_subbars - observed_subbars,
            "missing_subbar_times_utc": "|".join(iso_utc(timestamp) for timestamp in missing_times),
            "window_complete": bool(window_complete),
            "ohlc_valid": bool(ohlc_valid),
            "volume_valid": bool(volume_valid),
            "nonpositive_price": bool(nonpositive_price),
            "split_event": bool(split_factor > 0),
            "split_factor": split_factor,
            "quality_status": quality_status,
        })

    windows = pd.DataFrame(output_rows)
    complete_windows = int(windows["window_complete"].sum())
    expected_windows = int(len(windows))
    missing_windows = expected_windows - complete_windows
    coverage_ratio = complete_windows / expected_windows if expected_windows else 0.0
    stats = {
        "raw_rows": int(len(fetch.bars)),
        "deduplicated_raw_rows": int(len(bars)),
        "duplicate_source_timestamp_rows": duplicate_timestamps,
        "source_rows_outside_regular_session": outside_rth,
        "source_rows_inside_session_outside_selected_windows": int(excluded_session_times),
        "misaligned_source_rows_inside_session": int(misaligned_in_session),
        "expected_windows": expected_windows,
        "complete_windows": complete_windows,
        "missing_or_invalid_windows": missing_windows,
        "coverage_ratio": coverage_ratio,
        "quality_pass": bool(
            expected_windows > 0
            and complete_windows == expected_windows
            and misaligned_in_session == 0
        ),
    }
    return windows, stats


def attempt_row(
    *,
    ticker: str,
    is_benchmark: bool,
    attempt_order: int,
    attempt_id: str,
    source: str,
    feed: str,
    interval: str,
    status: str,
    reason: str,
    cache_hit: bool | None,
    stats: dict[str, Any] | None,
) -> dict[str, Any]:
    stats = stats or {}
    return {
        "ticker": ticker,
        "is_benchmark": is_benchmark,
        "attempt_order": attempt_order,
        "attempt_id": attempt_id,
        "source": source,
        "feed": feed,
        "interval": interval,
        "attempt_status": status,
        "reason": reason,
        "cache_hit": cache_hit,
        "raw_rows": stats.get("raw_rows"),
        "duplicate_source_timestamp_rows": stats.get("duplicate_source_timestamp_rows"),
        "source_rows_outside_regular_session": stats.get("source_rows_outside_regular_session"),
        "source_rows_inside_session_outside_selected_windows": stats.get(
            "source_rows_inside_session_outside_selected_windows"
        ),
        "misaligned_source_rows_inside_session": stats.get("misaligned_source_rows_inside_session"),
        "expected_windows": stats.get("expected_windows"),
        "complete_windows": stats.get("complete_windows"),
        "missing_or_invalid_windows": stats.get("missing_or_invalid_windows"),
        "coverage_ratio": stats.get("coverage_ratio"),
        "selected": False,
    }


def candidate_key(candidate: Candidate) -> tuple[float, int, int, int]:
    return (
        float(candidate.stats["coverage_ratio"]),
        -int(candidate.stats["misaligned_source_rows_inside_session"]),
        -candidate.source_rank,
        -candidate.interval_rank,
    )


def download_ticker(
    ticker: str,
    *,
    is_benchmark: bool,
    adapters: list[BaseAdapter],
    expected: pd.DataFrame,
    start_date: date,
    end_date: date,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    intervals: list[str],
    secrets: Iterable[str | None],
) -> tuple[pd.DataFrame | None, list[dict[str, Any]], dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    attempt_order = 0
    selected: Candidate | None = None
    best_partial: Candidate | None = None

    for source_rank, adapter in enumerate(adapters):
        available, availability_reason = adapter.availability()
        if not available:
            attempt_order += 1
            attempt_id = f"{ticker}:{attempt_order}"
            attempts.append(attempt_row(
                ticker=ticker,
                is_benchmark=is_benchmark,
                attempt_order=attempt_order,
                attempt_id=attempt_id,
                source=adapter.source,
                feed=getattr(adapter, "feed", ""),
                interval="",
                status="skipped",
                reason=availability_reason,
                cache_hit=None,
                stats=None,
            ))
            continue

        for interval_rank, interval in enumerate(intervals):
            attempt_order += 1
            attempt_id = f"{ticker}:{attempt_order}"
            try:
                fetch = adapter.fetch(
                    ticker,
                    start_date,
                    end_date,
                    start_utc,
                    end_utc,
                    interval,
                )
                windows, stats = align_provider_bars(fetch, expected)
                status = "success" if stats["quality_pass"] else "partial"
                reason = (
                    f"complete {stats['complete_windows']}/{stats['expected_windows']} windows; "
                    f"misaligned in-session source rows={stats['misaligned_source_rows_inside_session']}"
                )
                attempts.append(attempt_row(
                    ticker=ticker,
                    is_benchmark=is_benchmark,
                    attempt_order=attempt_order,
                    attempt_id=attempt_id,
                    source=fetch.source,
                    feed=fetch.feed,
                    interval=interval,
                    status=status,
                    reason=reason,
                    cache_hit=fetch.cache_hit,
                    stats=stats,
                ))
                candidate = Candidate(
                    windows=windows,
                    fetch=fetch,
                    stats=stats,
                    attempt_id=attempt_id,
                    source_rank=source_rank,
                    interval_rank=interval_rank,
                )
                if best_partial is None or candidate_key(candidate) > candidate_key(best_partial):
                    best_partial = candidate
                if stats["quality_pass"]:
                    selected = candidate
                    break
            except SourceError as exc:
                attempts.append(attempt_row(
                    ticker=ticker,
                    is_benchmark=is_benchmark,
                    attempt_order=attempt_order,
                    attempt_id=attempt_id,
                    source=adapter.source,
                    feed=getattr(adapter, "feed", ""),
                    interval=interval,
                    status="failed",
                    reason=redact(f"{exc.kind}: {exc}", secrets),
                    cache_hit=None,
                    stats=None,
                ))
                # Invalid credentials/permission will not improve by changing timeframe.
                if exc.kind in {"auth", "permission"}:
                    break
            except Exception as exc:
                attempts.append(attempt_row(
                    ticker=ticker,
                    is_benchmark=is_benchmark,
                    attempt_order=attempt_order,
                    attempt_id=attempt_id,
                    source=adapter.source,
                    feed=getattr(adapter, "feed", ""),
                    interval=interval,
                    status="failed",
                    reason=redact(f"unexpected_error: {type(exc).__name__}: {exc}", secrets),
                    cache_hit=None,
                    stats=None,
                ))
        if selected is not None:
            break

    chosen = selected or best_partial
    if chosen is None:
        final = {
            "final_status": "failed",
            "final_source": "",
            "final_feed": "",
            "final_interval": "",
            "final_complete_windows": 0,
            "final_expected_windows": int(len(expected)),
            "final_missing_windows": int(len(expected)),
            "final_coverage_ratio": 0.0,
            "provider_notes": [],
        }
        for row in attempts:
            row.update(final)
        return None, attempts, final

    final_status = "success" if chosen.stats["quality_pass"] else "partial_success"
    for row in attempts:
        row["selected"] = row["attempt_id"] == chosen.attempt_id
    valid_windows = chosen.windows[chosen.windows["window_complete"]]
    final = {
        "final_status": final_status,
        "final_source": chosen.fetch.source,
        "final_feed": chosen.fetch.feed,
        "final_interval": chosen.fetch.interval,
        "final_complete_windows": chosen.stats["complete_windows"],
        "final_expected_windows": chosen.stats["expected_windows"],
        "final_missing_windows": chosen.stats["missing_or_invalid_windows"],
        "final_coverage_ratio": chosen.stats["coverage_ratio"],
        "final_first_complete_bar_start_utc": valid_windows["bar_start_utc"].min() if not valid_windows.empty else "",
        "final_last_complete_bar_end_utc": valid_windows["bar_end_utc"].max() if not valid_windows.empty else "",
        "provider_notes": chosen.fetch.provider_notes,
    }
    for row in attempts:
        row.update(final)

    windows = chosen.windows.copy()
    windows.insert(0, "ticker", ticker)
    windows.insert(1, "is_benchmark", is_benchmark)
    windows["final_ticker_status"] = final_status
    return windows, attempts, final


def compute_returns(prices: pd.DataFrame, *, jump_threshold: float) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ticker, group in prices.groupby("ticker", sort=False):
        frame = group.copy().sort_values(["trading_date", "bar_start_utc"]).reset_index(drop=True)
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame["close_prev"] = np.nan
        frame["prev_bar_end_utc"] = ""
        frame["ret_cc"] = np.nan
        frame["log_ret_cc"] = np.nan
        frame["ret_oc"] = np.nan
        frame["log_ret_oc"] = np.nan
        frame["ret_overnight_open_prev_close"] = np.nan
        frame["log_ret_overnight_open_prev_close"] = np.nan
        frame["cc_gap_reason"] = ""
        frame["abnormal_jump_flag"] = False
        frame["return_requires_review"] = False

        previous_trading_date: str | None = None
        previous_row: pd.Series | None = None
        for index, row in frame.iterrows():
            current_complete = bool(row["window_complete"])
            positive_oc = current_complete and pd.notna(row["open"]) and pd.notna(row["close"]) and row["open"] > 0 and row["close"] > 0
            if positive_oc:
                frame.at[index, "ret_oc"] = row["close"] / row["open"] - 1.0
                frame.at[index, "log_ret_oc"] = math.log(row["close"] / row["open"])

            new_session = previous_trading_date != row["trading_date"]
            if new_session:
                prior_session_close: float | None = None
                if (
                    previous_row is not None
                    and bool(previous_row["window_complete"])
                    and previous_row["bar_end_utc"] == previous_row["session_close_utc"]
                    and pd.notna(previous_row["close"])
                    and previous_row["close"] > 0
                ):
                    prior_session_close = float(previous_row["close"])
                starts_at_session_open = row.get("bar_start_utc") == row.get("session_open_utc")
                if positive_oc and prior_session_close is not None and starts_at_session_open:
                    frame.at[index, "ret_overnight_open_prev_close"] = row["open"] / prior_session_close - 1.0
                    frame.at[index, "log_ret_overnight_open_prev_close"] = math.log(row["open"] / prior_session_close)
                frame.at[index, "cc_gap_reason"] = "first_window_of_trading_day"
                previous_row = None

            if previous_row is not None:
                frame.at[index, "prev_bar_end_utc"] = previous_row["bar_end_utc"]
                contiguous = previous_row["bar_end_utc"] == row["bar_start_utc"]
                previous_complete = bool(previous_row["window_complete"])
                if not current_complete:
                    frame.at[index, "cc_gap_reason"] = "current_window_incomplete"
                elif not previous_complete:
                    frame.at[index, "cc_gap_reason"] = "previous_window_incomplete"
                elif not contiguous:
                    frame.at[index, "cc_gap_reason"] = "noncontiguous_previous_window"
                elif pd.isna(previous_row["close"]) or previous_row["close"] <= 0 or row["close"] <= 0:
                    frame.at[index, "cc_gap_reason"] = "nonpositive_or_missing_close"
                else:
                    previous_close = float(previous_row["close"])
                    frame.at[index, "close_prev"] = previous_close
                    frame.at[index, "ret_cc"] = row["close"] / previous_close - 1.0
                    frame.at[index, "log_ret_cc"] = math.log(row["close"] / previous_close)
                    frame.at[index, "cc_gap_reason"] = ""

            log_return = frame.at[index, "log_ret_cc"]
            abnormal = pd.notna(log_return) and abs(float(log_return)) > jump_threshold
            split_review = bool(row.get("split_event", False)) and str(row.get("adjustment", "")).startswith("raw")
            frame.at[index, "abnormal_jump_flag"] = bool(abnormal)
            frame.at[index, "return_requires_review"] = bool(abnormal or split_review)

            previous_trading_date = row["trading_date"]
            previous_row = frame.loc[index]

        frame["return_quality_status"] = np.select(
            [
                frame["return_requires_review"],
                frame["cc_gap_reason"].eq("first_window_of_trading_day"),
                frame["cc_gap_reason"].ne(""),
                frame["log_ret_cc"].notna(),
            ],
            ["review", "first_window_no_cc", "gap_no_cc", "ok"],
            default="unavailable",
        )
        frames.append(frame)
    returns = pd.concat(frames, ignore_index=True) if frames else prices.copy()
    return returns


def make_wide_returns(
    returns: pd.DataFrame,
    ticker_order: list[str],
    *,
    value_column: str = "log_ret_cc",
) -> pd.DataFrame:
    if value_column not in returns.columns:
        raise ValueError(f"missing return column for wide output: {value_column}")
    eligible = returns[
        returns["window_complete"].astype(bool)
        & returns["duration_minutes"].eq(60)
    ].copy()
    if eligible.empty:
        return pd.DataFrame(columns=["bar_end_utc", *ticker_order])
    series = eligible.set_index(["bar_end_utc", "ticker"])[value_column]
    if series.index.duplicated().any():
        raise ValueError("duplicate ticker/bar_end_utc rows would make wide returns ambiguous")
    wide = series.unstack("ticker").sort_index()
    for ticker in ticker_order:
        if ticker not in wide.columns:
            wide[ticker] = np.nan
    wide = wide[[ticker for ticker in ticker_order if ticker in wide.columns]]
    wide = wide.reset_index()
    return wide


def empty_output_columns() -> list[str]:
    return [
        "ticker", "is_benchmark", "trading_date", "session_open_ny", "session_close_ny",
        "session_open_utc", "session_close_utc", "bar_start_ny", "bar_end_ny",
        "bar_start_utc", "bar_end_utc", "window_mode", "window_role",
        "premarket_minutes", "regular_minutes", "window_number", "duration_minutes",
        "short_window", "is_early_close", "open", "high", "low", "close", "volume",
        "source", "feed", "adjustment", "source_interval", "bar_construction",
        "source_timestamp_semantics", "first_source_timestamp_utc", "expected_subbars",
        "observed_subbars", "missing_subbars", "missing_subbar_times_utc", "window_complete",
        "ohlc_valid", "volume_valid", "nonpositive_price", "split_event", "split_factor",
        "quality_status", "final_ticker_status",
    ]


def build_quality_report(
    *,
    args: argparse.Namespace,
    range_metadata: dict[str, Any],
    schedule: pd.DataFrame,
    expected: pd.DataFrame,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    attempts: pd.DataFrame,
    finals: dict[str, dict[str, Any]],
    credentials: dict[str, Any],
    ticker_order: list[str],
    benchmark: str | None,
) -> dict[str, Any]:
    try:
        import yfinance as yf
        yfinance_version = getattr(yf, "__version__", "unknown")
    except Exception:
        yfinance_version = "not_installed"
    try:
        import pandas_market_calendars
        calendar_version = getattr(pandas_market_calendars, "__version__", "unknown")
    except Exception:
        calendar_version = "not_installed"

    early = expected[expected["is_early_close"]].drop_duplicates("trading_date")
    ticker_summary: dict[str, Any] = {}
    for ticker in ticker_order:
        final = finals.get(ticker, {})
        subset = prices[prices["ticker"] == ticker] if not prices.empty else pd.DataFrame()
        return_subset = returns[returns["ticker"] == ticker] if not returns.empty else pd.DataFrame()
        ticker_summary[ticker] = {
            **final,
            "is_benchmark": bool(benchmark and ticker == benchmark),
            "missing_window_count": int((~subset["window_complete"].astype(bool)).sum()) if not subset.empty else int(len(expected)),
            "short_window_count": int(subset["short_window"].astype(bool).sum()) if not subset.empty else 0,
            "abnormal_jump_count": int(return_subset["abnormal_jump_flag"].astype(bool).sum()) if not return_subset.empty else 0,
            "split_event_window_count": int(subset["split_event"].astype(bool).sum()) if not subset.empty else 0,
        }

    selected_attempts = attempts[attempts["selected"].astype(bool)] if not attempts.empty else pd.DataFrame()
    source_counts = selected_attempts.groupby(["source", "feed"], dropna=False)["ticker"].nunique().to_dict() if not selected_attempts.empty else {}
    source_counts_json = {f"{source}/{feed}": int(count) for (source, feed), count in source_counts.items()}

    failed = [ticker for ticker, final in finals.items() if final.get("final_status") == "failed"]
    partial = [ticker for ticker, final in finals.items() if final.get("final_status") == "partial_success"]
    unresolved: list[str] = []
    if failed:
        unresolved.append(f"Failed tickers: {', '.join(failed)}")
    if partial:
        unresolved.append(f"Partial-coverage tickers: {', '.join(partial)}")
    if any(final.get("final_feed") == "iex" for final in finals.values()):
        unresolved.append("At least one ticker uses Alpaca IEX, a single-exchange feed rather than consolidated SIP.")
    if any(final.get("final_source") == "fmp" for final in finals.values()):
        unresolved.append("FMP does not explicitly document whether its date label is a bar start; selected FMP rows passed exact schedule alignment and remain flagged with that timestamp convention.")
    if not unresolved:
        unresolved.append("No unresolved ticker-level download or alignment issue was detected by the implemented checks.")

    availability = {
        "alpaca": {
            "configured": bool(credentials["alpaca"]["configured"]),
            "environment_variables": [
                credentials["alpaca"].get("key_variable"),
                credentials["alpaca"].get("secret_variable"),
            ],
        },
        "twelve_data": {
            "configured": bool(credentials["twelve_data"]["configured"]),
            "environment_variable": credentials["twelve_data"].get("key_variable"),
        },
        "fmp": {
            "configured": bool(credentials["fmp"]["configured"]),
            "environment_variable": credentials["fmp"].get("key_variable"),
        },
        "secrets_written_to_outputs": False,
    }

    cc_gap_violations = 0
    cc_overnight_violations = 0
    overnight_nonfirst_violations = 0
    if not returns.empty:
        prior = returns.groupby("ticker", sort=False).shift(1)
        cc_eligible = (
            returns["trading_date"].eq(prior["trading_date"])
            & returns["bar_start_utc"].eq(prior["bar_end_utc"])
            & returns["window_complete"].astype(bool)
            & prior["window_complete"].fillna(False).astype(bool)
        )
        cc_gap_violations = int((returns["ret_cc"].notna() & ~cc_eligible).sum())
        first_window = returns.groupby(["ticker", "trading_date"], sort=False).cumcount().eq(0)
        cc_overnight_violations = int((returns["ret_cc"].notna() & first_window).sum())
        overnight_nonfirst_violations = int((returns["ret_overnight_open_prev_close"].notna() & ~first_window).sum())

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now_iso(),
        "command": {
            "months": args.months,
            "start": args.start.isoformat() if args.start else None,
            "end": args.end.isoformat() if args.end else None,
            "window_mode": args.window_mode,
            "benchmark": benchmark,
            "tickers": ticker_order,
            "output_directory": str(args.out),
            "refresh": bool(args.refresh),
        },
        "time_range": range_metadata,
        "calendar": {
            "name": "NYSE",
            "timezone": NY_TZ,
            "package": "pandas_market_calendars",
            "package_version": calendar_version,
            "session_count": int(len(schedule)),
            "expected_window_count_per_ticker": int(len(expected)),
            "expected_full_60_minute_windows_per_ticker": int(expected["duration_minutes"].eq(60).sum()),
            "expected_short_windows_per_ticker": int(expected["duration_minutes"].lt(60).sum()),
            "early_close_dates": early["trading_date"].tolist(),
        },
        "timestamp_definition": {
            "provider_timestamp": "bar start / left boundary; FMP is accepted only after exact schedule validation",
            "bar_start_ny": "start of exchange-aligned window in America/New_York",
            "bar_end_ny": "end of exchange-aligned window; close becomes available only at this time",
            "window_mode": args.window_mode,
            "normal_window_examples": (
                ["10:00-11:00", "11:00-12:00", "15:00-16:00"]
                if args.window_mode == WINDOW_MODE_CLOCK_10_16
                else (
                    ["07:00-08:00", "08:00-09:00", "09:00-10:00"]
                    if args.window_mode == WINDOW_MODE_BRIDGE_07_10
                    else ["09:30-10:30", "10:30-11:30", "15:30-16:00"]
                )
            ),
            "bridge_definition": (
                "07:00-08:00 and 08:00-09:00 are premarket; 09:00-10:00 contains "
                "30 premarket minutes and 30 regular-session minutes. All are for lag use only."
                if args.window_mode == WINDOW_MODE_BRIDGE_07_10
                else None
            ),
            "wide_timestamp": "bar_end_utc",
        },
        "price_and_return_definitions": {
            "adjustment_policy": "Request raw/unadjusted where the provider exposes a switch; never splice sources within one ticker.",
            "ret_cc": "close_t / close_prev - 1, only for contiguous windows within the same trading day",
            "log_ret_cc": "ln(close_t / close_prev), same eligibility as ret_cc",
            "ret_oc": "close_t / open_t - 1 for the current complete window",
            "log_ret_oc": "ln(close_t / open_t)",
            "window_composition": (
                "premarket_minutes and regular_minutes explicitly identify how much of each "
                "window lies before and after the 09:30 New York open."
            ),
            "overnight": (
                "Stored only when the first included bar begins at the exchange session open; "
                "clock and bridge modes intentionally leave it unavailable."
            ),
            "wide": {
                "hourly_returns_wide.csv": "Legacy-compatible log_ret_cc for complete 60-minute windows.",
                "hourly_log_ret_cc_wide.csv": "Explicitly named copy of complete-window log_ret_cc.",
                "hourly_log_ret_oc_wide.csv": "Complete-window log_ret_oc; primary six-window return for clock_10_16.",
            },
            "return_units": "decimal (0.01 = 1 percent)",
            "jump_review_threshold_abs_log_return": args.jump_threshold,
        },
        "source_priority": ["Yahoo Finance/yfinance", "Alpaca SIP", "Alpaca IEX", "Twelve Data", "Financial Modeling Prep"],
        "source_differences": {
            "yahoo": (
                "No key required; yfinance is an open-source client. This run explicitly uses "
                f"raw OHLC (auto_adjust=False) and prepost={args.window_mode == WINDOW_MODE_BRIDGE_07_10}."
            ),
            "alpaca_sip": "Consolidated eligible trades from all US exchanges; historical access still depends on account entitlement and recency.",
            "alpaca_iex": "Single IEX exchange only, so volume and sometimes OHLC differ from consolidated feeds.",
            "twelve_data": "Explicit UTC output, prepost=false and adjust=none; request windows are segmented below the 5,000-point cap.",
            "fmp": "Explicit nonadjusted=true and extended=false; timestamps are exchange-local and their label convention is accepted only after exact schedule alignment.",
        },
        "source_documentation": SOURCE_DOCS,
        "package_versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "requests": requests.__version__,
            "yfinance": yfinance_version,
        },
        "credential_availability": availability,
        "results": {
            "requested_tickers_excluding_benchmark": len([ticker for ticker in ticker_order if ticker != benchmark]),
            "benchmark": benchmark,
            "successful_tickers": sum(final.get("final_status") == "success" for final in finals.values()),
            "partial_success_tickers": len(partial),
            "failed_tickers": len(failed),
            "failed_ticker_symbols": failed,
            "selected_source_counts": source_counts_json,
            "price_rows": int(len(prices)),
            "complete_price_rows": int(prices["window_complete"].astype(bool).sum()) if not prices.empty else 0,
            "return_rows": int(len(returns)),
            "valid_log_ret_cc_rows": int(returns["log_ret_cc"].notna().sum()) if not returns.empty else 0,
            "valid_log_ret_oc_rows": int(returns["log_ret_oc"].notna().sum()) if not returns.empty else 0,
            "abnormal_jump_rows": int(returns["abnormal_jump_flag"].astype(bool).sum()) if not returns.empty else 0,
        },
        "ticker_summary": ticker_summary,
        "quality_checks": {
            "duplicate_processed_ticker_bar_start_rows": int(prices.duplicated(["ticker", "bar_start_utc"]).sum()) if not prices.empty else 0,
            "nonpositive_price_rows": int(prices["nonpositive_price"].astype(bool).sum()) if not prices.empty else 0,
            "invalid_ohlc_rows_with_observations": int((prices["observed_subbars"].gt(0) & ~prices["ohlc_valid"].astype(bool)).sum()) if not prices.empty else 0,
            "incomplete_window_rows": int((~prices["window_complete"].astype(bool)).sum()) if not prices.empty else 0,
            "cc_returns_crossing_missing_or_noncontiguous_window": cc_gap_violations,
            "cc_returns_crossing_overnight": cc_overnight_violations,
            "overnight_returns_on_nonfirst_window": overnight_nonfirst_violations,
            "short_windows_excluded_from_wide": True,
            "forward_fill_interpolation_or_zero_fill_used": False,
        },
        "unresolved_issues": unresolved,
        "reproduction_command": (
            "python download_hourly_returns.py "
            + (f"--start {args.start.isoformat()} " if args.start else f"--months {args.months} ")
            + (f"--end {args.end.isoformat()} " if args.end else "")
            + f"--window-mode {args.window_mode} --out {args.out}"
        ),
    }


def parse_ticker_list(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_TICKERS)
    tickers = [part.strip().upper() for part in value.split(",") if part.strip()]
    if not tickers:
        raise ValueError("--tickers did not contain any ticker symbols")
    return list(dict.fromkeys(tickers))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", type=int, default=1, help="Calendar months to look back when --start is omitted (default: 1)")
    parser.add_argument("--start", type=parse_iso_date, help="Inclusive requested start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=parse_iso_date, help="Inclusive requested end date (YYYY-MM-DD); capped at the latest complete session")
    parser.add_argument("--out", type=Path, default=Path("data/hourly"), help="Output directory")
    parser.add_argument(
        "--window-mode",
        choices=WINDOW_MODES,
        default=WINDOW_MODE_SESSION_OPEN,
        help=(
            "session_open keeps 09:30-anchored exchange windows; clock_10_16 keeps only "
            "10:00-11:00 through 15:00-16:00 New York; bridge_07_10 downloads "
            "07:00-08:00, 08:00-09:00 and 09:00-10:00 as lag-only extended-hours windows"
        ),
    )
    parser.add_argument("--benchmark", type=str, help="Optional benchmark ticker, e.g. QQQ")
    parser.add_argument("--tickers", type=str, help="Comma-separated override, useful for validation runs")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout seconds")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for transient errors (default: 3)")
    parser.add_argument("--jump-threshold", type=float, default=0.20, help="Absolute log-return threshold for review only")
    parser.add_argument("--refresh", action="store_true", help="Ignore raw caches and download again")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.months <= 0:
        raise ValueError("--months must be positive")
    if args.max_retries < 0 or args.max_retries > 3:
        raise ValueError("--max-retries must be between 0 and 3")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if load_dotenv is None or mcal is None:
        raise RuntimeError("Install requirements.txt before running this program")

    load_dotenv(Path(".env"), override=False)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "raw").mkdir(parents=True, exist_ok=True)

    ticker_order = parse_ticker_list(args.tickers)
    benchmark = args.benchmark.strip().upper() if args.benchmark else None
    if benchmark and benchmark not in ticker_order:
        ticker_order.append(benchmark)
    benchmark_set = {benchmark} if benchmark else set()

    schedule, range_metadata = resolve_schedule(
        months=args.months,
        requested_start=args.start,
        requested_end=args.end,
    )
    expected = build_expected_windows(schedule, window_mode=args.window_mode)
    five_minute_only = args.window_mode in {WINDOW_MODE_CLOCK_10_16, WINDOW_MODE_BRIDGE_07_10}
    include_prepost = args.window_mode == WINDOW_MODE_BRIDGE_07_10
    intervals = ["5m"] if five_minute_only else ["1h", "5m"]
    actual_start = schedule.index[0].date()
    actual_end = schedule.index[-1].date()
    start_utc = min(
        pd.Timestamp(schedule.iloc[0]["market_open"]).tz_convert("UTC"),
        pd.Timestamp(expected["bar_start_utc"].min()).tz_convert("UTC"),
    )
    end_utc = max(
        pd.Timestamp(schedule.iloc[-1]["market_close"]).tz_convert("UTC"),
        pd.Timestamp(expected["bar_end_utc"].max()).tz_convert("UTC"),
    )
    print(
        f"Actual NYSE range: {actual_start.isoformat()} through {actual_end.isoformat()} "
        f"({len(schedule)} sessions, {len(expected)} exchange-aligned windows per ticker, "
        f"mode={args.window_mode})"
    )
    print(f"UTC availability range: {iso_utc(start_utc)} through {iso_utc(end_utc)}")

    credentials = configured_credentials()
    secrets = [
        credentials["alpaca"].get("key"),
        credentials["alpaca"].get("secret"),
        credentials["twelve_data"].get("key"),
        credentials["fmp"].get("key"),
    ]
    adapters: list[BaseAdapter] = [
        YahooAdapter(
            args.out,
            timeout=args.timeout,
            max_retries=args.max_retries,
            refresh=args.refresh,
            include_prepost=include_prepost,
        ),
        AlpacaAdapter(args.out, timeout=args.timeout, max_retries=args.max_retries, refresh=args.refresh, feed="sip", credentials=credentials["alpaca"]),
        AlpacaAdapter(args.out, timeout=args.timeout, max_retries=args.max_retries, refresh=args.refresh, feed="iex", credentials=credentials["alpaca"]),
        TwelveDataAdapter(args.out, timeout=args.timeout, max_retries=args.max_retries, refresh=args.refresh, credentials=credentials["twelve_data"]),
        FMPAdapter(args.out, timeout=args.timeout, max_retries=args.max_retries, refresh=args.refresh, credentials=credentials["fmp"]),
    ]

    price_frames: list[pd.DataFrame] = []
    attempt_rows: list[dict[str, Any]] = []
    finals: dict[str, dict[str, Any]] = {}
    checkpoint_dir = args.out / "raw" / "_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for number, ticker in enumerate(ticker_order, start=1):
        print(f"[{number}/{len(ticker_order)}] {ticker}: trying providers in priority order")
        windows, attempts, final = download_ticker(
            ticker,
            is_benchmark=ticker in benchmark_set,
            adapters=adapters,
            expected=expected,
            start_date=actual_start,
            end_date=actual_end,
            start_utc=start_utc,
            end_utc=end_utc,
            intervals=intervals,
            secrets=secrets,
        )
        attempt_rows.extend(attempts)
        finals[ticker] = final
        if windows is not None:
            price_frames.append(windows)
            write_csv_atomic(windows, checkpoint_dir / f"{ticker}_{actual_start}_{actual_end}_prices.csv")
        write_json(checkpoint_dir / f"{ticker}_{actual_start}_{actual_end}_status.json", {
            "ticker": ticker,
            "final": final,
            "attempts": attempts,
        })
        print(
            f"    {final['final_status']}: {final['final_source']} {final['final_feed']} "
            f"{final['final_interval']} complete={final['final_complete_windows']}/{final['final_expected_windows']}"
        )

    if price_frames:
        prices = pd.concat(price_frames, ignore_index=True)
        prices["ticker_order"] = pd.Categorical(prices["ticker"], categories=ticker_order, ordered=True)
        prices = prices.sort_values(["ticker_order", "bar_start_utc"]).drop(columns="ticker_order").reset_index(drop=True)
    else:
        prices = pd.DataFrame(columns=empty_output_columns())

    attempts = pd.DataFrame(attempt_rows)
    returns = compute_returns(prices, jump_threshold=args.jump_threshold) if not prices.empty else prices.copy()
    empty_wide = pd.DataFrame(columns=["bar_end_utc", *ticker_order])
    wide_cc = (
        make_wide_returns(returns, ticker_order, value_column="log_ret_cc")
        if not returns.empty
        else empty_wide.copy()
    )
    wide_oc = (
        make_wide_returns(returns, ticker_order, value_column="log_ret_oc")
        if not returns.empty
        else empty_wide.copy()
    )

    price_path = args.out / "hourly_prices_long.csv"
    return_path = args.out / "hourly_returns_long.csv"
    wide_path = args.out / "hourly_returns_wide.csv"
    wide_cc_path = args.out / "hourly_log_ret_cc_wide.csv"
    wide_oc_path = args.out / "hourly_log_ret_oc_wide.csv"
    status_path = args.out / "download_status.csv"
    quality_path = args.out / "quality_report.json"
    write_csv_atomic(prices, price_path)
    write_csv_atomic(returns, return_path)
    write_csv_atomic(wide_cc, wide_path)
    write_csv_atomic(wide_cc, wide_cc_path)
    write_csv_atomic(wide_oc, wide_oc_path)
    write_csv_atomic(attempts, status_path)
    report = build_quality_report(
        args=args,
        range_metadata=range_metadata,
        schedule=schedule,
        expected=expected,
        prices=prices,
        returns=returns,
        attempts=attempts,
        finals=finals,
        credentials=credentials,
        ticker_order=ticker_order,
        benchmark=benchmark,
    )
    write_json(quality_path, report)

    success = sum(final.get("final_status") == "success" for final in finals.values())
    partial = sum(final.get("final_status") == "partial_success" for final in finals.values())
    failed = [ticker for ticker, final in finals.items() if final.get("final_status") == "failed"]
    print(f"Completed: success={success}, partial={partial}, failed={len(failed)}")
    if failed:
        print("Failed tickers: " + ", ".join(failed))
    print(f"Wrote outputs to {args.out}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
