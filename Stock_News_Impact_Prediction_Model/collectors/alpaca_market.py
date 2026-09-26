"""Alpaca 1-minute OHLCV bars (GET /v2/stocks/bars, timeframe=1Min ONLY).

Request strategy
----------------
Events are first reduced to the set of (ticker, trading-session date) pairs
they need (event session + BASELINE_LOOKBACK_SESSIONS prior sessions). Pairs are
grouped by date and requested as multi-symbol batches (ALPACA_SYMBOLS_PER_REQUEST
symbols per call, paginated), then split into one cache file per ticker/day:

    data/raw/market/AAPL_2024-09-24.csv

A cache file is the checkpoint for that pair: it is written atomically and is
never requested again. A pair that genuinely has no bars gets a header-only file
(so it is not re-requested either); a pair whose request *failed* gets no file and
is retried on the next run. OHLCV values are stored exactly as returned.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from collectors.alpaca_news import make_alpaca_client
from collectors.alpaca_reference import TradingCalendar
from collectors.api_utils import ApiError, FailedRequestLog, RateLimiter, StopRequested, STOP_EVENT
from config.settings import Settings
from storage.checkpoint import Checkpoint
from storage.csv_writer import atomic_write_df
from storage.timeutils import epoch_seconds

log = logging.getLogger(__name__)

BAR_COLUMNS = ["timestamp_utc", "open", "high", "low", "close", "volume", "trade_count", "vwap", "feed"]
TIMEFRAME = "1Min"  # the only interval this project uses


class Bars:
    """Numpy view of one session's 1-minute bars (t = bar start, UTC epoch seconds)."""

    __slots__ = ("t", "o", "h", "l", "c", "v")

    def __init__(self, df: pd.DataFrame):
        if len(df):
            self.t = epoch_seconds(df["timestamp_utc"])
        else:
            self.t = np.array([], dtype=np.int64)
        self.o = df["open"].to_numpy(dtype=float) if len(df) else np.array([])
        self.h = df["high"].to_numpy(dtype=float) if len(df) else np.array([])
        self.l = df["low"].to_numpy(dtype=float) if len(df) else np.array([])
        self.c = df["close"].to_numpy(dtype=float) if len(df) else np.array([])
        self.v = df["volume"].to_numpy(dtype=float) if len(df) else np.array([])
        order = np.argsort(self.t, kind="stable")
        for name in self.__slots__:
            setattr(self, name, getattr(self, name)[order])

    def __len__(self) -> int:
        return len(self.t)


class MarketDataCache:
    def __init__(self, market_dir: Path, max_items: int = 1024):
        self.dir = Path(market_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lru: OrderedDict[tuple[str, str], Bars | None] = OrderedDict()
        self.max_items = max_items

    def path(self, ticker: str, day: str) -> Path:
        return self.dir / f"{ticker}_{day}.csv"

    def has(self, ticker: str, day: str) -> bool:
        return self.path(ticker, day).exists()

    def load(self, ticker: str, day: str) -> Bars | None:
        """Bars for one session, or None if the pair was never downloaded."""
        key = (ticker, day)
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]
        p = self.path(ticker, day)
        bars = Bars(pd.read_csv(p)) if p.exists() else None
        self._lru[key] = bars
        if len(self._lru) > self.max_items:
            self._lru.popitem(last=False)
        return bars

    def save(self, ticker: str, day: str, rows: list[dict]) -> None:
        atomic_write_df(pd.DataFrame(rows, columns=BAR_COLUMNS), self.path(ticker, day))


class AlpacaMarketData:
    def __init__(self, settings: Settings, calendar: TradingCalendar, failed_log: FailedRequestLog,
                 limiter: RateLimiter | None = None):
        self.s = settings
        self.calendar = calendar
        self.cache = MarketDataCache(settings.market_dir)
        self.client = make_alpaca_client(settings, failed_log, limiter)
        self.cp = Checkpoint(settings.checkpoint_dir / "alpaca_market.json")
        self.feed = self.cp.get("feed_override") or settings.alpaca_data_feed
        self.feed_lock = threading.Lock()
        self.url = f"{settings.alpaca_data_url}/v2/stocks/bars"

    # ---------------------------------------------------------------- single
    def get_event_window(self, ticker: str, news_ts_utc: int) -> dict[str, Bars | None]:
        """Convenience API: bars needed for one (ticker, news timestamp) pair.

        Returns {session_date: Bars} for the aligned session and the baseline
        sessions, downloading any missing day into the cache first.
        """
        days = required_sessions(self.calendar, news_ts_utc, self.s.baseline_lookback_sessions)
        self.fetch_pairs({(ticker, d) for d in days})
        return {d: self.cache.load(ticker, d) for d in days}

    # ----------------------------------------------------------------- batch
    def fetch_pairs(self, pairs: set[tuple[str, str]]) -> dict:
        todo = sorted(p for p in pairs if not self.cache.has(*p))
        if not todo:
            return {"requested_pairs": len(pairs), "downloaded_pairs": 0, "failed_pairs": 0}
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=16)  # SIP data is delayed 15 min on free plans
        by_day: dict[str, list[str]] = defaultdict(list)
        skipped_future = 0
        for ticker, day in todo:
            i = self.calendar.date_index(day)
            if i is None or self.calendar.open_s[i] >= cutoff.timestamp():
                skipped_future += 1
                continue
            by_day[day].append(ticker)
        batches = []
        n = self.s.alpaca_symbols_per_request
        for day, tickers in sorted(by_day.items()):
            for k in range(0, len(tickers), n):
                batches.append((day, tickers[k:k + n]))
        log.info("Market data: %d ticker-days cached, %d to download in %d batched requests "
                 "(%d not yet available)", len(pairs) - len(todo), len(todo) - skipped_future, len(batches),
                 skipped_future)

        done_pairs = failed_pairs = 0
        with ThreadPoolExecutor(max_workers=self.s.max_workers, thread_name_prefix="alpaca-bars") as pool:
            futures = {pool.submit(self._fetch_batch, day, syms, cutoff): (day, syms) for day, syms in batches}
            try:
                for k, fut in enumerate(as_completed(futures), 1):
                    day, syms = futures[fut]
                    try:
                        done_pairs += fut.result()
                    except StopRequested:
                        pass
                    except (ApiError, OSError) as exc:
                        failed_pairs += len(syms)
                        log.error("Bars batch %s (%d symbols) failed: %s - will retry next run", day, len(syms), exc)
                    if k % 25 == 0 or k == len(batches):
                        log.info("Market data progress: %d/%d batches, %d ticker-days saved", k, len(batches), done_pairs)
                        self.cp.update({"last_progress": f"{k}/{len(batches)} batches"})
            except KeyboardInterrupt:
                STOP_EVENT.set()
                pool.shutdown(wait=True, cancel_futures=True)
                raise
            if STOP_EVENT.is_set():
                pool.shutdown(wait=True, cancel_futures=True)
        self.cp.update({"downloaded_pairs_total": self.cp.get("downloaded_pairs_total", 0) + done_pairs,
                        "failed_pairs_last_run": failed_pairs})
        return {"requested_pairs": len(pairs), "downloaded_pairs": done_pairs, "failed_pairs": failed_pairs,
                "not_yet_available": skipped_future}

    def _fetch_batch(self, day: str, symbols: list[str], cutoff: datetime) -> int:
        i = self.calendar.date_index(day)
        start = datetime.fromtimestamp(int(self.calendar.open_s[i]), tz=timezone.utc)
        end = datetime.fromtimestamp(int(self.calendar.close_s[i]), tz=timezone.utc)
        end = min(end, cutoff)
        bars: dict[str, list[dict]] = {s: [] for s in symbols}
        token = None
        while True:
            if STOP_EVENT.is_set():
                raise StopRequested()
            params = {"symbols": ",".join(symbols), "timeframe": TIMEFRAME,
                      "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "limit": 10000, "adjustment": "raw", "feed": self.feed, "sort": "asc"}
            if token:
                params["page_token"] = token
            try:
                payload = self.client.get_json(self.url, params, context=f"bars {day} x{len(symbols)}",
                                               record_failure=self.feed == "iex")
            except ApiError as exc:
                if exc.status == 403 and self.feed == "sip":
                    with self.feed_lock:
                        if self.feed == "sip":
                            log.warning("SIP feed not permitted for this Alpaca subscription - falling back to IEX "
                                        "for all remaining requests (recorded in the 'feed' column).")
                            self.feed = "iex"
                            self.cp.set("feed_override", "iex")
                    continue
                raise
            for sym, items in (payload.get("bars") or {}).items():
                for b in items:
                    bars.setdefault(sym, []).append({
                        "timestamp_utc": b.get("t"), "open": b.get("o"), "high": b.get("h"), "low": b.get("l"),
                        "close": b.get("c"), "volume": b.get("v"), "trade_count": b.get("n"),
                        "vwap": b.get("vw"), "feed": self.feed,
                    })
            token = payload.get("next_page_token")
            if not token:
                break
        for sym in symbols:  # write only after the whole batch succeeded
            self.cache.save(sym, day, bars.get(sym, []))
        return len(symbols)


def required_sessions(calendar: TradingCalendar, news_ts_utc: int, lookback: int) -> list[str]:
    """Aligned session + `lookback` prior sessions for a news timestamp."""
    i, _pos = calendar.locate(int(news_ts_utc))
    if i < 0:
        return []
    return [calendar.session_date(j) for j in calendar.previous_sessions(i, lookback) + [i]]
