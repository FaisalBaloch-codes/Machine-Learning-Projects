"""Timestamp alignment, post-news windows and market-reaction features.

All prices come from cached Alpaca 1-minute bars. Nothing is interpolated or
invented: a window price is the close of the last *actual* 1-minute bar at or
before the window end (the last traded price); if the window runs past the
session close the value is left empty (NaN).

Alignment (bar timestamps are bar START times, UTC)
---------------------------------------------------
* same_minute       news inside the regular session and a bar exists for the
                    news minute (e.g. news 14:32:17 -> bar 14:32:00-14:32:59)
* next_candle       news inside the session, no trade in that minute; first bar
                    after it (within MAX_ALIGNMENT_GAP_MINUTES)
* next_market_open  news pre-market / after-hours / weekend / holiday; first bar
                    of the next regular session
* unavailable       no usable bar (reason in `market_unavailable_reason`)

Feature timing - see README "Features vs target":
* PRE-NEWS (safe as model inputs): volatility, baseline_volume_per_min,
  pre_news_close, pre_news_return_15m, baseline_return_count, news_session_position
* POST-NEWS (outcomes, used ONLY for labelling): the T0 candle OHLCV, all
  return_*, high/low moves, volume_ratio, abnormal_return_*, gap_return
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from collectors.alpaca_market import AlpacaMarketData, Bars, MarketDataCache, required_sessions
from collectors.alpaca_reference import TradingCalendar
from collectors.api_utils import STOP_EVENT
from config.settings import Settings
from storage.checkpoint import Checkpoint
from storage.csv_writer import IncrementalCSVWriter, iter_csv_chunks, read_column_set
from storage.timeutils import epoch_seconds

log = logging.getLogger(__name__)

NY = "America/New_York"
SESSION_MINUTES = 390


def feature_columns(settings: Settings) -> list[str]:
    cols = [
        "event_id", "ticker", "news_published_timestamp_utc", "market_alignment", "market_unavailable_reason",
        "market_session_date", "news_session_position", "market_timestamp_utc", "alignment_lag_seconds",
        "open", "high", "low", "close", "volume",
        "pre_news_close", "pre_news_return_15m", "volatility", "baseline_return_count", "baseline_volume_per_min",
    ]
    cols += [f"return_{h}m" for h in settings.post_windows]
    cols += ["high_move_5m", "low_move_5m", "bars_in_5m", "event_volume", "expected_volume", "volume_ratio",
             "gap_return", "gap_abnormal"]
    cols += [f"abnormal_return_{h}m" for h in settings.post_windows]
    return cols


def _session_log_returns(b: Bars, upto: int | None = None) -> np.ndarray:
    """Per-minute-scaled log returns between consecutive bars of one session."""
    t, c = b.t, b.c
    if upto is not None:
        mask = t < upto
        t, c = t[mask], c[mask]
    if len(c) < 2:
        return np.array([])
    dt = np.diff(t) / 60.0
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(np.log(c)) / np.sqrt(np.maximum(dt, 1.0))
    return r[np.isfinite(r)]


def _asof_close(b: Bars, ts: int) -> float:
    k = int(np.searchsorted(b.t, ts, side="right")) - 1
    return float(b.c[k]) if k >= 0 else math.nan


def _position(news_ts: int, pos: str, i: int, cal: TradingCalendar) -> str:
    if pos == "in_session":
        return "regular_hours"
    et = pd.Timestamp(news_ts, unit="s", tz="UTC").tz_convert(NY)
    et_date = et.strftime("%Y-%m-%d")
    if et_date == cal.session_date(i):
        return "pre_market"
    j = cal.date_index(et_date)
    if j is not None and news_ts >= cal.close_s[j]:
        return "after_hours"
    return "non_trading_day"


def compute_event_features(ticker: str, news_ts: int, cal: TradingCalendar, cache: MarketDataCache,
                           s: Settings) -> dict | None:
    """Features for one ticker-event. Returns None if required bars are not downloaded yet."""
    out: dict = {"market_alignment": "unavailable", "market_unavailable_reason": ""}
    i, pos = cal.locate(news_ts)
    if i < 0:
        out["market_unavailable_reason"] = "outside_trading_calendar"
        return out
    out["market_session_date"] = cal.session_date(i)
    out["news_session_position"] = _position(news_ts, pos, i, cal)

    bars = cache.load(ticker, cal.session_date(i))
    prior_idx = cal.previous_sessions(i, s.baseline_lookback_sessions)
    prior = [cache.load(ticker, cal.session_date(j)) for j in prior_idx]
    if bars is None or any(p is None for p in prior):
        return None  # pending download
    open_i, close_i = int(cal.open_s[i]), int(cal.close_s[i])
    last_bar_start = close_i - 60

    # ------------------------------------------------------------ alignment
    gap = s.max_alignment_gap_minutes * 60
    minute = news_ts - news_ts % 60
    if pos == "in_session":
        k = int(np.searchsorted(bars.t, minute, side="left"))
        if k < len(bars) and bars.t[k] == minute:
            alignment = "same_minute"
        elif k < len(bars) and bars.t[k] - minute <= gap and bars.t[k] <= last_bar_start:
            alignment = "next_candle"
        else:
            alignment = None
            out["market_unavailable_reason"] = f"no_trade_within_{s.max_alignment_gap_minutes}m_of_news"
    else:
        k = int(np.searchsorted(bars.t, open_i, side="left"))
        if k < len(bars) and bars.t[k] - open_i <= gap and bars.t[k] <= last_bar_start:
            alignment = "next_market_open"
        else:
            alignment = None
            out["market_unavailable_reason"] = ("no_bars_for_session" if len(bars) == 0
                                                else f"no_trade_within_{s.max_alignment_gap_minutes}m_of_open")

    # ------------------------------------------------ pre-news (feature-safe)
    same_session_pre = Bars.__new__(Bars)
    mask = bars.t < minute
    for name in Bars.__slots__:
        setattr(same_session_pre, name, getattr(bars, name)[mask])
    if len(same_session_pre):
        pre_close = float(same_session_pre.c[-1])
    else:
        pre_close = next((float(p.c[-1]) for p in reversed(prior) if len(p)), math.nan)
    out["pre_news_close"] = pre_close
    if len(same_session_pre) and minute - 15 * 60 >= open_i:
        ref = _asof_close(same_session_pre, minute - 15 * 60)
        out["pre_news_return_15m"] = pre_close / ref - 1 if ref and not math.isnan(ref) else math.nan
    returns = [_session_log_returns(p) for p in prior] + [_session_log_returns(bars, upto=minute)]
    all_r = np.concatenate(returns) if returns else np.array([])
    out["baseline_return_count"] = int(len(all_r))
    out["volatility"] = float(np.std(all_r, ddof=1)) if len(all_r) >= 2 else math.nan
    prior_minutes = [(cal.close_s[j] - cal.open_s[j]) / 60 for j in prior_idx]
    vols = [p.v.sum() / m for p, m in zip(prior, prior_minutes) if m > 0]
    out["baseline_volume_per_min"] = float(np.mean(vols)) if vols else math.nan

    if alignment is None:
        return out

    # --------------------------------------------- T0 candle (post-news data)
    t0 = int(bars.t[k])
    p0 = float(bars.c[k])
    out.update({
        "market_alignment": alignment,
        "market_timestamp_utc": pd.Timestamp(t0, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "alignment_lag_seconds": t0 - news_ts,
        "open": float(bars.o[k]), "high": float(bars.h[k]), "low": float(bars.l[k]),
        "close": p0, "volume": float(bars.v[k]),
    })

    # ------------------------------------------------- post-news windows
    sigma = out["volatility"]
    for h in s.post_windows:
        target = t0 + h * 60
        if target > last_bar_start:
            out[f"return_{h}m"] = math.nan
            out[f"abnormal_return_{h}m"] = math.nan
            continue
        r = _asof_close(bars, target) / p0 - 1
        out[f"return_{h}m"] = r
        out[f"abnormal_return_{h}m"] = r / (sigma * math.sqrt(h)) if sigma and sigma > 0 else math.nan

    if t0 + 300 <= last_bar_start:
        w = (bars.t > t0) & (bars.t <= t0 + 300)
        out["bars_in_5m"] = int(w.sum())
        out["high_move_5m"] = float(bars.h[w].max() / p0 - 1) if w.any() else 0.0
        out["low_move_5m"] = float(bars.l[w].min() / p0 - 1) if w.any() else 0.0

    # Volume: event window vs. the same clock-time window in prior sessions.
    end = min(t0 + s.volume_window_minutes * 60, close_i)
    ev_vol = float(bars.v[(bars.t >= t0) & (bars.t < end)].sum())
    offset, length = t0 - open_i, end - t0
    expected = []
    for p, j in zip(prior, prior_idx):
        a = int(cal.open_s[j]) + offset
        if a + length <= cal.close_s[j]:
            expected.append(float(p.v[(p.t >= a) & (p.t < a + length)].sum()))
    out["event_volume"] = ev_vol
    exp_v = float(np.mean(expected)) if expected else math.nan
    out["expected_volume"] = exp_v
    out["volume_ratio"] = ev_vol / exp_v if exp_v and exp_v > 0 else math.nan

    if alignment == "next_market_open" and pre_close and not math.isnan(pre_close):
        out["gap_return"] = float(bars.o[k]) / pre_close - 1
        out["gap_abnormal"] = (out["gap_return"] / (sigma * math.sqrt(SESSION_MINUTES))
                               if sigma and sigma > 0 else math.nan)
    return out


# -------------------------------------------------------------------- stage
def run_market_stage(settings: Settings, cal: TradingCalendar, market: AlpacaMarketData) -> dict:
    cp = Checkpoint(settings.checkpoint_dir / "market_features.json")
    sig = settings.section_hash("post_windows", "baseline_lookback_sessions", "max_alignment_gap_minutes",
                                "volume_window_minutes")
    if cp.get("signature") not in (None, sig):
        log.warning("Market-feature settings changed; existing rows keep old values. "
                    "Run `python main.py --market --force` to recompute (cached bars are reused, no API calls).")
    cp.set("signature", cp.get("signature") or sig)
    cols = feature_columns(settings)
    writer = IncrementalCSVWriter(settings.market_features_path, cols, cp.get("committed_size"))
    done = read_column_set(settings.market_features_path, "event_id")

    parts = [c for c in iter_csv_chunks(settings.ticker_events_path, 200_000,
                                        usecols=["event_id", "ticker", "news_published_timestamp_utc"])]
    if not parts:
        log.warning("No ticker events found - run --ner first.")
        return {"status": "no_events"}
    ev = pd.concat(parts, ignore_index=True).drop_duplicates("event_id")
    ev = ev[~ev["event_id"].isin(done)].copy()
    ev["ts"] = epoch_seconds(ev["news_published_timestamp_utc"])
    log.info("Market stage: %d events already have features, %d to process", len(done), len(ev))

    pairs = set()
    for tkr, ts in zip(ev["ticker"], ev["ts"]):
        for d in required_sessions(cal, ts, settings.baseline_lookback_sessions):
            pairs.add((tkr, d))
    fetch_stats = market.fetch_pairs(pairs)

    ev = ev.sort_values(["ticker", "ts"])
    batch, written, pending = [], 0, 0
    status = "completed"
    for n, (eid, tkr, ts, ts_str) in enumerate(zip(ev["event_id"], ev["ticker"], ev["ts"],
                                                   ev["news_published_timestamp_utc"]), 1):
        if STOP_EVENT.is_set():
            status = "interrupted"
            break
        feats = compute_event_features(tkr, int(ts), cal, market.cache, settings)
        if feats is None:
            pending += 1
            continue
        feats.update({"event_id": eid, "ticker": tkr, "news_published_timestamp_utc": ts_str})
        batch.append(feats)
        if len(batch) >= 2000:
            cp.set("committed_size", writer.write_rows(batch))
            written += len(batch)
            batch = []
            log.info("Market features: %d/%d events processed", n, len(ev))
    if batch:
        cp.set("committed_size", writer.write_rows(batch))
        written += len(batch)
    if pending:
        log.warning("%d events still waiting for market data (failed/unavailable downloads). "
                    "Rerun `python main.py --market` to retry.", pending)
    return {"status": status, "events_with_features_this_run": written, "events_pending_market_data": pending,
            **fetch_stats}
