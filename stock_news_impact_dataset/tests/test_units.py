"""Focused unit tests for alignment, labelling, NER and de-duplication rules."""
import math
from datetime import date, datetime, timezone

import pandas as pd

from collectors.alpaca_market import BAR_COLUMNS, MarketDataCache
from collectors.alpaca_reference import TradingCalendar
from config.settings import load_settings
from labeling.impact_labeler import label_event
from processing.cleaner import normalize_url
from processing.market_features import compute_event_features
from tests.fake_apis import ASSETS, trading_days


def _ts(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def _calendar():
    days = trading_days(date(2024, 12, 16), date(2025, 1, 20))
    return TradingCalendar(pd.DataFrame({"date": [str(d) for d in days], "open": "09:30", "close": "16:00"}))


def _write_session(cache: MarketDataCache, ticker: str, day: str, minutes: list[int], open_utc: str, price=100.0):
    base = _ts(open_utc)
    rows = [{"timestamp_utc": pd.Timestamp(base + m * 60, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
             "open": price, "high": price * 1.001, "low": price * 0.999, "close": price * (1 + 0.0004 * ((m % 3) - 1)),
             "volume": 1000, "trade_count": 5, "vwap": price, "feed": "sip"} for m in minutes]
    pd.DataFrame(rows, columns=BAR_COLUMNS).to_csv(cache.path(ticker, day), index=False)


def test_same_minute_next_candle_and_unavailable(env, tmp_path):
    s = load_settings(env)
    cal = _calendar()
    cache = MarketDataCache(tmp_path / "m")
    for d in ("2025-01-02", "2025-01-03", "2025-01-06"):
        _write_session(cache, "XYZ", d, list(range(390)), f"{d}T14:30:00Z")
    _write_session(cache, "XYZ", "2025-01-07", [m for m in range(390) if m not in (122, 123)], "2025-01-07T14:30:00Z")

    f = compute_event_features("XYZ", _ts("2025-01-07T14:32:17Z"), cal, cache, s)
    assert f["market_alignment"] == "same_minute" and f["market_timestamp_utc"] == "2025-01-07T14:32:00Z"
    assert f["alignment_lag_seconds"] == -17

    f = compute_event_features("XYZ", _ts("2025-01-07T16:32:05Z"), cal, cache, s)  # minute 122 has no trade
    assert f["market_alignment"] == "next_candle" and f["market_timestamp_utc"] == "2025-01-07T16:34:00Z"

    f = compute_event_features("XYZ", _ts("2025-01-07T20:59:30Z"), cal, cache, s)  # 15:59 ET: no 5m window
    assert math.isnan(f["return_5m"])

    f = compute_event_features("XYZ", _ts("2025-01-04T15:00:00Z"), cal, cache, s)  # Saturday
    assert f["market_alignment"] == "next_market_open" and f["market_session_date"] == "2025-01-06"
    assert f["news_session_position"] == "non_trading_day"

    assert compute_event_features("ABC", _ts("2025-01-07T15:00:00Z"), cal, cache, s) is None  # not downloaded

    _write_session(cache, "EMPTY", "2025-01-07", [], "2025-01-07T14:30:00Z")
    for d in ("2025-01-03", "2025-01-06"):
        _write_session(cache, "EMPTY", d, [], f"{d}T14:30:00Z")
    f = compute_event_features("EMPTY", _ts("2025-01-07T15:00:00Z"), cal, cache, s)
    assert f["market_alignment"] == "unavailable" and "open" not in f  # no fabricated OHLCV


def _row(**kw):
    base = {"market_alignment": "same_minute", "volatility": 0.0005, "baseline_return_count": 700,
            "volume_ratio": 3.0, "return_5m": 0.0, "return_15m": 0.0, "return_30m": 0.0, "return_60m": 0.0}
    base.update(kw)
    for h in (5, 15, 30, 60):
        base[f"abnormal_return_{h}m"] = base[f"return_{h}m"] / (base["volatility"] * math.sqrt(h))
    return base


def test_labeler_multi_factor(env):
    s = load_settings(env)
    up = label_event(_row(return_5m=0.01, return_15m=0.012, return_30m=0.011, return_60m=0.009), s)
    assert up["label"] == "spike" and up["label_confidence"] >= 0.5
    down = label_event(_row(return_5m=-0.01, return_15m=-0.015, return_30m=-0.02, return_60m=-0.02), s)
    assert down["label"] == "crash"
    # abnormal but tiny volume and below absolute threshold -> not confirmed -> neutral
    weak = label_event(_row(volume_ratio=0.8, return_5m=0.006, return_15m=0.005, return_30m=0.004, return_60m=0.004), s)
    assert weak["label"] == "neutral" and "not confirmed" in weak["label_reason"]
    # big absolute move but for a very volatile stock -> not abnormal -> neutral
    vol = label_event(_row(volatility=0.02, return_5m=0.025, return_15m=0.03, return_30m=0.02, return_60m=0.02), s)
    assert vol["label"] == "neutral"
    # no market data -> no label at all (never guessed)
    none = label_event({"market_alignment": "unavailable", "market_unavailable_reason": "no_bars_for_session"}, s)
    assert none["label"] == "" and "unavailable" in none["label_reason"]


def test_ner_rules(env):
    from nlp.ticker_ner import TickerExtractor, TickerReference
    from collectors.alpaca_reference import classify_asset

    s = load_settings(env)
    assets = pd.DataFrame([{**a, "status": "active", "tradable": "True",
                            "is_excluded": classify_asset(a["symbol"], a["name"], a["exchange"])[0],
                            "exclude_reason": classify_asset(a["symbol"], a["name"], a["exchange"])[1]}
                           for a in ASSETS])
    ref = TickerReference(assets, s)
    assert "SPY" not in ref.valid and "AAPL" in ref.valid
    ex = TickerExtractor(ref, s)
    rows = [
        {"title": "CEO SAYS AI IS BIG FOR THE US", "description": "IT spending rises"},       # uppercase words only
        {"title": "Price target raised for several retailers", "description": ""},          # 'target' as a word
        {"title": "Apple and Microsoft lead gains", "description": "Shares of $TSLA fell"},
        {"title": "News", "description": "", "api_tickers_alpaca": '["SPY", "AAPL"]'},
        {"title": "Crypto", "description": "", "av_ticker_sentiment": '[{"ticker": "CRYPTO:BTC", "relevance_score": "0.9"}]'},
    ]
    res = [[f["ticker"] for f in r] for r in ex.extract_batch(rows)]
    assert res[0] == [] and res[1] == []
    assert set(res[2]) == {"AAPL", "MSFT", "TSLA"}
    assert res[3] == ["AAPL"] and res[4] == []


def test_dedup_does_not_merge_distinct_articles(env):
    from processing.deduplicator import find_duplicates

    s = load_settings(env)
    t = _ts("2025-01-07T13:00:00Z")
    keys = pd.DataFrame([
        {"news_id": "a", "provider": "alpaca", "provider_id": "1", "url": "x.com/a", "pub": "benzinga",
         "title": "stocks moving in tuesday s pre market session", "ts": t, "ntok": 8, "has_desc": True},
        {"news_id": "b", "provider": "alpaca", "provider_id": "2", "url": "x.com/b", "pub": "benzinga",
         "title": "stocks moving in tuesday s pre market session", "ts": t + 60, "ntok": 8, "has_desc": True},
        {"news_id": "c", "provider": "alpha_vantage", "provider_id": "3", "url": "y.com/c", "pub": "benzinga",
         "title": "stocks moving in tuesday s pre market session", "ts": t + 120, "ntok": 8, "has_desc": True},
    ])
    uf, merges = find_duplicates(keys, s)
    assert uf.find(0) != uf.find(1)          # same provider, different URLs: kept apart
    assert uf.find(2) in (uf.find(0), uf.find(1))  # cross-provider copy merged


def test_url_normalisation():
    assert normalize_url("https://www.Benzinga.com/news/x/?utm_source=a&id=3#top") == "benzinga.com/news/x?id=3"


def test_av_timezone_auto_detection(env):
    from processing.cleaner import detect_av_timezone

    s = load_settings(env)
    s.alpha_vantage_timezone = "auto"
    s.ensure_dirs()
    av, al = [], []
    for i in range(25):
        utc = pd.Timestamp("2025-01-07T15:00:00Z") + pd.Timedelta(minutes=7 * i)
        et = utc.tz_convert("America/New_York")
        av.append({"url": f"https://a.com/{i}", "time_published": et.strftime("%Y%m%dT%H%M%S")})
        al.append({"url": f"https://a.com/{i}", "created_at": utc.strftime("%Y-%m-%dT%H:%M:%SZ")})
    pd.DataFrame(av).to_csv(s.alpha_vantage_raw_path, index=False)
    pd.DataFrame(al).to_csv(s.alpaca_news_raw_path, index=False)
    tz, info = detect_av_timezone(s)
    assert tz == "America/New_York" and info["url_matches"] == 25
