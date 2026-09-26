"""Offline stand-in for the Alpha Vantage and Alpaca HTTP APIs - TESTS ONLY.

It mimics the real JSON response shapes so the whole pipeline can be exercised
without credentials or network. None of this is ever used by main.py; the
synthetic articles/bars live only in pytest's temporary directories.
"""
from __future__ import annotations

import zlib
from datetime import date, datetime, timedelta, timezone

import numpy as np

ASSETS = [
    {"symbol": "AAPL", "name": "Apple Inc. Common Stock", "exchange": "NASDAQ"},
    {"symbol": "MSFT", "name": "Microsoft Corporation Common Stock", "exchange": "NASDAQ"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation Common Stock", "exchange": "NASDAQ"},
    {"symbol": "TSLA", "name": "Tesla, Inc. Common Stock", "exchange": "NASDAQ"},
    {"symbol": "TGT", "name": "Target Corporation Common Stock", "exchange": "NYSE"},
    {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "exchange": "ARCA"},
    {"symbol": "AI", "name": "C3.ai, Inc. Class A Common Stock", "exchange": "NYSE"},
]

ALPACA_NEWS = [
    {"id": 1001, "headline": "Apple shares jump after surprise buyback announcement",
     "summary": "<p>Apple Inc. said it will expand its buyback, sending shares higher.</p>",
     "author": "Benzinga Newsdesk", "created_at": "2025-01-08T15:02:17Z", "updated_at": "2025-01-08T15:02:17Z",
     "url": "https://www.benzinga.com/news/25/01/apple-jumps", "source": "benzinga", "symbols": ["AAPL"]},
    {"id": 1002, "headline": "Nvidia shares slide after hours on new export curbs",
     "summary": "Chipmaker falls in extended trading, while artificial intelligence (AI) names wobble.",
     "author": "Benzinga Newsdesk", "created_at": "2025-01-08T22:10:00Z", "updated_at": "2025-01-08T22:10:00Z",
     "url": "https://www.benzinga.com/news/25/01/nvidia-slides", "source": "benzinga", "symbols": []},
    {"id": 1003, "headline": "Target Corporation (NYSE: TGT) and Tesla update guidance",
     "summary": "Analysts raise their price target on several retailers.",
     "author": "Staff", "created_at": "2025-01-07T16:45:00Z", "updated_at": "2025-01-07T16:45:00Z",
     "url": "https://www.benzinga.com/news/25/01/tgt-tsla", "source": "benzinga", "symbols": ["TGT", "TSLA"]},
    {"id": 1004, "headline": "Market wrap: futures steady", "summary": "", "author": "Staff",
     "created_at": "2025-01-07T13:00:00Z", "updated_at": "2025-01-07T13:00:00Z",
     "url": "https://www.benzinga.com/news/25/01/wrap", "source": "benzinga", "symbols": ["SPY"]},
    {"id": 1005, "headline": "", "summary": "Headline missing - must be rejected", "author": "",
     "created_at": "2025-01-07T14:00:00Z", "updated_at": "", "url": "https://www.benzinga.com/x", "source": "benzinga",
     "symbols": []},
]

AV_FEED = [
    {"title": "Apple shares jump after surprise buyback announcement",
     "url": "https://www.benzinga.com/news/25/01/apple-jumps?utm_source=feed", "time_published": "20250108T100217",
     "authors": ["Benzinga Newsdesk"], "summary": "Apple expands buyback.", "banner_image": "", "source": "Benzinga",
     "category_within_source": "News", "source_domain": "www.benzinga.com",
     "topics": [{"topic": "Earnings", "relevance_score": "0.9"}], "overall_sentiment_score": 0.41,
     "overall_sentiment_label": "Bullish",
     "ticker_sentiment": [{"ticker": "AAPL", "relevance_score": "0.95", "ticker_sentiment_score": "0.5",
                           "ticker_sentiment_label": "Bullish"}]},
    {"title": "Microsoft, bitcoin and the S&P 500: what to watch", "url": "https://www.fool.com/msft-watch",
     "time_published": "20250107T113000", "authors": ["A. Writer"],
     "summary": "Microsoft Azure growth, crypto swings and index levels.", "banner_image": "",
     "source": "Motley Fool", "category_within_source": "n/a", "source_domain": "www.fool.com", "topics": [],
     "overall_sentiment_score": 0.05, "overall_sentiment_label": "Neutral",
     "ticker_sentiment": [{"ticker": "MSFT", "relevance_score": "0.8", "ticker_sentiment_score": "0.1",
                           "ticker_sentiment_label": "Neutral"},
                          {"ticker": "CRYPTO:BTC", "relevance_score": "0.6", "ticker_sentiment_score": "0.2",
                           "ticker_sentiment_label": "Neutral"},
                          {"ticker": "SPY", "relevance_score": "0.5", "ticker_sentiment_score": "0.0",
                           "ticker_sentiment_label": "Neutral"}]},
]


def trading_days(start: date, end: date) -> list[date]:
    days, d = [], start
    while d <= end:
        if d.weekday() < 5 and d not in (date(2024, 12, 25), date(2025, 1, 1)):
            days.append(d)
        d += timedelta(days=1)
    return days


def make_bars(symbol: str, day: str, start: datetime, end: datetime) -> list[dict]:
    rng = np.random.RandomState(zlib.crc32(f"{symbol}{day}".encode()))
    minutes = int((end - start).total_seconds() // 60)
    if minutes <= 0:
        return []
    rets = rng.normal(0, 0.0005, minutes)
    vols = rng.randint(800, 1200, minutes).astype(float)
    level = np.ones(minutes)
    times = [start + timedelta(minutes=i) for i in range(minutes)]
    if symbol == "AAPL" and day == "2025-01-08":  # +3% jump right after the 15:02 UTC article
        for i, t in enumerate(times):
            if t >= datetime(2025, 1, 8, 15, 3, tzinfo=timezone.utc):
                level[i] = 1.03
            if datetime(2025, 1, 8, 15, 2, tzinfo=timezone.utc) <= t < datetime(2025, 1, 8, 15, 17, tzinfo=timezone.utc):
                vols[i] *= 6
    if symbol == "NVDA" and day == "2025-01-09":  # -4% opening gap, continued weakness, heavy volume
        for i in range(minutes):
            level[i] = 0.96 * (1 - 0.005 * min(i, 60) / 60)
        vols[:20] *= 5
    if symbol == "MSFT" and day == "2025-01-07":
        vols[:] = 1000  # plain session -> neutral
    price = 100 * np.exp(np.cumsum(rets)) * level
    bars = []
    for i, t in enumerate(times):
        c = round(float(price[i]), 4)
        o = round(float(price[i - 1]) if i else c, 4)
        bars.append({"t": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "o": o, "h": round(max(o, c) * 1.0002, 4),
                     "l": round(min(o, c) * 0.9998, 4), "c": c, "v": vols[i], "n": 10, "vw": c})
    return bars


class FakeAPIs:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.stop_after_alpaca_news_pages: int | None = None

    def __call__(self, client, url: str, params: dict | None = None, context: str = "", record_failure: bool = True):
        params = dict(params or {})
        self.calls.append((url, params))
        if "alphavantage" in url:
            return self._av(params)
        if url.endswith("/v2/assets"):
            return [{**a, "status": "active", "tradable": True, "class": "us_equity"} for a in ASSETS]
        if url.endswith("/v2/calendar"):
            s = date.fromisoformat(params["start"])
            e = date.fromisoformat(params["end"])
            return [{"date": str(d), "open": "09:30", "close": "16:00"} for d in trading_days(s, e)]
        if url.endswith("/v1beta1/news"):
            return self._alpaca_news(params)
        if url.endswith("/v2/stocks/bars"):
            return self._bars(params)
        raise AssertionError(f"unexpected URL {url}")

    def _av(self, params):
        f, t = params["time_from"], params["time_to"]
        feed = [a for a in AV_FEED if f <= a["time_published"][:13] < t]
        return {"items": str(len(feed)), "feed": feed}

    def _alpaca_news(self, params):
        from collectors.api_utils import STOP_EVENT

        start = datetime.fromisoformat(params["start"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(params["end"].replace("Z", "+00:00"))
        items = [n for n in ALPACA_NEWS
                 if start <= datetime.fromisoformat(n["created_at"].replace("Z", "+00:00")) < end]
        items.sort(key=lambda n: n["created_at"])
        page = int(params.get("page_token") or 0)
        size = 2  # small pages to exercise pagination
        chunk = items[page * size:(page + 1) * size]
        token = str(page + 1) if (page + 1) * size < len(items) else None
        n_pages = sum(1 for u, _ in self.calls if u.endswith("/v1beta1/news"))
        if self.stop_after_alpaca_news_pages and n_pages >= self.stop_after_alpaca_news_pages:
            STOP_EVENT.set()
        return {"news": chunk, "next_page_token": token}

    def _bars(self, params):
        symbols = params["symbols"].split(",")
        start = datetime.fromisoformat(params["start"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(params["end"].replace("Z", "+00:00"))
        day = (start - timedelta(hours=5)).strftime("%Y-%m-%d")
        # page 1: first symbol, page 2: the rest (exercises pagination)
        if params.get("page_token") == "p2":
            return {"bars": {s: make_bars(s, day, start, end) for s in symbols[1:]}, "next_page_token": None}
        return {"bars": {symbols[0]: make_bars(symbols[0], day, start, end)},
                "next_page_token": "p2" if len(symbols) > 1 else None}
