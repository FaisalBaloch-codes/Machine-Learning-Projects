"""Reference data from Alpaca: the US-equity universe and the trading calendar.

* data/raw/reference/us_equities.csv  - symbol validation list used by NER
* data/raw/reference/trading_calendar.csv - official session open/close per day
  (handles holidays and early closes)
"""
from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from collectors.api_utils import HttpClient
from config.settings import Settings
from storage.csv_writer import atomic_write_df
from storage.timeutils import epoch_seconds

log = logging.getLogger(__name__)

NY = "America/New_York"

FUND_NAME_RE = re.compile(
    r"\b(ETF|ETN|ETP|Exchange[- ]Traded|Index Fund|Fund|iShares|SPDR|ProShares|Direxion|Vanguard|"
    r"Invesco QQQ|WisdomTree|VanEck|Global X|First Trust|Schwab .*Index|Grayscale|Bitcoin|Ethereum|"
    r"Ultra(Pro|Short)|Leveraged|Inverse|2X|3X|-1X)\b", re.IGNORECASE)
NON_COMMON_RE = re.compile(r"\b(Warrants?|Rights?|Units?|Preferred|Depositary Shares? Each Representing|"
                           r"Notes? due|Debentures?|% Series)\b", re.IGNORECASE)


def classify_asset(symbol: str, name: str, exchange: str) -> tuple[bool, str]:
    """Returns (excluded, reason). Keeps common stock and ADRs of operating companies."""
    name = name or ""
    if not re.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?", symbol or ""):
        return True, "non_standard_symbol"
    if FUND_NAME_RE.search(name):
        return True, "etf_or_fund"
    if NON_COMMON_RE.search(name):
        return True, "non_common_security"
    return False, ""


def fetch_assets(settings: Settings, client: HttpClient, refresh: bool = False) -> pd.DataFrame:
    path = settings.reference_dir / "us_equities.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    log.info("Downloading Alpaca US-equity asset list ...")
    assets = client.get_json(f"{settings.trading_url}/v2/assets",
                             {"status": "active", "asset_class": "us_equity"}, context="assets")
    rows = []
    for a in assets:
        excluded, reason = classify_asset(a.get("symbol", ""), a.get("name", ""), a.get("exchange", ""))
        rows.append({
            "symbol": a.get("symbol", ""), "name": a.get("name", ""), "exchange": a.get("exchange", ""),
            "status": a.get("status", ""), "tradable": a.get("tradable", ""),
            "is_excluded": excluded, "exclude_reason": reason,
        })
    df = pd.DataFrame(rows).sort_values("symbol")
    atomic_write_df(df, path)
    log.info("Saved %d assets (%d usable equities) to %s", len(df), int((~df.is_excluded).sum()), path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


# ------------------------------------------------------------------ calendar


def fetch_calendar(settings: Settings, client: HttpClient | None) -> pd.DataFrame:
    path = settings.reference_dir / "trading_calendar.csv"
    need_start = settings.start_date - timedelta(days=21)
    need_end = settings.end_date + timedelta(days=10)
    if path.exists():
        cal = pd.read_csv(path, dtype=str)
        if len(cal) and cal["date"].min() <= str(need_start) and cal["date"].max() >= str(need_end):
            return cal
    if client is None:
        raise RuntimeError("Trading calendar not cached and no Alpaca client available")
    log.info("Downloading trading calendar %s .. %s from Alpaca", need_start, need_end)
    data = client.get_json(f"{settings.trading_url}/v2/calendar",
                           {"start": str(need_start), "end": str(need_end)}, context="calendar")
    cal = pd.DataFrame([{"date": d["date"], "open": d["open"], "close": d["close"]} for d in data])
    atomic_write_df(cal, path)
    return cal


class TradingCalendar:
    """Regular trading sessions as UTC epoch seconds."""

    def __init__(self, cal: pd.DataFrame):
        cal = cal.sort_values("date").reset_index(drop=True)
        self.dates = cal["date"].astype(str).tolist()
        opens = pd.to_datetime(cal["date"] + " " + cal["open"]).dt.tz_localize(NY).dt.tz_convert("UTC")
        closes = pd.to_datetime(cal["date"] + " " + cal["close"]).dt.tz_localize(NY).dt.tz_convert("UTC")
        self.open_s = epoch_seconds(opens)
        self.close_s = epoch_seconds(closes)
        self.index = {d: i for i, d in enumerate(self.dates)}

    @classmethod
    def from_file(cls, path: Path) -> "TradingCalendar":
        return cls(pd.read_csv(path, dtype=str))

    def locate(self, ts: int) -> tuple[int, str]:
        """Find the session relevant to a UTC epoch second.

        Returns (session_index, position) where position is 'in_session' when ts
        falls inside [open, close) of that session, otherwise 'before_open' and the
        index points at the next session that opens after ts. (-1, 'none') if the
        calendar does not cover ts.
        """
        i = int(np.searchsorted(self.close_s, ts, side="right"))  # first session closing after ts
        if i >= len(self.close_s):
            return -1, "none"
        if self.open_s[i] <= ts < self.close_s[i]:
            return i, "in_session"
        return i, "before_open"

    def previous_sessions(self, i: int, n: int) -> list[int]:
        return list(range(max(0, i - n), i))

    def session_date(self, i: int) -> str:
        return self.dates[i]

    def date_index(self, d: str | date) -> int | None:
        return self.index.get(str(d))
