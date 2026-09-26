"""Normalise raw provider files into one common schema (data/raw/raw_news.csv).

* Provider raw files are only read, never modified.
* Timestamps are parsed from the original provider strings and converted to
  UTC; the original string is kept in `published_timestamp_raw`.
* Invalid records are not silently dropped: they are written to
  data/processed/rejected_news.csv with a `reject_reason`.
"""
from __future__ import annotations

import hashlib
import html
import json
import logging
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import pandas as pd

from config.settings import Settings
from storage.checkpoint import atomic_write_text
from storage.csv_writer import AtomicCSVWriter, iter_csv_chunks

log = logging.getLogger(__name__)

NEWS_COLUMNS = [
    "news_id", "news_provider", "alpha_vantage_id", "alpaca_id", "title", "description",
    "published_timestamp_raw", "news_published_timestamp_utc", "source", "publisher", "source_domain",
    "author", "url", "url_normalized", "api_tickers", "av_ticker_sentiment", "overall_sentiment_score",
    "overall_sentiment_label", "topics", "category_within_source",
]
REJECT_COLUMNS = ["news_provider", "provider_id", "title", "url", "published_timestamp_raw", "reject_reason"]

TRACKING_PARAMS = re.compile(r"^(utm_|fbclid|gclid|mc_|cmpid|ncid|yptr|guccounter|src|ref|feed|rss)", re.I)
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
PUBLISHER_NAMES = {
    "benzinga": "Benzinga", "motleyfool": "Motley Fool", "zacks": "Zacks", "zackscommentary": "Zacks Commentary",
    "reuters": "Reuters", "cnbc": "CNBC", "marketwatch": "MarketWatch", "businesswire": "Business Wire",
    "globenewswire": "GlobeNewswire", "prnewswire": "PR Newswire", "seekingalpha": "Seeking Alpha",
    "yahoofinance": "Yahoo Finance", "bloomberg": "Bloomberg", "barrons": "Barron's", "investorsbusinessdaily": "Investor's Business Daily",
    "forbes": "Forbes", "fool": "Motley Fool", "thestreet": "TheStreet", "decrypt": "Decrypt",
}


# ------------------------------------------------------------ normalisation
def clean_text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = html.unescape(str(value))
    text = TAG_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def normalize_url(url) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip().lower()
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("m."):
        host = host[2:]
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query) if not TRACKING_PARAMS.match(k)))
    path = re.sub(r"/+$", "", parts.path) or "/"
    return urlunsplit(("", host, path.lower(), query, ""))[2:]


def normalize_title(title) -> str:
    t = clean_text(title).lower()
    t = re.sub(r"[‘’“”'\"`]", "", t)
    t = re.sub(r"[^a-z0-9$%.& ]+", " ", t)
    return WS_RE.sub(" ", t).strip()


def normalize_publisher(source) -> str:
    return re.sub(r"[^a-z0-9]", "", str(source or "").lower())


def publisher_display(source) -> str:
    key = normalize_publisher(source)
    if key in PUBLISHER_NAMES:
        return PUBLISHER_NAMES[key]
    s = clean_text(source)
    return s.title() if s.islower() else s


def make_news_id(url_norm: str, provider: str, provider_id: str) -> str:
    basis = url_norm or f"{provider}:{provider_id}"
    return "n_" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def url_domain(url) -> str:
    try:
        host = urlsplit(str(url)).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except ValueError:
        return ""


# ---------------------------------------------------------------- timestamps
def parse_av_time(series: pd.Series, tz: str) -> pd.Series:
    naive = pd.to_datetime(series, format="%Y%m%dT%H%M%S", errors="coerce")
    short = pd.to_datetime(series, format="%Y%m%dT%H%M", errors="coerce")
    naive = naive.fillna(short)
    if tz.upper() == "UTC":
        return naive.dt.tz_localize("UTC")
    return naive.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")


def parse_iso_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce", format="ISO8601")


def fmt_utc(ts: pd.Series) -> pd.Series:
    return ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def detect_av_timezone(settings: Settings) -> tuple[str, dict]:
    """Resolve ALPHA_VANTAGE_TIMEZONE.

    Alpha Vantage returns naive timestamps. With ALPHA_VANTAGE_TIMEZONE=auto we
    compare them with Alpaca's explicit-UTC timestamps for the SAME article URL
    and pick the zone that makes them agree. No timestamp is ever shifted by a
    guess: if there is not enough overlap we fall back to UTC and say so.
    """
    configured = settings.alpha_vantage_timezone
    if configured.lower() != "auto":
        return configured, {"mode": "configured", "timezone": configured}
    av_p, al_p = settings.alpha_vantage_raw_path, settings.alpaca_news_raw_path
    info: dict = {"mode": "auto"}
    if not (av_p.exists() and al_p.exists()):
        info.update({"timezone": "UTC", "reason": "only one provider collected; defaulting to UTC"})
        return "UTC", info
    av_urls: dict[str, str] = {}
    for chunk in iter_csv_chunks(av_p, 200_000, usecols=["url", "time_published"]):
        for u, t in zip(chunk["url"].fillna(""), chunk["time_published"].fillna("")):
            if u and t:
                av_urls[normalize_url(u)] = t
    pairs = []
    for chunk in iter_csv_chunks(al_p, 200_000, usecols=["url", "created_at"]):
        for u, t in zip(chunk["url"].fillna(""), chunk["created_at"].fillna("")):
            key = normalize_url(u)
            if key in av_urls:
                pairs.append((av_urls[key], t))
    info["url_matches"] = len(pairs)
    if len(pairs) < 20:
        info.update({"timezone": "UTC", "reason": "fewer than 20 URL matches between providers; defaulting to UTC"})
        return "UTC", info
    df = pd.DataFrame(pairs, columns=["av", "alpaca"])
    alp = parse_iso_utc(df["alpaca"])
    scores = {}
    for tz in ("UTC", "America/New_York"):
        diff = (parse_av_time(df["av"], tz) - alp).dt.total_seconds().abs() / 60
        scores[tz] = float(diff.median())
    best = min(scores, key=scores.get)
    info["median_abs_diff_minutes"] = scores
    if scores[best] > 30:
        info.update({"timezone": "UTC", "reason": f"no candidate zone agrees within 30 min ({scores}); defaulting to UTC"})
        return "UTC", info
    info.update({"timezone": best, "reason": "timezone with smallest median difference on URL-matched articles"})
    return best, info


# ------------------------------------------------------------ provider maps
def _av_frame(chunk: pd.DataFrame, tz: str) -> pd.DataFrame:
    ts = parse_av_time(chunk["time_published"].fillna(""), tz)

    def tickers(js):
        try:
            return json.dumps([t.get("ticker", "") for t in json.loads(js or "[]") if t.get("ticker")])
        except (json.JSONDecodeError, AttributeError):
            return "[]"

    out = pd.DataFrame({
        "news_provider": "alpha_vantage",
        "alpha_vantage_id": chunk["alpha_vantage_id"],
        "alpaca_id": "",
        "title": chunk["title"].map(clean_text),
        "description": chunk["summary"].map(clean_text),
        "published_timestamp_raw": chunk["time_published"].fillna(""),
        "_ts": ts,
        "source": chunk["source"].fillna(""),
        "source_domain": chunk["source_domain"].fillna(""),
        "author": chunk["authors"].map(lambda a: ", ".join(json.loads(a)) if isinstance(a, str) and a.startswith("[") else ""),
        "url": chunk["url"].fillna(""),
        "api_tickers": chunk["ticker_sentiment"].map(tickers),
        "av_ticker_sentiment": chunk["ticker_sentiment"].fillna("[]"),
        "overall_sentiment_score": chunk["overall_sentiment_score"],
        "overall_sentiment_label": chunk["overall_sentiment_label"].fillna(""),
        "topics": chunk["topics"].fillna("[]"),
        "category_within_source": chunk["category_within_source"].fillna(""),
    })
    out["_provider_id"] = out["alpha_vantage_id"]
    return out


def _alpaca_frame(chunk: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "news_provider": "alpaca",
        "alpha_vantage_id": "",
        "alpaca_id": chunk["alpaca_id"],
        "title": chunk["headline"].map(clean_text),
        "description": chunk["summary"].map(clean_text),
        "published_timestamp_raw": chunk["created_at"].fillna(""),
        "_ts": parse_iso_utc(chunk["created_at"].fillna("")),
        "source": chunk["source"].fillna(""),
        "source_domain": chunk["url"].map(url_domain),
        "author": chunk["author"].fillna(""),
        "url": chunk["url"].fillna(""),
        "api_tickers": chunk["symbols"].fillna("[]"),
        "av_ticker_sentiment": "",
        "overall_sentiment_score": "",
        "overall_sentiment_label": "",
        "topics": "",
        "category_within_source": "",
    })
    out["_provider_id"] = out["alpaca_id"]
    return out


def _finalise(df: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["url_normalized"] = df["url"].map(normalize_url)
    df["publisher"] = df["source"].map(publisher_display)
    df["news_id"] = [make_news_id(u, p, i) for u, p, i in zip(df["url_normalized"], df["news_provider"], df["_provider_id"])]
    reason = pd.Series("", index=df.index)
    raw_missing = df["published_timestamp_raw"].astype(str).str.strip() == ""
    reason[raw_missing] = "missing_timestamp"
    reason[(reason == "") & df["_ts"].isna()] = "malformed_timestamp"
    reason[(reason == "") & (df["title"].str.len() == 0)] = "missing_title"
    start = pd.Timestamp(settings.start_utc)
    end = pd.Timestamp(settings.end_utc_exclusive)
    in_range = (df["_ts"] >= start) & (df["_ts"] < end)
    reason[(reason == "") & ~in_range] = "outside_date_range"
    df["news_published_timestamp_utc"] = fmt_utc(df["_ts"])
    rejected = df[reason != ""].assign(reject_reason=reason[reason != ""], provider_id=df["_provider_id"])
    return df.loc[reason == "", NEWS_COLUMNS], rejected.reindex(columns=REJECT_COLUMNS)


def build_raw_news(settings: Settings) -> dict:
    """Stream both raw provider files into data/raw/raw_news.csv (normalised, pre-dedup)."""
    tz, tz_info = detect_av_timezone(settings)
    atomic_write_text(settings.processed_dir / "alpha_vantage_timezone.json", json.dumps(tz_info, indent=2))
    log.info("Alpha Vantage timestamps interpreted as %s (%s)", tz, tz_info.get("reason", tz_info.get("mode")))

    good_writer = AtomicCSVWriter(settings.raw_news_path)
    bad_writer = AtomicCSVWriter(settings.rejected_news_path)
    stats = {"alpha_vantage_raw_count": 0, "alpaca_raw_count": 0, "rejected": {}, "alpha_vantage_timezone": tz}
    try:
        sources = [(settings.alpha_vantage_raw_path, "alpha_vantage"), (settings.alpaca_news_raw_path, "alpaca")]
        for path, provider in sources:
            for chunk in iter_csv_chunks(path, settings.chunk_size):
                chunk = chunk.fillna("")
                stats[f"{provider}_raw_count"] += len(chunk)
                frame = _av_frame(chunk, tz) if provider == "alpha_vantage" else _alpaca_frame(chunk)
                good, bad = _finalise(frame, settings)
                if len(good):
                    good_writer.write(good)
                if len(bad):
                    bad_writer.write(bad)
                    for r, n in bad["reject_reason"].value_counts().items():
                        stats["rejected"][r] = stats["rejected"].get(r, 0) + int(n)
        good_writer.commit(empty_columns=NEWS_COLUMNS)
        bad_writer.commit(empty_columns=REJECT_COLUMNS)
    except BaseException:
        good_writer.abort()
        bad_writer.abort()
        raise
    stats["valid_count"] = good_writer.rows
    stats["invalid_count"] = sum(stats["rejected"].values())
    log.info("Cleaning: AV raw=%d, Alpaca raw=%d, valid=%d, rejected=%s", stats["alpha_vantage_raw_count"],
             stats["alpaca_raw_count"], stats["valid_count"], stats["rejected"])
    return stats
