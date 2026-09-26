"""Assemble the final dataset, validate it and write the quality report.

Outputs (data/processed/):
  final_stock_news_impact_dataset.csv   every ticker-event, with quality flags
  ml_ready_dataset.csv                  usable rows, ONLY pre-news features + target
  dataset_schema.json                   role of every column (feature / outcome / target / id)
  data_quality_report.json / .csv
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from config.settings import Settings
from storage.checkpoint import atomic_write_text
from storage.csv_writer import AtomicCSVWriter, atomic_write_df, iter_csv_chunks, read_column_set

log = logging.getLogger(__name__)

NY = "America/New_York"


def column_roles(settings: Settings) -> dict[str, tuple[str, str]]:
    """column -> (role, description). Roles:
    identifier | metadata | feature (known at publication) | outcome (post-news, labelling only) | target
    """
    roles = {
        "news_id": ("identifier", "Stable article id (hash of normalised URL)"),
        "event_id": ("identifier", "news_id + ticker (one row per article-ticker pair)"),
        "ticker": ("feature", "US-listed equity symbol affected by the article"),
        "company_name": ("feature", "Company name from Alpaca asset master"),
        "title": ("feature", "Article headline (text)"),
        "description": ("feature", "Article summary/description (text)"),
        "news_published_timestamp_utc": ("feature", "Publication time, UTC (ISO-8601)"),
        "news_published_timestamp_et": ("feature", "Publication time, US/Eastern"),
        "news_date": ("feature", "Publication date (UTC)"),
        "news_time": ("feature", "Publication time of day (UTC)"),
        "news_day_of_week": ("feature", "0=Monday (US/Eastern)"),
        "news_hour_et": ("feature", "Hour of day, US/Eastern"),
        "news_session_position": ("feature", "regular_hours / pre_market / after_hours / non_trading_day"),
        "news_provider": ("metadata", "API the canonical record came from: alpha_vantage | alpaca"),
        "news_providers_all": ("metadata", "All APIs that returned this article"),
        "publisher": ("feature", "Normalised name of the actual news publisher"),
        "source": ("metadata", "Publisher/source field exactly as returned by the API"),
        "source_domain": ("metadata", "Domain of the article URL"),
        "author": ("metadata", "Author(s) as returned by the API"),
        "url": ("metadata", "Article URL"),
        "alpha_vantage_id": ("identifier", "Alpha Vantage record id(s) (derived from URL; ';'-separated)"),
        "alpaca_id": ("identifier", "Alpaca news id(s) (';'-separated)"),
        "ticker_detection_method": ("metadata", "Evidence used to link the ticker"),
        "ticker_confidence": ("feature", "Confidence of the ticker link (0-1)"),
        "n_tickers_in_article": ("feature", "Number of tickers linked to the article"),
        "overall_sentiment_score": ("feature", "Alpha Vantage article sentiment (text-derived)"),
        "overall_sentiment_label": ("feature", "Alpha Vantage article sentiment label"),
        "av_ticker_relevance_score": ("feature", "Alpha Vantage relevance of this ticker"),
        "av_ticker_sentiment_score": ("feature", "Alpha Vantage ticker-specific sentiment"),
        "av_ticker_sentiment_label": ("feature", "Alpha Vantage ticker-specific sentiment label"),
        "topics": ("feature", "Alpha Vantage topics (JSON)"),
        "volatility": ("feature", "PRE-news per-minute log-return volatility (prior sessions + same-day pre-news bars)"),
        "baseline_volume_per_min": ("feature", "PRE-news average volume per minute in prior sessions"),
        "baseline_return_count": ("feature", "Number of 1-minute returns in the volatility baseline"),
        "pre_news_close": ("feature", "Last traded 1-minute close strictly BEFORE the news minute"),
        "pre_news_return_15m": ("feature", "Return over the 15 minutes BEFORE the news (same session)"),
        "market_alignment": ("metadata", "same_minute | next_candle | next_market_open | unavailable"),
        "market_unavailable_reason": ("metadata", "Why market data could not be aligned"),
        "market_session_date": ("metadata", "Trading session the event was aligned to"),
        "market_timestamp_utc": ("metadata", "Start time of the aligned T0 1-minute candle"),
        "alignment_lag_seconds": ("metadata", "T0 candle start minus news time (negative = candle began before news)"),
        "open": ("outcome", "T0 candle open (T0 candle overlaps/follows publication)"),
        "high": ("outcome", "T0 candle high"),
        "low": ("outcome", "T0 candle low"),
        "close": ("outcome", "T0 candle close = reference price for post-news returns"),
        "volume": ("outcome", "T0 candle volume"),
        "high_move_5m": ("outcome", "Max high in (T0, T0+5m] relative to T0 close"),
        "low_move_5m": ("outcome", "Min low in (T0, T0+5m] relative to T0 close"),
        "bars_in_5m": ("outcome", "Number of 1-minute bars in (T0, T0+5m]"),
        "event_volume": ("outcome", "Volume in the post-news volume window"),
        "expected_volume": ("outcome", "Same clock-time volume in prior sessions (baseline for volume_ratio)"),
        "volume_ratio": ("outcome", "event_volume / expected_volume"),
        "gap_return": ("outcome", "Opening gap vs last pre-news close (overnight news only)"),
        "gap_abnormal": ("outcome", "gap_return standardised by volatility*sqrt(390)"),
        "abnormal_return": ("outcome", "z-score of the horizon that determined the label"),
        "label": ("target", "spike | crash | neutral (market reaction following the news)"),
        "label_confidence": ("label_metadata", "Heuristic confidence of the label (0-1)"),
        "label_reason": ("label_metadata", "Human-readable explanation of the label"),
        "label_horizon_minutes": ("label_metadata", "Horizon that determined the label (0 = opening gap)"),
        "direction_consistency": ("label_metadata", "Share of horizons moving in the label direction"),
        "quality_flags": ("metadata", "';'-separated data-quality flags"),
        "is_usable": ("metadata", "True if the row has a label and passed all critical checks"),
    }
    for h in settings.post_windows:
        roles[f"return_{h}m"] = ("outcome", f"FUTURE return from T0 close to T0+{h}m (label generation only)")
        roles[f"abnormal_return_{h}m"] = ("outcome", f"return_{h}m / (volatility*sqrt({h}))")
    return roles


def final_columns(settings: Settings) -> list[str]:
    head = ["news_id", "event_id", "ticker", "company_name", "title", "description",
            "news_published_timestamp_utc", "news_published_timestamp_et", "news_date", "news_time",
            "news_day_of_week", "news_hour_et", "news_session_position",
            "news_provider", "news_providers_all", "publisher", "source", "source_domain", "author", "url",
            "alpha_vantage_id", "alpaca_id", "ticker_detection_method", "ticker_confidence", "n_tickers_in_article",
            "overall_sentiment_score", "overall_sentiment_label", "av_ticker_relevance_score",
            "av_ticker_sentiment_score", "av_ticker_sentiment_label", "topics",
            "volatility", "baseline_volume_per_min", "baseline_return_count", "pre_news_close", "pre_news_return_15m",
            "market_alignment", "market_unavailable_reason", "market_session_date", "market_timestamp_utc",
            "alignment_lag_seconds", "open", "high", "low", "close", "volume"]
    outcomes = [f"return_{h}m" for h in settings.post_windows]
    outcomes += ["high_move_5m", "low_move_5m", "bars_in_5m", "event_volume", "expected_volume", "volume_ratio",
                 "gap_return", "gap_abnormal"]
    outcomes += [f"abnormal_return_{h}m" for h in settings.post_windows] + ["abnormal_return"]
    tail = ["label", "label_confidence", "label_reason", "label_horizon_minutes", "direction_consistency",
            "quality_flags", "is_usable"]
    return head + outcomes + tail


def _count_rows(path) -> int:
    return sum(len(c) for c in iter_csv_chunks(path, 200_000, usecols=None)) if path.exists() else 0


def _valid_tickers(settings: Settings) -> set[str]:
    path = settings.reference_dir / "us_equities.csv"
    if not path.exists():
        return set()
    from nlp.ticker_ner import TickerReference

    return TickerReference(pd.read_csv(path, dtype=str, keep_default_na=False), settings).valid


def _quality_flags(df: pd.DataFrame, valid: set[str], start, end) -> pd.Series:
    flags = [[] for _ in range(len(df))]

    def add(mask, name):
        for i in np.flatnonzero(mask.to_numpy()):
            flags[i].append(name)

    ts = pd.to_datetime(df["news_published_timestamp_utc"], utc=True, errors="coerce")
    add(df["title"].fillna("").astype(str).str.strip() == "", "missing_title")
    add(df["news_published_timestamp_utc"].fillna("").astype(str) == "", "missing_timestamp")
    add(ts.isna() & (df["news_published_timestamp_utc"].fillna("").astype(str) != ""), "malformed_timestamp")
    add(~ts.isna() & ((ts < start) | (ts >= end)), "outside_date_range")
    if valid:
        add(~df["ticker"].isin(valid), "invalid_ticker")
    align = df["market_alignment"].fillna("")
    add(align == "", "missing_market_data_pending")
    add(align == "unavailable", "missing_market_data")
    o, h, l, c, v = (pd.to_numeric(df[k], errors="coerce") for k in ("open", "high", "low", "close", "volume"))
    has = ~o.isna()
    bad = has & ((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (l <= 0) | (h < l))
    add(bad, "impossible_ohlc")
    add(has & (v < 0), "negative_volume")
    add(df["label"].fillna("") == "", "unlabelled")
    return pd.Series([";".join(f) for f in flags], index=df.index)


CRITICAL = {"missing_title", "missing_timestamp", "malformed_timestamp", "outside_date_range", "invalid_ticker",
            "missing_market_data", "missing_market_data_pending", "impossible_ohlc", "negative_volume", "unlabelled"}


def build_dataset(settings: Settings) -> dict:
    cols = final_columns(settings)
    roles = column_roles(settings)
    valid_news = read_column_set(settings.dedup_news_path, "news_id")
    valid_tickers = _valid_tickers(settings)
    labeled = pd.read_csv(settings.labeled_events_path, low_memory=False) if settings.labeled_events_path.exists() \
        else pd.DataFrame(columns=["event_id"])
    labeled = labeled.drop_duplicates("event_id").drop(columns=["ticker", "news_published_timestamp_utc"],
                                                       errors="ignore").set_index("event_id")
    start, end = pd.Timestamp(settings.start_utc), pd.Timestamp(settings.end_utc_exclusive)

    # ML-ready file: pre-news features + target only. Outcome columns (T0 candle, future
    # returns, ...) are never included; ML_EXCLUDE_COLUMNS drops further columns on request.
    excluded = set(settings.ml_exclude_columns)
    feature_cols = [c for c in cols if roles.get(c, ("",))[0] in {"feature", "identifier"} and c not in excluded]
    ml_cols = feature_cols + [c for c in ("label", "label_confidence") if c not in excluded]

    final_w = AtomicCSVWriter(settings.final_dataset_path)
    ml_w = AtomicCSVWriter(settings.processed_dir / "ml_ready_dataset.csv")
    seen: set[str] = set()
    r = {"ticker_events_in_file": 0, "stale_events_skipped": 0, "duplicate_events_removed": 0, "final_rows": 0,
         "flags": {}, "labels": {}, "alignment": {}, "usable_events": 0}
    try:
        for chunk in iter_csv_chunks(settings.ticker_events_path, settings.chunk_size):
            r["ticker_events_in_file"] += len(chunk)
            stale = ~chunk["news_id"].isin(valid_news)
            r["stale_events_skipped"] += int(stale.sum())
            chunk = chunk[~stale]
            dup = chunk["event_id"].duplicated() | chunk["event_id"].isin(seen)
            r["duplicate_events_removed"] += int(dup.sum())
            chunk = chunk[~dup]
            seen.update(chunk["event_id"])
            if not len(chunk):
                continue
            df = chunk.join(labeled, on="event_id", how="left", rsuffix="_mkt")
            ts = pd.to_datetime(df["news_published_timestamp_utc"], utc=True, errors="coerce")
            et = ts.dt.tz_convert(NY)
            df["news_published_timestamp_et"] = et.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
            df["news_date"] = ts.dt.strftime("%Y-%m-%d")
            df["news_time"] = ts.dt.strftime("%H:%M:%S")
            df["news_day_of_week"] = et.dt.dayofweek
            df["news_hour_et"] = et.dt.hour
            df = df.reindex(columns=cols)
            df["quality_flags"] = _quality_flags(df, valid_tickers, start, end)
            df["is_usable"] = df["quality_flags"].map(lambda f: not (set(f.split(";")) & CRITICAL) if f else True)
            final_w.write(df)
            ml_w.write(df.loc[df["is_usable"], ml_cols])
            r["final_rows"] += len(df)
            r["usable_events"] += int(df["is_usable"].sum())
            for col, key in (("label", "labels"), ("market_alignment", "alignment")):
                for k, v in df[col].fillna("none").value_counts().items():
                    r[key][k] = r[key].get(k, 0) + int(v)
            for f in df["quality_flags"]:
                for name in filter(None, f.split(";")):
                    r["flags"][name] = r["flags"].get(name, 0) + 1
        final_w.commit(empty_columns=cols)
        ml_w.commit(empty_columns=ml_cols)
    except BaseException:
        final_w.abort()
        ml_w.abort()
        raise

    schema = {c: {"role": roles.get(c, ("metadata", ""))[0], "description": roles.get(c, ("", ""))[1]} for c in cols}
    atomic_write_text(settings.processed_dir / "dataset_schema.json", json.dumps(schema, indent=2))
    report = quality_report(settings, r)
    return report


def quality_report(settings: Settings, r: dict) -> dict:
    def load(name):
        p = settings.processed_dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    dedup = load("dedup_stats.json")
    av_raw = _count_rows(settings.alpha_vantage_raw_path)
    al_raw = _count_rows(settings.alpaca_news_raw_path)
    rejected = {}
    if settings.rejected_news_path.exists():
        rej = pd.read_csv(settings.rejected_news_path, usecols=["reject_reason"], dtype=str)
        rejected = {k: int(v) for k, v in rej["reject_reason"].value_counts().items()}
    flags = r["flags"]
    labels = r["labels"]
    matched = sum(v for k, v in r["alignment"].items() if k not in ("unavailable", "none"))
    usable = r["usable_events"]
    report = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date_range": [str(settings.start_date), str(settings.end_date)],
        "raw_news_count": av_raw + al_raw,
        "alpha_vantage_raw_count": av_raw,
        "alpaca_raw_count": al_raw,
        "rejected_news_by_reason": rejected,
        "deduplicated_news_count": dedup.get("final_unique_news_count"),
        "duplicate_count": dedup.get("duplicates_removed"),
        "duplicates_by_rule": dedup.get("duplicates_by_rule"),
        "ticker_events": r["final_rows"],
        "duplicate_events_removed": r["duplicate_events_removed"],
        "stale_events_skipped": r["stale_events_skipped"],
        "market_matched_events": matched,
        "missing_market_data": flags.get("missing_market_data", 0) + flags.get("missing_market_data_pending", 0),
        "missing_market_data_pending_download": flags.get("missing_market_data_pending", 0),
        "market_alignment_breakdown": r["alignment"],
        "neutral_count": labels.get("neutral", 0),
        "spike_count": labels.get("spike", 0),
        "crash_count": labels.get("crash", 0),
        "unlabelled_count": labels.get("none", 0),
        "invalid_count": sum(rejected.values()) + sum(flags.get(k, 0) for k in
                                                      ("invalid_ticker", "impossible_ohlc", "negative_volume",
                                                       "malformed_timestamp", "missing_title", "outside_date_range")),
        "quality_flag_counts": flags,
        "usable_labelled_events": usable,
        "market_matched_percentage": round(100 * matched / r["final_rows"], 2) if r["final_rows"] else 0.0,
        "target_40k_reached": usable >= 40_000,
        "note": (f"{usable:,} valid labelled events collected. Counts reflect only data actually returned by the "
                 "APIs; no rows were duplicated or synthesised to reach a target size."),
    }
    atomic_write_text(settings.processed_dir / "data_quality_report.json", json.dumps(report, indent=2))
    flat = []
    for k, v in report.items():
        if isinstance(v, dict):
            flat += [{"metric": f"{k}.{kk}", "value": vv} for kk, vv in v.items()]
        else:
            flat.append({"metric": k, "value": v if not isinstance(v, list) else " to ".join(v)})
    atomic_write_df(pd.DataFrame(flat), settings.processed_dir / "data_quality_report.csv")
    return report
