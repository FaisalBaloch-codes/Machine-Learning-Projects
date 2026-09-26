"""Conservative cross-provider news de-duplication.

Only the key columns are held in memory; full rows are streamed. Duplicate
clusters are built with union-find using these rules (in order):

1. same normalised URL
2. same normalised title AND timestamps within DEDUP_TIME_TOLERANCE_MINUTES
3. same normalised title AND same publisher within DEDUP_TITLE_SOURCE_WINDOW_HOURS
   (only titles with >= 5 tokens, so generic recurring headlines are not merged)
4. fuzzy title similarity >= FUZZY_TITLE_THRESHOLD, same publisher, within
   FUZZY_WINDOW_MINUTES, titles with >= MIN_TITLE_TOKENS_FOR_FUZZY tokens

Rules 2-4 never merge two records from the SAME provider that have DIFFERENT
URLs - those are distinct articles that merely look alike (e.g. daily
"Stocks moving premarket" posts). Every merge is logged in duplicate_map.csv.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from config.settings import Settings
from processing.cleaner import normalize_publisher, normalize_title
from storage.checkpoint import atomic_write_text
from storage.csv_writer import AtomicCSVWriter, atomic_write_df, iter_csv_chunks
from storage.timeutils import epoch_seconds

log = logging.getLogger(__name__)

KEY_COLS = ["news_id", "news_provider", "alpha_vantage_id", "alpaca_id", "title", "description", "source",
            "url_normalized", "news_published_timestamp_utc"]
EXTRA_COLUMNS = ["news_providers_all", "duplicate_count", "published_utc_alpha_vantage", "published_utc_alpaca",
                 "api_tickers_alpha_vantage", "api_tickers_alpaca"]


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[max(ra, rb)] = min(ra, rb)
        return True


def _load_keys(settings: Settings) -> pd.DataFrame:
    parts = []
    for chunk in iter_csv_chunks(settings.raw_news_path, 200_000, usecols=KEY_COLS):
        chunk = chunk.fillna("")
        parts.append(pd.DataFrame({
            "news_id": chunk["news_id"],
            "provider": chunk["news_provider"],
            "provider_id": np.where(chunk["news_provider"] == "alpaca", chunk["alpaca_id"], chunk["alpha_vantage_id"]),
            "url": chunk["url_normalized"],
            "title": chunk["title"].map(normalize_title),
            "pub": chunk["source"].map(normalize_publisher),
            "has_desc": chunk["description"].str.len() > 0,
            "ts": epoch_seconds(chunk["news_published_timestamp_utc"]),
        }))
    if not parts:
        return pd.DataFrame(columns=["news_id", "provider", "provider_id", "url", "title", "pub", "has_desc", "ts"])
    keys = pd.concat(parts, ignore_index=True)
    keys["ntok"] = keys["title"].str.split().str.len().fillna(0).astype(int)
    return keys


def _guard(keys: pd.DataFrame, a: int, b: int) -> bool:
    """True if a and b may be merged by a title-based rule."""
    if keys.at[a, "provider"] == keys.at[b, "provider"]:
        ua, ub = keys.at[a, "url"], keys.at[b, "url"]
        if ua and ub and ua != ub:
            return False
    return True


def find_duplicates(keys: pd.DataFrame, settings: Settings) -> tuple[UnionFind, list[tuple[int, int, str]]]:
    uf = UnionFind(len(keys))
    merges: list[tuple[int, int, str]] = []

    def link(a, b, rule):
        if uf.union(a, b):
            merges.append((a, b, rule))

    # 1. URL
    for _, idx in keys[keys["url"] != ""].groupby("url").groups.items():
        idx = list(idx)
        for other in idx[1:]:
            link(idx[0], other, "normalized_url")

    # 2. title + timestamp
    tol = settings.dedup_time_tolerance_minutes * 60
    sorted_keys = keys[keys["title"] != ""].sort_values(["title", "ts"])
    for _, grp in sorted_keys.groupby("title", sort=False):
        if len(grp) < 2:
            continue
        rows = list(zip(grp.index, grp["ts"]))
        for i in range(1, len(rows)):
            for j in range(i - 1, -1, -1):
                if rows[i][1] - rows[j][1] > tol:
                    break
                if _guard(keys, rows[i][0], rows[j][0]):
                    link(rows[j][0], rows[i][0], "title_timestamp")
                    break

    # 3. title + publisher
    win = settings.dedup_title_source_window_hours * 3600
    sub = keys[(keys["title"] != "") & (keys["pub"] != "") & (keys["ntok"] >= 5)].sort_values(["title", "pub", "ts"])
    for _, grp in sub.groupby(["title", "pub"], sort=False):
        if len(grp) < 2:
            continue
        rows = list(zip(grp.index, grp["ts"]))
        for i in range(1, len(rows)):
            if rows[i][1] - rows[i - 1][1] <= win and _guard(keys, rows[i][0], rows[i - 1][0]):
                link(rows[i - 1][0], rows[i][0], "title_publisher")

    # 4. fuzzy title within publisher + time window
    fwin = settings.fuzzy_window_minutes * 60
    thr = settings.fuzzy_title_threshold
    sub = keys[(keys["pub"] != "") & (keys["ntok"] >= settings.min_title_tokens_for_fuzzy)].sort_values(["pub", "ts"])
    for _, grp in sub.groupby("pub", sort=False):
        idx, ts, titles = grp.index.to_numpy(), grp["ts"].to_numpy(), grp["title"].tolist()
        left = 0
        for i in range(len(idx)):
            while ts[i] - ts[left] > fwin:
                left += 1
            for j in range(left, i):
                if uf.find(idx[i]) == uf.find(idx[j]) or titles[i] == titles[j]:
                    continue
                if fuzz.token_sort_ratio(titles[i], titles[j], score_cutoff=thr) and _guard(keys, idx[i], idx[j]):
                    link(idx[j], idx[i], f"fuzzy_title>={thr:g}")
    return uf, merges


def deduplicate(settings: Settings, provider_counts: dict | None = None) -> dict:
    keys = _load_keys(settings)
    n = len(keys)
    uf, merges = find_duplicates(keys, settings)
    roots = np.array([uf.find(i) for i in range(n)]) if n else np.array([], dtype=int)

    # Canonical record per cluster: has description > alpaca (structured symbols) > earliest timestamp.
    keys["root"] = roots
    keys["_rank_desc"] = (~keys["has_desc"]).astype(int)
    keys["_rank_prov"] = (keys["provider"] != "alpaca").astype(int)
    order = keys.sort_values(["root", "_rank_desc", "_rank_prov", "ts"])
    canonical = order.groupby("root").head(1)
    canonical_rows = set(canonical.index.tolist())
    root_to_canon = dict(zip(canonical["root"], canonical.index))

    # Merge metadata from all members of each cluster.
    cluster_info: dict[int, dict] = defaultdict(lambda: {"providers": set(), "av_ids": [], "alpaca_ids": [],
                                                         "ts_av": None, "ts_alpaca": None, "n": 0})
    for i, (root, prov, pid, ts) in enumerate(zip(roots, keys["provider"], keys["provider_id"], keys["ts"])):
        info = cluster_info[root]
        info["providers"].add(prov)
        info["n"] += 1
        if prov == "alpha_vantage":
            info["av_ids"].append(pid)
            info["ts_av"] = ts if info["ts_av"] is None else min(info["ts_av"], ts)
        else:
            info["alpaca_ids"].append(pid)
            info["ts_alpaca"] = ts if info["ts_alpaca"] is None else min(info["ts_alpaca"], ts)

    # Stream full rows: collect per-cluster api tickers, then write canonical rows.
    tickers_by_root: dict[int, dict[str, set]] = defaultdict(lambda: {"alpha_vantage": set(), "alpaca": set()})
    av_sentiment_by_root: dict[int, dict] = {}
    offset = 0
    for chunk in iter_csv_chunks(settings.raw_news_path, settings.chunk_size):
        chunk = chunk.fillna("")
        for k, (prov, tick, sent, score, label) in enumerate(zip(
                chunk["news_provider"], chunk["api_tickers"], chunk["av_ticker_sentiment"],
                chunk["overall_sentiment_score"], chunk["overall_sentiment_label"])):
            root = roots[offset + k]
            try:
                tickers_by_root[root][prov].update(json.loads(tick or "[]"))
            except json.JSONDecodeError:
                pass
            if prov == "alpha_vantage" and root not in av_sentiment_by_root:
                av_sentiment_by_root[root] = {"av_ticker_sentiment": sent, "overall_sentiment_score": score,
                                              "overall_sentiment_label": label}
        offset += len(chunk)

    writer = AtomicCSVWriter(settings.dedup_news_path)
    try:
        offset = 0
        for chunk in iter_csv_chunks(settings.raw_news_path, settings.chunk_size):
            chunk = chunk.fillna("")
            pos = np.arange(offset, offset + len(chunk))
            offset += len(chunk)
            keep_mask = np.array([p in canonical_rows for p in pos])
            out = chunk[keep_mask].copy()
            if not len(out):
                continue
            out_roots = roots[pos[keep_mask]]
            infos = [cluster_info[r] for r in out_roots]
            out["alpha_vantage_id"] = [";".join(sorted(set(i["av_ids"]))) for i in infos]
            out["alpaca_id"] = [";".join(sorted(set(i["alpaca_ids"]))) for i in infos]
            out["news_providers_all"] = [";".join(sorted(i["providers"])) for i in infos]
            out["duplicate_count"] = [i["n"] - 1 for i in infos]
            out["published_utc_alpha_vantage"] = [_fmt(i["ts_av"]) for i in infos]
            out["published_utc_alpaca"] = [_fmt(i["ts_alpaca"]) for i in infos]
            out["api_tickers_alpha_vantage"] = [json.dumps(sorted(tickers_by_root[r]["alpha_vantage"])) for r in out_roots]
            out["api_tickers_alpaca"] = [json.dumps(sorted(tickers_by_root[r]["alpaca"])) for r in out_roots]
            for col in ("av_ticker_sentiment", "overall_sentiment_score", "overall_sentiment_label"):
                out[col] = [av_sentiment_by_root.get(r, {}).get(col, v) for r, v in zip(out_roots, out[col])]
            writer.write(out)
        writer.commit()
    except BaseException:
        writer.abort()
        raise

    # Audit trail of every merge.
    mp = pd.DataFrame([{
        "removed_news_id": keys.at[b, "news_id"], "removed_provider": keys.at[b, "provider"],
        "removed_provider_id": keys.at[b, "provider_id"],
        "kept_news_id": keys.at[root_to_canon[roots[b]], "news_id"], "rule": rule,
    } for a, b, rule in merges])
    atomic_write_df(mp if len(mp) else pd.DataFrame(columns=["removed_news_id", "removed_provider",
                                                             "removed_provider_id", "kept_news_id", "rule"]),
                    settings.processed_dir / "duplicate_map.csv")

    by_rule = {}
    for _, _, rule in merges:
        by_rule[rule] = by_rule.get(rule, 0) + 1
    stats = {
        "alpha_vantage_raw_count": (provider_counts or {}).get("alpha_vantage_raw_count"),
        "alpaca_raw_count": (provider_counts or {}).get("alpaca_raw_count"),
        "valid_news_before_dedup": n,
        "valid_alpha_vantage": int((keys["provider"] == "alpha_vantage").sum()) if n else 0,
        "valid_alpaca": int((keys["provider"] == "alpaca").sum()) if n else 0,
        "duplicates_removed": n - len(canonical_rows),
        "duplicates_by_rule": by_rule,
        "cross_provider_clusters": sum(1 for i in cluster_info.values() if len(i["providers"]) > 1),
        "final_unique_news_count": len(canonical_rows),
    }
    atomic_write_text(settings.processed_dir / "dedup_stats.json", json.dumps(stats, indent=2))
    return stats


def _fmt(ts) -> str:
    if ts is None:
        return ""
    return pd.Timestamp(int(ts), unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def print_stats(stats: dict) -> None:
    print("\n=== De-duplication statistics ===")
    print(f"Alpha Vantage raw count : {stats.get('alpha_vantage_raw_count')}")
    print(f"Alpaca raw count        : {stats.get('alpaca_raw_count')}")
    print(f"Valid before dedup      : {stats.get('valid_news_before_dedup')}")
    print(f"Duplicates removed      : {stats.get('duplicates_removed')}  {stats.get('duplicates_by_rule')}")
    print(f"Cross-provider matches  : {stats.get('cross_provider_clusters')}")
    print(f"Final unique news count : {stats.get('final_unique_news_count')}\n")
