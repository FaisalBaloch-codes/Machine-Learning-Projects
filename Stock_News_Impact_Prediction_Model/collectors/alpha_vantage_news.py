"""Alpha Vantage NEWS_SENTIMENT collector.

The endpoint returns at most `limit` (<=1000) articles per call and has no page
token. We paginate with a time cursor: each work item is a (time_from, time_to)
window requested with sort=EARLIEST. If a response is saturated (== limit), the
window is re-queued starting at the minute of the last returned article, so the
remaining articles are fetched by the next call. Duplicates caused by the
overlapping minute are removed via the article id.

Timestamps: Alpha Vantage returns naive strings like 20240924T143217. They are
stored untouched in `time_published`; conversion to UTC happens in
processing/cleaner.py (see ALPHA_VANTAGE_TIMEZONE).
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone

from collectors.api_utils import (ApiError, FailedRequestLog, HttpClient, RateLimiter, StopRequested,
                                  interruptible_sleep)
from config.settings import Settings
from storage.checkpoint import Checkpoint
from storage.csv_writer import IncrementalCSVWriter, read_column_set

log = logging.getLogger(__name__)

BASE_URL = "https://www.alphavantage.co/query"
AV_TIME_FMT = "%Y%m%dT%H%M"

RAW_COLUMNS = [
    "news_provider", "alpha_vantage_id", "title", "url", "time_published", "authors", "summary",
    "banner_image", "source", "category_within_source", "source_domain", "topics",
    "overall_sentiment_score", "overall_sentiment_label", "ticker_sentiment",
    "request_time_from", "request_time_to", "collected_at_utc",
]


def alpha_vantage_id(url: str, title: str, time_published: str) -> str:
    """Alpha Vantage has no article id; derive a stable one from the article URL."""
    basis = url.strip() if url else f"{title}|{time_published}"
    return "av_" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:20]


class DailyQuotaExhausted(Exception):
    pass


def classify_message(payload: dict) -> tuple[str, str] | None:
    """Alpha Vantage signals limits/errors with HTTP 200 + a message field."""
    for key in ("Information", "Note", "Error Message"):
        if key in payload:
            msg = str(payload[key])
            low = msg.lower()
            if key == "Error Message" or "invalid" in low:
                return "error", msg
            if "per day" in low or "daily" in low or "requests per day" in low:
                return "daily_quota", msg
            if "per minute" in low or "per second" in low or "frequency" in low or "burst" in low or "rate limit" in low:
                return "minute_limit", msg
            if "premium" in low:
                return "daily_quota", msg
            return "error", msg
    return None


def to_raw_row(item: dict, window: tuple[str, str], collected_at: str) -> dict:
    url = item.get("url", "") or ""
    title = item.get("title", "") or ""
    tp = item.get("time_published", "") or ""
    return {
        "news_provider": "alpha_vantage",
        "alpha_vantage_id": alpha_vantage_id(url, title, tp),
        "title": title,
        "url": url,
        "time_published": tp,
        "authors": json.dumps(item.get("authors") or [], ensure_ascii=False),
        "summary": item.get("summary", ""),
        "banner_image": item.get("banner_image", ""),
        "source": item.get("source", ""),
        "category_within_source": item.get("category_within_source", ""),
        "source_domain": item.get("source_domain", ""),
        "topics": json.dumps(item.get("topics") or [], ensure_ascii=False),
        "overall_sentiment_score": item.get("overall_sentiment_score", ""),
        "overall_sentiment_label": item.get("overall_sentiment_label", ""),
        "ticker_sentiment": json.dumps(item.get("ticker_sentiment") or [], ensure_ascii=False),
        "request_time_from": window[0],
        "request_time_to": window[1],
        "collected_at_utc": collected_at,
    }


class AlphaVantageNewsCollector:
    def __init__(self, settings: Settings, failed_log: FailedRequestLog):
        self.s = settings
        self.failed_log = failed_log
        self.client = HttpClient(
            "alpha_vantage", RateLimiter(settings.av_requests_per_minute, 60.0), settings.request_timeout,
            settings.max_retries, settings.backoff_base_seconds, settings.backoff_max_seconds,
            failed_log=failed_log,
        )
        self.cp = Checkpoint(settings.checkpoint_dir / "alpha_vantage_news.json")

    # ------------------------------------------------------------ work queue
    def _initial_windows(self) -> list[list[str]]:
        # Pad by one day on each side: AV's time zone is resolved later, and the
        # cleaner filters to the exact UTC range.
        start = datetime.combine(self.s.start_date, datetime.min.time()) - timedelta(days=1)
        end = datetime.combine(self.s.end_date, datetime.min.time()) + timedelta(days=2)
        now = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=1)
        end = min(end, now)
        step = timedelta(hours=max(1, self.s.av_window_hours))
        windows, cur = [], start
        while cur < end:
            nxt = min(cur + step, end)
            windows.append([cur.strftime(AV_TIME_FMT), nxt.strftime(AV_TIME_FMT)])
            cur = nxt
        return windows

    def _query_signature(self) -> str:
        return json.dumps({"start": str(self.s.start_date), "end": str(self.s.end_date),
                           "tickers": self.s.av_tickers, "topics": self.s.av_topics,
                           "window_hours": self.s.av_window_hours})

    def _load_queue(self) -> list[list]:
        if self.cp.get("query_signature") != self._query_signature() or "queue" not in self.cp.data:
            if self.cp.get("queue"):
                log.warning("Alpha Vantage query settings changed - rebuilding work queue "
                            "(existing raw rows are kept and de-duplicated by id).")
            queue = [[a, b, 0] for a, b in self._initial_windows()]
            self.cp.update({"query_signature": self._query_signature(), "queue": queue,
                            "completed_windows": 0, "saturated_windows": 0, "articles_written": 0})
        return self.cp.get("queue")

    # ------------------------------------------------------------ requests
    def _count_daily_request(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily = self.cp.get("daily_requests", {})
        daily = {today: daily.get(today, 0) + 1}
        self.cp.set("daily_requests", daily, save=False)

    def _requests_today(self) -> int:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.cp.get("daily_requests", {}).get(today, 0)

    def _fetch(self, time_from: str, time_to: str) -> list[dict]:
        params = {"function": "NEWS_SENTIMENT", "time_from": time_from, "time_to": time_to,
                  "sort": "EARLIEST", "limit": self.s.av_limit, "apikey": self.s.alpha_vantage_api_key}
        if self.s.av_tickers:
            params["tickers"] = ",".join(self.s.av_tickers)
        if self.s.av_topics:
            params["topics"] = ",".join(self.s.av_topics)
        context = f"window {time_from}-{time_to}"
        for minute_retry in range(5):
            if self.s.av_daily_request_limit and self._requests_today() >= self.s.av_daily_request_limit:
                raise DailyQuotaExhausted(f"local AV_DAILY_REQUEST_LIMIT={self.s.av_daily_request_limit} reached")
            self._count_daily_request()
            payload = self.client.get_json(BASE_URL, params, context=context)
            kind = classify_message(payload) if isinstance(payload, dict) else ("error", "non-object response")
            if kind is None:
                return payload.get("feed", []) or []
            code, msg = kind
            if code == "daily_quota":
                raise DailyQuotaExhausted(msg)
            if code == "minute_limit":
                log.warning("Alpha Vantage per-minute limit hit, sleeping 65s: %s", msg[:150])
                interruptible_sleep(65)
                continue
            if "no articles" in msg.lower():
                return []
            raise ApiError(f"Alpha Vantage error: {msg}", 200, retryable=False, body=msg)
        raise ApiError("Alpha Vantage per-minute limit persisted after 5 waits", 429, retryable=True)

    # ------------------------------------------------------------------- run
    def run(self) -> dict:
        if not self.s.alpha_vantage_api_key:
            log.error("ALPHA_VANTAGE_API_KEY missing in .env - skipping Alpha Vantage news.")
            return {"status": "skipped_no_key"}

        writer = IncrementalCSVWriter(self.s.alpha_vantage_raw_path, RAW_COLUMNS,
                                      committed_size=self.cp.get("committed_size"))
        seen = read_column_set(self.s.alpha_vantage_raw_path, "alpha_vantage_id")
        queue = self._load_queue()
        log.info("Alpha Vantage: %d windows pending, %d articles already on disk", len(queue), len(seen))
        status = "completed"

        while queue:
            time_from, time_to, attempts = queue[0]
            try:
                feed = self._fetch(time_from, time_to)
            except DailyQuotaExhausted as exc:
                log.warning("Alpha Vantage daily quota exhausted (%s). Progress saved - rerun later "
                            "(or upgrade the key) to continue from window %s.", str(exc)[:200], time_from)
                status = "quota_exhausted"
                break
            except StopRequested:
                status = "interrupted"
                break
            except ApiError as exc:
                attempts += 1
                queue.pop(0)
                if attempts < self.s.av_max_attempts_per_window:
                    queue.append([time_from, time_to, attempts])
                    log.error("Alpha Vantage window %s-%s failed (%s); re-queued", time_from, time_to, exc)
                else:
                    failed = self.cp.get("failed_windows", [])
                    failed.append([time_from, time_to, str(exc)[:300]])
                    self.cp.set("failed_windows", failed, save=False)
                    log.error("Alpha Vantage window %s-%s failed permanently (logged in %s)",
                              time_from, time_to, self.s.failed_output)
                self.cp.update({"queue": queue})
                continue

            collected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            rows = []
            for item in feed:
                row = to_raw_row(item, (time_from, time_to), collected_at)
                if row["alpha_vantage_id"] in seen:
                    continue
                seen.add(row["alpha_vantage_id"])
                rows.append(row)
            size = writer.write_rows(rows) if rows else writer.size()

            queue.pop(0)
            saturated = len(feed) >= self.s.av_limit
            if saturated:
                last = max((i.get("time_published", "") for i in feed), default="")
                next_from = last[:13] if len(last) >= 13 else time_from
                if next_from <= time_from:
                    next_from = (datetime.strptime(time_from, AV_TIME_FMT) + timedelta(minutes=1)).strftime(AV_TIME_FMT)
                if next_from < time_to:
                    queue.insert(0, [next_from, time_to, 0])
                self.cp.set("saturated_windows", self.cp.get("saturated_windows", 0) + 1, save=False)
            else:
                self.cp.set("completed_windows", self.cp.get("completed_windows", 0) + 1, save=False)
            self.cp.update({"queue": queue, "committed_size": size,
                            "articles_written": self.cp.get("articles_written", 0) + len(rows)})
            log.info("Alpha Vantage %s-%s: %d returned, %d new%s | %d windows left",
                     time_from, time_to, len(feed), len(rows), " (saturated, continuing)" if saturated else "",
                     len(queue))

        if not queue and status == "completed":
            self.cp.set("finished", True)
        return {"status": status, "articles_on_disk": len(seen), "windows_left": len(queue),
                "failed_windows": len(self.cp.get("failed_windows", []))}
