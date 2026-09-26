"""Alpaca historical news collector (GET /v1beta1/news).

The date range is split into chunks (ALPACA_NEWS_CHUNK_DAYS). Each chunk is
paginated with `next_page_token`. After every page the rows are appended to
data/raw/alpaca_news.csv and the chunk's page token + the committed file size
are saved atomically, so an interrupted run resumes at the exact page.

Chunks run in one flat thread pool (MAX_WORKERS) that shares a single rate
limiter - there are no nested pools.
"""
from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from collectors.api_utils import ApiError, FailedRequestLog, HttpClient, RateLimiter, StopRequested, STOP_EVENT
from config.settings import Settings
from storage.checkpoint import Checkpoint
from storage.csv_writer import IncrementalCSVWriter, read_column_set

log = logging.getLogger(__name__)

RAW_COLUMNS = [
    "news_provider", "alpaca_id", "headline", "summary", "author", "created_at", "updated_at", "url",
    "source", "symbols", "images", "content", "request_chunk", "collected_at_utc",
]


def alpaca_headers(settings: Settings) -> dict:
    return {"APCA-API-KEY-ID": settings.alpaca_api_key, "APCA-API-SECRET-KEY": settings.alpaca_secret_key}


def make_alpaca_client(settings: Settings, failed_log: FailedRequestLog, limiter: RateLimiter | None = None) -> HttpClient:
    return HttpClient("alpaca", limiter or RateLimiter(settings.alpaca_requests_per_minute, 60.0),
                      settings.request_timeout, settings.max_retries, settings.backoff_base_seconds,
                      settings.backoff_max_seconds, headers=alpaca_headers(settings), failed_log=failed_log)


def to_raw_row(item: dict, chunk_key: str, collected_at: str, include_content: bool) -> dict:
    return {
        "news_provider": "alpaca",
        "alpaca_id": str(item.get("id", "")),
        "headline": item.get("headline", ""),
        "summary": item.get("summary", ""),
        "author": item.get("author", ""),
        "created_at": item.get("created_at", ""),
        "updated_at": item.get("updated_at", ""),
        "url": item.get("url", ""),
        "source": item.get("source", ""),
        "symbols": json.dumps(item.get("symbols") or []),
        "images": json.dumps(item.get("images") or []),
        "content": item.get("content", "") if include_content else "",
        "request_chunk": chunk_key,
        "collected_at_utc": collected_at,
    }


def _rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class AlpacaNewsCollector:
    def __init__(self, settings: Settings, failed_log: FailedRequestLog, limiter: RateLimiter | None = None):
        self.s = settings
        self.client = make_alpaca_client(settings, failed_log, limiter)
        self.cp = Checkpoint(settings.checkpoint_dir / "alpaca_news.json")
        self.commit_lock = threading.Lock()
        self.url = f"{settings.alpaca_data_url}/v1beta1/news"

    def _chunks(self) -> list[tuple[str, datetime, datetime]]:
        start = self.s.start_utc
        end = min(self.s.end_utc_exclusive, datetime.now(timezone.utc))
        step = timedelta(days=self.s.alpaca_news_chunk_days)
        chunks, cur = [], start
        while cur < end:
            nxt = min(cur + step, end)
            key = f"{cur:%Y-%m-%dT%H%M}_{nxt:%Y-%m-%dT%H%M}"
            chunks.append((key, cur, nxt))
            cur = nxt
        return chunks

    def _signature(self) -> str:
        return json.dumps({"symbols": self.s.alpaca_news_symbols, "chunk_days": self.s.alpaca_news_chunk_days,
                           "content": self.s.alpaca_news_include_content})

    def _commit(self, writer: IncrementalCSVWriter, rows: list[dict], chunk_key: str, token: str | None,
                done: bool, seen: set[str]) -> int:
        """Write a page and advance the chunk state as one unit."""
        with self.commit_lock:
            new_rows = [r for r in rows if r["alpaca_id"] not in seen]
            seen.update(r["alpaca_id"] for r in new_rows)
            size = writer.write_rows(new_rows) if new_rows else writer.size()
            chunks = self.cp.get("chunks", {})
            state = chunks.get(chunk_key, {"count": 0})
            state.update({"status": "done" if done else "in_progress", "page_token": token,
                          "count": state.get("count", 0) + len(new_rows)})
            chunks[chunk_key] = state
            self.cp.update({"chunks": chunks, "committed_size": size})
            return len(new_rows)

    def _run_chunk(self, writer, chunk_key: str, start: datetime, end: datetime, seen: set[str]) -> int:
        state = self.cp.get("chunks", {}).get(chunk_key, {})
        token = state.get("page_token")
        written = 0
        while True:
            if STOP_EVENT.is_set():
                raise StopRequested()
            params = {"start": _rfc3339(start), "end": _rfc3339(end), "limit": 50, "sort": "asc",
                      "include_content": str(self.s.alpaca_news_include_content).lower(),
                      "exclude_contentless": "false"}
            if self.s.alpaca_news_symbols:
                params["symbols"] = ",".join(self.s.alpaca_news_symbols)
            if token:
                params["page_token"] = token
            payload = self.client.get_json(self.url, params, context=f"news chunk {chunk_key}")
            items = payload.get("news", []) or []
            token = payload.get("next_page_token")
            collected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            rows = [to_raw_row(i, chunk_key, collected_at, self.s.alpaca_news_include_content) for i in items]
            written += self._commit(writer, rows, chunk_key, token, done=not token, seen=seen)
            if not token:
                return written

    def run(self) -> dict:
        if not (self.s.alpaca_api_key and self.s.alpaca_secret_key):
            log.error("ALPACA_API_KEY / ALPACA_SECRET_KEY missing in .env - skipping Alpaca news.")
            return {"status": "skipped_no_key"}
        if self.cp.get("signature") not in (None, self._signature()):
            log.warning("Alpaca news settings changed - re-scanning chunks (existing rows kept, de-duplicated by id).")
            self.cp.update({"chunks": {}})
        self.cp.set("signature", self._signature())

        writer = IncrementalCSVWriter(self.s.alpaca_news_raw_path, RAW_COLUMNS,
                                      committed_size=self.cp.get("committed_size"))
        seen = read_column_set(self.s.alpaca_news_raw_path, "alpaca_id")
        done = {k for k, v in self.cp.get("chunks", {}).items() if v.get("status") == "done"}
        pending = [c for c in self._chunks() if c[0] not in done]
        log.info("Alpaca news: %d chunks pending (%d done), %d articles already on disk",
                 len(pending), len(done), len(seen))

        status, failures, total_new = "completed", 0, 0
        with ThreadPoolExecutor(max_workers=self.s.max_workers, thread_name_prefix="alpaca-news") as pool:
            futures = {pool.submit(self._run_chunk, writer, key, a, b, seen): key for key, a, b in pending}
            try:
                for fut in as_completed(futures):
                    key = futures[fut]
                    try:
                        n = fut.result()
                        total_new += n
                        log.info("Alpaca news chunk %s done: %d new articles (total on disk %d)", key, n, len(seen))
                    except StopRequested:
                        status = "interrupted"
                    except (ApiError, OSError) as exc:
                        failures += 1
                        log.error("Alpaca news chunk %s failed: %s (will resume from its last saved page on "
                                  "the next run)", key, exc)
            except KeyboardInterrupt:
                STOP_EVENT.set()
                pool.shutdown(wait=True, cancel_futures=True)
                raise
            if STOP_EVENT.is_set():
                status = "interrupted"
                pool.shutdown(wait=True, cancel_futures=True)
        if failures and status == "completed":
            status = "partial"
        return {"status": status, "new_articles": total_new, "articles_on_disk": len(seen),
                "failed_chunks": failures}
