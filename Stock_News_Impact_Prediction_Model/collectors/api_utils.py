"""Shared HTTP plumbing: throttling, retries, backoff, graceful stop, failure log."""
from __future__ import annotations

import json
import logging
import random
import signal
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from storage.csv_writer import IncrementalCSVWriter

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
SECRET_PARAM_NAMES = {"apikey", "api_key", "key", "secret", "token"}

# --------------------------------------------------------------------------- stop


STOP_EVENT = threading.Event()


class StopRequested(Exception):
    """Raised inside workers once the user pressed Ctrl+C."""


def install_signal_handlers() -> None:
    """First Ctrl+C: finish the current request, checkpoint, exit. Second: hard stop."""

    def handler(signum, frame):
        if STOP_EVENT.is_set():
            log.warning("Second interrupt received - aborting immediately.")
            raise KeyboardInterrupt
        STOP_EVENT.set()
        log.warning("Interrupt received - finishing in-flight work and saving checkpoints "
                    "(press Ctrl+C again to force quit).")

    signal.signal(signal.SIGINT, handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handler)


def check_stop() -> None:
    if STOP_EVENT.is_set():
        raise StopRequested()


def interruptible_sleep(seconds: float) -> None:
    if STOP_EVENT.wait(max(0.0, seconds)):
        raise StopRequested()


# ---------------------------------------------------------------- rate limiting


class RateLimiter:
    """Sliding-window limiter shared by all threads that talk to one API."""

    def __init__(self, max_calls: int, period_seconds: float = 60.0):
        self.max_calls = max(1, int(max_calls))
        self.period = period_seconds
        self.calls: deque[float] = deque()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            check_stop()
            with self.lock:
                now = time.monotonic()
                while self.calls and now - self.calls[0] >= self.period:
                    self.calls.popleft()
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return
                wait = self.period - (now - self.calls[0]) + 0.01
            interruptible_sleep(wait)


# --------------------------------------------------------------------- failures


class ApiError(Exception):
    def __init__(self, message: str, status: int | None = None, retryable: bool = False, body: str = ""):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.body = body


FAILED_COLUMNS = ["timestamp_utc", "api", "endpoint", "params", "context", "status", "error", "attempts"]


class FailedRequestLog:
    """Every request that ultimately fails is recorded - never silently dropped."""

    def __init__(self, path: Path):
        self.writer = IncrementalCSVWriter(path, FAILED_COLUMNS)

    def record(self, api: str, endpoint: str, params: dict | None, context: str, status: Any,
               error: str, attempts: int) -> None:
        self.writer.write_rows([{
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "api": api,
            "endpoint": endpoint,
            "params": json.dumps(sanitize_params(params or {}), sort_keys=True),
            "context": context,
            "status": status,
            "error": str(error)[:1000],
            "attempts": attempts,
        }])


def sanitize_params(params: dict) -> dict:
    return {k: ("***" if k.lower() in SECRET_PARAM_NAMES else v) for k, v in params.items()}


# ------------------------------------------------------------------ HTTP client


class HttpClient:
    """GET-JSON client with throttling, retries, exponential backoff and jitter.

    Each thread gets its own requests.Session (Sessions are not guaranteed to be
    thread-safe). Credentials are passed via headers/params and never logged.
    """

    def __init__(self, api_name: str, limiter: RateLimiter, timeout: int, max_retries: int,
                 backoff_base: float, backoff_max: float, headers: dict | None = None,
                 failed_log: FailedRequestLog | None = None):
        self.api_name = api_name
        self.limiter = limiter
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.headers = headers or {}
        self.failed_log = failed_log
        self._local = threading.local()
        self.request_count = 0
        self._count_lock = threading.Lock()

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            s = requests.Session()
            s.headers.update({"Accept": "application/json", "User-Agent": "stock-news-impact-dataset/1.0"})
            s.headers.update(self.headers)
            self._local.session = s
        return self._local.session

    def _backoff(self, attempt: int, retry_after: str | None = None) -> float:
        if retry_after:
            try:
                return min(self.backoff_max, float(retry_after))
            except ValueError:
                pass
        delay = min(self.backoff_max, self.backoff_base * (2 ** attempt))
        return delay * (0.75 + random.random() * 0.5)

    def get_json(self, url: str, params: dict | None = None, context: str = "",
                 record_failure: bool = True) -> Any:
        params = params or {}
        last_error: Exception | None = None
        status: Any = None
        for attempt in range(self.max_retries + 1):
            check_stop()
            self.limiter.acquire()
            with self._count_lock:
                self.request_count += 1
            try:
                resp = self._session().get(url, params=params, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error, status = exc, type(exc).__name__
                wait = self._backoff(attempt)
                log.warning("%s %s: network error %s (attempt %d/%d), retrying in %.1fs",
                            self.api_name, context, type(exc).__name__, attempt + 1, self.max_retries + 1, wait)
                interruptible_sleep(wait)
                continue

            status = resp.status_code
            if status == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    last_error = ApiError(f"invalid JSON: {exc}", status, retryable=True, body=resp.text[:500])
                    interruptible_sleep(self._backoff(attempt))
                    continue

            body = resp.text[:500]
            if status in RETRYABLE_STATUS:
                wait = self._backoff(attempt, resp.headers.get("Retry-After"))
                if status == 429:
                    log.warning("%s rate limited (HTTP 429) on %s, backing off %.1fs", self.api_name, context, wait)
                else:
                    log.warning("%s HTTP %s on %s (attempt %d/%d), retrying in %.1fs",
                                self.api_name, status, context, attempt + 1, self.max_retries + 1, wait)
                last_error = ApiError(f"HTTP {status}", status, retryable=True, body=body)
                interruptible_sleep(wait)
                continue

            err = ApiError(f"HTTP {status}: {body}", status, retryable=False, body=body)
            if record_failure and self.failed_log:
                self.failed_log.record(self.api_name, url, params, context, status, err, attempt + 1)
            raise err

        err = ApiError(f"gave up after {self.max_retries + 1} attempts: {last_error}", status, retryable=True)
        if record_failure and self.failed_log:
            self.failed_log.record(self.api_name, url, params, context, status, err, self.max_retries + 1)
        raise err
