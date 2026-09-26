"""Stock-news impact dataset pipeline.

    python main.py                 # full pipeline (resumes from checkpoints)
    python main.py --news          # collect Alpha Vantage + Alpaca news, clean, de-duplicate
    python main.py --dedup         # re-run cleaning + de-duplication only
    python main.py --ner           # ticker identification
    python main.py --market        # Alpaca 1-minute bars + alignment + reaction features
    python main.py --label         # spike / crash / neutral labels
    python main.py --build         # final CSV + quality report
    python main.py --inspect       # summary + plots
    python main.py --check-api     # verify .env credentials against both APIs
    python main.py --ner --force   # redo a stage (old outputs are archived, never deleted)

Press Ctrl+C once to stop safely; rerun the same command to continue.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collectors.api_utils import (STOP_EVENT, ApiError, FailedRequestLog, RateLimiter, StopRequested,  # noqa: E402
                                  install_signal_handlers)
from config.settings import Settings, load_settings  # noqa: E402
from storage.checkpoint import Checkpoint, atomic_write_text  # noqa: E402
from storage.csv_writer import archive_file  # noqa: E402

log = logging.getLogger("pipeline")


def setup_logging(settings: Settings) -> Path:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    path = settings.log_dir / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return path


class Pipeline:
    def __init__(self, settings: Settings, force: bool = False):
        self.s = settings
        self.force = force
        self.failed_log = FailedRequestLog(settings.failed_requests_path)
        self.alpaca_limiter = RateLimiter(settings.alpaca_requests_per_minute, 60.0)  # shared by all Alpaca calls
        self.summary: dict = {}
        self._calendar = None

    # ------------------------------------------------------------ helpers
    def _alpaca_client(self):
        from collectors.alpaca_news import make_alpaca_client

        return make_alpaca_client(self.s, self.failed_log, self.alpaca_limiter)

    def _require_alpaca(self) -> bool:
        if not (self.s.alpaca_api_key and self.s.alpaca_secret_key):
            log.error("ALPACA_API_KEY / ALPACA_SECRET_KEY are missing in .env")
            return False
        return True

    def calendar(self):
        if self._calendar is None:
            from collectors.alpaca_reference import TradingCalendar, fetch_calendar

            client = self._alpaca_client() if self._require_alpaca() else None
            self._calendar = TradingCalendar(fetch_calendar(self.s, client))
        return self._calendar

    def reference(self):
        import pandas as pd

        from collectors.alpaca_reference import fetch_assets
        from nlp.ticker_ner import TickerReference

        path = self.s.reference_dir / "us_equities.csv"
        if path.exists():
            assets = pd.read_csv(path, dtype=str, keep_default_na=False)
        else:
            if not self._require_alpaca():
                raise RuntimeError("Alpaca keys are needed once to download the US-equity reference list")
            assets = fetch_assets(self.s, self._alpaca_client())
        return TickerReference(assets, self.s)

    # ------------------------------------------------------------- stages
    def stage_news(self, skip_av: bool, skip_alpaca: bool) -> None:
        from collectors.alpaca_news import AlpacaNewsCollector
        from collectors.alpha_vantage_news import AlphaVantageNewsCollector

        if self.force:
            for name in ("alpha_vantage_news.json", "alpaca_news.json"):
                Checkpoint(self.s.checkpoint_dir / name).reset()
            log.info("--force: news checkpoints reset; raw files are kept and new rows de-duplicated by id")
        if not skip_alpaca:
            log.info("=== Stage: Alpaca news ===")
            self.summary["alpaca_news"] = AlpacaNewsCollector(self.s, self.failed_log, self.alpaca_limiter).run()
        if STOP_EVENT.is_set():
            return
        if not skip_av:
            log.info("=== Stage: Alpha Vantage news ===")
            self.summary["alpha_vantage_news"] = AlphaVantageNewsCollector(self.s, self.failed_log).run()

    def stage_dedup(self, required: bool = True) -> None:
        from processing.cleaner import build_raw_news
        from processing.deduplicator import deduplicate, print_stats

        raws = [p for p in (self.s.alpha_vantage_raw_path, self.s.alpaca_news_raw_path) if p.exists()]
        if not raws:
            log.warning("No raw news files yet - run `python main.py --news` first.")
            return
        out = self.s.dedup_news_path
        newest_raw = max(p.stat().st_mtime for p in raws)
        if not required and out.exists() and out.stat().st_mtime >= newest_raw:
            log.info("De-duplicated news is up to date - skipping cleaning/dedup.")
            return
        log.info("=== Stage: cleaning + de-duplication ===")
        clean_stats = build_raw_news(self.s)
        stats = deduplicate(self.s, clean_stats)
        stats["rejected_by_reason"] = clean_stats["rejected"]
        stats["alpha_vantage_timezone"] = clean_stats["alpha_vantage_timezone"]
        print_stats(stats)
        self.summary["dedup"] = stats

    def stage_ner(self) -> None:
        from nlp.ticker_ner import run_ner

        if not self.s.dedup_news_path.exists():
            log.warning("No de-duplicated news - run --news first.")
            return
        if self.force:
            archive_file(self.s.ticker_events_path)
            archive_file(self.s.checkpoint_dir / "ner_processed.csv")
            Checkpoint(self.s.checkpoint_dir / "ner.json").reset()
        log.info("=== Stage: ticker identification (NER) ===")
        self.summary["ner"] = run_ner(self.s, self.reference())

    def stage_market(self) -> None:
        from collectors.alpaca_market import AlpacaMarketData
        from processing.market_features import run_market_stage

        if not self.s.ticker_events_path.exists():
            log.warning("No ticker events - run --ner first.")
            return
        if not self._require_alpaca():
            return
        if self.force:
            archive_file(self.s.market_features_path)
            Checkpoint(self.s.checkpoint_dir / "market_features.json").reset()
            log.info("--force: recomputing market features (cached 1-minute bars are reused)")
        log.info("=== Stage: Alpaca 1-minute market data + reaction features ===")
        cal = self.calendar()
        market = AlpacaMarketData(self.s, cal, self.failed_log, self.alpaca_limiter)
        self.summary["market"] = run_market_stage(self.s, cal, market)

    def stage_label(self) -> None:
        from labeling.impact_labeler import run_label_stage

        if self.force:
            Checkpoint(self.s.checkpoint_dir / "labeling.json").reset()
        log.info("=== Stage: impact labelling ===")
        self.summary["label"] = run_label_stage(self.s)

    def stage_build(self) -> None:
        from processing.dataset_builder import build_dataset

        if not self.s.ticker_events_path.exists():
            log.warning("Nothing to build yet.")
            return
        log.info("=== Stage: final dataset + quality report ===")
        report = build_dataset(self.s)
        self.summary["build"] = report
        print("\n=== Data quality report ===")
        for k in ("raw_news_count", "deduplicated_news_count", "duplicate_count", "ticker_events",
                  "market_matched_events", "missing_market_data", "spike_count", "crash_count", "neutral_count",
                  "unlabelled_count", "invalid_count", "usable_labelled_events"):
            print(f"{k:28s}: {report.get(k)}")
        print(report["note"], "\n")

    def stage_inspect(self) -> None:
        from processing.inspect_dataset import inspect

        inspect(self.s)

    def check_api(self) -> bool:
        ok = True
        print("\nCredential check (values are never printed):")
        for name, val in (("ALPHA_VANTAGE_API_KEY", self.s.alpha_vantage_api_key),
                          ("ALPACA_API_KEY", self.s.alpaca_api_key), ("ALPACA_SECRET_KEY", self.s.alpaca_secret_key)):
            print(f"  {name:22s}: {'set' if val else 'MISSING'}")
            ok &= bool(val)
        if self.s.alpaca_api_key and self.s.alpaca_secret_key:
            client = self._alpaca_client()
            for label, url, params in (
                    ("Alpaca news API", f"{self.s.alpaca_data_url}/v1beta1/news", {"limit": 1}),
                    ("Alpaca 1-min bars API", f"{self.s.alpaca_data_url}/v2/stocks/bars",
                     {"symbols": "AAPL", "timeframe": "1Min", "start": "2025-01-06T15:00:00Z",
                      "end": "2025-01-06T15:05:00Z", "feed": self.s.alpaca_data_feed}),
                    ("Alpaca trading API (calendar/assets)", f"{self.s.trading_url}/v2/calendar",
                     {"start": "2025-01-06", "end": "2025-01-07"})):
                try:
                    client.get_json(url, params, context="check-api", record_failure=False)
                    print(f"  {label:38s}: OK")
                except ApiError as exc:
                    ok = False
                    print(f"  {label:38s}: FAILED ({exc.status}) {str(exc)[:120]}")
        if self.s.alpha_vantage_api_key:
            from collectors.alpha_vantage_news import BASE_URL, classify_message
            from collectors.api_utils import HttpClient

            client = HttpClient("alpha_vantage", RateLimiter(5), self.s.request_timeout, 1, 1, 5)
            try:
                payload = client.get_json(BASE_URL, {"function": "NEWS_SENTIMENT", "limit": 1,
                                                     "apikey": self.s.alpha_vantage_api_key}, record_failure=False)
                msg = classify_message(payload)
                if msg:
                    ok = False
                    print(f"  {'Alpha Vantage NEWS_SENTIMENT':38s}: responded with {msg[0]}: {msg[1][:120]}")
                else:
                    print(f"  {'Alpha Vantage NEWS_SENTIMENT':38s}: OK (used 1 request of your quota)")
            except ApiError as exc:
                ok = False
                print(f"  {'Alpha Vantage NEWS_SENTIMENT':38s}: FAILED {str(exc)[:120]}")
        print()
        return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect and label a US stock news-impact dataset.")
    for flag, help_ in (("news", "collect news from Alpha Vantage and Alpaca (+ clean/dedup)"),
                        ("dedup", "re-run cleaning and de-duplication"), ("ner", "ticker identification"),
                        ("market", "Alpaca 1-minute bars + reaction features"), ("label", "impact labelling"),
                        ("build", "final dataset + quality report"), ("inspect", "print summary and save plots")):
        parser.add_argument(f"--{flag}", action="store_true", help=help_)
    parser.add_argument("--check-api", action="store_true", help="verify API credentials from .env")
    parser.add_argument("--force", action="store_true", help="redo the selected stages (outputs archived)")
    parser.add_argument("--skip-alpha-vantage", action="store_true", help="do not call Alpha Vantage")
    parser.add_argument("--skip-alpaca-news", action="store_true", help="do not collect Alpaca news")
    parser.add_argument("--env", default=None, help="path to an alternative .env file")
    args = parser.parse_args(argv)

    settings = load_settings(args.env)
    settings.ensure_dirs()
    lock = acquire_run_lock(settings)
    if lock is None:
        print("Another `python main.py` is already running on this data folder. Wait for it to finish "
              "(two runs at once would write the same rows twice).")
        return 2
    try:
        return _run(args, settings, parser)
    finally:
        lock.close()


def acquire_run_lock(settings: Settings):
    """Exclusive OS-level lock, released automatically if the process dies."""
    fh = open(settings.checkpoint_dir / "pipeline.lock", "a+")
    try:
        if os.name == "nt":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def _run(args, settings: Settings, parser) -> int:
    log_path = setup_logging(settings)
    install_signal_handlers()
    log.info("Log file: %s", log_path)
    log.info("Settings: %s", json.dumps(settings.public_dict(), default=str))

    pipe = Pipeline(settings, force=args.force)
    if args.check_api:
        return 0 if pipe.check_api() else 1

    stages = [s for s in ("news", "dedup", "ner", "market", "label", "build", "inspect") if getattr(args, s)]
    run_all = not stages
    if run_all and args.force:
        parser.error("--force must be combined with explicit stage flags, e.g. `--ner --force`")
    t0 = time.time()
    exit_code = 0
    try:
        if run_all or "news" in stages:
            pipe.stage_news(args.skip_alpha_vantage, args.skip_alpaca_news)
        if not STOP_EVENT.is_set() and (run_all or "news" in stages or "dedup" in stages or "ner" in stages):
            pipe.stage_dedup(required="dedup" in stages or "news" in stages)
        if not STOP_EVENT.is_set() and (run_all or "ner" in stages):
            pipe.stage_ner()
        if not STOP_EVENT.is_set() and (run_all or "market" in stages):
            pipe.stage_market()
        if not STOP_EVENT.is_set() and (run_all or "label" in stages):
            pipe.stage_label()
        if not STOP_EVENT.is_set() and (run_all or "build" in stages):
            pipe.stage_build()
        if not STOP_EVENT.is_set() and (run_all or "inspect" in stages):
            pipe.stage_inspect()
    except (KeyboardInterrupt, StopRequested):
        STOP_EVENT.set()
    except Exception:
        log.exception("Pipeline failed")
        exit_code = 1
    finally:
        pipe.summary["elapsed_seconds"] = round(time.time() - t0, 1)
        pipe.summary["interrupted"] = STOP_EVENT.is_set()
        atomic_write_text(settings.processed_dir / "last_run_summary.json",
                          json.dumps(pipe.summary, indent=2, default=str))
    if STOP_EVENT.is_set():
        log.warning("Stopped by user. All completed work is checkpointed - run the same command again to resume.")
        return 130
    log.info("Done in %.1f s", time.time() - t0)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
