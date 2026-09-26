"""Central configuration.

Every tunable value is read from environment variables (loaded from `.env` via
python-dotenv). Nothing secret is hard-coded; API keys only ever live in `.env`.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _int(name: str, default: int) -> int:
    return int(_str(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_str(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return _str(name, str(default)).lower() in {"1", "true", "yes", "y", "on"}


def _int_list(name: str, default: str) -> list[int]:
    return [int(x) for x in _str(name, default).split(",") if x.strip()]


def _str_list(name: str, default: str = "") -> list[str]:
    return [x.strip() for x in _str(name, default).split(",") if x.strip()]


def _date(name: str, default: str) -> date:
    raw = _str(name, default)
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{name} must be YYYY-MM-DD, got {raw!r}") from exc


@dataclass
class Settings:
    # --- credentials (never logged, never written to disk by the pipeline) ---
    alpha_vantage_api_key: str = field(repr=False, default="")
    alpaca_api_key: str = field(repr=False, default="")
    alpaca_secret_key: str = field(repr=False, default="")

    # --- date range (inclusive, interpreted as UTC calendar dates) ---
    start_date: date = date(2024, 9, 24)
    end_date: date = date(2026, 9, 25)

    # --- files ---
    output_dir: Path = PROJECT_ROOT / "data"
    news_output: str = "stock_news_dataset.csv"
    raw_news_output: str = "raw_news.csv"
    market_output: str = "market_data.csv"
    failed_output: str = "failed_requests.csv"
    log_dir: Path = PROJECT_ROOT / "logs"

    # --- generic HTTP behaviour ---
    max_workers: int = 8
    request_timeout: int = 30
    max_retries: int = 6
    backoff_base_seconds: float = 2.0
    backoff_max_seconds: float = 120.0
    chunk_size: int = 5000

    # --- Alpha Vantage ---
    alpha_vantage_timezone: str = "auto"  # auto | UTC | America/New_York
    av_requests_per_minute: int = 5
    av_daily_request_limit: int = 25  # 0 = no local cap (premium keys)
    av_window_hours: int = 168
    av_limit: int = 1000
    av_tickers: list[str] = field(default_factory=list)
    av_topics: list[str] = field(default_factory=list)
    av_max_attempts_per_window: int = 3

    # --- Alpaca ---
    alpaca_paper: bool = True
    alpaca_data_url: str = "https://data.alpaca.markets"
    alpaca_trading_url: str = ""
    alpaca_requests_per_minute: int = 180
    alpaca_news_symbols: list[str] = field(default_factory=list)
    alpaca_news_include_content: bool = False
    alpaca_news_chunk_days: int = 7
    alpaca_data_feed: str = "sip"  # sip | iex
    alpaca_symbols_per_request: int = 50

    # --- NER / ticker extraction ---
    use_spacy: bool = True
    spacy_model: str = "en_core_web_sm"
    min_ticker_confidence: float = 0.55
    max_tickers_per_article: int = 10
    include_etfs: bool = False
    allowed_exchanges: list[str] = field(default_factory=lambda: ["NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"])

    # --- deduplication ---
    dedup_time_tolerance_minutes: int = 10
    dedup_title_source_window_hours: int = 6
    fuzzy_title_threshold: float = 95.0
    fuzzy_window_minutes: int = 120
    min_title_tokens_for_fuzzy: int = 6

    # --- market data / features (1-minute bars only) ---
    post_windows: list[int] = field(default_factory=lambda: [1, 5, 15, 30, 60])
    baseline_lookback_sessions: int = 2
    max_alignment_gap_minutes: int = 5
    volume_window_minutes: int = 15
    min_baseline_bars: int = 60

    # --- labeling ---
    spike_return_threshold: float = 0.02
    crash_return_threshold: float = -0.02
    abnormal_return_z: float = 2.0
    volume_spike_ratio: float = 1.5
    label_horizons: list[int] = field(default_factory=lambda: [5, 15, 30, 60])
    min_abs_return: float = 0.003
    require_volume_confirmation: bool = False
    direction_consistency_min: float = 0.5
    label_include_overnight_gap: bool = True

    # --- ML-ready export ---
    ml_exclude_columns: list[str] = field(default_factory=lambda: [
        "news_id", "event_id", "publisher", "alpha_vantage_id", "alpaca_id"])

    # ------------------------------------------------------------------ paths
    @property
    def raw_dir(self) -> Path:
        return self.output_dir / "raw"

    @property
    def market_dir(self) -> Path:
        return self.raw_dir / "market"

    @property
    def reference_dir(self) -> Path:
        return self.raw_dir / "reference"

    @property
    def processed_dir(self) -> Path:
        return self.output_dir / "processed"

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def plots_dir(self) -> Path:
        return self.processed_dir / "plots"

    @property
    def alpha_vantage_raw_path(self) -> Path:
        return self.raw_dir / "alpha_vantage_news.csv"

    @property
    def alpaca_news_raw_path(self) -> Path:
        return self.raw_dir / "alpaca_news.csv"

    @property
    def raw_news_path(self) -> Path:
        return self.raw_dir / self.raw_news_output

    @property
    def rejected_news_path(self) -> Path:
        return self.processed_dir / "rejected_news.csv"

    @property
    def dedup_news_path(self) -> Path:
        return self.processed_dir / "deduplicated_news.csv"

    @property
    def ticker_events_path(self) -> Path:
        return self.processed_dir / self.news_output

    @property
    def market_features_path(self) -> Path:
        return self.processed_dir / self.market_output

    @property
    def labeled_events_path(self) -> Path:
        return self.processed_dir / "labeled_events.csv"

    @property
    def final_dataset_path(self) -> Path:
        return self.processed_dir / "final_stock_news_impact_dataset.csv"

    @property
    def failed_requests_path(self) -> Path:
        return self.output_dir / self.failed_output

    @property
    def trading_url(self) -> str:
        if self.alpaca_trading_url:
            return self.alpaca_trading_url.rstrip("/")
        return "https://paper-api.alpaca.markets" if self.alpaca_paper else "https://api.alpaca.markets"

    # ---------------------------------------------------------------- helpers
    @property
    def start_utc(self) -> datetime:
        return datetime(self.start_date.year, self.start_date.month, self.start_date.day, tzinfo=timezone.utc)

    @property
    def end_utc_exclusive(self) -> datetime:
        """End of END_DATE (exclusive bound), UTC."""
        from datetime import timedelta

        return datetime(self.end_date.year, self.end_date.month, self.end_date.day, tzinfo=timezone.utc) + timedelta(days=1)

    def ensure_dirs(self) -> None:
        for p in (self.raw_dir, self.market_dir, self.reference_dir, self.processed_dir,
                  self.checkpoint_dir, self.plots_dir, self.log_dir):
            p.mkdir(parents=True, exist_ok=True)

    def section_hash(self, *names: str) -> str:
        """Stable hash of selected settings, used to detect config changes between runs."""
        data = asdict(self)
        payload = {n: data[n] for n in names}
        return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def public_dict(self) -> dict:
        """Settings without credentials, safe to log."""
        data = asdict(self)
        for secret in ("alpha_vantage_api_key", "alpaca_api_key", "alpaca_secret_key"):
            data[secret] = "***set***" if data[secret] else "<missing>"
        return {k: str(v) if isinstance(v, (Path, date)) else v for k, v in data.items()}


LABEL_SETTING_NAMES = (
    "spike_return_threshold", "crash_return_threshold", "abnormal_return_z", "volume_spike_ratio",
    "label_horizons", "min_abs_return", "require_volume_confirmation", "direction_consistency_min",
    "label_include_overnight_gap", "min_baseline_bars",
)


def load_settings(env_file: str | os.PathLike | None = None) -> Settings:
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)

    output_dir = Path(_str("OUTPUT_DIR", "data"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    s = Settings(
        alpha_vantage_api_key=_str("ALPHA_VANTAGE_API_KEY"),
        alpaca_api_key=_str("ALPACA_API_KEY"),
        alpaca_secret_key=_str("ALPACA_SECRET_KEY"),
        start_date=_date("START_DATE", "2024-09-24"),
        end_date=_date("END_DATE", "2026-09-25"),
        output_dir=output_dir,
        log_dir=Path(_str("LOG_DIR", str(PROJECT_ROOT / "logs"))),
        news_output=_str("NEWS_OUTPUT", "stock_news_dataset.csv"),
        raw_news_output=_str("RAW_NEWS_OUTPUT", "raw_news.csv"),
        market_output=_str("MARKET_OUTPUT", "market_data.csv"),
        failed_output=_str("FAILED_OUTPUT", "failed_requests.csv"),
        max_workers=max(1, _int("MAX_WORKERS", 8)),
        request_timeout=_int("REQUEST_TIMEOUT", 30),
        max_retries=_int("MAX_RETRIES", 6),
        backoff_base_seconds=_float("BACKOFF_BASE_SECONDS", 2.0),
        backoff_max_seconds=_float("BACKOFF_MAX_SECONDS", 120.0),
        chunk_size=_int("CHUNK_SIZE", 5000),
        alpha_vantage_timezone=_str("ALPHA_VANTAGE_TIMEZONE", "auto"),
        av_requests_per_minute=_int("AV_REQUESTS_PER_MINUTE", 5),
        av_daily_request_limit=_int("AV_DAILY_REQUEST_LIMIT", 25),
        av_window_hours=_int("AV_WINDOW_HOURS", 168),
        av_limit=min(1000, _int("AV_LIMIT", 1000)),
        av_tickers=_str_list("AV_TICKERS"),
        av_topics=_str_list("AV_TOPICS"),
        av_max_attempts_per_window=_int("AV_MAX_ATTEMPTS_PER_WINDOW", 3),
        alpaca_paper=_bool("ALPACA_PAPER", True),
        alpaca_data_url=_str("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/"),
        alpaca_trading_url=_str("ALPACA_TRADING_URL", ""),
        alpaca_requests_per_minute=_int("ALPACA_REQUESTS_PER_MINUTE", 180),
        alpaca_news_symbols=_str_list("ALPACA_NEWS_SYMBOLS"),
        alpaca_news_include_content=_bool("ALPACA_NEWS_INCLUDE_CONTENT", False),
        alpaca_news_chunk_days=max(1, _int("ALPACA_NEWS_CHUNK_DAYS", 7)),
        alpaca_data_feed=_str("ALPACA_DATA_FEED", "sip").lower(),
        alpaca_symbols_per_request=max(1, _int("ALPACA_SYMBOLS_PER_REQUEST", 50)),
        use_spacy=_bool("USE_SPACY", True),
        spacy_model=_str("SPACY_MODEL", "en_core_web_sm"),
        min_ticker_confidence=_float("MIN_TICKER_CONFIDENCE", 0.55),
        max_tickers_per_article=_int("MAX_TICKERS_PER_ARTICLE", 10),
        include_etfs=_bool("INCLUDE_ETFS", False),
        allowed_exchanges=[x.upper() for x in _str_list("ALLOWED_EXCHANGES", "NYSE,NASDAQ,AMEX,ARCA,BATS")],
        dedup_time_tolerance_minutes=_int("DEDUP_TIME_TOLERANCE_MINUTES", 10),
        dedup_title_source_window_hours=_int("DEDUP_TITLE_SOURCE_WINDOW_HOURS", 6),
        fuzzy_title_threshold=_float("FUZZY_TITLE_THRESHOLD", 95.0),
        fuzzy_window_minutes=_int("FUZZY_WINDOW_MINUTES", 120),
        min_title_tokens_for_fuzzy=_int("MIN_TITLE_TOKENS_FOR_FUZZY", 6),
        post_windows=_int_list("POST_WINDOWS", "1,5,15,30,60"),
        baseline_lookback_sessions=max(1, _int("BASELINE_LOOKBACK_SESSIONS", 2)),
        max_alignment_gap_minutes=_int("MAX_ALIGNMENT_GAP_MINUTES", 5),
        volume_window_minutes=_int("VOLUME_WINDOW_MINUTES", 15),
        min_baseline_bars=_int("MIN_BASELINE_BARS", 60),
        spike_return_threshold=_float("SPIKE_RETURN_THRESHOLD", 0.02),
        crash_return_threshold=_float("CRASH_RETURN_THRESHOLD", -0.02),
        abnormal_return_z=_float("ABNORMAL_RETURN_Z", 2.0),
        volume_spike_ratio=_float("VOLUME_SPIKE_RATIO", 1.5),
        label_horizons=_int_list("LABEL_HORIZONS", "5,15,30,60"),
        min_abs_return=_float("MIN_ABS_RETURN", 0.003),
        require_volume_confirmation=_bool("REQUIRE_VOLUME_CONFIRMATION", False),
        direction_consistency_min=_float("DIRECTION_CONSISTENCY_MIN", 0.5),
        label_include_overnight_gap=_bool("LABEL_INCLUDE_OVERNIGHT_GAP", True),
        ml_exclude_columns=_str_list("ML_EXCLUDE_COLUMNS", "news_id,event_id,publisher,alpha_vantage_id,alpaca_id"),
    )
    if s.end_date < s.start_date:
        raise ValueError("END_DATE must be on or after START_DATE")
    missing_windows = [h for h in s.label_horizons if h not in s.post_windows]
    if missing_windows:
        raise ValueError(f"LABEL_HORIZONS {missing_windows} must also appear in POST_WINDOWS")
    if 5 not in s.post_windows:
        raise ValueError("POST_WINDOWS must include 5 (needed for high_move_5m/low_move_5m)")
    if s.alpaca_data_feed not in {"sip", "iex"}:
        raise ValueError("ALPACA_DATA_FEED must be 'sip' or 'iex'")
    return s
