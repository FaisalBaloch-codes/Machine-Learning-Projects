"""End-to-end pipeline tests against the offline fake APIs (no network, no real keys)."""
import json
from pathlib import Path

import pandas as pd

import main as pipeline_main
from config.settings import load_settings
from tests.conftest import TEST_KEYS


def _final(settings):
    return pd.read_csv(settings.final_dataset_path)


def test_full_pipeline(env, fake_api):
    assert pipeline_main.main(["--env", str(env)]) == 0
    s = load_settings(env)

    # raw provider files kept separately, with provider + original publisher
    av = pd.read_csv(s.alpha_vantage_raw_path)
    al = pd.read_csv(s.alpaca_news_raw_path)
    assert set(av["news_provider"]) == {"alpha_vantage"} and set(al["news_provider"]) == {"alpaca"}
    assert set(al["source"]) == {"benzinga"}
    assert "20250108T100217" in set(av["time_published"].astype(str))  # original string preserved

    # dedup: AV + Alpaca copy of the Apple article merged (URL rule)
    stats = json.loads((s.processed_dir / "dedup_stats.json").read_text())
    assert stats["duplicates_removed"] == 1
    dd = pd.read_csv(s.dedup_news_path)
    apple = dd[dd["title"].str.startswith("Apple shares jump")].iloc[0]
    assert apple["news_providers_all"] == "alpaca;alpha_vantage"
    assert apple["published_utc_alpha_vantage"] == apple["published_utc_alpaca"] == "2025-01-08T15:02:17Z"

    # rejected rows are kept with a reason, not silently dropped
    rej = pd.read_csv(s.rejected_news_path)
    assert "missing_title" in set(rej["reject_reason"])

    df = _final(s)
    # schema
    required = ["news_id", "ticker", "company_name", "title", "description", "news_published_timestamp_utc",
                "news_date", "news_time", "news_provider", "publisher", "source", "url", "ticker_detection_method",
                "ticker_confidence", "market_timestamp_utc", "open", "high", "low", "close", "volume",
                "return_1m", "return_5m", "return_15m", "return_30m", "return_60m", "high_move_5m",
                "low_move_5m", "volume_ratio", "volatility", "abnormal_return", "label", "label_confidence",
                "label_reason", "alpha_vantage_id", "alpaca_id", "market_alignment"]
    assert not [c for c in required if c not in df.columns]

    by = {t: r for t, r in zip(df["ticker"], df.to_dict("records"))}
    # ticker extraction: ETF (SPY), crypto and the "AI" acronym are excluded; multi-ticker article split
    assert set(df["ticker"]) == {"AAPL", "NVDA", "MSFT", "TGT", "TSLA"}
    tgt_tsla = df[df["title"].str.contains("Target Corporation")]
    assert set(tgt_tsla["ticker"]) == {"TGT", "TSLA"} and tgt_tsla["news_id"].nunique() == 1
    assert "exchange_tag" in by["TGT"]["ticker_detection_method"]
    assert by["NVDA"]["ticker_detection_method"] == "ner_alias"  # no API symbols -> found by NER
    assert "api_alpha_vantage" in by["MSFT"]["ticker_detection_method"]

    # 1-minute alignment
    assert by["AAPL"]["market_alignment"] == "same_minute"
    assert by["AAPL"]["market_timestamp_utc"] == "2025-01-08T15:02:00Z"  # 15:02:17 -> 15:02 candle
    assert by["NVDA"]["market_alignment"] == "next_market_open"
    assert by["NVDA"]["market_timestamp_utc"] == "2025-01-09T14:30:00Z"
    assert by["NVDA"]["news_session_position"] == "after_hours"

    # labels
    assert by["AAPL"]["label"] == "spike" and by["AAPL"]["volume_ratio"] > 1.5
    assert by["NVDA"]["label"] == "crash"
    assert by["MSFT"]["label"] == "neutral"
    assert "abnormal return" in by["AAPL"]["label_reason"]

    # quality report
    rep = json.loads((s.processed_dir / "data_quality_report.json").read_text())
    for k in ("raw_news_count", "deduplicated_news_count", "ticker_events", "market_matched_events",
              "missing_market_data", "neutral_count", "spike_count", "crash_count", "duplicate_count",
              "invalid_count"):
        assert k in rep
    assert rep["spike_count"] >= 1 and rep["crash_count"] >= 1
    assert (s.processed_dir / "data_quality_report.csv").exists()

    # ML-ready file contains no outcome columns
    ml = pd.read_csv(s.processed_dir / "ml_ready_dataset.csv")
    assert not [c for c in ml.columns if c.startswith("return_") or c in ("volume_ratio", "abnormal_return", "close")]
    for c in ("news_id", "event_id", "publisher", "alpha_vantage_id", "alpaca_id", "open", "high", "low", "volume"):
        assert c not in ml.columns
    assert "label" in ml.columns and "ticker" in ml.columns
    schema = json.loads((s.processed_dir / "dataset_schema.json").read_text())
    assert schema["return_15m"]["role"] == "outcome" and schema["label"]["role"] == "target"

    # market cache files per ticker/day
    assert (s.market_dir / "AAPL_2025-01-08.csv").exists()

    # plots
    for name in ("news_count_by_month", "label_distribution", "top_20_tickers", "top_publishers",
                 "market_reaction_distribution"):
        assert (s.plots_dir / f"{name}.png").exists()

    # secrets never written anywhere
    for p in Path(s.output_dir).rglob("*"):
        if p.is_file() and p.suffix in (".csv", ".json", ".log", ".bak"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            for secret in TEST_KEYS.values():
                assert secret not in text, f"secret leaked into {p}"


def test_resume_does_not_redownload(env, fake_api):
    assert pipeline_main.main(["--env", str(env)]) == 0
    n_calls = len(fake_api.calls)
    s = load_settings(env)
    before = _final(s)
    assert pipeline_main.main(["--env", str(env)]) == 0
    new_calls = fake_api.calls[n_calls:]
    # second run: no news pages, no bars requested again (only AV windows are complete too)
    assert not [u for u, _ in new_calls if u.endswith("/v2/stocks/bars") or u.endswith("/v1beta1/news")]
    assert not [u for u, _ in new_calls if "alphavantage" in u]
    after = _final(s)
    assert len(before) == len(after)


def test_ctrl_c_then_resume(env, fake_api):
    from collectors.api_utils import STOP_EVENT

    fake_api.stop_after_alpaca_news_pages = 1  # simulate Ctrl+C after the first page
    assert pipeline_main.main(["--env", str(env), "--news", "--skip-alpha-vantage"]) == 130
    s = load_settings(env)
    first = pd.read_csv(s.alpaca_news_raw_path)
    assert 0 < len(first) < 5
    cp = json.loads((s.checkpoint_dir / "alpaca_news.json").read_text())
    assert any(v.get("page_token") for v in cp["chunks"].values())

    STOP_EVENT.clear()
    fake_api.stop_after_alpaca_news_pages = None
    assert pipeline_main.main(["--env", str(env), "--news", "--skip-alpha-vantage"]) == 0
    full = pd.read_csv(s.alpaca_news_raw_path)
    assert len(full) == 5 and full["alpaca_id"].is_unique


def test_partial_write_is_truncated_on_resume(env, fake_api):
    assert pipeline_main.main(["--env", str(env), "--news", "--skip-alpha-vantage"]) == 0
    s = load_settings(env)
    good = s.alpaca_news_raw_path.read_bytes()
    with open(s.alpaca_news_raw_path, "ab") as fh:  # simulate a hard kill mid-row
        fh.write(b'news_provider_partial,"unterminated')
    assert pipeline_main.main(["--env", str(env), "--news", "--skip-alpha-vantage"]) == 0
    assert s.alpaca_news_raw_path.read_bytes() == good
    assert list(s.raw_dir.glob("alpaca_news.csv.*.orphan"))


def test_second_run_is_refused_while_locked(env, fake_api):
    s = load_settings(env)
    s.ensure_dirs()
    lock = pipeline_main.acquire_run_lock(s)
    try:
        assert pipeline_main.main(["--env", str(env), "--inspect"]) == 2
    finally:
        lock.close()
    assert pipeline_main.main(["--env", str(env), "--inspect"]) == 0
