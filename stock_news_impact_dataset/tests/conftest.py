import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEST_KEYS = {
    "ALPHA_VANTAGE_API_KEY": "TEST_AV_KEY_should_never_be_written",
    "ALPACA_API_KEY": "TEST_ALPACA_KEY_should_never_be_written",
    "ALPACA_SECRET_KEY": "TEST_ALPACA_SECRET_should_never_be_written",
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated settings: temp data dir, fake keys, small date range."""
    from collectors.api_utils import STOP_EVENT

    STOP_EVENT.clear()
    values = {
        **TEST_KEYS,
        "START_DATE": "2025-01-06", "END_DATE": "2025-01-10", "OUTPUT_DIR": str(tmp_path / "data"), "LOG_DIR": str(tmp_path / "logs"),
        "ALPHA_VANTAGE_TIMEZONE": "America/New_York", "AV_DAILY_REQUEST_LIMIT": "0",
        "AV_REQUESTS_PER_MINUTE": "1000", "ALPACA_REQUESTS_PER_MINUTE": "1000", "MAX_WORKERS": "2",
        "ALPACA_SYMBOLS_PER_REQUEST": "3", "MIN_BASELINE_BARS": "60",
    }
    for k, v in values.items():
        monkeypatch.setenv(k, v)
    env_file = tmp_path / "test.env"
    env_file.write_text("", encoding="utf-8")
    yield env_file
    STOP_EVENT.clear()


@pytest.fixture
def fake_api(monkeypatch):
    from collectors.api_utils import HttpClient
    from tests.fake_apis import FakeAPIs

    fake = FakeAPIs()
    monkeypatch.setattr(HttpClient, "get_json", lambda self, url, params=None, context="", record_failure=True:
                        fake(self, url, params, context, record_failure))
    return fake
