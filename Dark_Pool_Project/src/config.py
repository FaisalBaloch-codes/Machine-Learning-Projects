"""
Central configuration for the Dark Pool Signal project.
Edit TICKERS / START_DATE / END_DATE here to change scope without touching
any other script.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Universe: 40 liquid, heavily dark-pool-traded U.S. equities across sectors
# (large-cap tech, financials, healthcare, industrials, consumer, telecom)
# ---------------------------------------------------------------------------
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
    "JPM", "BAC", "WFC", "GS", "MS", "C",
    "XOM", "CVX",
    "JNJ", "PFE", "UNH",
    "V", "MA", "PG", "KO", "PEP",
    "DIS", "NFLX",
    "INTC", "AMD", "CSCO", "ORCL", "IBM",
    "WMT", "HD", "NKE", "MCD",
    "BA", "CAT", "GE",
    "T", "VZ", "CMCSA",
]
assert len(TICKERS) == 40

# ---------------------------------------------------------------------------
# Date range (weekly FINRA ATS data + daily OHLCV)
# ---------------------------------------------------------------------------
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"

# FINRA OTC Transparency API
FINRA_BASE_URL = "https://api.finra.org/data/group/otcMarket/name/weeklySummary"
FINRA_TIER = "T1"                 # NMS Tier 1 (S&P 500 / Russell 1000 names)
FINRA_SUMMARY_TYPE = "ATS_W_SMBL"  # Weekly ATS summary, by symbol

# Label construction
ABNORMAL_MOVE_VOL_MULTIPLE = 1.5   # |fwd 1wk return| > 1.5x trailing 12wk realized vol
VOL_LOOKBACK_WEEKS = 12

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUT_CHARTS = ROOT / "outputs" / "charts"
OUT_REPORTS = ROOT / "outputs" / "reports"
OUT_MODELS = ROOT / "outputs" / "models"

for p in [DATA_RAW, DATA_PROCESSED, OUT_CHARTS, OUT_REPORTS, OUT_MODELS]:
    p.mkdir(parents=True, exist_ok=True)
