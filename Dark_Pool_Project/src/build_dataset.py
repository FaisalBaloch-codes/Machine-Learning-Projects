"""
Step: Cleaning + Merging
Aligns weekly FINRA ATS data with daily price data (resampled to weekly),
handles reporting gaps, and produces a single clean weekly panel.

Reads the real FINRA + Yahoo Finance data produced by fetch_finra_data.py
and fetch_price_data.py.

Output: data/processed/weekly_panel_clean.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_RAW, DATA_PROCESSED


def _pick_file(*candidates):
    for c in candidates:
        p = DATA_RAW / c
        if p.exists():
            return p
    raise FileNotFoundError(f"None of {candidates} found in {DATA_RAW}. Run a fetch/generate script first.")


def load_finra():
    """Load FINRA weekly ATS (dark pool) data.

    Important: FINRA's weekly ATS summary reports one row PER REPORTING VENUE
    per symbol per week (many dark pools report separately). There is no
    "total consolidated market volume" field in this dataset -- that has to
    come from the price data instead (see load_prices / main). Here we sum
    dark-pool share volume across all reporting venues to get one
    ticker-week total.
    """
    path = _pick_file("finra_weekly_ats_raw.csv")
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    if "query_symbol" in df.columns:
        df = df.rename(columns={"query_symbol": "ticker"})
        if "issuesymbolidentifier" in df.columns:
            df = df.drop(columns=["issuesymbolidentifier"])
    elif "issuesymbolidentifier" in df.columns:
        df = df.rename(columns={"issuesymbolidentifier": "ticker"})

    rename_map = {
        "weekstartdate": "week_start",
        "totalweeklysharequantity": "ats_shares",
        "totaltradecountsum": "trade_count",
        "totalweeklytradecount": "trade_count",  # actual FINRA live column name
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "ticker" not in df.columns:
        raise ValueError("Could not find a ticker column in FINRA data.")
    if "ats_shares" not in df.columns:
        raise ValueError(
            f"Could not find a dark-pool share-volume column. Columns present: {list(df.columns)}"
        )

    keep_cols = ["ticker", "week_start", "ats_shares"] + (["trade_count"] if "trade_count" in df.columns else [])
    df = df[keep_cols].copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.dropna(subset=["ats_shares"])

    # Sum across reporting ATS venues -> one row per ticker-week
    agg = {"ats_shares": "sum"}
    if "trade_count" in df.columns:
        agg["trade_count"] = "sum"
    df = df.groupby(["ticker", "week_start"], as_index=False).agg(agg)

    print(f"Loaded FINRA data: {path.name} ({len(df):,} ticker-week rows after venue aggregation)")
    return df


def load_prices():
    path = _pick_file("price_ohlcv_raw.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # guard against stray non-date rows (e.g. leftover multi-index header artifacts)
    df = df[df["Date"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}")]
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])
    print(f"Loaded price data: {path.name} ({len(df):,} rows, {df['ticker'].nunique()} tickers)")
    return df


def weekly_price_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to weekly (Mon-start) bars per ticker with
    momentum/volatility features. Weekly 'volume' here also serves as the
    consolidated-market denominator for the dark-pool ratio."""
    price_df = price_df.sort_values(["ticker", "Date"]).copy()
    price_df["week_start"] = price_df["Date"] - pd.to_timedelta(price_df["Date"].dt.weekday, unit="D")
    price_df["daily_ret"] = price_df.groupby("ticker")["Close"].pct_change()

    weekly = (
        price_df.groupby(["ticker", "week_start"])
        .agg(
            open=("Open", "first"),
            close=("Close", "last"),
            high=("High", "max"),
            low=("Low", "min"),
            volume=("Volume", "sum"),
            realized_vol=("daily_ret", "std"),
        )
        .reset_index()
    )
    weekly = weekly.sort_values(["ticker", "week_start"])
    weekly["weekly_ret"] = weekly.groupby("ticker")["close"].pct_change()
    weekly["volume_z"] = weekly.groupby("ticker")["volume"].transform(
        lambda s: (s - s.rolling(12, min_periods=4).mean()) / s.rolling(12, min_periods=4).std()
    )
    weekly["momentum_4wk"] = weekly.groupby("ticker")["close"].transform(lambda s: s.pct_change(4))
    weekly["realized_vol_12wk"] = weekly.groupby("ticker")["weekly_ret"].transform(
        lambda s: s.rolling(12, min_periods=6).std()
    )
    return weekly


def main():
    finra_df = load_finra()
    price_df = load_prices()

    weekly_price = weekly_price_features(price_df)

    merged = pd.merge(
        weekly_price,
        finra_df,
        on=["ticker", "week_start"],
        how="inner",
    )

    # Handle reporting gaps: forward-fill short single-week FINRA gaps per ticker
    merged = merged.sort_values(["ticker", "week_start"])
    merged["ats_shares"] = merged.groupby("ticker")["ats_shares"].transform(lambda s: s.ffill(limit=1))
    merged = merged.dropna(subset=["ats_shares", "volume"])

    # No consolidated total-volume field exists in FINRA's ATS weekly summary --
    # use total weekly consolidated volume from the price data as the denominator.
    merged["dark_pool_ratio"] = merged["ats_shares"] / merged["volume"].replace(0, np.nan)
    merged["dark_pool_ratio"] = merged["dark_pool_ratio"].clip(0, 1)

    merged["dp_ratio_wow_change"] = merged.groupby("ticker")["dark_pool_ratio"].diff()
    merged["dp_ratio_4wk_avg"] = merged.groupby("ticker")["dark_pool_ratio"].transform(
        lambda s: s.rolling(4, min_periods=2).mean()
    )
    # Acceleration: is the week-over-week change itself accelerating?
    merged["dp_ratio_accel"] = merged.groupby("ticker")["dp_ratio_wow_change"].diff()
    # How unusual is this week's ratio relative to the stock's OWN trailing history
    # (distinct from the raw level, which varies a lot by ticker)
    merged["dp_ratio_zscore"] = merged.groupby("ticker")["dark_pool_ratio"].transform(
        lambda s: (s - s.rolling(12, min_periods=6).mean()) / s.rolling(12, min_periods=6).std()
    )
    # Cross-sectional signal: how does this ticker's ratio compare to the market that week
    merged["dp_ratio_market_relative"] = merged["dark_pool_ratio"] - merged.groupby("week_start")["dark_pool_ratio"].transform("mean")

    # Average trade size on dark pool venues (shares per print): higher values suggest
    # larger, more "institutional/block" prints rather than small retail-like flow
    if "trade_count" in merged.columns:
        merged["avg_trade_size"] = merged["ats_shares"] / merged["trade_count"].replace(0, np.nan)
        merged["avg_trade_size_z"] = merged.groupby("ticker")["avg_trade_size"].transform(
            lambda s: (s - s.rolling(12, min_periods=6).mean()) / s.rolling(12, min_periods=6).std()
        )

    # Slower-moving trend: 8-week average dark pool ratio, and its slope vs the 4-week average
    merged["dp_ratio_8wk_avg"] = merged.groupby("ticker")["dark_pool_ratio"].transform(
        lambda s: s.rolling(8, min_periods=4).mean()
    )
    merged["dp_ratio_trend_slope"] = merged["dp_ratio_4wk_avg"] - merged["dp_ratio_8wk_avg"]

    # Price-range position: where does this week's close sit within the trailing 12-week range
    merged["rolling_high_12wk"] = merged.groupby("ticker")["close"].transform(lambda s: s.rolling(12, min_periods=6).max())
    merged["rolling_low_12wk"] = merged.groupby("ticker")["close"].transform(lambda s: s.rolling(12, min_periods=6).min())
    merged["price_range_position"] = (
        (merged["close"] - merged["rolling_low_12wk"]) /
        (merged["rolling_high_12wk"] - merged["rolling_low_12wk"]).replace(0, np.nan)
    )

    required_cols = [
        "dark_pool_ratio", "dp_ratio_wow_change", "dp_ratio_accel", "dp_ratio_zscore",
        "dp_ratio_market_relative", "realized_vol_12wk", "momentum_4wk", "volume_z",
        "dp_ratio_trend_slope", "price_range_position",
    ]
    if "avg_trade_size_z" in merged.columns:
        required_cols.append("avg_trade_size_z")
    merged = merged.dropna(subset=required_cols)

    out_path = DATA_PROCESSED / "weekly_panel_clean.csv"
    merged.to_csv(out_path, index=False)

    print(f"\n[LIVE DATA] Clean weekly panel: {len(merged):,} rows, "
          f"{merged['ticker'].nunique()} tickers -> {out_path}")


if __name__ == "__main__":
    main()
