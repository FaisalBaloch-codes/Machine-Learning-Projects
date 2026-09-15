"""
Fetch daily OHLCV price data via yfinance for the ticker universe.

Output: data/raw/price_ohlcv_raw.csv

Requires outbound network access to Yahoo Finance's query hosts
(query1/query2.finance.yahoo.com). Will raise a clear error if unreachable.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TICKERS, START_DATE, END_DATE, DATA_RAW


def main():
    print(f"Fetching daily OHLCV for {len(TICKERS)} tickers, {START_DATE} to {END_DATE} ...")
    frames = []
    failures = []

    for i, ticker in enumerate(TICKERS, 1):
        print(f"[{i}/{len(TICKERS)}] {ticker}")
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if df.empty:
                raise ValueError("empty dataframe returned")
            if isinstance(df.columns, pd.MultiIndex):
                # newer yfinance returns (Field, Ticker) MultiIndex columns even
                # for a single ticker -- flatten to just the field name
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "Date"})  # first col post-reset is the date index
            df["ticker"] = ticker
            frames.append(df)
        except Exception as e:
            print(f"  FAILED: {ticker} -> {e}")
            failures.append(ticker)
        time.sleep(0.2)

    if not frames:
        raise RuntimeError(
            "No price data returned for any ticker. This almost always means the "
            "current environment cannot reach Yahoo Finance's endpoints (check "
            "network/egress settings) rather than a problem with the request."
        )

    all_df = pd.concat(frames, ignore_index=True)
    out_path = DATA_RAW / "price_ohlcv_raw.csv"
    all_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(all_df):,} rows -> {out_path}")
    if failures:
        print(f"Failed tickers ({len(failures)}): {failures}")


if __name__ == "__main__":
    main()
