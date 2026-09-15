"""
Step: Feature Engineering + Labeling
Builds the final modeling feature set and the abnormal-move target label:
    label = 1 if |forward 1-week return| > ABNORMAL_MOVE_VOL_MULTIPLE * trailing
                 12-week realized volatility (own history), else 0.

Output: data/processed/model_ready_dataset.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED, ABNORMAL_MOVE_VOL_MULTIPLE

FEATURE_COLS = [
    "dark_pool_ratio",
    "dp_ratio_wow_change",
    "dp_ratio_4wk_avg",
    "dp_ratio_8wk_avg",
    "dp_ratio_trend_slope",
    "dp_ratio_accel",
    "dp_ratio_zscore",
    "dp_ratio_market_relative",
    "avg_trade_size_z",
    "volume_z",
    "momentum_4wk",
    "realized_vol_12wk",
    "price_range_position",
]


def main():
    df = pd.read_csv(DATA_PROCESSED / "weekly_panel_clean.csv", parse_dates=["week_start"])
    df = df.sort_values(["ticker", "week_start"])

    # forward 1-week return = next week's weekly_ret, shifted back by one row per ticker
    df["fwd_return_1wk"] = df.groupby("ticker")["weekly_ret"].shift(-1)

    df["abnormal_move"] = (
        df["fwd_return_1wk"].abs() > ABNORMAL_MOVE_VOL_MULTIPLE * df["realized_vol_12wk"]
    ).astype(int)

    df["abnormal_move_direction"] = 0
    df.loc[(df["abnormal_move"] == 1) & (df["fwd_return_1wk"] > 0), "abnormal_move_direction"] = 1
    df.loc[(df["abnormal_move"] == 1) & (df["fwd_return_1wk"] < 0), "abnormal_move_direction"] = -1

    model_df = df.dropna(subset=FEATURE_COLS + ["fwd_return_1wk", "abnormal_move"]).copy()

    out_path = DATA_PROCESSED / "model_ready_dataset.csv"
    model_df.to_csv(out_path, index=False)

    pos_rate = model_df["abnormal_move"].mean()
    print(f"Model-ready dataset: {len(model_df):,} rows, {model_df['ticker'].nunique()} tickers")
    print(f"Features used ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Abnormal-move base rate: {pos_rate:.3%}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
