"""
Step: Exploratory Data Analysis
Generates charts examining dark-pool ratio behavior, its relationship to
abnormal moves, and general dataset structure.

Output: outputs/charts/eda_*.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED, OUT_CHARTS
from features_labels import FEATURE_COLS

sns.set_theme(style="whitegrid")


def main():
    df = pd.read_csv(DATA_PROCESSED / "model_ready_dataset.csv", parse_dates=["week_start"])

    # 1. Dark pool ratio distribution: abnormal vs normal weeks
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=df, x="dark_pool_ratio", hue="abnormal_move", fill=True,
                common_norm=False, alpha=0.4, ax=ax,
                palette={0: "#4C72B0", 1: "#DD8452"})
    ax.set_title("Dark Pool Ratio Distribution: Abnormal vs. Normal Weeks")
    ax.set_xlabel("Dark Pool Volume Ratio")
    fig.tight_layout()
    fig.savefig(OUT_CHARTS / "eda_01_dp_ratio_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Week-over-week dark pool ratio change vs abnormal move rate (binned)
    df["dp_change_bin"] = pd.qcut(df["dp_ratio_wow_change"], 10, duplicates="drop")
    bin_rates = df.groupby("dp_change_bin", observed=True)["abnormal_move"].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    bin_rates.plot(kind="bar", ax=ax, color="#55A868")
    ax.set_title("Abnormal-Move Rate by Week-over-Week Dark Pool Ratio Change (Decile)")
    ax.set_ylabel("Abnormal Move Rate")
    ax.set_xlabel("WoW Dark Pool Ratio Change (decile bins)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_CHARTS / "eda_02_dp_change_vs_abnormal_rate.png", dpi=150)
    plt.close(fig)

    # 3. Average dark pool ratio trend over time, market-wide
    trend = df.groupby("week_start")["dark_pool_ratio"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    trend.plot(ax=ax, color="#4C72B0")
    ax.set_title("Average Dark Pool Volume Ratio Over Time (All Tickers)")
    ax.set_ylabel("Dark Pool Ratio")
    ax.set_xlabel("Week")
    fig.tight_layout()
    fig.savefig(OUT_CHARTS / "eda_03_market_wide_dp_trend.png", dpi=150)
    plt.close(fig)

    # 4. Correlation heatmap of features + label
    feature_cols = FEATURE_COLS + ["abnormal_move"]
    corr = df[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(OUT_CHARTS / "eda_04_feature_correlation.png", dpi=150)
    plt.close(fig)

    # 5. Abnormal move base rate by ticker (top/bottom 10)
    rate_by_ticker = df.groupby("ticker")["abnormal_move"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    pd.concat([rate_by_ticker.head(10), rate_by_ticker.tail(10)]).plot(kind="barh", ax=ax, color="#C44E52")
    ax.set_title("Abnormal-Move Rate by Ticker (10 Lowest & 10 Highest)")
    ax.set_xlabel("Abnormal Move Rate")
    fig.tight_layout()
    fig.savefig(OUT_CHARTS / "eda_05_abnormal_rate_by_ticker.png", dpi=150)
    plt.close(fig)

    print(f"Saved 5 EDA charts -> {OUT_CHARTS}")


if __name__ == "__main__":
    main()
