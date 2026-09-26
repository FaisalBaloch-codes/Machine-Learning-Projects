"""Quick inspection of the final dataset: printed summary + diagnostic plots.

Run:  python main.py --inspect      (or)      python -m processing.inspect_dataset
Plots are written to data/processed/plots/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import Settings, load_settings  # noqa: E402

INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"
BAR = "#2f6fb0"  # single hue for magnitude-only bar charts
LABEL_COLORS = {"spike": "#2f6fb0", "neutral": "#9aa4ae", "crash": "#c4581c"}  # diverging poles + gray midpoint


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _hbar(series: pd.Series, title: str, path: Path, xlabel: str) -> None:
    s = series.sort_values()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(s) + 1)))
    ax.barh(s.index.astype(str), s.values, color=BAR, height=0.7)
    for y, v in enumerate(s.values):
        ax.text(v, y, f" {v:,}", va="center", fontsize=8, color=MUTED)
    _style(ax, title, xlabel=xlabel)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def inspect(settings: Settings | None = None, path: Path | None = None) -> dict:
    s = settings or load_settings()
    path = path or s.final_dataset_path
    if not path.exists():
        print(f"Final dataset not found: {path}. Run `python main.py --build` first.")
        return {}
    df = pd.read_csv(path, low_memory=False)
    plots = s.plots_dir
    plots.mkdir(parents=True, exist_ok=True)
    ts = pd.to_datetime(df["news_published_timestamp_utc"], utc=True, errors="coerce")
    matched = df["market_alignment"].isin(["same_minute", "next_candle", "next_market_open"])

    line = "=" * 64
    print(f"\n{line}\nDATASET INSPECTION: {path.name}\n{line}")
    print(f"Total rows              : {len(df):,}")
    print(f"Usable labelled rows    : {int(df['is_usable'].astype(str).eq('True').sum()):,}")
    print(f"Unique news articles    : {df['news_id'].nunique():,}")
    print(f"Unique tickers          : {df['ticker'].nunique():,}")
    print(f"Unique publishers       : {df['publisher'].nunique():,}")
    print(f"Date range (UTC)        : {ts.min()} -> {ts.max()}")
    print(f"Duplicate rows          : {int(df.duplicated().sum()):,} (duplicate event_id: {int(df['event_id'].duplicated().sum()):,})")
    print(f"Market-matched          : {matched.mean() * 100:.1f}%")
    print("\nMissing values (columns with any missing):")
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    print(miss.to_string() if len(miss) else "  none")
    print("\nLabel distribution:")
    print(df["label"].fillna("(unlabelled)").value_counts().to_string())
    print("\nMarket alignment:")
    print(df["market_alignment"].fillna("(pending)").value_counts().to_string())
    print("\nTop 20 tickers:")
    print(df["ticker"].value_counts().head(20).to_string())
    print("\nTop publishers:")
    print(df["publisher"].fillna("(unknown)").value_counts().head(20).to_string())
    print("\nNews per month (unique articles):")
    monthly = df.assign(month=ts.dt.strftime("%Y-%m")).drop_duplicates("news_id").groupby("month").size()
    print(monthly.to_string())

    # 1. news by month
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(monthly.index, monthly.values, color=BAR, width=0.8)
    _style(ax, "Unique news articles per month", ylabel="articles")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(plots / "news_count_by_month.png", dpi=130)
    plt.close(fig)

    # 2. label distribution
    counts = df["label"].value_counts().reindex(["crash", "neutral", "spike"]).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=[LABEL_COLORS[k] for k in counts.index], width=0.6)
    for x, v in enumerate(counts.values):
        ax.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=9, color=INK)
    _style(ax, "Label distribution", ylabel="events")
    fig.tight_layout()
    fig.savefig(plots / "label_distribution.png", dpi=130)
    plt.close(fig)

    # 3/4. top tickers / publishers
    _hbar(df["ticker"].value_counts().head(20), "Top 20 tickers by event count", plots / "top_20_tickers.png", "events")
    _hbar(df.drop_duplicates("news_id")["publisher"].fillna("(unknown)").value_counts().head(15),
          "Top news publishers (unique articles)", plots / "top_publishers.png", "articles")

    # 5. market reaction distribution (primary 15-minute return, by label)
    col = "return_15m" if "return_15m" in df else next(c for c in df.columns if c.startswith("return_"))
    fig, ax = plt.subplots(figsize=(8, 4))
    for lab in ("crash", "neutral", "spike"):
        vals = pd.to_numeric(df.loc[df["label"] == lab, col], errors="coerce").dropna() * 100
        if len(vals):
            ax.hist(vals.clip(-10, 10), bins=80, alpha=0.75, color=LABEL_COLORS[lab], label=f"{lab} ({len(vals):,})")
    _style(ax, f"Post-news market reaction ({col}, clipped to ±10%)", xlabel="return (%)", ylabel="events")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(plots / "market_reaction_distribution.png", dpi=130)
    plt.close(fig)
    print(f"\nPlots saved to {plots}\n")
    return {"rows": len(df), "plots_dir": str(plots)}


if __name__ == "__main__":
    inspect()
