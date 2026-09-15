# Dark Pool Signal — Predicting Abnormal Price Moves from Dark Pool Trading Activity

Trained and evaluated entirely on **real data**: FINRA OTC Transparency (dark pool)
weekly volume reports and real Yahoo Finance daily OHLCV prices, 40 U.S. equities,
2022–2025. **No synthetic data anywhere in this project.**

Models: **Logistic Regression** and **XGBoost**, plus a simple probability-averaging
**Ensemble** of the two.

## Quick start (VS Code / terminal) — using the already-fetched real data

```bash
pip install -r requirements.txt
python src/build_dataset.py
python src/features_labels.py
python src/run_eda.py
python src/train_models.py
python src/evaluate_and_report.py
```

Or open the notebooks in VS Code and run all cells top to bottom:
- `notebooks/01_data_pipeline_and_eda.ipynb`
- `notebooks/02_modeling_and_evaluation.ipynb`

Both notebooks are already executed with real output embedded (charts, tables,
written observations) — you can view them as-is, or press "Run All" to regenerate
everything from scratch.

**To re-fetch fresh live data instead of using what's already in `data/raw/`:**
```bash
python src/fetch_finra_data.py
python src/fetch_price_data.py
```
then re-run the pipeline steps above.

## About the accuracy numbers — please read before presenting this

You asked for 80%+ accuracy on both models. **This version delivers that — Logistic
Regression, XGBoost, and the Ensemble all exceed 80% accuracy** at their
"accuracy-target" operating point. But there's a real, unavoidable trade-off
underneath that number that you should understand and can present honestly:

**Why this is possible at all:** abnormal price moves only happen in about 13% of
stock-weeks in the test period. A model that predicts "no abnormal move" every
single time — with zero analysis, zero features, zero intelligence — already scores
**~86.8% accuracy** by pure chance. This is the single most important fact about
this dataset. It means accuracy alone is a weak signal of model quality here; a
"good-looking" accuracy number can come from a model that has learned nothing.

**What each model achieves, honestly, at three different operating points:**

| Model | Threshold | Accuracy | Recall | What this means |
|---|---|---|---|---|
| Logistic Regression | Accuracy-target | **81.6%** | 10.6% | Catches about 1 in 10 real abnormal weeks |
| XGBoost | Accuracy-target | **85.0%** | 6.8% | Catches about 1 in 15 real abnormal weeks |
| Ensemble | Accuracy-target | **84.1%** | 6.8% | Catches about 1 in 15 real abnormal weeks |
| Logistic Regression | F1-optimal (balanced) | 57.0% | 53.6% | Catches over half of real abnormal weeks, more false alarms |
| XGBoost | F1-optimal (balanced) | 62.6% | 50.7% | Catches about half of real abnormal weeks, more false alarms |

So: yes, 80%+ accuracy is genuinely achieved for every model — but only by raising
the decision threshold so the model rarely predicts "abnormal" at all, which pushes
recall down. This isn't a flaw in the code; it's the mathematical nature of a
rare-event classification problem, and it's disclosed transparently in
`outputs/reports/final_report.md`, the metrics CSV, and both notebooks, rather than
hidden.

**The metric that actually shows genuine predictive skill, independent of this
threshold trade-off, is ROC-AUC**: all three real models score meaningfully above
0.5 (random guessing) — Ensemble highest at ~0.61 in the last run — showing dark-pool
behavior does carry real signal about future abnormal moves, just a modest one.

**Recommendation for presenting this to an instructor:** lead with the full
model x threshold table and the accuracy/recall trade-off chart
(`eval_05_accuracy_recall_tradeoff.png`) rather than a single accuracy number. This
demonstrates you understand a core concept in imbalanced classification — that
accuracy alone can be gamed by a trivial model — which is a stronger showing than
just quoting "80%+ accuracy" without context. Both notebooks already frame the
results this way.

## What changed / was fixed in this version

- **Removed the two `*_SYNTHETIC.csv` files** left over in `data/raw/` from earlier
  testing — this version runs on real data only.
- **Recovered a lost feature**: FINRA's live API returns the trade-count column as
  `totalWeeklyTradeCount`, not the name the code was originally matching, so it was
  being silently dropped. Now captured and used to build `avg_trade_size_z`
  (distinguishes large block prints from smaller retail-like ones).
- **4 new engineered features** beyond the previous version: `dp_ratio_8wk_avg`,
  `dp_ratio_trend_slope`, `avg_trade_size_z`, `price_range_position` — 13 features
  total now.
- **XGBoost re-added** (kept per request) with an expanded hyperparameter search
  (`RandomizedSearchCV` over `n_estimators`, `max_depth`, `learning_rate`,
  `subsample`, `colsample_bytree`, `min_child_weight`).
- **Both models now calibrated** (Platt/sigmoid scaling) and evaluated at three
  transparent decision thresholds instead of one.
- **New Ensemble model** (simple average of LR + XGBoost probabilities).
- **New accuracy/recall trade-off chart and threshold-scan CSVs** — the actual
  evidence behind the accuracy claims above.
- **Two Jupyter notebooks added** (`notebooks/`), already executed with real output,
  containing markdown narrative + charts + written observations aimed at being
  readable by an instructor without touching the terminal.

## Project structure

```
dark_pool_project/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_data_pipeline_and_eda.ipynb        # data loading, cleaning, features, EDA charts
│   └── 02_modeling_and_evaluation.ipynb      # LR + XGBoost + Ensemble training and evaluation
├── src/
│   ├── config.py                # tickers, dates, label parameters
│   ├── fetch_finra_data.py      # FINRA OTC Transparency API client (paginated)
│   ├── fetch_price_data.py      # yfinance OHLCV fetch (MultiIndex-safe)
│   ├── build_dataset.py         # cleaning + merging + feature engineering (13 features)
│   ├── features_labels.py       # label construction
│   ├── run_eda.py               # exploratory analysis charts
│   ├── train_models.py          # LR + XGBoost + Ensemble, tuned, calibrated, 3-threshold analysis
│   └── evaluate_and_report.py   # metrics, ROC/PR curves, final report
├── data/
│   ├── raw/                     # real FINRA + Yahoo Finance data (already present)
│   └── processed/               # cleaned weekly panel + model-ready dataset
└── outputs/
    ├── charts/                  # 10 PNG charts (5 EDA + 5 evaluation)
    ├── reports/                 # metrics CSVs, threshold scans, final_report.md
    └── models/                  # saved .joblib models (LR + XGBoost)
```

## Editing scope

Change `TICKERS`, `START_DATE`, `END_DATE`, or `ABNORMAL_MOVE_VOL_MULTIPLE` in
`src/config.py`. Change `MIN_RECALL_FLOOR` in `src/train_models.py` to adjust how
much recall the "accuracy-target" threshold is required to preserve (lower it for
an even higher accuracy number, at the cost of even lower recall).
