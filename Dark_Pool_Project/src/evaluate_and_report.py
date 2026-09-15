"""
Step: Model Testing + Evaluation + Results/Conclusion
Computes Accuracy/Precision/Recall/F1/ROC-AUC for the naive baseline,
Logistic Regression, XGBoost, and a simple probability-averaging Ensemble,
each evaluated at THREE decision thresholds (default 0.5, F1-optimal,
accuracy-target) so the accuracy/recall trade-off is fully visible rather
than hidden behind one cherry-picked number.

Output:
    outputs/charts/eval_*.png
    outputs/reports/evaluation_metrics.csv
    outputs/reports/final_report.md
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, accuracy_score,
    roc_curve, precision_recall_curve, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUT_REPORTS, OUT_CHARTS, ABNORMAL_MOVE_VOL_MULTIPLE

MODEL_PROBA_COLS = {
    "Naive Baseline": "baseline_proba",
    "Logistic Regression": "logreg_proba",
    "XGBoost": "xgb_proba",
    "Ensemble (LR+XGB avg)": "ensemble_proba",
}

THRESHOLD_VARIANTS = {
    "0.5 (default)": "default",
    "F1-optimal (validation)": "f1opt",
    "Accuracy-target (validation)": "acctarget",
}


def compute_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    y_true = df["abnormal_move"]
    rows = []
    for model_name, proba_col in MODEL_PROBA_COLS.items():
        proba = df[proba_col]
        try:
            auc = roc_auc_score(y_true, proba) if proba.nunique() > 1 else float("nan")
        except ValueError:
            auc = float("nan")

        if model_name == "Naive Baseline":
            pred = df["baseline_pred"]
            rows.append({
                "model": model_name, "threshold_type": "n/a (always 0)",
                "accuracy": accuracy_score(y_true, pred),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
                "roc_auc": auc,
            })
            continue

        prefix = {"Logistic Regression": "logreg", "XGBoost": "xgb", "Ensemble (LR+XGB avg)": "ensemble"}[model_name]
        for label, suffix in THRESHOLD_VARIANTS.items():
            pred_col = f"{prefix}_pred_{suffix}"
            if pred_col not in df.columns:
                continue
            pred = df[pred_col]
            rows.append({
                "model": model_name, "threshold_type": label,
                "accuracy": accuracy_score(y_true, pred),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
                "roc_auc": auc,
            })
    return pd.DataFrame(rows)


def plot_roc_pr(df: pd.DataFrame):
    y_true = df["abnormal_move"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {"Logistic Regression": "#4C72B0", "XGBoost": "#DD8452", "Ensemble (LR+XGB avg)": "#55A868"}
    for model_name, proba_col in MODEL_PROBA_COLS.items():
        if model_name == "Naive Baseline" or df[proba_col].nunique() <= 1:
            continue
        fpr, tpr, _ = roc_curve(y_true, df[proba_col])
        axes[0].plot(fpr, tpr, label=model_name, color=colors.get(model_name))
        prec, rec, _ = precision_recall_curve(y_true, df[proba_col])
        axes[1].plot(rec, prec, label=model_name, color=colors.get(model_name))

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random")
    axes[0].set_title("ROC Curve (Held-Out Test Period)")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()
    axes[1].set_title("Precision-Recall Curve (Held-Out Test Period)")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT_CHARTS / "eval_01_roc_pr_curves.png", dpi=150)
    plt.close(fig)


def plot_metric_comparison(metrics_df: pd.DataFrame):
    pivot_data = metrics_df[metrics_df["threshold_type"] != "n/a (always 0)"].copy()
    pivot_data["label"] = pivot_data["model"] + " | " + pivot_data["threshold_type"]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    pivot_data.set_index("label")[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(kind="bar", ax=ax)
    ax.axhline(metrics_df.loc[metrics_df["model"] == "Naive Baseline", "accuracy"].iloc[0],
               color="red", linestyle="--", linewidth=1, label="Naive baseline accuracy")
    ax.set_title("Model x Threshold Comparison")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_CHARTS / "eval_02_metric_comparison.png", dpi=150)
    plt.close(fig)


def plot_accuracy_recall_tradeoff():
    """The key transparency chart: shows how accuracy and recall trade off as
    the decision threshold changes, for each model."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, fname, color in [
        ("Logistic Regression", "threshold_scan_logreg.csv", "#4C72B0"),
        ("XGBoost", "threshold_scan_xgboost.csv", "#DD8452"),
        ("Ensemble (LR+XGB avg)", "threshold_scan_ensemble.csv", "#55A868"),
    ]:
        path = OUT_REPORTS / fname
        if not path.exists():
            continue
        scan = pd.read_csv(path)
        ax.plot(scan["threshold"], scan["accuracy"], color=color, linestyle="-", label=f"{name} — accuracy")
        ax.plot(scan["threshold"], scan["recall"], color=color, linestyle="--", alpha=0.6, label=f"{name} — recall")
    ax.set_title("Accuracy vs. Recall as Decision Threshold Changes (Validation Set)")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_CHARTS / "eval_05_accuracy_recall_tradeoff.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrices(df: pd.DataFrame):
    panels = [
        ("Naive Baseline", "baseline_pred"),
        ("Logistic Regression\n(F1-optimal)", "logreg_pred_f1opt"),
        ("XGBoost\n(F1-optimal)", "xgb_pred_f1opt"),
        ("Ensemble\n(F1-optimal)", "ensemble_pred_f1opt"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, (name, col) in zip(axes, panels):
        cm = confusion_matrix(df["abnormal_move"], df[col])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Abnormal"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Abnormal"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(OUT_CHARTS / "eval_03_confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_feature_importance():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    coef_path = OUT_REPORTS / "logreg_coefficients.csv"
    if coef_path.exists():
        coefs = pd.read_csv(coef_path, index_col=0)["coefficient"].sort_values(key=abs).tail(12)
        colors = ["#C44E52" if v < 0 else "#4C72B0" for v in coefs.values]
        coefs.plot(kind="barh", ax=axes[0], color=colors)
        axes[0].set_title("Logistic Regression: Top 12 Coefficients")
        axes[0].set_xlabel("Coefficient (standardized)")

    imp_path = OUT_REPORTS / "xgboost_feature_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path, index_col=0)["importance"].sort_values().tail(12)
        imp.plot(kind="barh", ax=axes[1], color="#DD8452")
        axes[1].set_title("XGBoost: Feature Importance")

    fig.tight_layout()
    fig.savefig(OUT_CHARTS / "eval_04_feature_importance_both_models.png", dpi=150)
    plt.close(fig)


def write_report(metrics_df: pd.DataFrame, df: pd.DataFrame):
    naive_acc = metrics_df.loc[metrics_df["model"] == "Naive Baseline", "accuracy"].iloc[0]
    base_rate = df["abnormal_move"].mean()

    lines = []
    lines.append("# Dark Pool Signal — Final Report\n")
    lines.append("> Trained and evaluated on real FINRA OTC Transparency (dark pool) data "
                  "and real Yahoo Finance OHLCV price data. No synthetic data used.\n")

    lines.append("## 1. Dataset Summary")
    lines.append(f"- Test period rows: {len(df):,}")
    lines.append(f"- Abnormal-move base rate (test): {base_rate:.2%}")
    lines.append(f"- Label definition: |forward 1-week return| > {ABNORMAL_MOVE_VOL_MULTIPLE}x "
                 "trailing 12-week realized volatility")
    lines.append(f"- Naive baseline accuracy (always predicts 'normal'): {naive_acc:.2%}\n")

    lines.append("## 2. Full Model x Threshold Comparison (Held-Out Chronological Test Set)")
    lines.append(metrics_df.round(4).to_markdown(index=False))
    lines.append("")
    lines.append(
        f"> **Why three thresholds per model?** With abnormal moves at only ~{base_rate:.1%} of "
        f"weeks, a model can trivially exceed **{naive_acc:.1%} accuracy** simply by predicting "
        "'normal' almost always -- that's what the naive baseline does, with zero forecasting "
        "value. To show the real trade-off transparently, each model is reported at: **0.5** "
        "(arbitrary default), **F1-optimal** (best balance of precision/recall, tuned on the "
        "validation set), and **accuracy-target** (the highest accuracy achievable on the "
        "validation set while keeping recall at or above 10%, so it isn't just predicting "
        "nothing). See `eval_05_accuracy_recall_tradeoff.png` for the full curve -- accuracy "
        "and recall move in opposite directions as the threshold rises, and any single number "
        "quoted without its threshold and recall alongside it is not a complete picture of "
        "model quality."
    )
    lines.append("")

    best_row = metrics_df.loc[metrics_df["model"] != "Naive Baseline", :].loc[
        metrics_df.loc[metrics_df["model"] != "Naive Baseline", "roc_auc"].idxmax()
    ]
    lines.append("## 3. Key Finding")
    lines.append(
        f"**{best_row['model']}** achieved the highest ROC-AUC (**{best_row['roc_auc']:.3f}**, "
        "vs. 0.500 for random guessing), indicating the strongest ranking ability among the "
        "models tested. ROC-AUC is threshold-independent and is the primary metric for judging "
        "real predictive skill on this rare-event problem; the accuracy-target rows above show "
        "what happens to precision/recall if the decision threshold is pushed to chase a high "
        "accuracy number instead."
    )
    lines.append("")

    lines.append("## 4. Files Produced")
    lines.append("- `outputs/charts/eda_01..05_*.png` — exploratory analysis")
    lines.append("- `outputs/charts/eval_01_roc_pr_curves.png` — ROC / Precision-Recall curves")
    lines.append("- `outputs/charts/eval_02_metric_comparison.png` — all models x all thresholds")
    lines.append("- `outputs/charts/eval_03_confusion_matrices.png` — confusion matrices (F1-optimal threshold)")
    lines.append("- `outputs/charts/eval_04_feature_importance_both_models.png` — LR coefficients + XGBoost importances")
    lines.append("- `outputs/charts/eval_05_accuracy_recall_tradeoff.png` — accuracy vs. recall by threshold")
    lines.append("- `outputs/reports/evaluation_metrics.csv`, `chosen_thresholds.csv`, `threshold_scan_*.csv`")
    lines.append("- `outputs/reports/model_predictions_test.csv` — row-level test predictions, all models/thresholds")
    lines.append("- `outputs/models/logistic_regression.joblib`, `xgboost_model.joblib`")

    (OUT_REPORTS / "final_report.md").write_text("\n".join(lines))


def main():
    df = pd.read_csv(OUT_REPORTS / "model_predictions_test.csv")
    metrics_df = compute_metrics_table(df)
    metrics_df.to_csv(OUT_REPORTS / "evaluation_metrics.csv", index=False)
    print(metrics_df.round(4).to_string(index=False))

    plot_roc_pr(df)
    plot_metric_comparison(metrics_df)
    plot_accuracy_recall_tradeoff()
    plot_confusion_matrices(df)
    plot_feature_importance()
    write_report(metrics_df, df)

    print(f"\nSaved evaluation charts -> {OUT_CHARTS}")
    print(f"Saved final report -> {OUT_REPORTS / 'final_report.md'}")


if __name__ == "__main__":
    main()
