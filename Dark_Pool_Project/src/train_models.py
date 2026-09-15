"""
Step: Model Training + Validation/Tuning
Chronological train/validation/test split. Trains a naive baseline, a tuned
+ calibrated Logistic Regression, and a tuned + calibrated XGBoost model,
plus a simple probability-averaging ensemble of the two.

For each real model we report metrics at THREE decision thresholds so the
accuracy/recall trade-off is fully transparent rather than hidden behind a
single cherry-picked number:
    - default 0.5
    - F1-optimal (maximizes F1 on the validation set)
    - accuracy-target (highest threshold that still keeps recall >= a floor,
      i.e. the best accuracy you can get without the model collapsing to
      "never predict abnormal")

Output:
    outputs/models/*.joblib
    outputs/reports/model_predictions_test.csv
    outputs/reports/threshold_analysis.csv
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED, OUT_MODELS, OUT_REPORTS
from features_labels import FEATURE_COLS

MIN_RECALL_FLOOR = 0.10  # accuracy-target threshold must keep at least this much recall


def chronological_split(df: pd.DataFrame, train_frac=0.6, val_frac=0.2):
    df = df.sort_values("week_start")
    weeks = df["week_start"].drop_duplicates().sort_values().to_numpy()
    n = len(weeks)
    train_end = weeks[int(n * train_frac)]
    val_end = weeks[int(n * (train_frac + val_frac))]
    train = df[df["week_start"] <= train_end]
    val = df[(df["week_start"] > train_end) & (df["week_start"] <= val_end)]
    test = df[df["week_start"] > val_end]
    return train, val, test


def calibrate(fitted_estimator, X_val, y_val):
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(fitted_estimator), method="sigmoid")
    except ImportError:
        cal = CalibratedClassifierCV(fitted_estimator, method="sigmoid", cv="prefit")
    cal.fit(X_val, y_val)
    return cal


def analyze_thresholds(y_val, val_proba, min_recall=MIN_RECALL_FLOOR):
    """Return (f1_optimal_threshold, accuracy_target_threshold, scan_table)."""
    candidates = np.linspace(0.02, 0.98, 49)
    rows = []
    for t in candidates:
        pred = (val_proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "accuracy": accuracy_score(y_val, pred),
            "recall": recall_score(y_val, pred, zero_division=0),
            "f1": f1_score(y_val, pred, zero_division=0),
        })
    scan = pd.DataFrame(rows)

    f1_thresh = scan.loc[scan["f1"].idxmax(), "threshold"]

    feasible = scan[scan["recall"] >= min_recall]
    if len(feasible):
        acc_thresh = feasible.loc[feasible["accuracy"].idxmax(), "threshold"]
    else:
        acc_thresh = scan.loc[scan["recall"].idxmax(), "threshold"]  # fallback

    return f1_thresh, acc_thresh, scan


def main():
    df = pd.read_csv(DATA_PROCESSED / "model_ready_dataset.csv", parse_dates=["week_start"])
    train, val, test = chronological_split(df)

    print(f"Train: {len(train):,} rows ({train['week_start'].min().date()} - {train['week_start'].max().date()})")
    print(f"Val:   {len(val):,} rows ({val['week_start'].min().date()} - {val['week_start'].max().date()})")
    print(f"Test:  {len(test):,} rows ({test['week_start'].min().date()} - {test['week_start'].max().date()})")

    X_train, y_train = train[FEATURE_COLS], train["abnormal_move"]
    X_val, y_val = val[FEATURE_COLS], val["abnormal_move"]
    X_test, y_test = test[FEATURE_COLS], test["abnormal_move"]

    results = test[["ticker", "week_start", "fwd_return_1wk", "abnormal_move"]].copy()
    results["baseline_pred"] = 0
    results["baseline_proba"] = 0.0

    threshold_summary = {}

    # ============================= Logistic Regression =============================
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ("clf", LogisticRegression(max_iter=3000)),
    ])
    lr_grid = {
        "clf__C": [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        "clf__class_weight": ["balanced", None],
        "clf__solver": ["lbfgs"],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    lr_search = GridSearchCV(lr_pipe, lr_grid, cv=tscv, scoring="roc_auc", n_jobs=-1)
    lr_search.fit(X_train, y_train)
    print(f"\nBest Logistic Regression params: {lr_search.best_params_}")
    print(f"Best CV ROC-AUC (train folds): {lr_search.best_score_:.4f}")

    lr_best = lr_search.best_estimator_
    lr_best.fit(X_train, y_train)
    lr_cal = calibrate(lr_best, X_val, y_val)

    lr_val_proba = lr_cal.predict_proba(X_val)[:, 1]
    lr_f1_t, lr_acc_t, lr_scan = analyze_thresholds(y_val, lr_val_proba)
    lr_scan.to_csv(OUT_REPORTS / "threshold_scan_logreg.csv", index=False)

    lr_test_proba = lr_cal.predict_proba(X_test)[:, 1]
    results["logreg_proba"] = lr_test_proba
    results["logreg_pred_default"] = (lr_test_proba >= 0.5).astype(int)
    results["logreg_pred_f1opt"] = (lr_test_proba >= lr_f1_t).astype(int)
    results["logreg_pred_acctarget"] = (lr_test_proba >= lr_acc_t).astype(int)
    threshold_summary["Logistic Regression"] = {"f1_threshold": lr_f1_t, "acc_threshold": lr_acc_t}

    joblib.dump(lr_cal, OUT_MODELS / "logistic_regression.joblib")

    try:
        feat_names = lr_best.named_steps["poly"].get_feature_names_out(FEATURE_COLS)
        coefs = lr_best.named_steps["clf"].coef_[0]
        imp = pd.Series(coefs, index=feat_names).sort_values(key=abs, ascending=False)
        imp.to_csv(OUT_REPORTS / "logreg_coefficients.csv", header=["coefficient"])
    except Exception as e:
        print(f"Could not extract LR coefficients: {e}")

    # ================================= XGBoost =================================
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_grid = {
        "n_estimators": [150, 300, 500],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.02, 0.05, 0.1],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
        "min_child_weight": [1, 5],
    }
    xgb_base = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=pos_weight, random_state=42, n_jobs=-1,
    )
    # Randomized-style reduced grid via GridSearchCV would be huge (3*3*3*2*2*2=216
    # combos x 5 folds); use a curated subset for tractable runtime instead of full grid.
    from sklearn.model_selection import RandomizedSearchCV
    xgb_search = RandomizedSearchCV(
        xgb_base, xgb_grid, n_iter=25, cv=tscv, scoring="roc_auc",
        random_state=42, n_jobs=-1,
    )
    xgb_search.fit(X_train, y_train)
    print(f"\nBest XGBoost params: {xgb_search.best_params_}")
    print(f"Best CV ROC-AUC (train folds): {xgb_search.best_score_:.4f}")

    xgb_best = xgb_search.best_estimator_
    xgb_best.fit(X_train, y_train)
    xgb_cal = calibrate(xgb_best, X_val, y_val)

    xgb_val_proba = xgb_cal.predict_proba(X_val)[:, 1]
    xgb_f1_t, xgb_acc_t, xgb_scan = analyze_thresholds(y_val, xgb_val_proba)
    xgb_scan.to_csv(OUT_REPORTS / "threshold_scan_xgboost.csv", index=False)

    xgb_test_proba = xgb_cal.predict_proba(X_test)[:, 1]
    results["xgb_proba"] = xgb_test_proba
    results["xgb_pred_default"] = (xgb_test_proba >= 0.5).astype(int)
    results["xgb_pred_f1opt"] = (xgb_test_proba >= xgb_f1_t).astype(int)
    results["xgb_pred_acctarget"] = (xgb_test_proba >= xgb_acc_t).astype(int)
    threshold_summary["XGBoost"] = {"f1_threshold": xgb_f1_t, "acc_threshold": xgb_acc_t}

    joblib.dump(xgb_cal, OUT_MODELS / "xgboost_model.joblib")

    importances = pd.Series(xgb_best.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    importances.to_csv(OUT_REPORTS / "xgboost_feature_importance.csv", header=["importance"])

    # ============================= Ensemble (avg of both) =============================
    ens_val_proba = (lr_val_proba + xgb_val_proba) / 2
    ens_f1_t, ens_acc_t, ens_scan = analyze_thresholds(y_val, ens_val_proba)
    ens_scan.to_csv(OUT_REPORTS / "threshold_scan_ensemble.csv", index=False)

    ens_test_proba = (lr_test_proba + xgb_test_proba) / 2
    results["ensemble_proba"] = ens_test_proba
    results["ensemble_pred_default"] = (ens_test_proba >= 0.5).astype(int)
    results["ensemble_pred_f1opt"] = (ens_test_proba >= ens_f1_t).astype(int)
    results["ensemble_pred_acctarget"] = (ens_test_proba >= ens_acc_t).astype(int)
    threshold_summary["Ensemble (LR+XGB avg)"] = {"f1_threshold": ens_f1_t, "acc_threshold": ens_acc_t}

    out_path = OUT_REPORTS / "model_predictions_test.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved test-set predictions -> {out_path}")

    thresh_df = pd.DataFrame(threshold_summary).T.reset_index().rename(columns={"index": "model"})
    thresh_df.to_csv(OUT_REPORTS / "chosen_thresholds.csv", index=False)
    print("\nChosen thresholds (from validation set):")
    print(thresh_df.to_string(index=False))


if __name__ == "__main__":
    main()
