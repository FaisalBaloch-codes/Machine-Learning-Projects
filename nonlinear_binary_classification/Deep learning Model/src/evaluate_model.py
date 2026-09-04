"""
evaluate_model.py
------------------
Loads the trained model and the unseen TEST data, and measures how well the
model generalizes to data it has never seen during training.

Metrics explained:
    * Accuracy  -> percentage of test samples classified correctly.
    * Precision -> out of everything the model predicted as class 1, what
                   fraction was actually class 1? (Are we trustworthy when
                   we say "yes"?)
    * Recall    -> out of everything that was actually class 1, what
                   fraction did the model catch? (Do we miss real cases?)
    * F1-score  -> a single number that balances precision and recall.
    * AUC       -> "Area Under the ROC Curve". Roughly: if you pick one
                   random class-0 point and one random class-1 point, AUC is
                   the probability the model gives the class-1 point a
                   higher predicted probability. 1.0 = perfect, 0.5 = random
                   guessing.

Outputs:
    * outputs/evaluation_metrics.csv
    * outputs/classification_report.txt
    * plots/06_confusion_matrix.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from tensorflow import keras

from config import (
    TEST_CSV,
    MODEL_PATH,
    EVALUATION_METRICS_CSV,
    CLASSIFICATION_REPORT_TXT,
    PLOTS_DIR,
)


def load_test_data():
    df = pd.read_csv(TEST_CSV)
    X = df[["x", "y"]].values
    y = df["target"].values
    return X, y


def plot_confusion_matrix(cm: np.ndarray, save_path) -> None:
    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ["Predicted 0", "Predicted 1"])
    plt.yticks(tick_marks, ["Actual 0", "Actual 1"])

    # Write the actual numbers (TN, FP, FN, TP) inside each cell.
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to: {save_path}")


def evaluate():
    X_test, y_test = load_test_data()
    model = keras.models.load_model(MODEL_PATH)

    # model.evaluate() gives us the loss + the metrics we compiled with
    # (accuracy, precision, recall, auc), computed on the TEST set.
    results = model.evaluate(X_test, y_test, verbose=0)
    metric_names = model.metrics_names  # e.g. ['loss', 'accuracy', 'precision', 'recall', 'auc']
    keras_metrics = dict(zip(metric_names, results))

    # Get predicted probabilities, then convert to class labels using a
    # 0.5 threshold (the standard choice unless told otherwise).
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Compute the same metrics again independently with scikit-learn, as a
    # cross-check, and to get F1-score (which is not a Keras metric here).
    metrics = {
        "test_loss": keras_metrics.get("loss"),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_pred_proba),
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(EVALUATION_METRICS_CSV, index=False)
    print(f"Evaluation metrics saved to: {EVALUATION_METRICS_CSV}")
    print(metrics_df.to_string(index=False))

    # Human-readable classification report (precision/recall/f1 per class).
    report = classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"])
    with open(CLASSIFICATION_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("Classification Report\n")
        f.write("======================\n\n")
        f.write(report)
        f.write("\n\nOverall metrics\n----------------\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"Classification report saved to: {CLASSIFICATION_REPORT_TXT}")

    # Confusion matrix plot.
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, PLOTS_DIR / "06_confusion_matrix.png")

    return metrics, y_pred_proba, y_pred, y_test


if __name__ == "__main__":
    evaluate()
