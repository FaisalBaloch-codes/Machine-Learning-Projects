"""
predict.py
----------
Uses the trained model to make predictions on the test dataset, saves them
to a CSV file, and creates two visualizations:

    1. plots/08_actual_vs_predicted.png
       Side-by-side scatter plots showing the REAL classes vs the classes
       the model PREDICTED, so you can visually spot any mistakes.

    2. plots/07_decision_boundary.png
       Shows the curved boundary the network learned to separate class 0
       from class 1. This is the most important plot for a NONLINEAR
       classification problem: it visually proves the network learned a
       curved boundary, not just a straight line.

What is a "decision boundary"?
    Imagine coloring every single point on the graph based on what the
    model would predict for it. The line where the color changes from
    "class 0" to "class 1" is the decision boundary. For a nonlinear
    dataset like ours, a good model should produce a CURVED boundary that
    follows the shape of the two moons.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

from config import TEST_CSV, MODEL_PATH, PREDICTIONS_CSV, PLOTS_DIR


def make_predictions():
    df = pd.read_csv(TEST_CSV)
    X_test = df[["x", "y"]].values
    y_test = df["target"].values

    model = keras.models.load_model(MODEL_PATH)
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    predictions_df = pd.DataFrame({
        "x": df["x"],
        "y": df["y"],
        "actual_target": y_test,
        "predicted_probability": y_pred_proba,
        "predicted_target": y_pred,
    })
    predictions_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Predictions saved to: {PREDICTIONS_CSV}")

    return predictions_df, model


def plot_actual_vs_predicted(predictions_df: pd.DataFrame, save_path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)

    colors = {0: "#3b82f6", 1: "#f97316"}

    for cls, color in colors.items():
        subset = predictions_df[predictions_df["actual_target"] == cls]
        axes[0].scatter(subset["x"], subset["y"], c=color, label=f"Class {cls}", alpha=0.7, edgecolors="k", linewidths=0.3)
    axes[0].set_title("Actual Classes (Test Set)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for cls, color in colors.items():
        subset = predictions_df[predictions_df["predicted_target"] == cls]
        axes[1].scatter(subset["x"], subset["y"], c=color, label=f"Predicted {cls}", alpha=0.7, edgecolors="k", linewidths=0.3)
    axes[1].set_title("Predicted Classes (Test Set)")
    axes[1].set_xlabel("x")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved actual-vs-predicted plot to: {save_path}")


def plot_decision_boundary(model, predictions_df: pd.DataFrame, save_path) -> None:
    x_min = predictions_df["x"].min() - 0.5
    x_max = predictions_df["x"].max() + 0.5
    y_min = predictions_df["y"].min() - 0.5
    y_max = predictions_df["y"].max() + 0.5

    # Build a fine grid of points covering the whole plot area.
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Ask the model for a prediction at every single grid point.
    grid_probs = model.predict(grid_points, verbose=0).reshape(xx.shape)

    plt.figure(figsize=(7.5, 6.5))
    # Background shading: shows the model's predicted probability everywhere.
    contour = plt.contourf(xx, yy, grid_probs, levels=25, cmap="RdBu_r", alpha=0.75)
    plt.colorbar(contour, label="Predicted probability of Class 1")
    # The 0.5 line IS the decision boundary.
    plt.contour(xx, yy, grid_probs, levels=[0.5], colors="black", linewidths=2)

    class0 = predictions_df[predictions_df["actual_target"] == 0]
    class1 = predictions_df[predictions_df["actual_target"] == 1]
    plt.scatter(class0["x"], class0["y"], c="#3b82f6", edgecolors="k", linewidths=0.5, label="Class 0 (actual)")
    plt.scatter(class1["x"], class1["y"], c="#f97316", edgecolors="k", linewidths=0.5, label="Class 1 (actual)")

    plt.title("Learned Nonlinear Decision Boundary (Test Set)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved decision boundary plot to: {save_path}")


def main():
    predictions_df, model = make_predictions()
    plot_actual_vs_predicted(predictions_df, PLOTS_DIR / "08_actual_vs_predicted.png")
    plot_decision_boundary(model, predictions_df, PLOTS_DIR / "07_decision_boundary.png")


if __name__ == "__main__":
    main()
