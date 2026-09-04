"""
visualize_dataset.py
---------------------
Creates a scatter plot of the dataset so we can SEE why it is nonlinear.

If you look at the resulting plot (plots/01_dataset_distribution.png) you
will see two curved, interleaving "moon" shapes. Try to imagine drawing a
single straight line on that picture that puts all of class 0 on one side
and all of class 1 on the other — it is impossible! That is what "nonlinear"
means in practice, and it is why a plain straight-line model would fail on
this data, but a neural network with hidden layers can succeed.
"""

import matplotlib
matplotlib.use("Agg")  # allows saving plots without opening a window
import matplotlib.pyplot as plt
import pandas as pd

from config import FULL_DATASET_CSV, PLOTS_DIR


def plot_dataset(df: pd.DataFrame, save_path):
    plt.figure(figsize=(7, 6))

    class0 = df[df["target"] == 0]
    class1 = df[df["target"] == 1]

    plt.scatter(class0["x"], class0["y"], c="#3b82f6", label="Class 0", alpha=0.7, edgecolors="k", linewidths=0.3)
    plt.scatter(class1["x"], class1["y"], c="#f97316", label="Class 1", alpha=0.7, edgecolors="k", linewidths=0.3)

    plt.title("Nonlinear Binary Classification Dataset (Two Moons)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved dataset distribution plot to: {save_path}")


def main():
    df = pd.read_csv(FULL_DATASET_CSV)
    plot_dataset(df, PLOTS_DIR / "01_dataset_distribution.png")


if __name__ == "__main__":
    main()
