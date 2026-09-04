"""
generate_dataset.py
--------------------
Creates a random, but reproducible, NONLINEAR binary classification dataset.

What does "nonlinear" mean here?
    A "linear" classifier can only separate two classes with a single
    straight line (in 2D). Our dataset is shaped like two interleaving
    crescent moons ("two moons" dataset), so no single straight line can
    correctly separate class 0 from class 1. A neural network with hidden
    layers and activation functions, however, CAN learn a curved boundary
    that separates the two moons correctly. This is exactly what makes the
    assignment interesting: it forces the network to prove it can learn a
    nonlinear decision boundary.

This script:
    1. Generates 1000 (x, y) points belonging to class 0 or class 1 using
       scikit-learn's `make_moons` function.
    2. Saves the full dataset to data/nonlinear_dataset_1000.csv
    3. Splits the data into 80% train / 20% test.
    4. Saves data/train.csv and data/test.csv
    5. Saves a small preview (first 10 rows) to outputs/dataset_preview.csv
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from config import (
    SEED,
    N_SAMPLES,
    NOISE,
    TEST_SIZE,
    FULL_DATASET_CSV,
    TRAIN_CSV,
    TEST_CSV,
    DATASET_PREVIEW_CSV,
)


def generate_dataset() -> pd.DataFrame:
    """Generate the full 1000-row nonlinear dataset and return it as a DataFrame."""

    # Set the seed for NumPy as well, so that any extra randomness we add
    # later (if any) is also reproducible.
    np.random.seed(SEED)

    # `make_moons` creates two interleaving half-circle ("moon") shapes.
    # X will have shape (1000, 2) -> the two columns are our features x, y.
    # y will have shape (1000,)   -> the class label, 0 or 1.
    X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)

    df = pd.DataFrame(X, columns=["x", "y"])
    df["target"] = y.astype(int)

    return df


def split_and_save(df: pd.DataFrame) -> None:
    """Split the dataset into train/test sets and save all CSV files."""

    # Save the FULL dataset first, exactly as required by the assignment.
    df.to_csv(FULL_DATASET_CSV, index=False)

    # Save a small preview file (first 10 rows) so it's easy to quickly
    # inspect the shape of the data without opening the full CSV.
    df.head(10).to_csv(DATASET_PREVIEW_CSV, index=False)

    # `stratify=df["target"]` makes sure that both the train and test sets
    # keep roughly the same proportion of class 0 / class 1 as the full
    # dataset. This avoids, by bad luck, ending up with a test set that is
    # almost entirely one class.
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["target"],
    )

    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print(f"Full dataset saved:  {FULL_DATASET_CSV}  ({len(df)} rows)")
    print(f"Training set saved:  {TRAIN_CSV}  ({len(train_df)} rows)")
    print(f"Testing set saved:   {TEST_CSV}  ({len(test_df)} rows)")


def main():
    df = generate_dataset()

    # A few sanity checks so we can be sure the requirements are respected.
    assert len(df) == N_SAMPLES, "Dataset must contain exactly 1000 samples."
    assert set(df["target"].unique()) <= {0, 1}, "Target must be binary (0/1)."
    assert list(df.columns) == ["x", "y", "target"], "Columns must be x, y, target."

    split_and_save(df)


if __name__ == "__main__":
    main()
