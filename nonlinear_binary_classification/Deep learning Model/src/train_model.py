"""
train_model.py
---------------
Loads the training data, builds the required neural network, trains it, and
saves:
    * the trained model               -> models/nonlinear_binary_classifier.keras
    * the training history (CSV)      -> outputs/training_history.csv
    * the model summary (text file)   -> outputs/model_summary.txt
    * training/validation plots (PNG) -> plots/02..05

What happens during "training"?
    1. Forward propagation: the network takes an (x, y) point and pushes it
       through the 4 hidden layers and the output layer, producing a
       predicted probability.
    2. Loss: we compare that probability with the real answer (0 or 1) using
       "binary_crossentropy", a formula that gives a bigger penalty the more
       confidently wrong the prediction was.
    3. Backpropagation: TensorFlow automatically calculates how much each
       weight in the network contributed to the error.
    4. The Adam optimizer then nudges every weight a small amount in the
       direction that reduces the loss.
    5. This repeats for many "epochs" (full passes over the training data)
       until the loss stops improving, or we hit the epoch limit.
"""

import io
import contextlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras

from config import (
    TRAIN_CSV,
    MODEL_PATH,
    TRAINING_HISTORY_CSV,
    MODEL_SUMMARY_TXT,
    PLOTS_DIR,
    MAX_EPOCHS,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    VALIDATION_SPLIT,
)
from build_model import build_model, compile_model


def load_training_data():
    """Load train.csv and split it into features (X) and target (y)."""
    df = pd.read_csv(TRAIN_CSV)
    X = df[["x", "y"]].values
    y = df["target"].values
    return X, y


def save_model_summary(model: keras.Model) -> None:
    """Capture Keras' printed model.summary() output into a text file."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        model.summary()
    summary_text = buffer.getvalue()

    explanation = """
How to read this summary
-------------------------
* "Layer (type)"    -> the name we gave the layer and what kind it is (Dense).
* "Output Shape"    -> (None, 6) means "however many samples you give it,
                        each one produces 6 numbers" (one per neuron).
* "Param #"         -> the number of trainable weights + biases in that
                        layer. For a Dense layer going from N inputs to M
                        neurons, the parameter count is (N * M) weights plus
                        M biases, i.e. N*M + M.
                        Example: Input(2) -> Dense(6):
                                 2*6 weights + 6 biases = 18 parameters.
                        Example: Dense(6) -> Dense(6):
                                 6*6 weights + 6 biases = 42 parameters.
* "Total params"    -> the sum of all parameters in the whole network.
                        These are exactly the numbers that get adjusted
                        during training (via backpropagation + Adam).
"""
    with open(MODEL_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(summary_text)
        f.write(explanation)

    print(f"Model summary saved to: {MODEL_SUMMARY_TXT}")


def plot_history(history_df: pd.DataFrame) -> None:
    """Create the 4 required training/validation plots."""

    def _plot(column, val_column, title, ylabel, filename):
        plt.figure(figsize=(7, 5))
        plt.plot(history_df["epoch"], history_df[column], label=column, color="#2563eb")
        if val_column in history_df.columns:
            plt.plot(history_df["epoch"], history_df[val_column], label=val_column, color="#f97316")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / filename, dpi=150)
        plt.close()
        print(f"Saved plot: {PLOTS_DIR / filename}")

    _plot("loss", "val_loss", "Training Loss over Epochs", "Loss", "02_training_loss.png")
    _plot("accuracy", "val_accuracy", "Training Accuracy over Epochs", "Accuracy", "03_training_accuracy.png")

    # Dedicated validation-only plots (as required by the assignment).
    if "val_loss" in history_df.columns:
        plt.figure(figsize=(7, 5))
        plt.plot(history_df["epoch"], history_df["val_loss"], color="#f97316")
        plt.title("Validation Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "04_validation_loss.png", dpi=150)
        plt.close()
        print(f"Saved plot: {PLOTS_DIR / '04_validation_loss.png'}")

    if "val_accuracy" in history_df.columns:
        plt.figure(figsize=(7, 5))
        plt.plot(history_df["epoch"], history_df["val_accuracy"], color="#f97316")
        plt.title("Validation Accuracy over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "05_validation_accuracy.png", dpi=150)
        plt.close()
        print(f"Saved plot: {PLOTS_DIR / '05_validation_accuracy.png'}")


def train():
    X_train, y_train = load_training_data()

    model = build_model(input_dim=X_train.shape[1])
    model = compile_model(model)

    save_model_summary(model)

    # EarlyStopping watches the validation loss and stops training once it
    # stops improving for `EARLY_STOPPING_PATIENCE` epochs in a row. This
    # avoids wasting time training long past the point of improvement and
    # helps prevent overfitting, WITHOUT changing the required architecture.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping],
        verbose=2,
    )

    # Save the trained model to disk so evaluate_model.py / predict.py can
    # load it later without re-training.
    model.save(MODEL_PATH)
    print(f"Trained model saved to: {MODEL_PATH}")

    # Save training history as a CSV file.
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", range(1, len(history_df) + 1))
    history_df.to_csv(TRAINING_HISTORY_CSV, index=False)
    print(f"Training history saved to: {TRAINING_HISTORY_CSV}")

    plot_history(history_df)

    return model, history_df


if __name__ == "__main__":
    train()
