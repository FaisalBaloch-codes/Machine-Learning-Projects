"""
run_pipeline.py
----------------
Runs the ENTIRE project from start to finish with a single command:

    python src/run_pipeline.py

Steps performed (in order):
    1. Generate the 1000-row nonlinear dataset -> data/nonlinear_dataset_1000.csv
    2. Split it into train.csv / test.csv
    3. Visualize the raw dataset -> plots/01_dataset_distribution.png
    4. Build + train the neural network -> models/nonlinear_binary_classifier.keras
    5. Evaluate the model on the test set -> outputs/evaluation_metrics.csv, etc.
    6. Generate predictions + decision boundary plots
    7. Write docs/results.md with the real numbers from this run

This script uses only relative imports/paths built with `pathlib`
(see config.py), so it works correctly no matter which computer or folder
you run it from, as long as you run it from the project's root folder.
"""

import sys
from pathlib import Path

# Make sure "src" is on the Python path so `import config` etc. works
# whether this script is run as `python src/run_pipeline.py` or
# `python run_pipeline.py` from inside the src/ folder.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import generate_dataset
import visualize_dataset
import train_model
import evaluate_model
import predict
import write_results_doc


def banner(step_number: int, text: str) -> None:
    print("\n" + "=" * 70)
    print(f"STEP {step_number}: {text}")
    print("=" * 70)


def main():
    banner(1, "Generating the nonlinear dataset (1000 samples)")
    generate_dataset.main()

    banner(2, "Visualizing the raw dataset")
    visualize_dataset.main()

    banner(3, "Building and training the neural network")
    train_model.train()

    banner(4, "Evaluating the model on the unseen test set")
    evaluate_model.evaluate()

    banner(5, "Generating predictions, decision boundary and comparison plots")
    predict.main()

    banner(6, "Writing docs/results.md with the real results from this run")
    write_results_doc.main()

    print("\nAll done! Check the data/, models/, outputs/, plots/ and docs/ folders.")


if __name__ == "__main__":
    main()
