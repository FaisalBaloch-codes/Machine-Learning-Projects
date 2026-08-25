"""
Category Prediction Model - Main Pipeline
==========================================

Runs the complete Logistic Regression classification pipeline:

1. Create 80/20 stratified split
2. Create 90/10 validation split
3. Calculate TF-IDF using min_df and max_df
4. Train L1 Logistic Regression
5. Train L2 Logistic Regression
6. Compare L1 vs L2 on validation data
7. Select L1 based on Macro F1
8. Test selected L1 model on untouched 1,000 test articles

Run the complete project with:

    python main.py
"""

import subprocess
import sys


def run_script(script_name):
    """Run a Python script using the current Python environment."""

    print("\n" + "=" * 70)
    print(f"RUNNING: {script_name}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, f"src/{script_name}"],
        check=False
    )

    if result.returncode != 0:
        print(f"\nERROR: {script_name} failed.")
        sys.exit(result.returncode)

    print(f"\nCOMPLETED: {script_name}")


def main():

    print("=" * 70)
    print("NEWS CATEGORY PREDICTION - COMPLETE PIPELINE")
    print("=" * 70)

    print("\nThis pipeline will run:")
    print("1. Data splitting and TF-IDF")
    print("2. L1 vs L2 Logistic Regression validation")
    print("3. Final testing on untouched test data")

    # -----------------------------------------------------------
    # STEP 1
    # -----------------------------------------------------------

    run_script("validation_split.py")

    # -----------------------------------------------------------
    # STEP 2
    # -----------------------------------------------------------

    run_script("validate_logistic.py")

    # -----------------------------------------------------------
    # STEP 3
    # -----------------------------------------------------------

    run_script("test_logistic.py")

    # -----------------------------------------------------------
    # DONE
    # -----------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMPLETE PIPELINE FINISHED")
    print("=" * 70)

    print("\nYour Logistic Regression pipeline is complete.")

    print("\nGenerated Prediction Files:")
    print("  - results/L1_model_predictions.csv")
    print("  - results/L2_model_predictions.csv")
    print("  - results/log_regmodel_predictions.csv")

    print("\nPipeline Flow:")
    print("Dataset")
    print("   |")
    print("   v")
    print("80% Training / 20% Test")
    print("   |")
    print("   v")
    print("90% Training / 10% Validation")
    print("   |")
    print("   v")
    print("TF-IDF (min_df=2, max_df=0.95)")
    print("   |")
    print("   v")
    print("L1 vs L2 Logistic Regression")
    print("   |")
    print("   v")
    print("Validation comparison")
    print("   |")
    print("   v")
    print("L1 selected")
    print("   |")
    print("   v")
    print("Final test & prediction CSV generation")


if __name__ == "__main__":
    main()