# Nonlinear Binary Classification Using Deep Learning and TensorFlow

## Objective

This project implements a small deep neural network with **TensorFlow /
Keras** that learns to classify points from a randomly generated,
**nonlinear** ("two moons") dataset into one of two classes (0 or 1). It
was built to satisfy a BSCS Deep Learning assignment requiring:

* A 1,000-sample, 2-feature (`x`, `y`) nonlinear binary classification
  dataset, generated reproducibly.
* A neural network with exactly **4 hidden layers**, each with exactly
  **6 neurons**, each using a **different activation function**.
* A full evaluation (accuracy, precision, recall, F1, AUC), plots, and a
  saved trained model.

Every file in this repository — the dataset CSVs, the trained model, the
plots, and the metrics — was produced by actually running the code, not
invented by hand.

## Technologies

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

## Dataset

* **1,000 samples**, 2 features named exactly `x` and `y`, and a binary
  `target` column (0 or 1).
* Generated with scikit-learn's `make_moons`, which produces two
  interleaving crescent ("moon") shapes — a classic nonlinear pattern that
  cannot be separated by a single straight line.
* Reproducible: everything uses a fixed random seed (`SEED = 42`, see
  `src/config.py`).
* Split into `data/train.csv` (800 rows, 80%) and `data/test.csv` (200
  rows, 20%), using stratified sampling so both classes stay balanced.

See `plots/01_dataset_distribution.png` for a scatter plot proving the
nonlinear shape.

## Neural Network

```text
Input: 2 features (x, y)

Hidden Layer 1: 6 neurons -> ReLU
Hidden Layer 2: 6 neurons -> Tanh
Hidden Layer 3: 6 neurons -> Sigmoid
Hidden Layer 4: 6 neurons -> ELU

Output Layer:   1 neuron  -> Sigmoid
```

* Loss function: `binary_crossentropy`
* Optimizer: `Adam`
* Metrics tracked during training: Accuracy, Precision, Recall, AUC
* Total trainable parameters: **151** (see `outputs/model_summary.txt` for
  the full breakdown of every layer)

Full beginner-friendly explanations of every concept (neuron, activation
function, loss, backpropagation, etc.) live in `docs/model_explanation.md`.

## Workflow

```text
Dataset generation (make_moons, 1000 samples)
 ↓
Train/Test split (80% / 20%)
 ↓
Dataset visualization
 ↓
Build the 4-hidden-layer neural network
 ↓
Forward propagation (input -> hidden layers -> output)
 ↓
Loss calculation (binary_crossentropy)
 ↓
Backpropagation (gradient calculation)
 ↓
Weight updates (Adam optimizer)
 ↓
Training (with EarlyStopping to avoid wasted epochs)
 ↓
Evaluation on the unseen test set
 ↓
Predictions, decision boundary & comparison plots
 ↓
Results documentation (docs/results.md)
```

See `docs/workflow.md` for a plain-language, step-by-step walkthrough.

## Project Structure

```text
nonlinear_binary_classification/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── nonlinear_dataset_1000.csv   # full 1000-row dataset
│   ├── train.csv                    # 800 rows (80%)
│   └── test.csv                     # 200 rows (20%)
│
├── src/
│   ├── config.py                    # shared settings & file paths
│   ├── generate_dataset.py          # creates the nonlinear dataset
│   ├── visualize_dataset.py         # scatter plot of raw data
│   ├── build_model.py               # defines the required architecture
│   ├── train_model.py               # trains the network, saves plots
│   ├── evaluate_model.py            # evaluates on the test set
│   ├── predict.py                   # predictions + decision boundary
│   ├── write_results_doc.py         # writes docs/results.md from real numbers
│   └── run_pipeline.py              # runs everything, start to finish
│
├── models/
│   └── nonlinear_binary_classifier.keras   # the trained model
│
├── outputs/
│   ├── dataset_preview.csv
│   ├── predictions.csv
│   ├── evaluation_metrics.csv
│   ├── classification_report.txt
│   ├── model_summary.txt
│   └── training_history.csv
│
├── plots/
│   ├── 01_dataset_distribution.png
│   ├── 02_training_loss.png
│   ├── 03_training_accuracy.png
│   ├── 04_validation_loss.png
│   ├── 05_validation_accuracy.png
│   ├── 06_confusion_matrix.png
│   ├── 07_decision_boundary.png
│   └── 08_actual_vs_predicted.png
│
└── docs/
    ├── workflow.md
    ├── model_explanation.md
    └── results.md
```

## Installation (Windows + VS Code)

1. Open the project folder in VS Code (`File > Open Folder...`).
2. Open a terminal inside VS Code (`` Terminal > New Terminal `` or `` Ctrl+` ``).
3. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

4. Activate it:

   ```bash
   .venv\Scripts\activate
   ```

   (On macOS/Linux, use `source .venv/bin/activate` instead.)

5. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

## Running the project

From the project's root folder, run:

```bash
python src/run_pipeline.py
```

This single command performs the ENTIRE workflow: it generates the
dataset, splits it, visualizes it, builds and trains the network, saves
the model, evaluates it on the test set, generates predictions, creates
every plot, and writes `docs/results.md` with the real numbers from that
run.

After it finishes, you will find:

* `data/nonlinear_dataset_1000.csv`, `train.csv`, `test.csv`
* `models/nonlinear_binary_classifier.keras`
* Every file listed in `outputs/` and `plots/`
* An updated `docs/results.md`

You can also run any individual step on its own, e.g.:

```bash
python src/generate_dataset.py
python src/train_model.py
python src/evaluate_model.py
python src/predict.py
```

## Actual results from this run

| Metric | Value |
|---|---|
| Test Accuracy | 98.50% |
| Test Loss | 0.0395 |
| Precision | 0.9802 |
| Recall | 0.9900 |
| F1-score | 0.9851 |
| AUC | 0.9995 |
| Epochs actually trained | 96 (out of a max of 150, stopped early) |

Full details in `docs/results.md`.

## Optional: Visualizing the same ideas in TensorFlow Playground

[TensorFlow Playground](https://playground.tensorflow.org/) is a
browser-based tool your teacher may have shown you. It is **not** used to
implement this assignment — the actual implementation is 100% Python +
TensorFlow/Keras, as required. However, it's a great way to build
intuition, since you can watch training happen live in your browser.

If you'd like to explore the same concepts visually:

1. Open <https://playground.tensorflow.org/>.
2. Under **DATA**, choose the "two moons"-like dataset (the "spiral" or
   the circular one also work, but "exclusive or"/moons-style shapes are
   closest to this project).
3. Under **HIDDEN LAYERS**, click the `+` button until you have 4 hidden
   layers, and set each one to 6 neurons using the `-`/`+` controls under
   each layer.
4. Change the **Activation** dropdown at the top — note that Playground
   applies one activation to the whole network at a time, unlike our
   project which deliberately uses a different one per layer.
5. Click the ▶ (play) button and watch the decision boundary evolve as
   training progresses.

This is purely for learning/visualization — remember to tell your teacher
that the graded implementation is the Python/TensorFlow project in this
repository.
