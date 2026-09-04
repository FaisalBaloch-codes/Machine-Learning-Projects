# Project Workflow (Explained Simply)

This page walks through everything the project does, step by step, in plain
language — no jargon left unexplained.

```text
1. We generate 1000 points.
2. Each point has two numbers: x and y.
3. Each point belongs to class 0 or class 1 (this is our "target").
4. We split the data: 800 points for training, 200 for testing.
5. The neural network receives x and y as input.
6. The values pass through four hidden layers, one after another.
7. Each hidden layer transforms the numbers using 6 neurons and an
   activation function (ReLU, then Tanh, then Sigmoid, then ELU).
8. The output layer produces a single number between 0 and 1 — the
   probability that the point belongs to class 1.
9. During training, that prediction is compared with the real answer.
10. The "loss" tells the model how wrong its prediction was.
11. Backpropagation calculates how much each weight in the network
    contributed to that error.
12. The Adam optimizer nudges every weight slightly, to reduce the error.
13. This whole process (steps 6-12) repeats for many epochs.
14. Finally, we test the trained model on the 200 points it has NEVER
    seen before, to check whether it really learned the pattern.
```

## Why do we split the data into train and test?

Imagine studying for an exam using only the exact questions that will be
on the test. You might memorize the answers without actually
understanding the subject. A machine learning model has the same risk —
this is called **overfitting**. By testing the model on data it has never
seen (the test set), we get an honest measurement of how well it actually
learned the underlying pattern, rather than just memorizing examples.

## Why is the dataset nonlinear?

Our dataset is shaped like two crescent moons that curve into each other
(see `plots/01_dataset_distribution.png`). No single straight line can
separate the two moons correctly — you would need a *curved* line. That
curviness is what "nonlinear" means, and it's exactly the kind of problem
that a simple linear model (like plain logistic regression) struggles
with, but a small neural network with hidden layers can solve.

## The full pipeline, visually

```text
generate_dataset.py            -> data/nonlinear_dataset_1000.csv
        |                           data/train.csv, data/test.csv
        v
visualize_dataset.py           -> plots/01_dataset_distribution.png
        |
        v
train_model.py                 -> models/nonlinear_binary_classifier.keras
  (uses build_model.py)            outputs/training_history.csv
                                    outputs/model_summary.txt
                                    plots/02..05 (loss/accuracy plots)
        |
        v
evaluate_model.py               -> outputs/evaluation_metrics.csv
                                    outputs/classification_report.txt
                                    plots/06_confusion_matrix.png
        |
        v
predict.py                      -> outputs/predictions.csv
                                    plots/07_decision_boundary.png
                                    plots/08_actual_vs_predicted.png
        |
        v
write_results_doc.py            -> docs/results.md
```

Running `python src/run_pipeline.py` performs every one of these steps
automatically, in the correct order.
