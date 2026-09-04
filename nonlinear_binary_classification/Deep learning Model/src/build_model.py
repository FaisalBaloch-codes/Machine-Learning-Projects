"""
build_model.py
---------------
Defines the exact neural network architecture required by the assignment:

    Input (2 features: x, y)
        -> Dense(6, activation="relu")
        -> Dense(6, activation="tanh")
        -> Dense(6, activation="sigmoid")
        -> Dense(6, activation="elu")
        -> Dense(1, activation="sigmoid")   (output layer)

Why this architecture?
    * 4 hidden layers give the network enough "depth" to combine simple
      patterns into more complex ones, which is necessary to learn a curved
      (nonlinear) boundary like the two-moons shape.
    * 6 neurons per hidden layer is simply the width chosen by the
      assignment — enough neurons to represent a reasonably complex curve
      without being an unnecessarily large network for such a small, simple
      2-feature dataset.
    * Using a DIFFERENT activation function in each hidden layer is a
      deliberate teaching choice: it lets us see, in one single network, how
      ReLU, Tanh, Sigmoid and ELU each reshape the data as it flows through
      the layers.
    * The output layer has exactly 1 neuron with a Sigmoid activation
      because Sigmoid squashes any real number into the range (0, 1), which
      can be read as "the probability that this point belongs to class 1".
      That is exactly what we need for BINARY classification.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import SEED, HIDDEN_LAYER_UNITS, HIDDEN_LAYER_ACTIVATIONS, OUTPUT_ACTIVATION


def set_global_seeds(seed: int = SEED) -> None:
    """Set the random seed for NumPy, Python's random module and TensorFlow.

    Neural network training involves several sources of randomness (initial
    weight values, order in which batches are shuffled, etc.). Setting all
    of these seeds makes the training process reproducible: running the
    project again should give (very close to) the same results.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def build_model(input_dim: int = 2) -> keras.Model:
    """Build and return the required neural network, uncompiled."""

    set_global_seeds()

    model = keras.Sequential(name="nonlinear_binary_classifier")

    # Input layer: tells Keras to expect 2 input features (x and y).
    model.add(keras.Input(shape=(input_dim,), name="input_features"))

    # Hidden layers: exactly 4 layers, each with 6 neurons, each using a
    # different activation function (as required by the assignment).
    for i, activation in enumerate(HIDDEN_LAYER_ACTIVATIONS, start=1):
        model.add(
            layers.Dense(
                HIDDEN_LAYER_UNITS,
                activation=activation,
                name=f"hidden_layer_{i}_{activation}",
            )
        )

    # Output layer: 1 neuron, Sigmoid activation -> outputs a probability
    # between 0 and 1 that the sample belongs to class 1.
    model.add(layers.Dense(1, activation=OUTPUT_ACTIVATION, name="output_layer"))

    return model


def compile_model(model: keras.Model) -> keras.Model:
    """Compile the model with an optimizer, loss function and metrics."""

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def main():
    """Quick manual check: build the model and print its summary."""
    model = build_model()
    model = compile_model(model)
    model.summary()

    # A few automated checks confirming the architecture matches the
    # assignment's strict requirements.
    hidden_layers = model.layers[:-1]
    assert len(hidden_layers) == 4, "There must be exactly 4 hidden layers."
    for layer in hidden_layers:
        assert layer.units == 6, "Every hidden layer must have exactly 6 neurons."
    activations_used = [layer.activation.__name__ for layer in hidden_layers]
    assert len(set(activations_used)) == 4, "All 4 hidden layers must use different activations."
    print("\nArchitecture check passed:", activations_used)


if __name__ == "__main__":
    main()
