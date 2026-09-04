# Model Explanation (Beginner Friendly)

This document explains every concept behind the project, in simple terms.

## The data

**What is a feature?**
A feature is one piece of information we measure about each data point.
Here, every point has exactly two features, `x` and `y` — think of them as
the horizontal and vertical position of a dot on a graph.

**What are `x` and `y`?**
They are just two numbers (coordinates) describing where a point sits on a
2D plane. They don't represent anything from the real world here — they are
simply the two inputs our nonlinear pattern is built from.

**What is the target?**
The target (also called the "label") is the correct answer we want the
model to predict: `0` or `1`, representing which of the two classes a
point belongs to.

**What does binary classification mean?**
"Binary" means there are only two possible categories. "Classification"
means the model's job is to decide which category a new point belongs to.
So binary classification = sorting things into one of exactly two boxes.

**What makes the dataset nonlinear?**
The two classes are shaped like two interleaving crescent moons. You
cannot draw one straight line that puts all of class 0 on one side and all
of class 1 on the other — the boundary between them has to curve. That
curviness is what "nonlinear" means.

## The network's building blocks

**What is a neuron?**
A neuron is a small computing unit. It takes several numbers in, multiplies
each one by a "weight", adds them up together with a "bias", and then
passes the result through an activation function to produce one output
number.

**What is a hidden layer?**
A hidden layer is a group of neurons that sits between the input and the
output. It's called "hidden" because we don't directly see its numbers —
we only see the final prediction. Hidden layers let the network build up
increasingly abstract representations of the data.

**Why are there 4 hidden layers?**
This assignment specifically asks for 4 hidden layers so that we can watch
how a value gets progressively reshaped by four different activation
functions in a row, and so the network has enough depth to bend a straight
line into the curved boundary the two-moons data needs.

**Why are there exactly 6 neurons per hidden layer?**
Six neurons is simply the width chosen by the assignment — enough capacity
to represent a moderately curved decision boundary for a small, 2-feature
dataset, without making the network unnecessarily large.

**What is an activation function?**
It's a mathematical function applied to a neuron's raw output. Without it,
stacking layers would be mathematically equivalent to a single layer (just
multiplying matrices together), so the network could only ever learn
straight lines. Activation functions introduce "bends", which is what lets
deep networks learn curved, nonlinear patterns.

**Why do we need activation functions?**
Because our data is nonlinear! Only by adding these mathematical "bends"
between layers can the network approximate a curved decision boundary like
the one needed to separate two interleaving moons.

**Why does each hidden layer use a different activation function?**
This is a deliberate design/teaching choice for this assignment. It lets us
see, within a single small network, how four very different activation
"shapes" — ReLU, Tanh, Sigmoid, ELU — each reshape the data as it flows
through the network.

**What does ReLU do?**
ReLU ("Rectified Linear Unit") outputs the input unchanged if it's
positive, and outputs 0 if it's negative. It's simple, fast to compute, and
very popular because it helps avoid a training problem called "vanishing
gradients".

**What does Tanh do?**
Tanh squashes any input into the range -1 to 1, and is centered on 0. It's
useful because its symmetric, zero-centered output can make learning more
stable than functions that only output positive numbers.

**What does Sigmoid do?**
Sigmoid squashes any input into the range 0 to 1, following an S-shaped
curve. Because its output can be read as a probability, it's especially
useful in the output layer of a binary classifier.

**What does ELU do?**
ELU ("Exponential Linear Unit") behaves like ReLU for positive inputs, but
for negative inputs it smoothly curves towards a small negative value
instead of hitting a hard 0. This can help the network keep learning even
when a neuron's input is negative.

**Why does the output layer use Sigmoid?**
Because we need a single number between 0 and 1 that represents "the
probability this point belongs to class 1". Sigmoid is exactly built for
that: it always outputs a value in the (0, 1) range, which is a valid
probability.

## How the network learns

**What is forward propagation?**
It's the process of pushing an input (x, y) through every layer of the
network, in order, until you get a final prediction. Each layer's output
becomes the next layer's input.

**What is loss?**
Loss is a single number that measures how wrong a prediction was. We use
"binary_crossentropy", which gives a small penalty for a confident correct
answer and a large penalty for a confident WRONG answer.

**What is backpropagation?**
After computing the loss, backpropagation works backwards through the
network, calculating exactly how much each individual weight contributed
to that error. This uses calculus (the chain rule) under the hood, but
TensorFlow handles all of the math for us automatically.

**What are weights and biases?**
Weights are the numbers each neuron multiplies its inputs by; biases are
extra numbers added afterward. Together, weights and biases are the
network's "knowledge" — training is the process of finding good values for
all of them.

**What is an optimizer?**
An optimizer is the algorithm that decides HOW to adjust the weights and
biases, given the information from backpropagation, in order to reduce the
loss.

**What is the Adam optimizer?**
Adam is a widely-used optimizer that adapts how big a step to take for each
individual weight, based on how that weight's gradient has behaved
recently. It generally converges faster and more reliably than plain
gradient descent, which is why it's a common default choice.

**What is an epoch?**
An epoch is one complete pass through the entire training dataset. If
training runs for 96 epochs, it means the model has seen every training
example 96 times (though the weights are updated more frequently than
that, once per batch).

**What is batch size?**
Instead of updating the weights after looking at the ENTIRE training set at
once, we split the data into small "batches" (16 examples at a time in
this project) and update the weights after each batch. This makes training
faster and often more stable.

**What is training data?**
The portion of the dataset (800 points, 80%) the model is allowed to learn
from directly — its weights are adjusted based on this data.

**What is testing data?**
The portion of the dataset (200 points, 20%) the model NEVER sees during
training. We only use it afterwards, to honestly check how well the model
generalizes to new, unseen data.

## Evaluating the results

**What does accuracy mean?**
The percentage of test points the model classified correctly (predicted
class equals actual class).

**What is a confusion matrix?**
A small table that breaks down exactly how many predictions fell into each
of 4 categories: True Positive, True Negative, False Positive, False
Negative. See `plots/06_confusion_matrix.png`.

* **True Positive (TP):** actual class 1, predicted class 1 (correct).
* **True Negative (TN):** actual class 0, predicted class 0 (correct).
* **False Positive (FP):** actual class 0, but predicted class 1 (mistake).
* **False Negative (FN):** actual class 1, but predicted class 0 (mistake).

From these four numbers:
* Accuracy = (TP + TN) / (TP + TN + FP + FN)
* Precision = TP / (TP + FP) — "when we said 1, how often were we right?"
* Recall = TP / (TP + FN) — "of all real 1's, how many did we catch?"
* F1-score = the harmonic mean of precision and recall, balancing both.

**What are precision and recall?**
See above — precision cares about false alarms (FP), recall cares about
missed cases (FN). Depending on the real-world problem, you might care more
about one than the other (e.g. in medical screening you usually want very
high recall, even if it costs some precision).

**What is a decision boundary?**
It's the imaginary line (or curve) in the input space where the model's
prediction flips from "class 0" to "class 1". `plots/07_decision_boundary.png`
shows this directly: the black line is where the predicted probability
crosses 0.5. Because it curves to follow the two moons, it visually proves
the network learned a genuinely nonlinear boundary.

## The exact architecture used in this project

```text
Input: 2 features (x, y)

Hidden Layer 1: 6 neurons -> ReLU
Hidden Layer 2: 6 neurons -> Tanh
Hidden Layer 3: 6 neurons -> Sigmoid
Hidden Layer 4: 6 neurons -> ELU

Output Layer:   1 neuron  -> Sigmoid
```

Data flows straight down this list: the 2 input numbers are transformed by
layer 1 into 6 numbers, those 6 numbers are transformed by layer 2 into
another 6 numbers, and so on, until the output layer compresses everything
down into the final single probability.
