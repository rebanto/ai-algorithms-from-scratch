# Simple Neural Network Implementation

This repository contains a basic implementation of a two-layer neural network for binary classification, leveraging NumPy for numerical operations and scikit-learn for synthetic dataset generation.

## Overview

This implementation provides the fundamental building blocks of a neural network:

-   **`Dense` Layer:** A fully connected linear layer performing the affine transformation: $\text{output} = \text{input} \cdot \text{weights} + \text{biases}$.
-   **`ReLU` Activation:** The Rectified Linear Unit activation function, defined as $\text{output} = \max(0, \text{input})$.
-   **`Sigmoid` Activation:** The sigmoid activation function, $\text{output} = \frac{1}{1 + e^{-\text{input}}}$. Commonly used in binary classification for outputting probabilities between 0 and 1.
-   **`BinaryCrossentropy` Loss:** The loss function for binary classification, measuring the discrepancy between predicted probabilities and true binary labels:
    $$L = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$
    where $y_i$ represents the true label and $\hat{y}_i$ is the predicted probability.
-   **`SGD` Optimizer:** The Stochastic Gradient Descent optimizer, a basic optimization algorithm that iteratively updates the network's weights and biases based on the gradient of the loss function.

The script demonstrates the practical application of these components by training a simple two-layer neural network on the `make_moons` dataset from scikit-learn, a benchmark dataset for non-linear binary classification.

## Getting Started

Ensure you have the following libraries installed in your Python environment:

-   **NumPy:** For efficient numerical computations. Install via pip:

    ```
    pip install numpy
    ```

-   **scikit-learn:** Used here for generating the synthetic `make_moons` dataset. Install via pip:

    ```
    pip install scikit-learn
    ```

## Usage

To execute the code, simply run the Python file:

```
python basic_neural_network.py
```

Upon execution, the script will:

1. Generate a non-linearly separable binary classification dataset using make_moons.
2. Initialize a two-layer neural network.
   -  The hidden layer uses the ReLU activation function
   -  The output layer employs the Sigmoid activation function.
3. Define the Binary Cross-entropy loss function to quantify the model's errors.
4. Initialize the SGD optimizer to manage the learning process.
5. Train the neural network for a specified number of epochs.
   -  During training, the loss and accuracy will be printed every 1000 epochs to monitor progress.

## Code Structure

The codebase is organized into distinct classes, each representing a core component of the neural network:

#### Dense Class

Implements a fully connected layer, including:

* forward pass (computes outputs)
* backward pass (computes gradients)

#### ReLU Class

Implements the ReLU activation function, with:

* forward and backward methods

#### Sigmoid Class

Implements the Sigmoid activation function, including:

* forward and backward computations

#### BinaryCrossentropy Class

Implements the binary cross-entropy loss function, with:

* loss computation
* Gradient calculation (backward pass)

#### SGD Class

Implements the Stochastic Gradient Descent optimizer:

* `step()` method updates the weights and biases using the computed gradients
