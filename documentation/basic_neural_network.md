# The Bridge: Multi-Layer Neural Networks

This is where all the previous pieces finally come together. After building linear and logistic models, I realized they’re just single-layer networks. To solve more complex problems (like the non-linear "moons" dataset), I needed to stack these layers and add some hidden neurons. This is a simple two-layer MLP (Multi-Layer Perceptron) that finally feels like "Deep Learning."

Building this was about modularity: taking the matrix multiplications and activation functions from the earlier projects and turning them into a system where I could easily add more layers.

## What's Inside?

I broke the network down into modular pieces so I could swap them around later:

- **`Dense` Layer**: The workhorse. It just does $Y = XW + B$. The "weights" are what the network actually learns.
- **`ReLU` (Rectified Linear Unit)**: My favorite activation function because it's so simple ($\text{max}(0, x)$) but it's the reason we can learn non-linear patterns.
- **`Sigmoid`**: Squashes any number down to between 0 and 1. Perfect for the final output when you just want a probability.
- **`BinaryCrossentropy`**: The loss function. It's basically a way to shout at the network: "You were 90% sure this was a 1, but it was actually a 0! That's a huge error!" 
- **`SGD` (Stochastic Gradient Descent)**: The optimizer that actually does the work of nudging the weights in the right direction after every batch.

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
