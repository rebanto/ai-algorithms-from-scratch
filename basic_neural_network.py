import numpy as np
from sklearn.datasets import make_moons # a simple classification dataset

class Dense: # fully connected layer
    def __init__(self, in_features, out_features):
        # weights initialized with small random values. this helps prevent initial large gradients
        # that could destabilize training.
        self.weights = 0.1 * np.random.randn(in_features, out_features)
        self.biases = np.zeros((1, out_features))

    def forward(self, x):
        self.inputs = x
        self.output = np.dot(x, self.weights) + self.biases

    def backward(self, d_out):
        # essentially applying the chain rule.
        # we're figuring out how much each weight and bias contributed to the final error.
        self.dweights = np.dot(self.inputs.T, d_out)
        self.dbiases = np.sum(d_out, axis=0, keepdims=True)
        self.dinputs = np.dot(d_out, self.weights.T)

class ReLU:
    def forward(self, x):
        self.inputs = x
        self.output = np.maximum(0, x)

    def backward(self, d_out):
        # gradient passes through only where input was positive.
        # helps solve the "vanishing gradient" problem.
        self.dinputs = d_out * (self.inputs > 0)

class Sigmoid: # outputs values between 0 and 1
    def forward(self, x):
        self.inputs = x
        self.output = 1 / (1 + np.exp(-x))

    def backward(self, d_out):
        self.dinputs = d_out * self.output * (1 - self.output)

class BinaryCrossentropy:
    def forward(self, y_pred, y_true):
        # clip predictions to prevent log(0), which would result in infinite loss.
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        self.dinputs = (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred)) / samples

class MeanSquaredError:
    def forward(self, y_pred, y_true):
        # L = 1/N * sum((y_true - y_pred)^2)
        self.diff = y_pred - y_true
        return np.mean(self.diff**2)

    def backward(self, y_pred, y_true):
        # dL/d(y_pred) = 2/N * sum(y_pred - y_true)
        samples = len(y_pred)
        self.dinputs = (2 * self.diff) / samples

class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, layer):
        layer.weights -= self.lr * layer.dweights
        layer.biases -= self.lr * layer.dbiases

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
y = y.reshape(-1, 1)

# network architecture
dense1, relu = Dense(2, 32), ReLU()
dense2, sigmoid = Dense(32, 1), Sigmoid()
loss_fn = BinaryCrossentropy()
optimizer = SGD(lr=0.1)
NUM_EPOCHS = 20001

# training loop
for epoch in range(NUM_EPOCHS):
    # forward pass
    dense1.forward(X); relu.forward(dense1.output)
    dense2.forward(relu.output); sigmoid.forward(dense2.output)
    loss = loss_fn.forward(sigmoid.output, y)

    # calculate and print accuracy periodically
    acc = np.mean((sigmoid.output > 0.5).astype(int) == y)
    if epoch % 1000 == 0:
        print(f"epoch {epoch}: loss={loss:.3f}, acc={acc:.3f}")

    # backward pass
    loss_fn.backward(sigmoid.output, y)
    sigmoid.backward(loss_fn.dinputs)
    dense2.backward(sigmoid.dinputs)
    relu.backward(dense2.dinputs)
    dense1.backward(relu.dinputs)

    # optimization (params are updated based on calculated gradients)
    optimizer.step(dense1)
    optimizer.step(dense2)
