import numpy as np
from sklearn.datasets import make_moons # a simple classification dataset
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import pandas as pd

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

df = pd.DataFrame(X, columns=['X_1', 'X_2'])
df['y'] = y

# first 10 rows
df_head = df.head(10)

print(df_head.to_markdown(index=False, numalign="left", stralign="left"))

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', alpha=0.6, edgecolors='w', linewidth=0.5)
plt.title('Make Moons Dataset')
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.grid(True, linestyle='--', alpha=0.6)

legend1 = plt.legend(*scatter.legend_elements(), title='Classes')
plt.gca().add_artist(legend1)

# Ensure plots directory exists and save the dataset scatter image
os.makedirs('plots', exist_ok=True)
plt.savefig(os.path.join('plots', 'dataset_scatter.png'), bbox_inches='tight', dpi=150)
plt.show()


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


print("Final evaluation and plotting decision boundary...")
dense1.forward(X)
relu.forward(dense1.output)
dense2.forward(relu.output)
sigmoid.forward(dense2.output)
final_loss = loss_fn.forward(sigmoid.output, y)
final_acc = np.mean((sigmoid.output > 0.5).astype(int) == y)
print(f"final: loss={final_loss:.3f}, acc={final_acc:.3f}")


pad = 0.5
x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]


dense1.forward(grid); relu.forward(dense1.output)
dense2.forward(relu.output); sigmoid.forward(dense2.output)
Z = sigmoid.output.reshape(xx.shape)


plt.figure(figsize=(8, 6))
cf = plt.contourf(xx, yy, Z, levels=50, cmap='viridis', alpha=0.3)
boundary = plt.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=1)
scatter = plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='viridis', marker='o', alpha=0.6, edgecolors='w', linewidth=0.5)
plt.title('Decision Boundary — Trained Model')
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.grid(True, linestyle='--', alpha=0.6)

cbar = plt.colorbar(cf)
cbar.set_label('P(class=1)')

class_handles, class_labels = scatter.legend_elements()
boundary_handle = Line2D([0], [0], color='k', lw=1)
plt.legend([*class_handles, boundary_handle], [*class_labels, 'Decision boundary'], title='Classes/Boundary')

plt.savefig(os.path.join('plots', 'decision_boundary.png'), bbox_inches='tight', dpi=150)
plt.show()

