import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import os

# logistic regression is basically linear regression for classification.
# instead of outputting a raw number, we squash the output through a sigmoid
# so it becomes a probability between 0 and 1. then we threshold at 0.5.
#
# i realized after building the neural network that logistic regression is
# literally just a neural network with NO hidden layers. it's the degenerate
# case where the network is just: input -> linear -> sigmoid -> output.
# that connection blew my mind a little.

class Sigmoid:
    def forward(self, z):
        # clip to prevent exp overflow for very large negative values
        self.out = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        return self.out

    def backward(self, d_out):
        # derivative of sigmoid: s * (1 - s). clean.
        return d_out * self.out * (1 - self.out)


class BinaryCrossentropy:
    def forward(self, y_pred, y_true):
        # clip so we never take log(0) -- that would be -inf
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_pred, y_true):
        n = len(y_pred)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # this simplifies a lot when combined with sigmoid's backward.
        # separately: d(BCE)/d(pred) = -(y/pred) + (1-y)/(1-pred)
        return (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred)) / n


class LogisticRegression:
    def __init__(self, lr=0.1, epochs=5000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None
        self.loss_history = []
        self.acc_history = []
        self._sigmoid = Sigmoid()
        self._loss_fn = BinaryCrossentropy()

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for epoch in range(self.epochs):
            # forward
            z = X @ self.w + self.b
            y_pred = self._sigmoid.forward(z)

            loss = self._loss_fn.forward(y_pred, y)
            acc  = np.mean((y_pred > 0.5).astype(int) == y)
            self.loss_history.append(loss)
            self.acc_history.append(acc)

            # backward -- chain rule through BCE -> sigmoid -> linear
            d_pred = self._loss_fn.backward(y_pred, y)
            d_z    = self._sigmoid.backward(d_pred)

            dw = X.T @ d_z
            db = np.sum(d_z)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 500 == 0:
                print(f"  epoch {epoch:5d} | loss = {loss:.4f} | acc = {acc:.4f}")

    def predict_proba(self, X):
        z = X @ self.w + self.b
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ---- data ----

X, y = make_moons(n_samples=600, noise=0.2, random_state=42)

# ---- train ----

print("Training logistic regression...")
model = LogisticRegression(lr=0.1, epochs=5000)
model.fit(X, y)
print(f"\nFinal accuracy: {model.score(X, y):.4f}")

# ---- plots ----

os.makedirs('plots', exist_ok=True)

# 1. decision boundary
pad = 0.5
x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict_proba(grid).reshape(xx.shape)

plt.figure(figsize=(8, 6))
cf = plt.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.4)
plt.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu',
            edgecolors='w', linewidth=0.5, alpha=0.85, s=35)
plt.colorbar(cf, label='P(class = 1)')
plt.title('Logistic Regression — Decision Boundary')
plt.xlabel('X₁')
plt.ylabel('X₂')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join('plots', 'decision_boundary.png'), dpi=150)
plt.show()

# 2. training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(model.loss_history, color='steelblue', linewidth=1.5)
ax1.set_title('Training Loss (BCE)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.plot(model.acc_history, color='seagreen', linewidth=1.5)
ax2.set_title('Training Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join('plots', 'training_curves.png'), dpi=150)
plt.show()
