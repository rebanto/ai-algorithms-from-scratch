import numpy as np
import matplotlib.pyplot as plt
import os

# linear regression is probably the most "fundamental" thing in all of ML.
# everything else kind of builds on it. the idea is embarrassingly simple:
# fit a line (or hyperplane) to your data by minimizing squared error.
# i'm implementing two ways to do it -- gradient descent (iterative) and
# the normal equation (closed-form, exact answer in one shot).

class LinearRegression:
    def __init__(self, learning_rate=0.05, epochs=2000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = None   # slope(s)
        self.b = None   # intercept
        self.loss_history = []

    def _predict(self, X):
        return X @ self.w + self.b

    def fit(self, X, y):
        n, d = X.shape
        # start weights at zero -- could use random init but for linear regression
        # the loss surface is convex so it honestly doesn't matter where you start
        self.w = np.zeros(d)
        self.b = 0.0

        for epoch in range(self.epochs):
            y_pred = self._predict(X)
            residuals = y_pred - y

            # MSE loss: L = (1/n) * sum((y_pred - y)^2)
            loss = np.mean(residuals ** 2)
            self.loss_history.append(loss)

            # gradients -- just calculus on the MSE formula
            # dL/dw = (2/n) * X^T @ residuals
            # dL/db = (2/n) * sum(residuals)
            dw = (2 / n) * (X.T @ residuals)
            db = (2 / n) * np.sum(residuals)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 200 == 0:
                print(f"  epoch {epoch:4d} | loss = {loss:.5f}")

    def predict(self, X):
        return self._predict(X)

    def mse(self, X, y):
        return np.mean((self._predict(X) - y) ** 2)


def normal_equation(X, y):
    # the closed-form solution: w = (X^T X)^{-1} X^T y
    # this gives the *exact* minimum in one calculation -- no iteration needed.
    # the catch: it requires matrix inversion which is O(d^3), so it blows up
    # when you have thousands of features. gradient descent scales much better.
    # augmenting X with a column of 1s to absorb the bias term into w
    X_aug = np.column_stack([np.ones(len(X)), X])
    # pinv is safer than inv -- handles near-singular cases
    return np.linalg.pinv(X_aug.T @ X_aug) @ (X_aug.T @ y)


# ---- data generation ----

np.random.seed(42)
X_raw = 2 * np.random.rand(200, 1)
# true relationship: y = 3 + 5x, with Gaussian noise on top
y = 3 + 5 * X_raw[:, 0] + np.random.randn(200) * 0.8

split = 160
X_train, X_test = X_raw[:split], X_raw[split:]
y_train, y_test = y[:split], y[split:]

# ---- training ----

print("Training via gradient descent...")
model = LinearRegression(learning_rate=0.05, epochs=2000)
model.fit(X_train, y_train)

# one-shot closed-form solution
ne_params = normal_equation(X_train, y_train)  # [bias, weight]

print(f"\nGradient descent  →  w = {model.w[0]:.4f},  b = {model.b:.4f}")
print(f"Normal equation   →  w = {ne_params[1]:.4f},  b = {ne_params[0]:.4f}")
print(f"True values       →  w = 5.0000,  b = 3.0000")
print(f"\nTest MSE (gradient descent): {model.mse(X_test, y_test):.5f}")

# ---- plots ----

os.makedirs('plots', exist_ok=True)

# 1. loss curve
plt.figure(figsize=(8, 4))
plt.plot(model.loss_history, color='steelblue', linewidth=1.5)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join('plots', 'loss_curve.png'), dpi=150)
plt.show()

# 2. regression fit comparison
x_line = np.linspace(0, 2, 200).reshape(-1, 1)
y_gd  = model.predict(x_line)
y_ne  = ne_params[0] + ne_params[1] * x_line[:, 0]
y_true_line = 3 + 5 * x_line[:, 0]

plt.figure(figsize=(9, 6))
plt.scatter(X_train, y_train, alpha=0.4, color='steelblue', label='Train data', s=25)
plt.scatter(X_test,  y_test,  alpha=0.5, color='coral',     label='Test data',  s=30, marker='x')
plt.plot(x_line, y_gd,        'k-',  linewidth=2,   label='Gradient Descent')
plt.plot(x_line, y_ne,        'r--', linewidth=2,   label='Normal Equation')
plt.plot(x_line, y_true_line, 'g--', linewidth=1.5, label='True function', alpha=0.7)
plt.title('Linear Regression — Gradient Descent vs Normal Equation')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join('plots', 'regression_fit.png'), dpi=150)
plt.show()
