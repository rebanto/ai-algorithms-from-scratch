import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import os

# KNN is the laziest algorithm in existence and i mean that in the best way.
# there is no training. the model literally just memorizes the dataset and then
# at prediction time, finds the K closest training points and takes a vote.
# it sounds too simple to work, but it actually does surprisingly well.
#
# the main thing i had to wrap my head around: smaller K = more complex boundary
# (high variance, low bias), larger K = smoother boundary (low variance, high bias).
# a single neighbor (K=1) fits the training data perfectly -- always right when
# predicting on training points -- but overfits badly on test data.

class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # no gradient descent, no parameters to optimize.
        # this is called "lazy learning" -- defer all computation to predict time.
        self.X_train = X
        self.y_train = y

    def _distances(self, x):
        # euclidean distance from a single point x to every training point
        # broadcasting handles the whole matrix at once: no python loop needed
        return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

    def predict(self, X):
        preds = []
        for x in X:
            dists = self._distances(x)
            # argsort gives indices sorted by distance, take first k
            k_idx    = np.argsort(dists)[:self.k]
            k_labels = self.y_train[k_idx].astype(int)
            # majority vote -- bincount counts occurrences of each integer label
            vote = np.bincount(k_labels).argmax()
            preds.append(vote)
        return np.array(preds)

    def predict_proba(self, X):
        # "probability" = fraction of k neighbors that are class 1.
        # not a real probability in a rigorous sense, but useful for plotting.
        probs = []
        for x in X:
            dists    = self._distances(x)
            k_idx    = np.argsort(dists)[:self.k]
            k_labels = self.y_train[k_idx]
            probs.append(np.mean(k_labels))
        return np.array(probs)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ---- data ----

X, y = make_moons(n_samples=600, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

os.makedirs('plots', exist_ok=True)

# ---- decision boundary grid setup ----

pad = 0.5
x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
# 150x150 to keep prediction time manageable (150^2 = 22500 points each)
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                     np.linspace(y_min, y_max, 150))
grid = np.c_[xx.ravel(), yy.ravel()]

# ---- plot decision boundaries for different K values ----

k_values = [1, 5, 15, 31]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, k in enumerate(k_values):
    knn = KNN(k=k)
    knn.fit(X_train, y_train)
    Z = knn.predict_proba(grid).reshape(xx.shape)
    test_acc = knn.score(X_test, y_test)

    ax = axes[idx]
    ax.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.4)
    ax.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=1.5)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdBu',
               edgecolors='w', linewidth=0.4, alpha=0.8, s=25)
    ax.set_title(f'K = {k}  |  Test acc = {test_acc:.3f}', fontsize=12)
    ax.set_xlabel('X₁')
    ax.set_ylabel('X₂')
    ax.grid(True, linestyle='--', alpha=0.4)

plt.suptitle('KNN — Effect of K on Decision Boundary', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join('plots', 'knn_boundaries.png'), dpi=150, bbox_inches='tight')
plt.show()

# ---- accuracy vs K curve ----

k_range   = range(1, 51, 2)
train_acc = []
test_acc  = []

for k in k_range:
    knn = KNN(k=k)
    knn.fit(X_train, y_train)
    train_acc.append(knn.score(X_train, y_train))
    test_acc.append(knn.score(X_test,  y_test))

best_k   = list(k_range)[np.argmax(test_acc)]
best_acc = max(test_acc)
print(f"Best K on test set: K={best_k}  (acc={best_acc:.4f})")

plt.figure(figsize=(9, 5))
plt.plot(list(k_range), train_acc, label='Train Accuracy', marker='o', markersize=4, linewidth=1.5)
plt.plot(list(k_range), test_acc,  label='Test Accuracy',  marker='s', markersize=4, linewidth=1.5)
plt.axvline(best_k, color='gray', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
plt.xlabel('K (number of neighbors)')
plt.ylabel('Accuracy')
plt.title('KNN — Accuracy vs K (Bias–Variance Tradeoff)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join('plots', 'accuracy_vs_k.png'), dpi=150)
plt.show()
