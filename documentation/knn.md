# K-Nearest Neighbors From Scratch

KNN might be the simplest machine learning algorithm that's actually useful. There are no parameters to learn, no gradient descent, no training at all. The entire "model" is just the training dataset stored in memory. At prediction time, you find the K closest training points and let them vote.

It sounds almost too naive. But it works — and more importantly, understanding it really clearly shows you what machine learning is actually doing: finding patterns in data via similarity.

## The Algorithm

KNN is a "lazy" learner—it doesn't actually learn a model during training.

**Training:** Just store $(X_{train}, y_{train})$. That’s it.
**Prediction for a new point $x$:**
1. Compute the Euclidean distance from $x$ to every training point:
$$
d(x, x_i) = \sqrt{\sum_{j=1}^{d} (x_j - x_{ij})^2}
$$
2. Find the $K$ points with the smallest distances.
3. Take a majority vote among their labels.
4. Return the winner.

For "probabilities," I use the fraction of the K neighbors that belong to class 1. It's not a real probability in any rigorous sense, but it's smooth and useful for plotting decision boundaries.

## The Bias–Variance Tradeoff

This is where KNN becomes a really clean teaching example. The hyperparameter $K$ directly controls the tradeoff between **bias** and **variance**:

| K | Bias | Variance | Behavior |
|---|------|----------|----------|
| 1 | Low | High | Perfect on training data. Every training point predicts itself. Wildly overfits. |
| moderate | Balanced | Balanced | Usually the sweet spot |
| large | High | Low | Very smooth boundary. Eventually just predicts the majority class everywhere. |

I plotted decision boundaries for $K = 1, 5, 15, 31$ side by side. The $K=1$ boundary is jagged and noisy. $K=31$ is smooth but misses the curve structure of the moons. Somewhere in the middle performs best on test data.

## What I Built

```
KNN
├── fit(X, y)             # store training data (no actual computation)
├── predict(X)            # majority vote among K nearest neighbors
├── predict_proba(X)      # fraction of neighbors voting for class 1
└── score(X, y)           # accuracy
```

I also sweep over K from 1 to 50 and plot train vs test accuracy. The classic U-shape where K=1 has 100% train accuracy but mediocre test accuracy, and very large K has low accuracy everywhere.

## Visualizations

- **`plots/knn_boundaries.png`** — 2×2 grid showing how the decision boundary changes with K. Worth staring at for a while.
- **`plots/accuracy_vs_k.png`** — Train/test accuracy as a function of K. The divergence between them at small K is textbook overfitting.

## Computational Note

## Complexity trade-offs

KNN has zero training cost but high *prediction* cost. For each test point, you have to compute distances to every single training point. That makes prediction $O(n \cdot d)$, where $n$ is the number of training examples. Real-world libraries use KD-trees to speed this up, but here I used brute force to keep the implementation clean and understandable.

## Running It

```
pip install numpy matplotlib scikit-learn
python knn.py
```
