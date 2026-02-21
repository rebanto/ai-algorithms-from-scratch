# Logistic Regression From Scratch

After getting linear regression working, the next logical step was to try classification. Logistic regression is basically the bridge between simple line-fitting and "real" neural networks. It’s the same basic setup as linear regression, but with a twist: instead of predicting a continuous number, we’re predicting the probability of a category.

It's essentially a neural network with zero hidden layers. It: input → linear transformation → sigmoid → binary probability. 

## The Problem

We want to predict a binary label $y \in \{0, 1\}$ from features $X$. Linear regression would just output a raw number, which doesn't make sense as a probability. So we pass the linear output through a **sigmoid** to squash it to $(0, 1)$:

$$\hat{y} = \sigma(Xw + b) = \frac{1}{1 + e^{-(Xw + b)}}$$

## Loss Function

The right loss function here is **binary cross-entropy**, not MSE. Using MSE with a sigmoid output gives a non-convex loss surface. BCE stays convex:

$$L = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

When the model is very confident and wrong, this blows up — which is exactly the behavior you want. Strong penalties for confident mistakes.

## Backpropagation

The backward pass has a nice simplification. If you differentiate BCE and then cascade through the sigmoid, the combined gradient of the loss with respect to the pre-sigmoid output $z = Xw + b$ is just:

$$\frac{\partial L}{\partial z} = \frac{\hat{y} - y}{n}$$

Clean. I computed it the long way (through `BinaryCrossentropy.backward` then `Sigmoid.backward`) just to make the code structure explicit, but the final expression is satisfying.

## What I Built

```
Sigmoid                       # forward and backward
BinaryCrossentropy            # forward and backward

LogisticRegression
├── fit(X, y)                 # gradient descent with both layers
├── predict_proba(X)          # sigmoid output ∈ (0, 1)
├── predict(X)                # thresholded at 0.5
└── score(X, y)               # accuracy
```

Tested on the `make_moons` dataset — same dataset as the neural network so I could directly compare. Logistic regression gets a lower accuracy on moons because it can only learn a *linear* decision boundary, but the moons are non-linearly separable. The neural network wins here because it has a hidden layer with ReLU that lets it learn curves.

## Visualizations

- **`plots/decision_boundary.png`** — You can see the boundary is a smooth curve but not perfectly fitting the moons. This is not a bad model, it's just a fundamentally limited one.
- **`plots/training_curves.png`** — Loss and accuracy over epochs. Loss drops fast then plateaus; accuracy rises and levels off.

## Running It

```
pip install numpy matplotlib scikit-learn
python logistic_regression.py
```
