# Logistic Regression From Scratch

After getting linear regression working, the next logical step was to try classification. Logistic regression is basically the bridge between simple line-fitting and "real" neural networks. It’s the same basic setup as linear regression, but with a twist: instead of predicting a continuous number, we’re predicting the probability of a category.

It's essentially a neural network with zero hidden layers. It: input → linear transformation → sigmoid → binary probability. 

## The Problem

We want to predict a binary label $y \in \{0, 1\}$ from features $X$. Linear regression would just output a raw number, which doesn't make sense as a probability. So we pass the linear output through a **sigmoid** to squash it to $(0, 1)$:

$$
\hat{y} = \sigma(Xw + b) = \frac{1}{1 + e^{-(Xw + b)}}
$$

## Loss Function: Shouting at the Model

The "wrong" way to do this is with MSE. If you use MSE with a sigmoid, the loss surface gets all bumpy and it becomes easy to get stuck. Instead, we use **binary cross-entropy** (BCE). It's a convex loss (smooth bowl) that works perfectly with classification because it punishes confident mistakes exponentially. If the model is 99% sure it's a "0" when it's actually a "1," the loss blows up.

## Backpropagation (The Chain Rule)

The math for the backward pass is actually really clean. If you differentiate the BCE loss and then multiply it by the derivative of the sigmoid (classic chain rule), a bunch of terms cancel out. The combined gradient for the whole "layer" is just:

$$
\frac{\partial L}{\partial z} = \frac{\hat{y} - y}{n}
$$

I implemented it step-by-step (`BCE.backward` → `Sigmoid.backward`) just to follow the logic, but it's satisfying that after all that calculus, we're left with something so simple.

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
