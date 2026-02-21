# Linear Regression From Scratch

This was probably the first "real" ML algorithm I implemented and honestly it gave me one of the clearest views into what machine learning actually *is*. If you strip away all the hype, linear regression is basically just: find a line that fits your data as well as possible. How do you find it? You define "as well as possible" mathematically (mean squared error), then use calculus to figure out which direction to nudge the parameters to make it better.

## The Core Idea

Given a dataset of input-output pairs $(x_i, y_i)$, we want to find weights $w$ and bias $b$ such that:

$$
\hat{y} = Xw + b
$$

is as close to $y$ as possible. "Close" is measured by **mean squared error**:

$$
L = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
$$

The goal is to minimize $L$ by adjusting $w$ and $b$.

## Two Ways to Solve It

### 1. Gradient Descent (Iterative)

Take the derivative of $L$ with respect to $w$ and $b$, then repeatedly move in the opposite direction of the gradient. The gradients are:

$$
\frac{\partial L}{\partial w} = \frac{2}{n} X^T (\hat{y} - y), \qquad \frac{\partial L}{\partial b} = \frac{2}{n} \sum(\hat{y} - y)
$$

Then update: $w \leftarrow w - \alpha \cdot \frac{\partial L}{\partial w}$, where $\alpha$ is the learning rate.

This is the same update rule that every neural network uses. It's just much easier to see here because there's only one layer.

### 2. Normal Equation (Closed-Form)

If you take the gradient of $L$ and set it equal to zero, you can actually solve for the exact optimal $w$ analytically:

$$
w^* = (X^T X)^{-1} X^T y
$$

This gives the exact answer in a single matrix operation. No iteration, no learning rate to tune. The catch is that matrix inversion is $O(d^3)$ — it becomes completely impractical when you have thousands of features. Gradient descent scales much better.

I implemented both and compared them. They converge to essentially the same values, which is a good sanity check.

## What I Built

```
LinearRegression
├── fit(X, y)          # runs gradient descent
├── predict(X)         # y_pred = X @ w + b
└── mse(X, y)          # evaluate mean squared error

normal_equation(X, y)  # standalone function for the closed-form solution
```

The training data is $y = 3 + 5x + \epsilon$ (Gaussian noise), so I know the true answer. After training, gradient descent recovers $w \approx 5$ and $b \approx 3$, which is satisfying.

## Visualizations

- **`plots/loss_curve.png`** — MSE dropping over epochs. Should be a smooth curve downward; if it's oscillating, the learning rate is too high.
- **`plots/regression_fit.png`** — Scatter plot of data with three lines overlaid: gradient descent fit, normal equation fit, and the true function. They're basically identical, which is the point.

## What Surprised Me

The loss surface for linear regression is a perfect convex bowl — there's only one global minimum and no local minima. This is why gradient descent *always* converges here, regardless of where you start. That's a luxury that disappears the moment you add a nonlinear activation function.

Also, the normal equation involving $X^T X$ is just the **covariance matrix** of the features. I didn't notice that at first but it makes the linear algebra feel much more meaningful.

## Running It

```
pip install numpy matplotlib scikit-learn
python linear_regression.py
```
