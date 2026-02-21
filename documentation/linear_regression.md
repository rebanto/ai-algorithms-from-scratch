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

### 2. Normal Equation (The "Got it in One" Method)

If you're willing to do a bit of matrix algebra, you can actually solve for the exact optimal $w$ in a single shot without any training:

$$
w^* = (X^T X)^{-1} X^T y
$$

This is the "closed-form" solution. It's incredibly satisfying because it gives the exact answer with no iteration or learning rate to tune. But there's a huge catch: matrix inversion is $O(d^3)$. If you have 10,000 features, this will practically melt your CPU. That's why we still use gradient descent—it scales properly to the big stuff.

I implemented both just to see if they'd agree. They converged to nearly identical values, which was a great "Aha!" moment.

## What I Built

I wrote a simple class to handle the heavy lifting:
```python
LinearRegression
├── fit(X, y)          # the main loop for gradient descent
├── predict(X)         # simple y_pred = X @ w + b
└── mse(X, y)          # how we evaluate performance (MSE)
```

The training data I used is $y = 3 + 5x + \epsilon$ (the $\epsilon$ is just a bit of noise). After training, the model recovered $w \approx 5$ and $b \approx 3$. 

## What I Learned

The coolest thing about linear regression is that the "loss surface" is a perfect convex bowl. There's only one bottom (global minimum) and no trap doors (local minima). This is why it always works—you just fall down the hill until you hit the bottom. 

Also, it turns out that $X^T X$ in the normal equation is basically the **covariance matrix** of the features. I didn't realize at first that the "shortcut" solution was connected to statistics like that, but it makes the linear algebra feel a lot more grounded.

## Running It

```
pip install numpy matplotlib scikit-learn
python linear_regression.py
```
