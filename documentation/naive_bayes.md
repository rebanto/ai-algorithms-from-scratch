# Gaussian Naive Bayes From Scratch

Naive Bayes was one of those algorithms I thought would be straightforward, and it mostly was — except for the probability underflow issue that I ran into immediately. Multiplying together 50 small probabilities gives you a number so tiny that floating point just rounds it to zero. The fix is to work in log space, which turns products into sums. Seems obvious in retrospect.

The "naive" part refers to an assumption the model makes: it treats every feature as **conditionally independent** given the class. That means it factors the joint probability as:

$$P(x_1, x_2, \ldots, x_d \mid c) = \prod_{j=1}^{d} P(x_j \mid c)$$

This is almost never actually true. Features are correlated. But the model works anyway, often surprisingly well, because even a wrong independence assumption can still put you in the right ballpark.

## Bayes' Theorem

The whole algorithm follows directly from Bayes' theorem:

$$P(c \mid x) = \frac{P(x \mid c) \cdot P(c)}{P(x)}$$

We want the class $c$ that maximizes $P(c \mid x)$. Since $P(x)$ is constant across all classes, we just need to maximize the numerator:

$$\hat{c} = \arg\max_c \; P(c) \cdot P(x \mid c)$$

- $P(c)$ is the **prior** — just the fraction of training examples in class $c$
- $P(x \mid c)$ is the **likelihood** — how probable is this input given the class

For the Gaussian version, we assume features within each class follow a normal distribution:

$$P(x_j \mid c) = \frac{1}{\sqrt{2\pi\sigma_{jc}^2}} \exp\!\left(-\frac{(x_j - \mu_{jc})^2}{2\sigma_{jc}^2}\right)$$

So "training" just means computing $\mu_{jc}$ (mean) and $\sigma_{jc}^2$ (variance) for each feature $j$ and class $c$. That's just two passes over the data.

## Log Space

Since we're multiplying many probabilities, convert everything to logs. Products become sums:

$$\log P(c \mid x) \propto \log P(c) + \sum_{j=1}^{d} \log P(x_j \mid c)$$

This avoids underflow and is also faster because addition is cheaper than multiplication.

## What I Built

```
GaussianNaiveBayes
├── fit(X, y)              # compute priors, means, variances per class
├── _log_gaussian(x, μ, σ²) # log PDF of Gaussian
├── _log_posterior(x, c)   # log P(c) + Σ log P(x_j | c)
├── predict(X)             # argmax over log posteriors
├── predict_proba(X)       # softmax of log posteriors -> actual probabilities
└── score(X, y)            # accuracy
```

Tested on `make_blobs` with 3 classes. I also visualize the **1-sigma Gaussian ellipses** — little dashed ellipses centered at each class mean, with axes showing the learned standard deviation per feature. It's a direct window into what the model has learned.

## Visualizations

- **`plots/naive_bayes_boundary.png`** — Decision boundary with class assignments, plus the learned Gaussian ellipses and class mean markers. This is probably the most informative plot in the project.
- **`plots/feature_distributions.png`** — The actual Gaussian PDFs the model learned for each feature. Histograms of real data overlaid to show the fit.

## Running It

```
pip install numpy matplotlib scikit-learn
python naive_bayes.py
```
