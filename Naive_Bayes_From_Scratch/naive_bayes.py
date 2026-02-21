import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.datasets import make_blobs
import os

# naive bayes is built on a beautifully simple idea: Bayes' theorem.
# P(class | features) ∝ P(class) * P(features | class)
# the "naive" part is the independence assumption -- we pretend that each
# feature is independent given the class. that's almost never actually true,
# but it works shockingly well in practice.
#
# the gaussian version assumes the features within each class follow a normal
# distribution. so we just need to learn the mean and variance per feature per
# class during training. prediction is just evaluating Gaussian PDFs.
#
# one thing i had to be careful about: multiplying many small probabilities
# together causes underflow (numbers too small for float64). the fix is to
# work in log space -- sum of logs instead of product of probabilities.

class GaussianNaiveBayes:
    def __init__(self):
        self.classes   = None
        self.priors    = {}   # log P(class)
        self.means     = {}   # mu per feature per class
        self.variances = {}   # sigma^2 per feature per class

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_total = len(y)

        for c in self.classes:
            X_c = X[y == c]
            # prior: just the fraction of training examples in this class
            self.priors[c]    = np.log(len(X_c) / n_total)
            self.means[c]     = np.mean(X_c, axis=0)
            # small epsilon so we never divide by zero in the PDF
            self.variances[c] = np.var(X_c, axis=0) + 1e-9

    def _log_gaussian(self, x, mean, var):
        # log of Gaussian PDF: avoids underflow from multiplying tiny probabilities
        # log N(x; mu, sigma^2) = -0.5*log(2*pi*sigma^2) - (x-mu)^2 / (2*sigma^2)
        return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)

    def _log_posterior(self, x, c):
        # Bayes: log P(c|x) ∝ log P(c) + log P(x|c)
        # the naive assumption lets us decompose log P(x|c) as a sum over features
        log_likelihood = np.sum(self._log_gaussian(x, self.means[c], self.variances[c]))
        return self.priors[c] + log_likelihood

    def predict(self, X):
        preds = []
        for x in X:
            log_posts = {c: self._log_posterior(x, c) for c in self.classes}
            preds.append(max(log_posts, key=log_posts.get))
        return np.array(preds)

    def predict_proba(self, X):
        # convert log-posteriors back to actual probabilities via softmax
        proba = []
        for x in X:
            log_posts = np.array([self._log_posterior(x, c) for c in self.classes])
            # subtract max for numerical stability before exponentiating
            log_posts -= log_posts.max()
            exp_posts  = np.exp(log_posts)
            proba.append(exp_posts / exp_posts.sum())
        return np.array(proba)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


if __name__ == '__main__':
    # ---- data ----

    X, y = make_blobs(n_samples=600, centers=3, cluster_std=1.5, random_state=42)

    # ---- train ----

    model = GaussianNaiveBayes()
    model.fit(X, y)
    print(f"Training accuracy: {model.score(X, y):.4f}")

    os.makedirs('plots', exist_ok=True)

    # ---- decision boundary ----

    pad = 1.0
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(grid).reshape(xx.shape)

    cmap = plt.cm.get_cmap('tab10', len(model.classes))

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.contourf(xx, yy, Z, alpha=0.2, cmap='tab10',
                levels=np.arange(-0.5, len(model.classes)), vmin=0, vmax=len(model.classes)-1)
    ax.contour(xx, yy, Z, colors='k', linewidths=1,
               levels=np.arange(0.5, len(model.classes)-0.5))

    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10',
                         edgecolors='white', linewidth=0.5, s=50, alpha=0.9,
                         vmin=0, vmax=len(model.classes)-1)

    # draw 1-sigma Gaussian ellipses for each class -- shows what the model "sees"
    for c in model.classes:
        std = np.sqrt(model.variances[c])
        ellipse = Ellipse(
            xy=model.means[c],
            width=2 * std[0], height=2 * std[1],
            edgecolor=cmap(c), facecolor='none', linewidth=2.5,
            linestyle='--', zorder=5
        )
        ax.add_patch(ellipse)
        ax.plot(*model.means[c], 'x', color=cmap(c),
                markersize=11, markeredgewidth=2.5, zorder=6, label=f'Class {c} mean')

    ax.set_title('Gaussian Naive Bayes — Decision Boundary + 1σ Ellipses')
    ax.set_xlabel('X₁')
    ax.set_ylabel('X₂')
    plt.colorbar(scatter, ax=ax, ticks=list(model.classes), label='Class')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'naive_bayes_boundary.png'), dpi=150)
    plt.show()

    # ---- per-feature distribution plot ----

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    feature_names = ['X₁', 'X₂']
    x_vals = np.linspace(X.min() - 1, X.max() + 1, 300)

    for feat_idx, ax in enumerate(axes):
        for c in model.classes:
            mu  = model.means[c][feat_idx]
            var = model.variances[c][feat_idx]
            # plot learned Gaussian PDF for this feature/class pair
            pdf = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * (x_vals - mu) ** 2 / var)
            ax.plot(x_vals, pdf, color=cmap(c), linewidth=2, label=f'Class {c}')
            ax.axvline(mu, color=cmap(c), linestyle=':', alpha=0.7)

        # overlay actual data histogram
        for c in model.classes:
            ax.hist(X[y == c, feat_idx], bins=25, density=True,
                    alpha=0.2, color=cmap(c))

        ax.set_title(f'Learned Gaussians — Feature {feature_names[feat_idx]}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'feature_distributions.png'), dpi=150)
    plt.show()
