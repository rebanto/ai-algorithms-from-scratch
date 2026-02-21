"""
run_all.py - Master runner for AI From Scratch
================================================
Runs a fast demo of every implemented algorithm, prints info about
what each one does and what dataset it's using, generates organized
plots in plots/<algorithm>/, and includes a cool "extra" visualization
per algorithm (saliency maps for CNN, hidden state heatmap for RNN, etc.)

Usage:
    python run_all.py              # run all algorithms
    python run_all.py linear       # run only linear regression
    python run_all.py cnn rnn      # run only CNN and RNN
"""

import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse

# Global flag for interactivity
INTERACTIVE = False

# ---- path setup so we can import from each algorithm folder ----
BASE = os.path.dirname(os.path.abspath(__file__))
for folder in ['Linear_Regression_From_Scratch', 'Logistic_Regression_From_Scratch',
               'KNN_From_Scratch', 'Naive_Bayes_From_Scratch',
               'CNN_From_Scratch', 'RNN_From_Scratch']:
    sys.path.insert(0, os.path.join(BASE, folder))

from linear_regression    import LinearRegression, normal_equation
from logistic_regression  import LogisticRegression
from knn                  import KNN
from naive_bayes          import GaussianNaiveBayes
from cnn                  import (Conv2D, MaxPool2D, Flatten, Dense,
                                   ReLU, Softmax, CrossEntropyLoss, Adam)
from rnn                  import RNNCell, one_hot, char2idx, idx2char, data, vocab_size, corpus

from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split

# ---- output directory ----
PLOTS_DIR = os.path.join(BASE, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---- terminal formatting ----
CYAN   = '\033[96m'
YELLOW = '\033[93m'
GREEN  = '\033[92m'
BOLD   = '\033[1m'
RESET  = '\033[0m'
LINE   = '=' * 65


def header(title, description, dataset, extra=''):
    print(f'\n{BOLD}{CYAN}{LINE}{RESET}')
    print(f'{BOLD}{CYAN}  {title}{RESET}')
    print(f'{LINE}')
    print(f'{YELLOW}  What it does:{RESET} {description}')
    print(f'{YELLOW}  Dataset:     {RESET} {dataset}')
    if extra:
        print(f'{YELLOW}  Extra:       {RESET} {extra}')
    print(f'{LINE}\n')


def save(fig, algo_name, filename, close=True):
    d = os.path.join(PLOTS_DIR, algo_name)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    
    print(f'  {GREEN}saved ->{RESET} plots/{algo_name}/{filename}')
    
    if INTERACTIVE:
        plt.show()
    
    if close:
        plt.close(fig)
    return path


def tick(label):
    print(f'  {GREEN}[DONE]{RESET} {label}')


# =======================================================================
# 1. LINEAR REGRESSION
# =======================================================================

def run_linear_regression():
    header(
        '1. Linear Regression',
        'Fit a line to data by minimizing Mean Squared Error via gradient descent.\n'
        '               Also computes the closed-form Normal Equation for comparison.',
        'Synthetic 1D: y = 3 + 5x + Gaussian noise  (200 samples)',
        'Bonus: 3D loss surface visualizing the convex MSE bowl'
    )
    t0 = time.time()

    np.random.seed(42)
    X_raw = 2 * np.random.rand(200, 1)
    y     = 3 + 5 * X_raw[:, 0] + np.random.randn(200) * 0.8
    X_train, X_test = X_raw[:160], X_raw[160:]
    y_train, y_test = y[:160],     y[160:]

    model = LinearRegression(learning_rate=0.05, epochs=2000)
    model.fit(X_train, y_train)
    ne = normal_equation(X_train, y_train)

    test_mse = model.mse(X_test, y_test)
    print(f'  Gradient descent  ->  w={model.w[0]:.4f}, b={model.b:.4f}')
    print(f'  Normal equation   ->  w={ne[1]:.4f}, b={ne[0]:.4f}')
    print(f'  True values       ->  w=5.0000, b=3.0000')
    print(f'  Test MSE: {test_mse:.5f}')

    x_line       = np.linspace(0, 2, 200).reshape(-1, 1)
    y_gd         = model.predict(x_line)
    y_ne         = ne[0] + ne[1] * x_line[:, 0]
    y_true_line  = 3 + 5 * x_line[:, 0]

    # plot 1: regression fit
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.scatter(X_train, y_train, alpha=0.4, color='steelblue', s=20, label='Train')
    ax.scatter(X_test,  y_test,  alpha=0.6, color='coral', s=25, marker='x', label='Test')
    ax.plot(x_line, y_gd,       'k-',  lw=2,   label='Gradient Descent')
    ax.plot(x_line, y_ne,       'r--', lw=2,   label='Normal Equation')
    ax.plot(x_line, y_true_line,'g--', lw=1.5, alpha=0.7, label='True function')
    ax.set_title('Regression Fit: GD vs Normal Equation')
    ax.set_xlabel('X'); ax.set_ylabel('y')
    ax.legend(fontsize=8); ax.grid(True, ls='--', alpha=0.5)

    # plot 2: loss curve
    ax2 = axes[1]
    ax2.plot(model.loss_history, color='steelblue', lw=1.5)
    ax2.set_title('Training Loss (MSE) over Epochs')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('MSE')
    ax2.grid(True, ls='--', alpha=0.5)

    fig.suptitle('Linear Regression From Scratch', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save(fig, '01_linear_regression', 'fit_and_loss.png')
    tick('Regression fit + loss curve')

    # BONUS: 3D loss surface -- shows the convex bowl
    w_vals = np.linspace(3.0, 7.0, 60)
    b_vals = np.linspace(1.0, 5.0, 60)
    WW, BB = np.meshgrid(w_vals, b_vals)
    Z_surf = np.array([
        [np.mean((X_train[:, 0] * w + b - y_train) ** 2)
         for w in w_vals] for b in b_vals
    ])

    fig = plt.figure(figsize=(9, 6))
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.plot_surface(WW, BB, Z_surf, cmap='viridis', alpha=0.8, edgecolor='none')
    ax3d.scatter([model.w[0]], [model.b], [model.mse(X_train, y_train)],
                 color='red', s=60, zorder=5, label='GD solution')
    ax3d.scatter([ne[1]], [ne[0]], [np.mean((X_train[:, 0] * ne[1] + ne[0] - y_train) ** 2)],
                 color='lime', s=60, zorder=5, marker='^', label='Normal Eq')
    ax3d.set_xlabel('Weight (w)'); ax3d.set_ylabel('Bias (b)'); ax3d.set_zlabel('MSE Loss')
    ax3d.set_title('Loss Surface (convex bowl - no local minima!)')
    ax3d.legend()
    save(fig, '01_linear_regression', 'loss_surface_3d.png')
    tick('3D loss surface visualization')

    print(f'\n  Done in {time.time()-t0:.1f}s')


# =======================================================================
# 2. LOGISTIC REGRESSION
# =======================================================================

def run_logistic_regression():
    header(
        '2. Logistic Regression',
        'Sigmoid-activated linear classifier trained with Binary Cross-Entropy\n'
        '               and gradient descent. Same math as a 1-layer neural network.',
        'make_moons (600 samples, noise=0.2) — 2-class nonlinear dataset',
        'Probability contour map showing confidence across input space'
    )
    t0 = time.time()

    X, y = make_moons(n_samples=600, noise=0.2, random_state=42)
    model = LogisticRegression(lr=0.1, epochs=5000)
    model.fit(X, y)
    acc = model.score(X, y)
    print(f'  Final accuracy: {acc:.4f}')

    pad = 0.5
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid   = np.c_[xx.ravel(), yy.ravel()]
    Z      = model.predict_proba(grid).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    cf = ax.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.4)
    ax.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=1.5)
    sc = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='w', lw=0.5, alpha=0.85, s=30)
    plt.colorbar(cf, ax=ax, label='P(class=1)')
    ax.set_title(f'Decision Boundary  (acc={acc:.3f})')
    ax.set_xlabel('X1'); ax.set_ylabel('X2')
    ax.grid(True, ls='--', alpha=0.4)

    ax2 = axes[1]
    ax2.plot(model.loss_history, color='steelblue', lw=1.5, label='BCE Loss')
    ax2b = ax2.twinx()
    ax2b.plot(model.acc_history, color='seagreen', lw=1.5, linestyle='--', label='Accuracy')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('BCE Loss', color='steelblue')
    ax2b.set_ylabel('Accuracy', color='seagreen')
    ax2.set_title('Training Curves')
    ax2.grid(True, ls='--', alpha=0.4)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    fig.suptitle('Logistic Regression From Scratch', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save(fig, '02_logistic_regression', 'boundary_and_training.png')
    tick('Decision boundary + training curves')

    print(f'\n  Done in {time.time()-t0:.1f}s')


# =======================================================================
# 3. KNN
# =======================================================================

def run_knn():
    header(
        '3. K-Nearest Neighbors',
        'No training — memorize the data, classify by majority vote of K nearest\n'
        '               neighbors. Bias-variance tradeoff is controlled entirely by K.',
        'make_moons (600 samples, noise=0.3) — 80/20 train/test split',
        'Side-by-side boundaries for K=1,5,15,31 + accuracy vs K sweep'
    )
    t0 = time.time()

    X, y = make_moons(n_samples=600, noise=0.3, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    pad = 0.5
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 120), np.linspace(y_min, y_max, 120))
    grid   = np.c_[xx.ravel(), yy.ravel()]

    k_values = [1, 5, 15, 31]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, k in enumerate(k_values):
        knn = KNN(k=k)
        knn.fit(X_tr, y_tr)
        Z    = knn.predict_proba(grid).reshape(xx.shape)
        acc  = knn.score(X_te, y_te)
        ax   = axes.ravel()[idx]
        ax.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.4)
        ax.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=1.5)
        ax.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap='RdBu',
                   edgecolors='w', lw=0.4, alpha=0.8, s=20)
        ax.set_title(f'K={k}  |  Test acc={acc:.3f}', fontsize=11)
        ax.set_xlabel('X1'); ax.set_ylabel('X2')
        ax.grid(True, ls='--', alpha=0.3)
    fig.suptitle('KNN — Decision Boundaries (effect of K)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save(fig, '03_knn', 'decision_boundaries.png')
    tick('Decision boundaries for K=1,5,15,31')

    k_range = range(1, 51, 2)
    tr_acc, te_acc = [], []
    for k in k_range:
        knn = KNN(k=k); knn.fit(X_tr, y_tr)
        tr_acc.append(knn.score(X_tr, y_tr))
        te_acc.append(knn.score(X_te, y_te))
    best_k = list(k_range)[np.argmax(te_acc)]
    print(f'  Best K on test set: K={best_k}  (acc={max(te_acc):.4f})')

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(list(k_range), tr_acc, marker='o', ms=4, lw=1.5, label='Train Accuracy')
    ax.plot(list(k_range), te_acc, marker='s', ms=4, lw=1.5, label='Test Accuracy')
    ax.axvline(best_k, color='gray', ls='--', alpha=0.7, label=f'Best K={best_k}')
    ax.set_xlabel('K'); ax.set_ylabel('Accuracy')
    ax.set_title('KNN — Accuracy vs K  (Bias–Variance Tradeoff)', fontweight='bold')
    ax.legend(); ax.grid(True, ls='--', alpha=0.5)
    save(fig, '03_knn', 'accuracy_vs_k.png')
    tick('Accuracy vs K sweep')

    print(f'\n  Done in {time.time()-t0:.1f}s')


# =======================================================================
# 4. NAIVE BAYES
# =======================================================================

def run_naive_bayes():
    header(
        '4. Gaussian Naive Bayes',
        'Applies Bayes\' theorem with a Gaussian likelihood per feature per class.\n'
        '               Uses log-space arithmetic to avoid probability underflow.',
        'make_blobs (600 samples, 3 classes, cluster_std=1.5)',
        'X1-sigma Gaussian ellipses showing the model\'s learned class distributions'
    )
    t0 = time.time()

    X, y = make_blobs(n_samples=600, centers=3, cluster_std=1.5, random_state=42)
    model = GaussianNaiveBayes()
    model.fit(X, y)
    acc = model.score(X, y)
    print(f'  Training accuracy: {acc:.4f}')

    pad = 1.0
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid   = np.c_[xx.ravel(), yy.ravel()]
    Z      = model.predict(grid).reshape(xx.shape)
    cmap   = plt.cm.get_cmap('tab10', len(model.classes))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.contourf(xx, yy, Z, alpha=0.15, cmap='tab10',
                levels=np.arange(-0.5, len(model.classes)), vmin=0, vmax=len(model.classes)-1)
    ax.contour(xx, yy, Z, colors='k', linewidths=1,
               levels=np.arange(0.5, len(model.classes)-0.5))
    sc = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10',
                    edgecolors='white', lw=0.5, s=40, alpha=0.9,
                    vmin=0, vmax=len(model.classes)-1)
    for c in model.classes:
        std = np.sqrt(model.variances[c])
        ax.add_patch(Ellipse(xy=model.means[c], width=2*std[0], height=2*std[1],
                             edgecolor=cmap(c), facecolor='none', lw=2.5, ls='--', zorder=5))
        ax.plot(*model.means[c], 'x', color=cmap(c), ms=10, markeredgewidth=2.5, zorder=6)
    ax.set_title(f'Decision Boundary + 1-sigma Ellipses  (acc={acc:.3f})')
    ax.set_xlabel('X1'); ax.set_ylabel('X2')
    ax.grid(True, ls='--', alpha=0.4)

    ax2 = axes[1]
    x_vals = np.linspace(X.min()-1, X.max()+1, 300)
    for feat_idx, label in enumerate(['Feature X1', 'Feature X2']):
        ax2.clear() if feat_idx else None
    # feature distributions side by side in second axis
    for feat_idx in range(2):
        sub_ax = axes[1] if feat_idx == 0 else None
    # just do both features stacked in one plot
    fig2, f_axes = plt.subplots(1, 2, figsize=(12, 4))
    for feat_idx, fax in enumerate(f_axes):
        for c in model.classes:
            mu, var = model.means[c][feat_idx], model.variances[c][feat_idx]
            pdf = (1/np.sqrt(2*np.pi*var)) * np.exp(-0.5*(x_vals-mu)**2/var)
            fax.plot(x_vals, pdf, color=cmap(c), lw=2, label=f'Class {c}')
            fax.axvline(mu, color=cmap(c), ls=':', alpha=0.6)
        for c in model.classes:
            fax.hist(X[y==c, feat_idx], bins=20, density=True, alpha=0.18, color=cmap(c))
        fax.set_title(f'Learned Gaussian — Feature X{feat_idx+1}')
        fax.set_xlabel('Value'); fax.set_ylabel('Density')
        fax.legend(fontsize=8); fax.grid(True, ls='--', alpha=0.4)
    f_axes[0].set_title(f'Gaussian Naive Bayes — Feature Distributions', fontweight='bold')

    ax2.set_visible(False)  # hide the unused second subplot in main fig
    fig.suptitle('Gaussian Naive Bayes From Scratch', fontsize=13, fontweight='bold')
    plt.figure(fig.number); plt.tight_layout()
    save(fig, '04_naive_bayes', 'decision_boundary.png')
    tick('Decision boundary + Gaussian ellipses')

    fig2.suptitle('Gaussian Naive Bayes — Learned Feature Distributions', fontweight='bold')
    plt.figure(fig2.number); plt.tight_layout()
    save(fig2, '04_naive_bayes', 'feature_distributions.png')
    tick('Per-feature Gaussian distribution plots')

    print(f'\n  Done in {time.time()-t0:.1f}s')


# =======================================================================
# 5. CNN (fast demo: fewer filters, epochs, samples)
# =======================================================================

def _cnn_forward(x, conv1, relu1, pool1, flatten, dense1, relu2, dense2, softmax):
    x = conv1.forward(x); x = relu1.forward(x)
    x = pool1.forward(x); x = flatten.forward(x)
    x = dense1.forward(x); x = relu2.forward(x)
    x = dense2.forward(x); x = softmax.forward(x)
    return x

def _cnn_backward(d, conv1, relu1, pool1, flatten, dense1, relu2, dense2, softmax):
    d = softmax.backward(d); d = dense2.backward(d)
    d = relu2.backward(d);   d = dense1.backward(d)
    d = flatten.backward(d); d = pool1.backward(d)
    d = relu1.backward(d);   d = conv1.backward(d)


def run_cnn():
    header(
        '5. Convolutional Neural Network',
        'Conv2D (stride tricks + einsum) -> ReLU -> MaxPool -> Dense -> Softmax.\n'
        '               Backprop through convolution via spatial gradient scattering.',
        'MNIST handwritten digits 0-9  (5000 train / 1000 test, 28×28 grayscale)',
        'Gradient Saliency Maps - which pixels the network looks at for each digit'
    )
    t0 = time.time()

    print('  Loading MNIST...')
    try:
        from tensorflow.keras.datasets import mnist as km
        (Xtr, ytr), (Xte, yte) = km.load_data()
    except ImportError:
        from sklearn.datasets import fetch_openml
        print('  (TF not found — using sklearn, may take a moment)')
        d = fetch_openml('mnist_784', version=1, as_frame=False, cache=True)
        Xall = d.data.reshape(-1, 28, 28); yall = d.target.astype(int)
        Xtr, ytr, Xte, yte = Xall[:60000], yall[:60000], Xall[60000:], yall[60000:]

    TRAIN_N, TEST_N = 5000, 1000
    Xtr = Xtr[:TRAIN_N].astype(np.float32) / 255.0; ytr = ytr[:TRAIN_N]
    Xte = Xte[:TEST_N].astype(np.float32)  / 255.0; yte = yte[:TEST_N]
    print(f'  Loaded {TRAIN_N} train / {TEST_N} test samples')

    conv1   = Conv2D(num_filters=8, filter_size=3)
    relu1   = ReLU(); pool1 = MaxPool2D(pool_size=2, stride=2)
    flatten = Flatten()
    dense1  = Dense(8 * 13 * 13, 64); relu2 = ReLU()
    dense2  = Dense(64, 10); softmax = Softmax()
    loss_fn = CrossEntropyLoss(); opt = Adam(lr=0.001)

    EPOCHS, BS = 5, 64
    n_batches  = TRAIN_N // BS
    loss_hist, acc_hist = [], []

    print(f'  Training {EPOCHS} epochs...')
    for epoch in range(EPOCHS):
        perm = np.random.permutation(TRAIN_N)
        Xs, ys = Xtr[perm], ytr[perm]
        epoch_loss = 0.0
        opt.t += 1
        for b in range(n_batches):
            xb  = Xs[b*BS:(b+1)*BS]; yb = ys[b*BS:(b+1)*BS]
            out = _cnn_forward(xb, conv1, relu1, pool1, flatten, dense1, relu2, dense2, softmax)
            epoch_loss += loss_fn.forward(out, yb)
            d = loss_fn.backward(out, yb)
            _cnn_backward(d, conv1, relu1, pool1, flatten, dense1, relu2, dense2, softmax)
            for p, g, k in [(dense2, 'dw', 'd2w'), (dense2, 'db', 'd2b'),
                            (dense1, 'dw', 'd1w'), (dense1, 'db', 'd1b')]:
                setattr(p, 'w' if g=='dw' else 'b',
                        opt.update(p.w if g=='dw' else p.b,
                                   p.dw if g=='dw' else p.db, k))
            conv1.filters = opt.update(conv1.filters, conv1.dfilters, 'c1f')
            conv1.biases  = opt.update(conv1.biases,  conv1.dbiases,  'c1b')

        correct = sum(
            (np.argmax(_cnn_forward(Xte[i:i+BS], conv1, relu1, pool1, flatten,
                                    dense1, relu2, dense2, softmax), axis=1) == yte[i:i+BS]).sum()
            for i in range(0, TEST_N, BS)
        )
        acc = correct / TEST_N
        loss_hist.append(epoch_loss / n_batches); acc_hist.append(acc)
        print(f'    Epoch {epoch+1}/{EPOCHS}  loss={loss_hist[-1]:.4f}  acc={acc:.4f}')

    # training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(range(1, EPOCHS+1), loss_hist, marker='o', color='steelblue')
    ax1.set_title('Training Loss'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Cross-Entropy')
    ax1.grid(True, ls='--', alpha=0.5)
    ax2.plot(range(1, EPOCHS+1), acc_hist, marker='s', color='seagreen')
    ax2.set_title('Test Accuracy'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1); ax2.grid(True, ls='--', alpha=0.5)
    fig.suptitle('CNN on MNIST — Training', fontsize=13, fontweight='bold'); plt.tight_layout()
    save(fig, '05_cnn', 'training_curves.png')
    tick('Training curves')

    # learned filters
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i, ax in enumerate(axes.ravel()):
        f = conv1.filters[i]; vm = np.abs(f).max()
        ax.imshow(f, cmap='RdBu', vmin=-vm, vmax=vm)
        ax.set_title(f'Filter {i}', fontsize=9); ax.axis('off')
    fig.suptitle('Learned Convolutional Filters', fontsize=12, fontweight='bold')
    plt.tight_layout(); save(fig, '05_cnn', 'learned_filters.png')
    tick('Learned filters')

    # sample predictions
    np.random.seed(7)
    idx    = np.random.choice(TEST_N, 10, replace=False)
    preds  = np.argmax(_cnn_forward(Xte[idx], conv1, relu1, pool1, flatten,
                                    dense1, relu2, dense2, softmax), axis=1)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(Xte[idx[i]], cmap='gray')
        col = 'green' if preds[i] == yte[idx[i]] else 'red'
        ax.set_title(f'True:{yte[idx[i]]}  Pred:{preds[i]}', color=col, fontsize=9)
        ax.axis('off')
    fig.suptitle('Sample Predictions (green=correct, red=wrong)', fontsize=12, fontweight='bold')
    plt.tight_layout(); save(fig, '05_cnn', 'sample_predictions.png')
    tick('Sample predictions')

    # BONUS: Gradient Saliency Maps
    # Backprop the output score for the predicted class all the way to the input pixels.
    # Bright = important pixel, dark = irrelevant.
    print('  Computing gradient saliency maps...')
    n_saliency = 8
    saliency_imgs, digits, predicted = [], [], []
    shown = {d: False for d in range(10)}
    for ii in range(TEST_N):
        label = int(yte[ii])
        if shown[label]: continue
        xb  = Xte[ii:ii+1]
        out = _cnn_forward(xb, conv1, relu1, pool1, flatten, dense1, relu2, dense2, softmax)
        pred = int(np.argmax(out))
        if pred != label: continue   # only correct predictions
        # backward with a one-hot gradient for the true class
        d_out = np.zeros_like(out); d_out[0, label] = 1.0
        _cnn_backward(d_out, conv1, relu1, pool1, flatten, dense1, relu2, dense2, softmax)
        saliency = np.abs(conv1.dinputs[0])   # (28, 28) absolute gradient magnitude
        saliency_imgs.append(saliency); digits.append(label); predicted.append(pred)
        shown[label] = True
        if sum(shown.values()) >= n_saliency: break

    fig, axes = plt.subplots(2, len(saliency_imgs), figsize=(14, 5))
    for i in range(len(saliency_imgs)):
        axes[0, i].imshow(Xte[np.where(yte == digits[i])[0][0]], cmap='gray')
        axes[0, i].set_title(f'Digit: {digits[i]}', fontsize=9); axes[0, i].axis('off')
        vm = saliency_imgs[i].max()
        axes[1, i].imshow(saliency_imgs[i], cmap='hot', vmin=0, vmax=vm)
        axes[1, i].set_title('Saliency', fontsize=9); axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Input', fontsize=9)
    axes[1, 0].set_ylabel('Gradient\nSaliency', fontsize=9)
    fig.suptitle('Gradient Saliency Maps — Which Pixels the CNN Pays Attention To',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(); save(fig, '05_cnn', 'saliency_maps.png')
    tick('Gradient saliency maps (cool!)')

    print(f'\n  Final test accuracy: {acc_hist[-1]:.4f}')
    print(f'  Done in {time.time()-t0:.1f}s')


# =======================================================================
# 6. RNN (character-level text generation)
# =======================================================================

def run_rnn():
    header(
        '6. Recurrent Neural Network (character-level)',
        'Vanilla RNN trained with BPTT + gradient clipping to predict the next\n'
        '               character in a sequence. Samples new text from the trained model.',
        'Hamlet excerpt (embedded in script) — ~900 chars, 45 unique characters',
        'Hidden state activation heatmap: see what the network\'s "memory" looks like'
    )
    t0 = time.time()

    HIDDEN = 128; SEQ = 25; LR = 0.1; N = 30000

    rnn      = RNNCell(input_size=vocab_size, hidden_size=HIDDEN, output_size=vocab_size)
    smooth   = -np.log(1.0 / vocab_size) * SEQ
    h_prev   = np.zeros(HIDDEN); ptr = 0
    loss_log = []

    print(f'  Corpus: {len(corpus)} chars, vocab={vocab_size}')
    print(f'  Training {N} iterations...')

    for i in range(N):
        if ptr + SEQ + 1 >= len(data):
            ptr = 0; h_prev = np.zeros(HIDDEN)
        inp_idx = data[ptr:ptr+SEQ]; tgt_idx = data[ptr+1:ptr+SEQ+1]
        xs = one_hot(inp_idx, vocab_size)
        _, hs, _, ps = rnn.forward(xs, h_prev)
        dWxh, dWhh, dWhy, dbh, dby, loss, h_prev = rnn.backward(xs, hs, ps, tgt_idx)
        rnn.update(dWxh, dWhh, dWhy, dbh, dby, lr=LR)
        smooth = 0.999 * smooth + 0.001 * loss * SEQ; ptr += SEQ
        if i % 1000 == 0:
            loss_log.append((i, smooth))
            sample = rnn.sample(h_prev, inp_idx[0], 80, temp=1.0)
            print(f'    iter {i:5d} | loss={smooth:.4f} | sample: {repr(sample[:50])}')

    print(f'\n  --- Generated text (200 chars, temp=1.0) ---')
    generated = rnn.sample(np.zeros(HIDDEN), data[0], 200, temp=1.0)
    print(f'  {generated}\n')

    # loss curve
    fig, ax = plt.subplots(figsize=(9, 4))
    iters = [x[0] for x in loss_log]; losses = [x[1] for x in loss_log]
    ax.plot(iters, losses, color='steelblue', lw=2)
    ax.set_title('RNN Training Loss (smoothed cross-entropy)', fontweight='bold')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Loss')
    ax.grid(True, ls='--', alpha=0.5); plt.tight_layout()
    save(fig, '06_rnn', 'loss_curve.png')
    tick('Training loss curve')

    # BONUS: hidden state activation heatmap
    h = np.zeros(HIDDEN); hidden_states = []
    for ci in data[:300]:
        xv = np.zeros(vocab_size); xv[ci] = 1
        h  = np.tanh(xv @ rnn.Wxh + h @ rnn.Whh + rnn.bh)
        hidden_states.append(h[:48])   # first 48 units for visibility
    H_mat = np.array(hidden_states).T  # (48, 300)

    # annotate with the actual characters underneath
    chars_shown = [idx2char[int(c)] for c in data[:300]]

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(H_mat, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='tanh activation')

    # label every 10th character on x-axis
    tick_positions = list(range(0, 300, 10))
    tick_labels    = [corpus[i] if corpus[i] != '\n' else '>' for i in tick_positions]
    ax.set_xticks(tick_positions); ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_xlabel('Character position in corpus (labeled every 10)')
    ax.set_ylabel('Hidden unit index (first 48 of 128)')
    ax.set_title('RNN Hidden State Activations - each row is one neuron\'s memory over time',
                 fontweight='bold')
    plt.tight_layout(); save(fig, '06_rnn', 'hidden_activations.png')
    tick('Hidden state activation heatmap')

    # character probability bar chart for a sample input
    seed_char = 'T'
    seed_idx  = char2idx[seed_char]
    h_test    = np.zeros(HIDDEN)
    x_test    = np.zeros(vocab_size); x_test[seed_idx] = 1
    h_test    = np.tanh(x_test @ rnn.Wxh + h_test @ rnn.Whh + rnn.bh)
    logits    = (h_test @ rnn.Why + rnn.by) / 1.0  # temp=1.0
    e         = np.exp(logits - logits.max()); probs = e / e.sum()

    top_n  = 15
    top_idx = np.argsort(probs)[-top_n:][::-1]
    top_ch  = [idx2char[i] if idx2char[i] != '\n' else '>' for i in top_idx]
    top_pr  = probs[top_idx]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(top_ch, top_pr, color='steelblue', edgecolor='white')
    ax.set_title(f'RNN Next-Character Probabilities after seeing: "{seed_char}"',
                 fontweight='bold')
    ax.set_xlabel('Next character'); ax.set_ylabel('Probability')
    for bar, p in zip(bars, top_pr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{p:.2f}', ha='center', va='bottom', fontsize=8)
    ax.grid(True, axis='y', ls='--', alpha=0.5); plt.tight_layout()
    save(fig, '06_rnn', 'next_char_probs.png')
    tick(f'Next-character probability distribution after "{seed_char}"')

    print(f'\n  Done in {time.time()-t0:.1f}s')


# =======================================================================
# MAIN
# =======================================================================

RUNNERS = {
    'linear':   run_linear_regression,
    'logistic': run_logistic_regression,
    'knn':      run_knn,
    'bayes':    run_naive_bayes,
    'cnn':      run_cnn,
    'rnn':      run_rnn,
}

ALL_ORDER = ['linear', 'logistic', 'knn', 'bayes', 'cnn', 'rnn']

if __name__ == '__main__':
    requested = [a.lower() for a in sys.argv[1:]]
    
    # If arguments provided, run CLI mode (no popups)
    if requested:
        print(f'\n{BOLD}{CYAN}{"="*65}')
        print(' AI From Scratch - Master Runner (CLI Mode)')
        print(f'{"="*65}{RESET}')
        print(f' Plots saved to: {os.path.join(BASE, "plots")}/\n')
        
        to_run = [k for k in ALL_ORDER if any(r in k for r in requested)]
        if not to_run:
            print(f'Unknown algorithm(s). Choose from: {", ".join(ALL_ORDER)}')
            sys.exit(1)
            
        total_t0 = time.time()
        for key in to_run:
            RUNNERS[key]()
    
    # Otherwise, enter INTERACTIVE mode (with popups!)
    else:
        INTERACTIVE = True
        while True:
            # Clear terminal for a fresh menu
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f'\n{BOLD}{CYAN}{"="*65}')
            print(' AI From Scratch - INTERACTIVE RUNNER')
            print(f'{"="*65}{RESET}')
            print(' Select an algorithm to run (and see its graphs!):')
            print(f'{LINE}')
            for i, algo in enumerate(ALL_ORDER, 1):
                print(f'  [{i}] {algo.replace("_", " ").title()}')
            print(f'  [A] Run All')
            print(f'  [Q] Exit')
            print(f'{LINE}')
            
            choice = input('\n Choice > ').strip().lower()
            
            if choice == 'q':
                print('\n Goodbye!\n')
                break
            
            to_run = []
            if choice == 'a':
                to_run = ALL_ORDER
            elif choice.isdigit() and 1 <= int(choice) <= len(ALL_ORDER):
                to_run = [ALL_ORDER[int(choice)-1]]
            else:
                print(f'\n {YELLOW}Invalid choice!{RESET} Press Enter to try again...')
                input()
                continue
                
            total_t0 = time.time()
            for key in to_run:
                RUNNERS[key]()
            
            print(f'\n{BOLD}{GREEN}{"="*65}')
            print(f' Finished in {time.time()-total_t0:.1f}s')
            print(f'{"="*65}{RESET}')
            input('\n Press Enter to return to menu...')

    print(f'\n{BOLD}{GREEN}{"="*65}')
    print(f'  Session ended. Plots available in: {PLOTS_DIR}')
    print(f'{"="*65}{RESET}\n')
