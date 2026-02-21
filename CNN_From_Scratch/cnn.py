import numpy as np
import matplotlib.pyplot as plt
import os

# CNNs were the thing I was most scared to build from scratch.
# Backprop through a convolution sounds terrifying but it's actually just
# more chain rule -- the same idea as everywhere else.
#
# The architecture here: Conv2D -> ReLU -> MaxPool -> Flatten -> Dense -> ReLU -> Dense -> Softmax
# Trained on MNIST (handwritten digits 0-9).
#
# Key insight I kept forgetting: the conv layer doesn't have a separate weight
# for every pixel position. The same filter slides over the whole image --
# that's the "weight sharing" that makes CNNs so parameter-efficient.
#
# I use numpy stride tricks for the forward pass to extract patches efficiently.
# The alternative (4 nested Python loops) works but takes forever on MNIST.

# -----------------------------------------------------------------------
# Activations
# -----------------------------------------------------------------------

class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, d_out):
        return d_out * (self.x > 0)


class Softmax:
    def forward(self, x):
        # subtract row max before exp to prevent overflow -- numerically identical result
        e = np.exp(x - x.max(axis=1, keepdims=True))
        self.out = e / e.sum(axis=1, keepdims=True)
        return self.out

    def backward(self, d_out):
        # when paired with cross-entropy, the combined gradient is (pred - one_hot)/n
        # the loss function's backward already handles this, so softmax just passes through
        return d_out


# -----------------------------------------------------------------------
# Layers
# -----------------------------------------------------------------------

class Conv2D:
    """Single-channel 2D convolution (no padding, stride=1)."""

    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        fs = filter_size
        # He initialization -- good default for layers followed by ReLU
        self.filters = np.random.randn(num_filters, fs, fs) * np.sqrt(2 / (fs * fs))
        self.biases  = np.zeros(num_filters)

    def forward(self, x):
        # x: (batch, H, W)
        self.x = x
        batch, H, W = x.shape
        fs = self.filter_size
        out_H = H - fs + 1
        out_W = W - fs + 1

        # extract all (fs x fs) patches from x in one shot using stride tricks.
        # result shape: (batch, out_H, out_W, fs, fs) -- every patch we'll ever need.
        shape   = (batch, out_H, out_W, fs, fs)
        strides = (x.strides[0], x.strides[1], x.strides[2], x.strides[1], x.strides[2])
        self.patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides).copy()

        # einsum: for each batch item and output position, dot the patch with each filter
        # 'bhwij,fij->bfhw'  -- contracting over the (fs x fs) spatial dims
        self.out = (np.einsum('bhwij,fij->bfhw', self.patches, self.filters)
                    + self.biases[None, :, None, None])
        return self.out

    def backward(self, d_out):
        # d_out: (batch, num_filters, out_H, out_W)
        fs = self.filter_size
        batch, H, W = self.x.shape
        _, _, out_H, out_W = d_out.shape

        # gradient w.r.t. filters: cross-correlate input patches with output gradient
        self.dfilters = np.einsum('bhwij,bfhw->fij', self.patches, d_out)
        self.dbiases  = d_out.sum(axis=(0, 2, 3))

        # gradient w.r.t. input: scatter each upstream gradient through its corresponding filter
        # only looping over spatial output positions (out_H * out_W iterations)
        self.dinputs = np.zeros_like(self.x)
        for i in range(out_H):
            for j in range(out_W):
                # d_out[:, :, i, j]: (batch, num_filters)
                # result: (batch, fs, fs)
                self.dinputs[:, i:i+fs, j:j+fs] += np.einsum('bf,fij->bij', d_out[:, :, i, j], self.filters)

        return self.dinputs


class MaxPool2D:
    """2x2 max pooling with stride 2."""

    def __init__(self, pool_size=2, stride=2):
        self.ps = pool_size
        self.s  = stride

    def forward(self, x):
        # x: (batch, C, H, W)
        self.x = x
        batch, C, H, W = x.shape
        ps, s = self.ps, self.s
        out_H = (H - ps) // s + 1
        out_W = (W - ps) // s + 1

        self.out  = np.zeros((batch, C, out_H, out_W))
        self.mask = np.zeros_like(x, dtype=bool)

        for i in range(out_H):
            for j in range(out_W):
                region   = x[:, :, i*s:i*s+ps, j*s:j*s+ps]       # (batch, C, ps, ps)
                max_vals = region.max(axis=(2, 3), keepdims=True)  # (batch, C, 1, 1)
                self.out[:, :, i, j] = max_vals[:, :, 0, 0]
                # remember which positions held the max (needed for backward)
                self.mask[:, :, i*s:i*s+ps, j*s:j*s+ps] |= (region == max_vals)

        return self.out

    def backward(self, d_out):
        ps, s = self.ps, self.s
        _, _, out_H, out_W = d_out.shape
        self.dinputs = np.zeros_like(self.x)

        for i in range(out_H):
            for j in range(out_W):
                mask_region = self.mask[:, :, i*s:i*s+ps, j*s:j*s+ps].astype(float)
                # normalize so tied maxes share the gradient equally
                mask_region /= mask_region.sum(axis=(2, 3), keepdims=True).clip(min=1)
                self.dinputs[:, :, i*s:i*s+ps, j*s:j*s+ps] += (
                    mask_region * d_out[:, :, i, j][:, :, None, None]
                )

        return self.dinputs


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, d_out):
        return d_out.reshape(self.shape)


class Dense:
    def __init__(self, in_features, out_features):
        self.w = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
        self.b = np.zeros(out_features)

    def forward(self, x):
        self.x = x
        return x @ self.w + self.b

    def backward(self, d_out):
        self.dw = self.x.T @ d_out
        self.db = d_out.sum(axis=0)
        return d_out @ self.w.T


# -----------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------

class CrossEntropyLoss:
    def forward(self, y_pred, y_true):
        # y_true: integer class labels (not one-hot)
        # pick the probability at each sample's true class
        n = len(y_true)
        y_pred = np.clip(y_pred, 1e-7, 1.0)
        self.loss = -np.sum(np.log(y_pred[np.arange(n), y_true])) / n
        return self.loss

    def backward(self, y_pred, y_true):
        # combined softmax + cross-entropy gradient simplifies nicely:
        # d/d(logit) = (softmax_output - one_hot) / n
        n    = len(y_true)
        grad = y_pred.copy()
        grad[np.arange(n), y_true] -= 1
        return grad / n


# -----------------------------------------------------------------------
# Optimizer (Adam)
# -----------------------------------------------------------------------

class Adam:
    """Adaptive moment estimation optimizer."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0
        self.m     = {}
        self.v     = {}

    def update(self, param, grad, key):
        if key not in self.m:
            self.m[key] = np.zeros_like(grad)
            self.v[key] = np.zeros_like(grad)

        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad ** 2

        m_hat = self.m[key] / (1 - self.beta1 ** self.t)
        v_hat = self.v[key] / (1 - self.beta2 ** self.t)

        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return param




if __name__ == '__main__':
    print("Loading MNIST...")
    try:
        from tensorflow.keras.datasets import mnist as km
        (X_train, y_train), (X_test, y_test) = km.load_data()
    except ImportError:
        from sklearn.datasets import fetch_openml
        print("TF not found, using sklearn...")
        d = fetch_openml('mnist_784', version=1, as_frame=False, cache=True)
        X_train = d.data.reshape(-1, 28, 28); y_train = d.target.astype(int)
        X_test = X_train[60000:]; y_test = y_train[60000:]
        X_train = X_train[:60000]; y_train = y_train[:60000]

    import numpy as np_local
    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32')  / 255.0
    TRAIN_N, TEST_N = 12000, 2000
    X_train, y_train = X_train[:TRAIN_N], y_train[:TRAIN_N]
    X_test,  y_test  = X_test[:TEST_N],   y_test[:TEST_N]
    print(f"Using {TRAIN_N} train / {TEST_N} test samples")

    conv1  = Conv2D(8, 3); relu1 = ReLU(); pool1 = MaxPool2D(2, 2); flat = Flatten()
    den1   = Dense(8*13*13, 128); relu2 = ReLU(); den2 = Dense(128, 10)
    smx    = Softmax(); loss_fn = CrossEntropyLoss(); opt = Adam(0.001)
    EPOCHS = 10; BS = 64; N_B = TRAIN_N // BS
    lh, ah = [], []

    def fwd(x):
        x = conv1.forward(x); x = relu1.forward(x); x = pool1.forward(x)
        x = flat.forward(x);  x = den1.forward(x);  x = relu2.forward(x)
        x = den2.forward(x);  x = smx.forward(x); return x

    def bwd(d):
        d = smx.backward(d);  d = den2.backward(d); d = relu2.backward(d)
        d = den1.backward(d); d = flat.backward(d); d = pool1.backward(d)
        d = relu1.backward(d); d = conv1.backward(d)

    print("Training...")
    for epoch in range(EPOCHS):
        perm = np.random.permutation(TRAIN_N)
        Xs, ys = X_train[perm], y_train[perm]; el = 0.0; opt.t += 1
        for b in range(N_B):
            xb = Xs[b*BS:(b+1)*BS]; yb = ys[b*BS:(b+1)*BS]
            out = fwd(xb); el += loss_fn.forward(out, yb)
            bwd(loss_fn.backward(out, yb))
            den2.w = opt.update(den2.w, den2.dw, 'd2w'); den2.b = opt.update(den2.b, den2.db, 'd2b')
            den1.w = opt.update(den1.w, den1.dw, 'd1w'); den1.b = opt.update(den1.b, den1.db, 'd1b')
            conv1.filters = opt.update(conv1.filters, conv1.dfilters, 'c1f')
            conv1.biases  = opt.update(conv1.biases,  conv1.dbiases,  'c1b')
        correct = sum((np.argmax(fwd(X_test[i:i+BS]), axis=1) == y_test[i:i+BS]).sum() for i in range(0, TEST_N, BS))
        acc = correct / TEST_N; lh.append(el/N_B); ah.append(acc)
        print(f"Epoch {epoch+1}/{EPOCHS}  loss={lh[-1]:.4f}  acc={acc:.4f}")

    import os, matplotlib.pyplot as plt
    os.makedirs('plots', exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(range(1, EPOCHS+1), lh, marker='o', color='steelblue')
    a1.set_title('Training Loss'); a1.set_xlabel('Epoch'); a1.grid(True, ls='--', alpha=0.5)
    a2.plot(range(1, EPOCHS+1), ah, marker='s', color='seagreen')
    a2.set_title('Test Accuracy'); a2.set_xlabel('Epoch'); a2.set_ylim(0,1); a2.grid(True, ls='--', alpha=0.5)
    plt.suptitle('CNN on MNIST', fontsize=13); plt.tight_layout()
    plt.savefig(os.path.join('plots', 'training_curves.png'), dpi=150); plt.show()

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i, ax in enumerate(axes.ravel()):
        f = conv1.filters[i]; vm = abs(f).max()
        ax.imshow(f, cmap='RdBu', vmin=-vm, vmax=vm); ax.set_title(f'Filter {i}'); ax.axis('off')
    plt.suptitle('Learned Filters'); plt.tight_layout()
    plt.savefig(os.path.join('plots', 'learned_filters.png'), dpi=150); plt.show()

    idx = np.random.choice(TEST_N, 10, replace=False); preds = np.argmax(fwd(X_test[idx]), axis=1)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(X_test[idx[i]], cmap='gray')
        col = 'green' if preds[i] == y_test[idx[i]] else 'red'
        ax.set_title(f'T:{y_test[idx[i]]} P:{preds[i]}', color=col, fontsize=9); ax.axis('off')
    plt.suptitle('Sample Predictions'); plt.tight_layout()
    plt.savefig(os.path.join('plots', 'sample_predictions.png'), dpi=150); plt.show()
    print(f"Final test accuracy: {ah[-1]:.4f}")
