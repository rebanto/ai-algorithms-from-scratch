import numpy as np
import matplotlib.pyplot as plt
import os

# character-level RNN from scratch.
# the idea: train on a text corpus, learn to predict the next character given
# all the characters so far. then sample from it to generate new text.
#
# the tricky part compared to feedforward networks is that the hidden state
# carries information across time steps -- the network has memory.
# backprop through time (BPTT) is just unrolling that computation graph
# and applying the chain rule across every timestep.
#
# training: feed one character at a time, predict the next one.
# the loss at each step is cross-entropy between predicted and actual next char.
#
# i kept getting exploding gradients, which is why there's gradient clipping.
# without it the weights would just blow up after a few batches.

# -----------------------------------------------------------------------
# Corpus
# -----------------------------------------------------------------------

# a small slice of hamlet -- long enough to train on, short enough to fit in memory
corpus = """To be, or not to be, that is the question:
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep,
No more; and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to: tis a consummation
Devoutly to be wished. To die, to sleep;
To sleep, perchance to dream. Ay, there is the rub,
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There is the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
The oppressors wrong, the proud mans contumely,
The pangs of despised love, the laws delay,
The insolence of office, and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin.""".strip()

# map every unique character to an integer index
chars    = sorted(set(corpus))
vocab_size = len(chars)
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in enumerate(chars)}
data     = np.array([char2idx[c] for c in corpus], dtype=np.int32)

print(f"Corpus length : {len(corpus)} characters")
print(f"Vocabulary    : {vocab_size} unique characters")
print(f"Characters    : {''.join(chars)}")


# -----------------------------------------------------------------------
# One-hot encoding helper
# -----------------------------------------------------------------------

def one_hot(idx, size):
    """Convert integer indices to one-hot rows: (N,) -> (N, size)"""
    o = np.zeros((len(idx), size))
    o[np.arange(len(idx)), idx] = 1
    return o


# -----------------------------------------------------------------------
# RNN Cell
# -----------------------------------------------------------------------

class RNNCell:
    """
    Single-layer vanilla RNN.
    h_t = tanh(Wxh @ x_t  +  Whh @ h_{t-1}  +  bh)
    y_t = Why @ h_t + by        (logits, before softmax)
    """

    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.input_size  = input_size
        self.output_size = output_size

        scale = 0.01
        # input-to-hidden weights
        self.Wxh = np.random.randn(input_size,  hidden_size) * scale
        # hidden-to-hidden (recurrent) weights
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale
        # hidden-to-output weights
        self.Why = np.random.randn(hidden_size, output_size) * scale
        # biases
        self.bh  = np.zeros(hidden_size)
        self.by  = np.zeros(output_size)

    def forward(self, inputs, h_prev):
        """
        inputs : list of T one-hot vectors, each (input_size,)
        h_prev : initial hidden state (hidden_size,)
        returns xs, hs, ys, ps -- all the intermediate values needed for BPTT
        """
        T   = len(inputs)
        xs  = inputs                            # input at each step
        hs  = np.zeros((T + 1, self.hidden_size))
        ys  = np.zeros((T, self.output_size))   # logits
        ps  = np.zeros((T, self.output_size))   # softmax probs

        hs[-1] = h_prev  # store previous hidden state at index -1 (python wraps around)

        for t in range(T):
            hs[t] = np.tanh(xs[t] @ self.Wxh + hs[t-1] @ self.Whh + self.bh)
            ys[t] = hs[t] @ self.Why + self.by
            # softmax for probabilities
            e = np.exp(ys[t] - ys[t].max())
            ps[t] = e / e.sum()

        return xs, hs, ys, ps

    def backward(self, xs, hs, ps, targets, clip=5.0):
        """
        Backpropagation Through Time (BPTT).
        targets: list of T integer indices (the true next characters)
        """
        T = len(xs)

        # zero out gradients
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh  = np.zeros_like(self.bh)
        dby  = np.zeros_like(self.by)
        dh_next = np.zeros(self.hidden_size)

        loss = 0.0

        for t in reversed(range(T)):
            # cross-entropy loss at this timestep
            loss -= np.log(ps[t, targets[t]] + 1e-9)

            # gradient of loss w.r.t. logits (softmax + CE combined)
            dy = ps[t].copy()
            dy[targets[t]] -= 1

            dWhy += np.outer(hs[t], dy)
            dby  += dy

            # gradient flowing back into the hidden state from output + future hidden
            dh = dy @ self.Why.T + dh_next

            # backprop through tanh: dtanh = (1 - tanh^2) * upstream
            dtanh = (1 - hs[t] ** 2) * dh

            dbh  += dtanh
            dWxh += np.outer(xs[t], dtanh)
            dWhh += np.outer(hs[t-1], dtanh)
            dh_next = dtanh @ self.Whh.T

        # gradient clipping -- without this the gradients can explode catastrophically
        # on long sequences. clip everything to [-clip, clip].
        for grad in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(grad, -clip, clip, out=grad)

        return dWxh, dWhh, dWhy, dbh, dby, loss / T, hs[T - 1]

    def update(self, dWxh, dWhh, dWhy, dbh, dby, lr):
        self.Wxh -= lr * dWxh
        self.Whh -= lr * dWhh
        self.Why -= lr * dWhy
        self.bh  -= lr * dbh
        self.by  -= lr * dby

    def sample(self, h, seed_idx, n_chars):
        """Generate n_chars characters starting from seed_idx."""
        x    = np.zeros(self.input_size)
        x[seed_idx] = 1
        generated = [seed_idx]

        for _ in range(n_chars):
            h = np.tanh(x @ self.Wxh + h @ self.Whh + self.bh)
            y = h @ self.Why + self.by
            e = np.exp(y - y.max())
            p = e / e.sum()
            # sample from the distribution rather than always taking argmax
            # this gives more natural, varied text
            idx = np.random.choice(self.output_size, p=p)
            x = np.zeros(self.input_size)
            x[idx] = 1
            generated.append(idx)

        return ''.join(idx2char[i] for i in generated)


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

HIDDEN_SIZE = 128
SEQ_LEN     = 25    # characters per training chunk
LR          = 1e-2
N_ITER      = 3000

rnn = RNNCell(input_size=vocab_size, hidden_size=HIDDEN_SIZE, output_size=vocab_size)

loss_history = []
smooth_loss  = -np.log(1.0 / vocab_size) * SEQ_LEN  # initial expected loss (random)
h_prev       = np.zeros(HIDDEN_SIZE)
pointer      = 0

print(f"\nTraining RNN (hidden={HIDDEN_SIZE}, seq_len={SEQ_LEN}, lr={LR})...\n")

for i in range(N_ITER):
    # wrap around the corpus
    if pointer + SEQ_LEN + 1 >= len(data):
        pointer = 0
        h_prev  = np.zeros(HIDDEN_SIZE)

    inputs_idx  = data[pointer:pointer + SEQ_LEN]
    targets_idx = data[pointer + 1:pointer + SEQ_LEN + 1]

    xs = one_hot(inputs_idx, vocab_size)

    # forward
    xs_list, hs, ys, ps = rnn.forward(xs, h_prev)

    # backward
    dWxh, dWhh, dWhy, dbh, dby, loss, h_prev = rnn.backward(xs, hs, ps, targets_idx)

    # Adagrad-style update (simple, stable for RNNs)
    rnn.update(dWxh, dWhh, dWhy, dbh, dby, lr=LR)

    smooth_loss = 0.999 * smooth_loss + 0.001 * loss * SEQ_LEN
    pointer += SEQ_LEN

    if i % 200 == 0:
        sample_text = rnn.sample(h_prev, inputs_idx[0], 100)
        print(f"iter {i:5d} | loss = {smooth_loss:.4f}")
        print(f"  Sample: {repr(sample_text)}\n")
        loss_history.append((i, smooth_loss))

# -----------------------------------------------------------------------
# Final generation
# -----------------------------------------------------------------------

print("\n--- Final Generated Text (200 chars) ---")
seed = data[0]
generated = rnn.sample(np.zeros(HIDDEN_SIZE), seed, 200)
print(generated)

# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------

os.makedirs('plots', exist_ok=True)

iters  = [h[0] for h in loss_history]
losses = [h[1] for h in loss_history]

plt.figure(figsize=(9, 5))
plt.plot(iters, losses, color='steelblue', linewidth=1.8)
plt.title('RNN Training Loss (smoothed cross-entropy)')
plt.xlabel('Iteration')
plt.ylabel('Smoothed Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join('plots', 'loss_curve.png'), dpi=150)
plt.show()

# visualize hidden state activations on the corpus
print("\nVisualizing hidden state activations...")
h = np.zeros(HIDDEN_SIZE)
hidden_states = []
for idx in data[:200]:
    x     = np.zeros(vocab_size)
    x[idx] = 1
    h = np.tanh(x @ rnn.Wxh + h @ rnn.Whh + rnn.bh)
    hidden_states.append(h[:32])   # first 32 units

H_mat = np.array(hidden_states).T  # (32, 200)

plt.figure(figsize=(14, 5))
plt.imshow(H_mat, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
plt.colorbar(label='tanh activation')
plt.title('Hidden State Activations (first 32 units, first 200 chars of corpus)')
plt.xlabel('Character position')
plt.ylabel('Hidden unit')
plt.tight_layout()
plt.savefig(os.path.join('plots', 'hidden_activations.png'), dpi=150)
plt.show()
