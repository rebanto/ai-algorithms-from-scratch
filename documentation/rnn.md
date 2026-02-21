# Recurrent Neural Network From Scratch

RNNs were the hardest thing I've built in this series, and not because the math is uniquely difficult — it's just that the bookkeeping gets complicated fast. A feedforward network maps one input to one output. An RNN processes a *sequence*, maintaining a hidden state that accumulates information across every step. That hidden state is what gives the network memory.

I trained a character-level RNN on a slice of Hamlet. The task: given all the characters you've seen so far, predict the next one. Once trained, you can sample from it to generate new text that (sort of) sounds like Shakespeare.

## The Architecture

At each time step $t$, the RNN takes the current input $x_t$ and the previous hidden state $h_{t-1}$, and produces a new hidden state $h_t$ and an output $y_t$:

$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$
$$
y_t = W_{hy} h_t + b_y
$$

Then softmax over $y_t$ gives probabilities over the vocabulary.

The same weight matrices $W_{xh}$, $W_{hh}$, $W_{hy}$ are shared across *every* time step. That's what makes it recurrent — the same computation applied at each step, with state flowing through.

## Backpropagation Through Time (BPTT)

This is where it gets interesting. To compute gradients, you unroll the computation graph across all $T$ time steps and apply the chain rule backwards through the whole sequence. At each step $t$, the gradient of the loss flows back through:
1. The output projection ($W_{hy}$)
2. The tanh nonlinearity
3. Into the previous hidden state (via $W_{hh}$)

The gradient that reaches hidden state $h_{t-1}$ carries information from all future time steps.

In code this looks like walking backwards through $t = T-1, T-2, \ldots, 0$ and accumulating:

$$
\delta_t = \frac{\partial L}{\partial y_t} W_{hy}^T + \delta_{t+1} W_{hh}^T, \qquad \text{tanh backprop: } (1 - h_t^2) \odot \delta_t
$$

## Gradient Clipping

Vanilla RNNs famously suffer from **exploding gradients**. When you multiply the recurrent weight matrix by itself $T$ times (during BPTT), the gradient can grow exponentially. Without clipping, the weights blow up after just a few batches.

The fix: just clip all gradients to $[-c, c]$ before applying them. It's blunt but it works:

```python
np.clip(grad, -5.0, 5.0, out=grad)
```

(Vanishing gradients — where the gradient shrinks to zero over long sequences — are a different and harder problem, which is why LSTMs were invented. This implementation is a vanilla RNN, so it really only "remembers" the last ~20-30 characters reliably.)

## Character-Level Language Modeling

The input is one-hot encoded: a vector of length `vocab_size` with a single 1 at the position of the current character. The target at each step is just the *next* character in the sequence.

To generate text: start with a seed character, run a forward pass, sample from the output probability distribution (not just argmax — sampling gives more varied text), feed the sampled character back as the next input, repeat.

## Training Details

I used a hidden size of 128 and sequence length of 25. The learning rate is 0.01 with gradient clipping at 5. Training runs for 3000 iterations on the embedded Hamlet corpus.

Early in training the generated text is basically random noise. By iteration 1000 it starts getting spaces and punctuation in the right places. By the end it's producing something that vaguely resembles English sentence structure, even if it's mostly nonsense words.

## What I Built

```
RNNCell
├── forward(inputs, h_prev)        # runs T steps, returns all intermediates
├── backward(xs, hs, ps, targets)  # BPTT with gradient clipping
├── update(gradients, lr)          # SGD update
└── sample(h, seed_idx, n_chars)   # generate text by sampling from the model
```

## Visualizations

- **`plots/loss_curve.png`** — Smoothed cross-entropy loss over training. Should be a steady decline with some noise.
- **`plots/hidden_activations.png`** — Heatmap of the first 32 hidden units over the first 200 characters of the corpus. You can actually see different units activating for different patterns (spaces, vowels, punctuation). This is my favorite plot in the whole project.

## Running It

```
pip install numpy matplotlib
python rnn.py
```

No dataset download needed — the corpus is embedded in the script.
