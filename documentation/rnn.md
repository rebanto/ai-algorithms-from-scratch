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

This is where the bookkeeping gets wild. To compute gradients, you have to "unroll" the whole sequence and pass the error back through every single time step. 

Think of it like replaying a tape and correcting mistakes at every frame. The gradient at any step $t$ doesn't just care about that step—it carries information from all the *future* steps it contributed to. In code, I walk backwards from the end of the sequence to the beginning, accumulating the error as I go:

$$
\delta_t = \frac{\partial L}{\partial y_t} W_{hy}^T + \delta_{t+1} W_{hh}^T, \qquad \text{tanh backprop: } (1 - h_t^2) \odot \delta_t
$$

## The Exploding Gradient Problem

Vanilla RNNs have a famous math problem: if you multiply a number by itself 25 times, it either disappears or grows into a monster. When calculating gradients over long sequences, they can "explode" and turn into `NaN` in an instant.

The fix is "Gradient Clipping"—it sounds fancy, but it just means "if the gradient is bigger than 5, force it to be 5." It’s a simple safety valve that keeps the training stable.

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
