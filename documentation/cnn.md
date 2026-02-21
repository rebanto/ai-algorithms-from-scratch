# Convolutional Neural Network From Scratch

This was the biggest jump in complexity for spatial data. Backpropagating through a convolution sounds almost unapproachable when you first hear about it, but it's still just the chain rule applied to a sliding window. It takes everything I learned from the basic MLP and pushes it into the 2D world of images.

The payoff is huge. CNNs are why we went from barely being able to classify blurry images to superhuman performance on ImageNet. The key idea is "weight sharing"—using the same set of weights everywhere on the image.

## Why Not Just Use a Dense Layer on Images?

You could take a 28×28 image, flatten it to 784 numbers, and feed it into a regular dense layer. This actually works okay on MNIST. The problem is it generalizes terribly — if the digit is shifted 2 pixels to the right, the network sees what looks like a completely different input. There's no built-in notion of spatial translation invariance.

A conv layer fixes this by applying the same filter pattern at every position. If it learns to detect a vertical edge at position (5, 5), it'll also detect vertical edges at position (12, 20) without needing separate weights for that.

## The Convolution Operation

A convolutional layer has $F$ filters of size $k \times k$. For each filter and each spatial position $(i, j)$ in the input:

$$
\text{output}[f, i, j] = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \text{input}[i+m, j+n] \cdot \text{filter}[f, m, n] + b_f
$$

Slide this window across the entire image for every filter. The resulting output is called a **feature map**.

For a 28×28 input with 8 filters of size 3×3, the output is 8 feature maps each of size 26×26.

## Forward Pass (The Speed Trick)

Doing this with 4 nested Python loops works, but it’s painfully slow (like "take a coffee break" slow). The smarter way is to extract all the small patches from the image matrix in one shot. 

I used `numpy.lib.stride_tricks.as_strided`, which creates a "view" of all the patches we need without actually copying any data in memory. Then, a single `np.einsum` (Einstein summation) computes all the dot products for every filter at once. It’s basically a super-optimized matrix multiplication that handles the sliding window for us.

## Backward Pass (Passing the Error Back)

The gradient with respect to the **filters** is basically just looking at which parts of the input caused the error. The gradient with respect to the **input** is a bit trickier because each pixel contributes to multiple overlapping windows. I had to "scatter" the errors back through the filters to figure out where they came from.

## Max Pooling

After the conv layer, a 2×2 max pool with stride 2 halves the spatial dimensions. It takes the maximum value in each 2×2 region. During the backward pass, the gradient only flows to the position that was the max — all others get zero. I store a boolean mask during the forward pass to remember which position won.

## Architecture

```
Input (28×28)
  → Conv2D (8 filters, 3×3)  → 8×26×26
  → ReLU
  → MaxPool2D (2×2, stride 2) → 8×13×13
  → Flatten → 1352
  → Dense (1352 → 128) → ReLU
  → Dense (128 → 10) → Softmax
```

## Optimizer: Why Adam?

I used the **Adam** optimizer instead of basic SGD. Adam is like SGD but with an "internal memory." It keeps track of the average of the gradients (momentum) and the average of the squared gradients (to scale the learning rate).

Basically, if a weight is changing a lot, Adam slows it down. If it's barely moving, Adam gives it a nudge. It’s much more stable and let me use a higher learning rate without the model exploding.

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$
$$
w \leftarrow w - \alpha \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
$$

## Results

Training on 12,000 MNIST examples (not the full 60,000 — pure numpy is slow) for 10 epochs hits around 95-96% test accuracy. Not state of the art, but not bad for a from-scratch numpy implementation with a pretty small training set.

## Visualizations

- **`plots/training_curves.png`** — Loss and test accuracy over epochs
- **`plots/learned_filters.png`** — The 8 learned conv filters after training. Early-epoch filters look like noise; trained filters show edge-like patterns
- **`plots/sample_predictions.png`** — Random test images with true label vs predicted label (green=correct, red=wrong)

## Running It

```
pip install numpy matplotlib
# either tensorflow (for fast data loading):
pip install tensorflow
# or scikit-learn (slower data download):
pip install scikit-learn

python cnn.py
```
