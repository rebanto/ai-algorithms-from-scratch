# AI From Scratch (NumPy Edition)

I wanted to see how these algorithms actually work under the hood. No TensorFlow, no PyTorch, no "magic" libraries—just raw **NumPy** and the math I learned from diving deep into linear algebra and calculus.

Most ML tutorials just tell you to import a library and call `.fit()`, which feels like cheating. This project is my attempt to build the "engines" from scratch: implementing the gradients, the backprop, and the decision logic from the ground up.

## Check it Out (The Master Runner)

I built a cool interactive script to show everything off. You can pick an algorithm, watch it train, and see the graphs pop up in real-time.

```bash
# Start the interactive menu
python run_all.py
```

If you're in a hurry and just want to run one thing from the terminal:
```bash
python run_all.py linear cnn    # runs specific ones
python run_all.py rnn           # runs the Hamlet text generator
```

---

## The "Under the Hood" Stuff

I wrote down the math and the "why" for each algorithm. If you want to see the derivatives or how the convolution stride tricks work, check these out:

- [1. Linear Regression](documentation/linear_regression.md) (The starting point: fitting a line)
- [2. Logistic Regression](documentation/logistic_regression.md) (Stepping up to classification)
- [3. K-Nearest Neighbors](documentation/knn.md) (Geometric similarity)
- [4. Gaussian Naive Bayes](documentation/naive_bayes.md) (Probabilistic classification)
- [5. The Bridge: Neural Networks](documentation/basic_neural_network.md) (Combining it all into a modular MLP)
- [6. Convolutional Neural Network](documentation/cnn.md) (The spatial challenge for images)
- [7. Recurrent Neural Network](documentation/rnn.md) (The "final boss" of bookkeeping and memory)

---

## Visualizing the Math

I spent a lot of time on the plots because seeing the math is way better than just looking at loss numbers:
- **3D Loss Bowls**: Watch the gradient descent fall down the convex surface.
- **Saliency Maps**: See exactly which pixels the CNN thinks are important.
- **Hidden State Heatmaps**: Peek into how the RNN's "brain" remembers characters.

---

## The Setup

```
.
├── documentation/       # Where I explain all the math
├── plots/               # All the generated graphs
├── run_all.py           # The main entry point
└── [Algos]_From_Scratch/ # The actual source code
```
