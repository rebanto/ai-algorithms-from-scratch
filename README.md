# AI From Scratch — NumPy Implementations

A collection of core machine learning and deep learning algorithms implemented entirely from scratch using **NumPy**. This project aims to demonstrate the mathematical foundations of AI without the "magic" of high-level frameworks.

## 🚀 Getting Started

The project features a master runner script that coordinates all implementations.

### Interactive Mode
Launch the interactive menu to pick algorithms and see live Matplotlib visualizations:
```bash
python run_all.py
```

### CLI Mode
Run specific algorithms (or all) directly from the command line:
```bash
python run_all.py linear cnn    # runs specific algorithms
python run_all.py rnn           # runs the character-level RNN
```

---

## 📚 Algorithm Documentation

All detailed math and implementation explanations have been consolidated into the [documentation/](documentation/) directory:

- [1. Linear Regression](documentation/linear_regression.md)
- [2. Logistic Regression](documentation/logistic_regression.md)
- [3. K-Nearest Neighbors](documentation/knn.md)
- [4. Gaussian Naive Bayes](documentation/naive_bayes.md)
- [5. Convolutional Neural Network](documentation/cnn.md)
- [6. Recurrent Neural Network](documentation/rnn.md)
- [7. Basic Neural Network](documentation/basic_neural_network.md) (Multi-layer Perceptron)

---

## 🎨 Visualizations

The runner script generates rich plots for every algorithm, stored in the `plots/` directory:
- **3D Loss Surfaces**: Visualize the convex optimization landscape.
- **Decision Boundaries**: See how models separate different classes.
- **Saliency Maps**: View which pixels a CNN focuses on for digit recognition.
- **Activation Heatmaps**: Peek into an RNN's internal memory during text generation.

---

## 📁 Repository Structure

```
.
├── documentation/       # Centralized math & theory docs
├── plots/               # Generated visualizations
├── Linear_Regression_From_Scratch/
├── Logistic_Regression_From_Scratch/
├── KNN_From_Scratch/
├── Naive_Bayes_From_Scratch/
├── CNN_From_Scratch/
├── RNN_From_Scratch/
├── Basic_Neural_Network_From_Scratch/
└── run_all.py           # Master interactive runner
```
