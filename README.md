# FYS-STK4155 Project 2: Neural networks with backpropagation

**Group members:** Simon Berg, Arian Farhanieh

## Description

We've writen a FFNN, with backpropagation and train it with plain gradient descent, plain SGD and SGD with RMSprop and Adam. The gradients from backpropagation are checked against automatic differentiation with JAX. We use the network for both regression and classification.

**Regression:** the one-dimensional Runge function $f(x) = 1/(1+25x^2)$ with noise, compared with OLS, Ridge and Lasso from project 1. We test the sigmoid, ReLU and leaky ReLU activations, different depths and widths, and L1 and L2 penalties. Learning rates, architectures and penalties are chosen on a validation set taken from the training data; the test set is only used for the selected models.

**Classification:** the full MNIST data set of handwritten digits, with a softmax output and the cross-entropy cost. We compare against logistic regression (our network with no hidden layers) and Scikit-Learn, and show the confusion matrix of the best model.

## Installation

Python 3.14 works, for any other version we cant guarantee compatibility.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Note: The figures use LaTeX for their text (`text.usetex`), so a LaTeX installation is needed. Otherwise, set `"text.usetex": False` in `plot_style()`.

## Files

| File | Description |
|---|---|
| `Code/Project2.ipynb` | All code and results from all parts of the project: the network, the training loop Run from top to bottom from inside `Code/` (about 10 minutes on 20 cores). |
| `Code/figures/` | The figures used in the report, saved by the notebook as PDF. |
| `requirements.txt` | The Python packages, with the versions used. |

Running the notebook will automaticlly download the MNIST data set\\
All random numbers and data splits use the seed 2026.
