# Gradient descent from scratch

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-0a2c46?style=flat)

Libraries like PyTorch hide the optimiser behind one line of code. I wanted to see what actually happens to the parameters on every step, so I wrote the optimisers myself in NumPy and watched them train a linear regression model.

I built this during the ITI Data Science diploma, as a series of practical exercises.

## What's inside

| Notebook | What it covers |
|---|---|
| `Momentum and NAG.ipynb` | Batch gradient descent, then Momentum and Nesterov accelerated gradient (NAG), on a single-variable regression. I tried momentum values from 0.5 to 0.9 to see how it changes the path to the minimum. |
| `Adagrad, RMSProp, and Adam.ipynb` | The three adaptive learning-rate methods, each built on the previous one. I compared learning rates of 0.05, 0.005 and 0.0005, and played with Adam's `beta1` and `beta2`. |
| `Comparative Analysis between BFGS & Adam.ipynb` | A vectorised Adam and a BFGS optimiser, both training a multivariable regression on the same data, with the same learning rate and stopping tolerance. |

The first two notebooks fit a line to `y = -x + 2`. The last one uses `MultiVarLR.csv`, a small dataset with 24 rows, three features and one target.

## The main result: BFGS vs Adam

Both optimisers used a learning rate of 0.5 and a tolerance of 1e-4. They stop when the loss change or the gradient norm drops below the tolerance.

| Optimiser | Iterations to converge | R² |
|---|---|---|
| BFGS | 43 (the loop counter prints 42) | 0.990 |
| Adam | 388 | 0.989 |

They end up with nearly the same fit, but BFGS gets there about nine times faster. That makes sense. BFGS builds an estimate of the curvature (the inverse Hessian) as it goes, so it takes better-scaled steps. Adam only rescales each parameter using running averages of the gradient.

## Caveats

- This is **one small dataset** with four parameters. I wouldn't claim the same ratio holds elsewhere.
- I compared **iterations, not time**. A BFGS iteration costs more, because it updates a matrix every step. With four parameters this barely matters. With millions it would be a problem, and it's why Adam is the usual choice for neural networks.
- The problem is a convex least-squares fit, which is where BFGS does best. I haven't tested either method on a hard, non-convex loss.
- The notebooks print every iteration, so the saved outputs are long. Jupyter truncates them.

## How to run

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook
```

Open any notebook and run the cells in order. The BFGS notebook reads the data from `/content/MultiVarLR.csv`, which is the Google Colab path. If you run it locally, change that line to `MultiVarLR.csv`.

## What I'd like to do next

Try these optimisers on a non-convex loss, such as a small neural network, and plot their paths over the loss surface.
