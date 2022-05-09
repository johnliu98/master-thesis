import numpy as np
import matplotlib.pyplot as plt

from src.bayesian_optimization import BayesianLinearRegression


regr = BayesianLinearRegression(1, 1, 1, 1)

x = np.linspace(-np.pi, np.pi, 100)
y_true = 0.5 * np.sin(x) - x

plt.figure()
plt.plot(x, y_true)
plt.show()
