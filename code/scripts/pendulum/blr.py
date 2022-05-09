import numpy as np
import matplotlib.pyplot as plt

from src.bayesian_optimization import BayesianLinearRegression


regr = BayesianLinearRegression(2, 1, 1, 1)

f = lambda x: 2 * np.sin(x) - 3 * x
par_true = np.array([-3, 2]).reshape(-1, 1)

x_true = np.linspace(-np.pi, np.pi, 100)
y_true = f(x_true)

feat_true = np.empty(shape=(x_true.shape[0], 2))
feat_true[:, 0] = x_true
feat_true[:, 1] = np.sin(x_true)

N = 10
x_meas = np.random.uniform(-np.pi, np.pi, size=(N,))
y_meas = f(x_meas) + np.random.normal(0, 1, size=x_meas.shape)

feat_meas = np.empty(shape=(x_meas.shape[0], 2))
feat_meas[:, 0] = x_meas
feat_meas[:, 1] = np.sin(x_meas)

err = [np.linalg.norm(regr.params - par_true)]
for i in range(N):
    regr.learn(feat_meas, y_meas)
    err.append(np.linalg.norm(regr.params - par_true))

regr.learn(feat_meas, y_meas)
print(regr.params)

y_est, y_std = regr.predict(feat_true)

plt.figure()
plt.plot(x_true, y_true)
plt.plot(x_true, y_est)
plt.scatter(x_meas, y_meas, facecolor="None", edgecolor="k")
plt.grid()

plt.show()
