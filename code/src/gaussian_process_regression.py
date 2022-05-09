import random
import numpy as np
import scipy
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
import warnings
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import WhiteKernel

from src.mpc import PendulumFilter
from utils.angle import minus_pi_to_pi

warnings.filterwarnings("ignore")


class BangBangPolicy:

    KERNEL = RBF(5, length_scale_bounds="fixed") + WhiteKernel(
        noise_level_bounds="fixed"
    )
    # KERNEL = Matern(length_scale=20, length_scale_bounds="fixed", nu=0.5) + WhiteKernel(
    #     noise_level=1, noise_level_bounds="fixed"
    # )
    MODEL = GaussianProcessRegressor(KERNEL)

    T = 150
    N_SAMPLES = int(1e3)

    def __init__(self):
        self.X = np.empty((0, 2))
        self.y = np.empty((0))

    def surrogate(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = self.MODEL.predict(X, return_std=True)
            return mean, std

    def acquisition(self, X, X_samples):
        yhat, _ = self.surrogate(X)
        best = max(yhat)
        mu, std = self.surrogate(X_samples)
        z = (mu - best) / (std + 1e-9)
        cdf = scipy.stats.norm.cdf(z)
        pdf = scipy.stats.norm.pdf(z)

        # Expected Improvement
        return (mu - best) * cdf + std * pdf

    def get_next_x(self, X):
        k2s = [random.randint(0, self.T) for _ in range(self.N_SAMPLES)]
        k1s = [random.randint(0, k) for k in k2s]
        X_samples = np.array([k1s, k2s]).T
        scores = self.acquisition(X, X_samples)
        ix = np.argmax(scores)
        return X_samples[ix]

    def learn(self, X, y):
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        self.X = np.vstack((self.X, X))
        self.y = np.hstack((self.y, y))
        self.MODEL.fit(self.X, self.y)
        x = self.get_next_x(self.X).reshape(1, -1)
        return x[0, 0], x[0, 1]

    def predict(self):
        i = np.argmax(self.y)
        return self.X[i]

    def plot(self):
        X_plot = np.meshgrid(np.arange(self.T), np.arange(self.T))
        X_eval = np.array(X_plot).reshape(2, -1).T
        est, _ = self.surrogate(X_eval)
        est = est.reshape(self.T, self.T)

        plt.figure("cost")
        plt.pcolormesh(*X_plot, est)
        plt.scatter(*self.X.T, 1, color="k")
        plt.xlim(0, self.T)
        plt.ylim(0, self.T)
        plt.colorbar()
        plt.show()


class BangBang:

    NX = 2
    NU = 2
    T = 70
    X0 = np.zeros((2,))
    XREF = np.pi

    def __init__(self, sys, filter):
        self.sys = sys
        self.filter = filter
        self._run = self.sys._update.mapaccum(self.T)
        self.data = None

        kernel = (
            RBF(2, length_scale_bounds="fixed")
            + ConstantKernel(0.5, constant_value_bounds="fixed")
            + WhiteKernel(0.1, noise_level_bounds="fixed")
        )
        self.model = GaussianProcessRegressor(kernel=kernel)

    def run(self, p):
        u = np.zeros((self.T,))
        u[: p[0]] = self.filter.INPUT_CONS["low"]
        u[p[0] :] = self.filter.INPUT_CONS["high"]

        x = self._run(self.X0, u)
        x = np.array(x)
        return x, u

    def objective(self, X, noise=0.05):
        noise = np.random.normal(loc=0, scale=noise, size=(X.shape[0],))
        total_costs = []
        for p in X:
            x, _ = self.run(p)
            total_costs.append(
                sum((minus_pi_to_pi(self.XREF - x[0])) ** 2) / self.T / 10
            )
        total_costs = np.array(total_costs) + noise

        return -total_costs

    def surrogate(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.model.predict(X, return_std=True)

    def acquisition(self, X, X_samples):
        # calculate the best surrogate score found so far
        yhat, _ = self.surrogate(X)
        best = max(yhat)
        mu, std = self.surrogate(X_samples)
        z = (mu - best) / (std + 1e-9)
        cdf = scipy.stats.norm.cdf(z)
        pdf = scipy.stats.norm.pdf(z)

        # Expected Improvement
        # return (mu - best) * cdf + std * pdf

        # Entropy Search
        tmp = 0.5 * z * pdf / cdf - np.log(cdf)
        breakpoint()
        return np.mean(tmp, axis=1, keepdims=True)

    def get_next_x(self, X):
        # X_samples = np.meshgrid(np.arange(self.T), np.arange(self.T))
        # X_samples = np.array(X_samples).reshape(2, -1).T

        X_samples = np.arange(self.T).reshape(-1, 1)
        scores = self.acquisition(X, X_samples)
        ix = np.argmax(scores)

        # self._not_vistied = np.delete(self._not_vistied, ix, 0)

        return X_samples[ix]

    def generate_data(self, n):
        X = np.random.randint(self.T, size=(n, 1))
        y = self.objective(X)
        return X, y

    def learn(self, iterations):
        X, y = self.generate_data(100)

        self.model.fit(X, y)

        for i in range(iterations):
            x = self.get_next_x(X).reshape(1, -1)
            cost = self.objective(x)
            print(">n=%.0f, x=%.0f, f()=%3f" % (i, x.squeeze(), cost))
            X = np.vstack((X, x))
            y = np.hstack((y, cost))
            self.model.fit(X, y)

        ix = np.argmax(y)
        print("Best result: x=%.3f, y=%.3f" % (X[ix], y[ix]))

        self.data = X

        return X[ix]

    def plot(self):
        X_samples = np.arange(self.T).reshape(-1, 1)
        yest, _ = self.surrogate(X_samples)
        ytrue = self.objective(X_samples, noise=0)

        ix = np.argmax(ytrue)
        print("True best: x=%.3f, y=%.3f" % (X_samples[ix], ytrue[ix]))

        plt.plot(X_samples, ytrue)
        plt.plot(X_samples, yest)
        plt.grid()
        plt.show()


class SafeBangBang(BangBang):
    def __init__(self, sys, filter):
        super().__init__(sys, filter)

        self.filter = PendulumFilter(self.sys, 55)

    def safe_run(self, p):
        ul = np.zeros((self.T,))
        ul[: p[0]] = -self.filter.INPUT_CONS
        ul[p[0] : p[1]] = self.filter.INPUT_CONS
        ul[p[1] :] = 0
        x = np.zeros((self.NX, self.T + 1))
        u = np.zeros((self.T,))
        x[:, 0] = self.X0
        for i in range(self.T):
            _, v = self.filter.optimize(x[:, i], ul[i])
            v = v.full().squeeze()
            x[:, i + 1] = self.sys.update(x[:, i], v[0])
            u[i] = v[0]

        return x, u

    def objective(self, X, noise=0.0):
        total_costs = []
        for p in X:
            x, _ = self.safe_run(p)
            total_costs.append(sum(abs(normalize_angle(self.XREF - x[0]))) / self.T)
            input_cost = sum((u - ul) ** 2)
            total_costs.append((state_cost + input_cost) / self.T)
