import numpy as np
import scipy
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
import warnings
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import WhiteKernel

from src.mpc import SafetyFilter
from utils.angle import normalize_angle

warnings.filterwarnings("ignore")


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
        u[p[0] : p[1]] = self.filter.INPUT_CONS["high"]
        u[p[1] :] = 0

        x = self._run(self.X0, u)
        x = np.array(x)
        x[0] = normalize_angle(x[0])
        return x, u

    def objective(self, X, noise=0.05):
        noise = np.random.normal(loc=0, scale=noise, size=(X.shape[0],))
        total_costs = []
        for p in X:
            x, _ = self.run(p)
            total_costs.append(sum((normalize_angle(self.XREF - x[0])) ** 2) / self.T / 10)
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
        score = (mu - best) * cdf + std * pdf
        return score

    def get_next_x(self, X):
        X_samples = np.meshgrid(np.arange(self.T), np.arange(self.T))
        X_samples = np.array(X_samples).reshape(2, -1).T
        scores = self.acquisition(X, X_samples)
        ix = np.argmax(scores)
        # self._not_vistied = np.delete(self._not_vistied, ix, 0)
        return X_samples[ix]

    def generate_data(self, n):
        X = np.random.randint(self.T, size=(n, 2))
        y = self.objective(X)
        return X, y

    def learn(self, iterations):
        X, y = self.generate_data(100)

        self.model.fit(X, y)

        for i in range(iterations):
            x = self.get_next_x(X).reshape(1, -1)
            cost = self.objective(x)
            print(">n=%.0f, x=[%.0f, %.0f], f()=%3f" % (i, *x.squeeze(), cost))
            X = np.vstack((X, x))
            y = np.hstack((y, cost))
            self.model.fit(X, y)

        ix = np.argmax(y)
        print("Best result: x=[%.3f, %.3f], y=%.3f" % (*X[ix], y[ix]))

        self.data = X

        return X[ix]

    def plot(self):
        # 2, 1, figsize=(5, 5), subplot_kw={"projection": "3d"}
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))
        X_plot = np.meshgrid(np.arange(self.T), np.arange(self.T))
        X_samples = np.array(X_plot).reshape(2, -1).T
        yest, _ = self.surrogate(X_samples)
        ytrue = self.objective(X_samples, noise=0)

        ix = np.argmax(ytrue)
        print("True best: x=[%.3f, %.3f], y=%.3f" % (*X_samples[ix], ytrue[ix]))

        n = int(np.sqrt(yest.shape[0]))
        yest = yest.reshape(n, n)
        ytrue = ytrue.reshape(n, n)

        # trueplot = ax1.plot_surface(*X_plot, ytrue, linewidth=0, cmap="viridis")
        # ax2.plot_surface(*X_plot, yest, linewidth=0, cmap="viridis")

        trueplot = ax1.pcolormesh(*X_plot, ytrue)
        estplot = ax2.pcolormesh(*X_plot, yest)
        ax1.scatter(*self.data.T, s=1, color='r', alpha=0.2)

        fig.colorbar(trueplot, ax=ax1)
        fig.colorbar(estplot, ax=ax2)

        for ax in (ax1, ax2):
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.grid()

        plt.show()


class SafeBangBang(BangBang):
    def __init__(self, sys, filter):
        super().__init__(sys, filter)

        self.filter = SafetyFilter(self.sys, 55)

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
            print("hej")
            x, _ = self.safe_run(p)
            total_costs.append(sum(abs(normalize_angle(self.XREF - x[0]))) / self.T)
            input_cost = sum((u - ul) ** 2)
            total_costs.append((state_cost + input_cost) / self.T)
