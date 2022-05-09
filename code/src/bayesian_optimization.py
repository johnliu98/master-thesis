from typing import Tuple, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class BayesianLinearRegression:
    def __init__(
        self, n_features: int, n_outputs: int, alpha: float, beta: float
    ) -> None:
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.alpha = alpha
        self.beta = beta
        self.mean = np.zeros((n_features, n_outputs))
        self.cov_inv = alpha * np.identity(n_features)

    def learn(self, X: np.ndarray, y: Union[float, np.ndarray]) -> None:
        X = np.atleast_2d(X)
        y = np.atleast_2d(y)

        # update inverse covariance
        cov_inv = self.cov_inv + self.beta * X.T @ X

        # update mean
        cov = np.linalg.inv(cov_inv)

        mean = cov @ (self.cov_inv @ self.mean + self.beta * X.T @ y)

        self.cov_inv = cov_inv
        self.mean = mean

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(X)

        # compute predictive mean
        y_pred_mean = X @ self.mean

        # compute predictive standard deviation
        cov = np.linalg.inv(self.cov_inv)
        y_pred_var = 1 / self.beta + (X @ cov * X).sum(axis=1)
        y_pred_std = np.sqrt(y_pred_var)


        y_pred_mean = np.squeeze(y_pred_mean)
        y_pred_std = np.squeeze(y_pred_std)

        return y_pred_mean, y_pred_std

    @property
    def params(self):
        return self.mean

    @property
    def covariance(self):
        return np.linalg.inv(self.cov_inv)


class PendulumDynamicsRegression(BayesianLinearRegression):

    N_FEATURES: int = 4
    N_OUTPUTS: int = 2

    N_PLOT: int = 100

    def __init__(self, *args) -> None:
        BayesianLinearRegression.__init__(self, self.N_FEATURES, self.N_OUTPUTS, *args)

        self.xs = np.empty(shape=(0, 2))
        self.us = np.empty(shape=(0, 1))
        self.fs = np.empty(shape=(0, 2))

    def features(self, x: np.ndarray, u: Union[float, np.ndarray]) -> np.ndarray:
        x = np.atleast_2d(x)
        u = np.atleast_1d(u)

        # model pendulum dynamics as
        # f(x, u) = params^T [x1 sin(x0) u0]
        feat = np.empty((x.shape[0], self.n_features))
        feat[:, 0] = x[:, 0]
        feat[:, 1] = x[:, 1]
        feat[:, 2] = np.sin(x[:, 0])
        feat[:, 3] = u[0]

        return feat

    def learn(
        self,
        x: np.ndarray,
        u: Union[float, np.ndarray],
        f: np.ndarray,
    ) -> None:

        feat = self.features(x, u)
        super().learn(feat, f)

        self.xs = np.vstack([self.xs, x])
        self.us = np.vstack([self.us, u.reshape(-1, 1)])
        self.fs = np.vstack([self.fs, f])

    def predict(self, x: np.ndarray, u: np.ndarray):
        feat = self.features(x, u)
        y_pred_mean, y_pred_std = super().predict(feat)
        return y_pred_mean, y_pred_std

    def initialize_figure(self) -> None:
        self.fig = plt.figure(figsize=(10, 5), num="dynamics")
        self.fig.suptitle("Dynamics")
        plt.ion()

        X = np.meshgrid(
            np.linspace(-4, 4, self.N_PLOT), np.linspace(-10, 10, self.N_PLOT)
        )
        self.ax_heatplot = self.fig.add_subplot(121)
        self.heatplot = self.ax_heatplot.pcolor(
            *X,
            np.zeros(X[0].shape),
            norm=mpl.colors.LogNorm(1e-3, 1e-2),
            cmap="viridis",
            # shading="auto",
        )
        self.ax_heatplot.grid()
        self.dataplot_2d = self.ax_heatplot.scatter(
            [], [], facecolors="None", edgecolor="k", lw=0.25
        )
        self.predplot = self.ax_heatplot.scatter([], [], s=5, facecolors="r")
        self.stateplot = self.ax_heatplot.scatter([], [], s=5, facecolors="white")
        self.fig.colorbar(self.heatplot, ax=self.ax_heatplot)

        self.ax_heatplot.set_xlim(-4, 4)
        self.ax_heatplot.set_ylim(-10, 10)
        self.ax_heatplot.set_box_aspect(1)

        self.ax_surfplot = self.fig.add_subplot(122, projection="3d")
        self.surface = self.ax_surfplot.plot_surface(*X, np.zeros(X[0].shape))
        self.ax_surfplot.set_xlabel("x")
        self.ax_surfplot.set_ylabel("y")
        self.ax_surfplot.set_xlim(-4, 4)
        self.ax_surfplot.set_ylim(-10, 10)

        # X_eval = np.array(X).reshape(2, -1).T
        # U = np.zeros(self.N_PLOT**2)
        # feat = self.features(X_eval, U)
        # par_true = np.array([[1, 0], [0.02, 0.94666667], [0, -0.3924], [0, 0.53333333]])
        # mean = feat @ par_true
        # mean = mean[:, 1].reshape(self.N_PLOT, self.N_PLOT)
        # self.ax_surfplot.plot_surface(*X, mean)

    def animate_error(self) -> None:
        X = np.meshgrid(
            np.linspace(-4, 4, self.N_PLOT), np.linspace(-10, 10, self.N_PLOT)
        )
        X_eval = np.array(X).reshape(2, -1).T
        U = np.zeros(self.N_PLOT**2)
        mean, std = self.predict(X_eval, U)
        mean, std = mean[:, 1].reshape(self.N_PLOT, self.N_PLOT), std.reshape(
            self.N_PLOT, self.N_PLOT
        )

        self.heatplot.set_array(std.ravel())
        self.dataplot_2d.set_offsets(self.xs)

        self.surface.remove()
        self.surface = self.ax_surfplot.plot_surface(*X, mean, color="#1f77b4")

        plt.draw()

        plt.show()
        plt.pause(0.001)

    def animate_pred(self, x_pred) -> None:
        self.predplot.set_offsets(x_pred.T)
        self.stateplot.set_offsets(x_pred.T[0])

        plt.show()
        plt.pause(0.001)
