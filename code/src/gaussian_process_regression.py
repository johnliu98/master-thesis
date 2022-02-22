from typing import Tuple
import numpy as np
import sklearn.metrics
from sklearn.gaussian_process.kernels import RBF


class GPR:
    def __init__(self) -> None:
        self.kernel = RBF(1)

    def learn(
        self, Xtrain: np.ndarray, ytrain: np.ndarray, Xtest: np.ndarray, noise: float
    ) -> Tuple[np.ndarray, np.ndarray]:

        K = self.kernel(Xtrain, Xtrain)
        K_s = self.kernel(Xtrain, Xtest)
        K_ss = self.kernel(Xtest, Xtest)

        L = np.linalg.cholesky(K + noise * np.eye(Xtrain.shape[0]))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, ytrain))

        mu = (K_s.T @ alpha).squeeze()

        v = np.linalg.solve(L, K_s)
        cov = K_ss - v.T @ v
        stdv = np.sqrt(np.diag(cov)).squeeze()

        return mu, stdv

    def marginalize(self) -> None:
        pass

    def sparse(self) -> None:
        pass

    def predict(self) -> None:
        pass
