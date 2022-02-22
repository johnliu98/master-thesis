import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def main():
    # Create training data
    nt = 50
    f = lambda x: x * np.sin(x)
    noise = 1.0

    Xtrain = np.random.uniform(-3, 13, size=(nt, 1))
    ytrain = f(Xtrain) + np.random.normal(0, noise, size=(nt, 1))

    kernel = 1.0 * RBF(1.0) + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr.fit(Xtrain, ytrain)

    # Generate test data
    ns = 1000
    Xtest = np.linspace(-7.5, 17.5, ns).reshape(-1, 1)
    ytest, std = gpr.predict(Xtest, return_std=True)
    ytest = ytest.ravel()

    RMSE = np.sqrt(np.mean((ytest - f(Xtest).ravel()) ** 2))
    print(f"Root Mean squared error: {RMSE:.3f}")

    # Plot regression
    plt.figure()
    plt.fill_between(
        Xtest.ravel(),
        ytest - 1.96 * std,
        ytest + 1.96 * std,
        alpha=0.5,
        color="#dddddd",
        label="confidence interval",
    )
    plt.scatter(Xtrain, ytrain, 20, "k", alpha=0.5, label="training")
    plt.plot(Xtest, f(Xtest), label="true")
    plt.plot(Xtest, ytest, "r--", label="test")
    plt.grid()
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
