# example of bayesian optimization for a 1d function from scratch
import numpy as np
import scipy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF
from warnings import catch_warnings
from warnings import simplefilter
import matplotlib.pyplot as plt

from pprint import pprint

# objective function
def objective(x, noise=0.1):
    noise = np.random.normal(loc=0, scale=noise, size=(x.shape[0],))
    J = (
        x[:, 0]
        * x[:, 1]
        * np.sin(5 * np.pi * x[:, 0]) ** 6.0
        * np.cos(3 * np.pi * x[:, 1]) ** 3
    )
    # J = x[:, 0] ** 2 + x[:, 1] ** 2
    return J + noise


# surrogate or approximation for the objective function
def surrogate(model, X):
    # catch any warning generated when making a prediction
    with catch_warnings():
        # ignore generated warnings
        simplefilter("ignore")
        return model.predict(X, return_std=True)


# probability of improvement acquisition function
def acquisition(X, Xsamples, model):
    # calculate the best surrogate score found so far
    yhat, _ = surrogate(model, X)
    best = max(yhat)
    # calculate mean and stdev via surrogate function
    mu, std = surrogate(model, Xsamples)
    z = (mu - best) / (std + 1e-9)
    # calculate the probability of improvement
    cdf = scipy.stats.norm.cdf(z)
    pdf = scipy.stats.norm.pdf(z)
    score = (mu - best) * cdf + std * pdf
    return score


# optimize the acquisition function
def opt_acquisition(X, y, model):
    # np.random.random search, generate np.random.random samples
    Xsamples = np.random.uniform(low=0, high=1, size=(1000, 2))
    # calculate the acquisition function for each sample
    scores = acquisition(X, Xsamples, model)
    # locate the index of the largest scores
    ix = np.argmax(scores)
    return Xsamples[ix]


# plot real observations vs surrogate function
def plot(X, y, model):
    Nx, Ny = 100, 100
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), subplot_kw={"projection": "3d"})
    # scatter plot of inputs and real objective function
    # line plot of surrogate function across domain
    Xplot = np.meshgrid(np.linspace(0, 1, Nx), np.linspace(0, 1, Ny))
    Xsamples = np.array(Xplot).reshape(2, -1).T
    yest, _ = surrogate(model, Xsamples)
    ytrue = objective(Xsamples, noise=0)

    yest = yest.reshape(Nx, Ny)
    ytrue = ytrue.reshape(Nx, Ny)

    trueplot = ax1.plot_surface(*Xplot, ytrue, linewidth=0, cmap='viridis')
    ax2.plot_surface(*Xplot, yest, linewidth=0, cmap='viridis')

    fig.colorbar(trueplot, ax=[ax1, ax2])

    for ax in (ax1, ax2):
        ax.scatter(*X.T, y, s=0.5, alpha=0.5)
        ax.set_zlim(-0.5, 0.5)
        ax.grid()

    plt.show()


# sample the domain sparsely with noise
X = np.random.uniform(low=0, high=1, size=(100, 2))
y = objective(X)
# define the model
kernel = 1.0 * RBF(1.0) + ConstantKernel(1.0) + WhiteKernel(1.0)
model = GaussianProcessRegressor()
# fit the model
model.fit(X, y)
# plot before hand
plot(X, y, model)

# perform the optimization process
for i in range(200):
    # select the next point to sample
    x = opt_acquisition(X, y, model).reshape(1, -1)
    # sample the point
    actual = objective(x)
    # summarize the finding
    est, _ = surrogate(model, x)
    print(">n=%.0f, x=[%.3f, %.3f], f()=%3f, actual=%.3f" % (i, *x.squeeze(), est, actual))
    # add the data to the dataset
    X = np.vstack((X, x))
    y = np.hstack((y, actual))
    # update the model
    model.fit(X, y)

# best result
ix = np.argmax(y)
print("Best Result: x=[%.3f, %.3f], y=%.3f" % (*X[ix], y[ix]))

X_true = np.meshgrid(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000))
X_true = np.array(X_true).reshape(2, -1).T
y_true = objective(X_true, noise=0)
ix = np.argmax(y_true)
print("True Result: x=[%.3f, %.3f], y=%.3f" % (*X_true[ix], y_true[ix]))

# plot all samples and the final surrogate function
plot(X, y, model)
