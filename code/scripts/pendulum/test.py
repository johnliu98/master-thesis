import numpy as np
import matplotlib.pyplot as plt

from src.systems import Pendulum

sys = Pendulum()

N = 100
X = np.meshgrid(np.linspace(-4, 4, N), np.linspace(-10, 10, N))
X_eval = np.array(X).reshape(2, -1).T
U = np.zeros(N**2)

X_next = np.empty(X_eval.shape)
for i, (x, u) in enumerate(zip(X_eval, U)):
    X_next[i] = sys.update(x, u)
F = (X_next - X_eval) / sys.DT
theta = F[:, 1].reshape(N, N)

fig = plt.figure()

ax = fig.add_subplot(111, projection="3d")
surface = ax.plot_surface(*X, theta)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(-4, 4)
ax.set_ylim(-10, 10)
plt.show()

