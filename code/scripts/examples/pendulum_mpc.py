"""
    Dynamic Pendulum System
    Author: John Liu
"""

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

from src.systems import Pendulum
from src.mpc import MPC

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--time", dest="time", type=float, nargs="?", default=3)
args = parser.parse_args()

# Simulation parameters
tf = args.time
x0 = np.array([-np.pi / 3, 0])
xref = np.array([np.pi, 0])

# Control parameters
N = 25

# Instantiate model and controller
sys = Pendulum()
T = int(tf / sys.DT)
controller = MPC(sys, N)

sys.initialize_figure()
comp_times = []

# Run simulation with MPC control
x = ca.DM(x0)
for _ in range(T):
    tic = time.time()
    x, u = controller.optimize(x, xref)
    x, u = x.full(), u.full()
    toc = time.time()

    sys.animate(x, u)
    x = sys.update(x[:, 0], u[:, 0])

    comp_times.append(toc - tic)

# Plot computation time
plt.ioff()
fig = plt.figure(figsize=(9, 4))
ax1 = plt.subplot2grid((1, 2), (0, 0))
ax1.plot(np.linspace(0, tf, T), comp_times)
ax1.set_xlabel("time [s]")
ax1.set_ylabel("computation time [s]")
ax1.grid()

ax2 = plt.subplot2grid((1, 2), (0, 1))
ax2.hist(comp_times, bins=25, density=True)
ax2.set_xlabel("computation time [s]")
ax2.set_ylabel("probability [%]")

plt.tight_layout()
plt.show()

# print(f"Computation time: {(toc-tic):.2f} s")
