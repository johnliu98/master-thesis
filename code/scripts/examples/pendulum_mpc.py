"""
    Dynamic Pendulum System
    Author: John Liu
"""

import argparse
import numpy as np
import casadi as ca

from src.systems import Pendulum
from src.mpc import MPC

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--time", dest="time", type=float, nargs="?", default=3)
args = parser.parse_args()

# Simulation parameters
tf = args.time
x0 = np.array([-30 / 180 * np.pi, 0])
xref = np.array([np.pi, 0])

# Control parameters
N = 10

# Instantiate model and controller
sys = Pendulum()
T = int(tf / sys.DT)
controller = MPC(sys, N)

sys.initialize_figure()

# Run simulation with MPC control
x = x0
th_prev = x[0]
for _ in range(T):
    xs, us = controller.optimize(x, xref, verbose=0)
    xs, us = xs.full(), us.full()

    sys.animate(xs, us)
    x = np.array(sys.update(xs[:, 0], us[:, 0]))
