import time
import numpy as np
import casadi as ca

from src.systems import Pendulum
from src.mpc import MPC

TIME_UNITS = ["s", "ms", "µs", "ns"]
time_iter = iter(range(len(TIME_UNITS)))

# define parameters
T = int(1.5e2)
N = 25

# initialize dynamic system
sys = Pendulum()
controller = MPC(sys, N)

# set initia values
x = ca.DM([140 / 180 * np.pi, 0])
xref = ca.DM([np.pi, 0])
u = ca.DM([0.6])

# bench mark computation time
start = time.time()
for _ in range(T):
    controller.optimize(x, xref)
end = time.time()
total_time = end - start
time_per_run = total_time / T

# print results
print(f"Total time: {total_time:.3f} s")

time_index = next(time_iter)
while time_per_run < 1:
    time_per_run *= 1000
    time_index = next(time_iter)
print(f"{time_per_run:.3f} {TIME_UNITS[time_index]} per run over {T:.0f} total runs")
