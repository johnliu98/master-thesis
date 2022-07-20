import argparse

import pickle
import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

from src.trajectory import create_trajectory

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--logs", type=str, nargs="+")
parser.add_argument("-n", "--figname", type=str, nargs="?", default="")
parser.add_argument("-f", "--fileformat", type=str, nargs="?", default="png")
args = parser.parse_args()

FIGURE_FORMAT = args.fileformat

if len(args.logs) != 2:
    raise SystemExit("Please enter two log file.\nExample: -l[--log] <file> <file>")

logs = []
with open(f"./{args.logs[0]}", "rb") as file:
    logs.append(pickle.load(file))

with open(f"./{args.logs[1]}", "rb") as file:
    logs.append(pickle.load(file))

# Define curvature function
s = ca.SX.sym("s")
delta = 5.5
trans_1 = 10
trans_2 = 39
c = 0.1 * (
    ca.heaviside(s - trans_1)
    - 2 * ca.heaviside(s - (trans_1 + delta))
    + ca.heaviside(s - (trans_1 + 2 * delta))
    - ca.heaviside(s - trans_2)
    + 2 * ca.heaviside(s - (trans_2 + delta))
    - ca.heaviside(s - (trans_2 + 2 * delta))
)
curv = ca.Function("curv", [s], [c])

# Create trajectory
MAX_S: float = 60
DS: float = 0.005
s_d = np.linspace(0, MAX_S, int(MAX_S / DS))
curvatures = curv(s_d).full().squeeze()
traj = create_trajectory(curvatures, DS)

# Compute vehicle pose in cartesian coordinates
opt_runs = []
for runs in logs:
    costs = np.array([r["cost"] for r in runs])
    opt_i = np.argmin(costs)
    opt_runs.append(runs[opt_i])

for run in opt_runs:
    run["poses"] = []
    states = run["states"]
    for state in states:
        s, n, a, _, _, _ = state
        f = traj.pose(s)
        x = f.x - n * np.sin(f.yaw)
        y = f.y + n * np.cos(f.yaw)
        yaw = f.yaw + a
        run["poses"].append(np.array([x, y, yaw]))

fig = plt.figure(figsize=(10, 2))
ax = plt.gca()
ax.set_aspect("equal")
traj.plot(ax)

unsafe_run = opt_runs[0]
xs = [p[0] for p in unsafe_run["poses"]]
ys = [p[1] for p in unsafe_run["poses"]]
ax.plot(xs, ys, linewidth=1, color="b", linestyle="--")

safe_run = opt_runs[1]
xs = np.array([p[0] for p in safe_run["poses"]])
ys = np.array([p[1] for p in safe_run["poses"]])
us = np.array(safe_run["inputs"])
uls = np.array(safe_run["learning-inputs"])
ul_mags = np.linalg.norm(us - uls, axis=1)
ax.plot(xs, ys, linewidth=1, color="g", linestyle="dashdot")
ax.scatter(xs[:-1], ys[:-1], s=5e5 * ul_mags, color="g", linewidth=0.2, facecolors="none")


ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_ylim(-3, 7)

plt.legend(loc="upper left")
plt.grid()
plt.tight_layout()

if args.figname:
    fig.savefig(
        f"./figures/{args.figname}.{FIGURE_FORMAT}", format=FIGURE_FORMAT, dpi=100
    )
else:
    plt.show()
