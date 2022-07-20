import argparse

import pickle
import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

from src.trajectory import create_trajectory

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log", type=str, nargs="?", default="")
parser.add_argument("-n", "--figname", type=str, nargs="?", default="")
parser.add_argument("-f", "--fileformat", type=str, nargs="?", default="png")
args = parser.parse_args()

if args.log == "":
    raise SystemExit("Please enter log file.\nExample: -l[--log] <file>")

LOG_FILE = args.log
FIGURE_FORMAT = args.fileformat

with open(f"./{LOG_FILE}", "rb") as file:
    runs = pickle.load(file)

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
for run in runs:
    run["poses"] = []
    states = run["states"]
    for state in states:
        s, n, a, _, _, _ = state
        f = traj.pose(s)
        x = f.x - n * np.sin(f.yaw)
        y = f.y + n * np.cos(f.yaw)
        yaw = f.yaw + a
        run["poses"].append(np.array([x, y, yaw]))

# Find optimal trajectory
cost = 80
i_opt = -1
for i, run in enumerate(runs):
    if run["cost"] < cost:
        cost = run["cost"]
        i_opt = i
run_opt = runs[i_opt]

print(f"#### Optimal Control ({len(runs)} samples) ####")
k_opt = run_opt["k"]
cost_opt = run_opt["cost"]
print(f"k_yaws={k_opt[0]:.9f}, k_y={k_opt[1]:.9f}, cost={cost_opt:.3f}")

# plt.figure()
# plt.plot([r["k"][0] for r in runs])
# plt.plot([r["k"][1] for r in runs])
# plt.grid()
# plt.show()

fig = plt.figure(figsize=(10, 2))
ax = plt.gca()
ax.set_aspect("equal")
traj.plot(ax)

for run in runs:
    xs = [p[0] for p in run["poses"]]
    ys = [p[1] for p in run["poses"]]
    ax.plot(xs, ys, linewidth=1, color="r", alpha=0.3)

# xs_opt = [p[0] for p in run_opt["poses"]]
# ys_opt = [p[1] for p in run_opt["poses"]]
# ax.plot(xs_opt, ys_opt, linewidth=1.2, linestyle="dashed", color="b")

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
