import argparse
import copy

import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
import imageio

from src.systems import Bruce
from src.controller import Stanley, Pid
from src.mpc import VehicleFilter

from utils.args import check_bool_arg

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num-iterations", type=str, nargs="?", default="1")
parser.add_argument("-t", "--time", type=float, nargs="?", default="50")
parser.add_argument("-s", "--simulation", type=str, nargs="?", default="false")
parser.add_argument("-a", "--animate", type=str, nargs="?", default="")
args = parser.parse_args()

check_bool_arg(args.simulation)

s = ca.SX.sym("s")
c = 0.15 * (ca.heaviside(s - 25) - 2 * ca.heaviside(s - 30) + ca.heaviside(s - 35))
# c = 0.05 * (ca.heaviside(s) - 2 * ca.heaviside(s - 40 * np.pi) + ca.heaviside(s - 80 * np.pi))
k = ca.Function("curv", [s], [c])

N = 60
sys = Bruce(k)
filter = VehicleFilter(sys, N)
ctl_steer = Stanley(k_y=1, k_yaw=1)
ctl_throttle = Pid(k=5)

if args.simulation == "true":
    sys.initialize_figure()

i_fig = 0
u_prev = np.array([0, 0])
for _ in range(int(args.num_iterations)):

    while sys.time < args.time and sys.dist_to_end > 0.2:

        vel_x = sys.state[3]
        n, a = sys.err_front_axis
        steer = ctl_steer.compute_control(n, a, vel_x)
        throttle = ctl_throttle.compute_control(0, sys.desired_vel_x - vel_x)
        # ul = [steer, throttle]
        ul = np.array([0, 0])

        x_pred, u_pred = filter.compute_control(sys.state, ul, u_prev)
        u = u_pred[:, 0]
        u_prev = copy.copy(u)

        st = sys.state
        inp = u
        print(f"#### System Info (t={sys.time:.2f}) ####")
        print(f"s={st[0]:.3f}, n={st[1]:.3f}, a={st[2]:.3f}, vel_x={st[3]:.3f}")
        # print(f"vel_x={st[3]:.3f}, vel_y={st[4]:.3f}, yaw_rate={st[5]:.3f}")
        print(f"d_f={inp[0]:.3f}, tau={inp[1]:.3f}\n")

        if args.simulation == "true":
            sys.animate(x_pred, u_pred)

        if args.animate:
            plt.savefig(f"./figures/{i_fig}.png")
            i_fig += 1

        sys.update(u)

breakpoint()

# build gif
if args.animate:
    with imageio.get_writer(f"./gifs/{args.animate}.gif", mode="I") as writer:
        for i in range(i_fig):
            image = imageio.imread(f"./figures/{i}.png")
            writer.append_data(image)
