import argparse

import pickle
import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
import imageio

from src.systems import Kinematic, Dynamic
from src.controller import Stanley, Pid
from src.mpc import VehicleFilter
from src.gaussian_process_regression import StanleyPolicy

from utils.args import check_bool_arg

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num-iterations", type=str, nargs="?", default="1")
parser.add_argument("-t", "--time", type=float, nargs="?", default="30")
parser.add_argument("-s", "--simulation", type=str, nargs="?", default="false")
parser.add_argument("-f", "--filter", type=str, nargs="?", default="false")
parser.add_argument("-a", "--animate", type=str, nargs="?", default="")
parser.add_argument("--save", type=str, nargs="?", default="false")
parser.add_argument("--load", type=str, nargs="?", default="false")
args = parser.parse_args()

check_bool_arg(args.simulation)
check_bool_arg(args.filter)
check_bool_arg(args.save)
check_bool_arg(args.load)

s = ca.SX.sym("s")
delta = 5.5
trans_1 = 10
trans_2 = 39
c = 0.1 * (
    ca.heaviside(s - trans_1)
    - 2 * ca.heaviside(s - (trans_1 + delta))
    + ca.heaviside(s - (trans_1 + 2 * delta))
    -ca.heaviside(s - trans_2)
    + 2 * ca.heaviside(s - (trans_2 + delta))
    - ca.heaviside(s - (trans_2 + 2 * delta))
)
curv = ca.Function("curv", [s], [c])

LOG_NAME = "temp"

N = 80
sys = Dynamic(curv)
sim = Kinematic(curv)
filter = VehicleFilter(sim, N)
agent = StanleyPolicy()

if args.load == "true":
    with open(f"./logs/{LOG_NAME}.pkl", "rb") as file:
        runs = pickle.load(file)

    k_yaws = np.array([r["k"][0] for r in runs])
    k_ys = np.array([r["k"][1] for r in runs])
    costs = np.array([r["cost"] for r in runs])
    k_yaw, k_y = agent.learn(np.vstack([k_yaws, k_ys]).T, -costs)
else:
    k_yaw = 1
    k_y = 1
    runs = []
# k_yaw = 0.516993332
# k_y = 0.192385877

if args.simulation == "true":
    sys.initialize_figure()

i_fig = 0
for i in range(int(args.num_iterations)):

    ctl_steer = Stanley(k_yaw=k_yaw, k_y=k_y)
    ctl_throttle = Pid(k=5)

    ul_prev = sys._u_prev
    cost = 0
    sys.reset()
    data = {
        "k": None,
        "cost": None,
        "states": [sys.state],
        "inputs": [],
        "learning-inputs": [],
    }
    while sys.time < args.time and sys.state[0] < sys.MAX_S:

        vel_x = sys.state[3]
        n, a = sys.err_front_axis
        steer = ctl_steer.compute_control(n, a, vel_x)
        steer = np.clip(steer, -sys.MAX_STEER, sys.MAX_STEER)

        throttle = ctl_throttle.compute_control(0, sys.desired_vel_x - vel_x)
        throttle = np.clip(throttle, sys.MIN_THROTTLE, sys.MAX_THROTTLE)
        ul = np.array([steer, throttle])
        ul = sys.check_input_constraints(ul, ul_prev)
        ul_prev = ul

        if args.filter == "true":
            x_pred, u_pred = filter.compute_control(sys.kinematic_state, ul, sys.u_prev)
            u = u_pred[:, 0]
        else:
            u = ul

        # st = sys.state
        # inp = u
        # print(f"#### System Info (i={i+1:.0f}, t={sys.time:.3f}) ####")
        # print(f"s={st[0]:.3f}, n={st[1]:.3f}, a={st[2]:.3f}, vel_x={st[3]:.3f}")
        # print(f"vel_x={st[3]:.3f}, vel_y={st[4]:.3f}, yaw_rate={st[5]:.3f}")
        # print(f"d_f={inp[0]:.3f}, tau={inp[1]:.3f}\n")

        sys.update(u)

        # n, a = sys.err_front_axis
        n, _ = sys.err
        err_cost = n**2
        cost += err_cost + (u[0] - ul[0]) ** 2 + (u[1] - ul[1]) ** 2

        if args.simulation == "true":
            if args.filter == "true":
                sys.animate(ul, x_pred, u_pred)
            else:
                sys.animate(ul)

        if args.animate:
            plt.savefig(f"./gifs/tmp/{i_fig}.png")
            i_fig += 1

        data["states"].append(sys.state)
        data["inputs"].append(u)
        data["learning-inputs"].append(ul)

    cost /= sys.time

    data["k"] = [k_yaw, k_y]
    data["cost"] = cost
    runs.append(data)

    print(f"#### System Info (i={i+1:.0f}) ####")
    print(f"k_yaw={k_yaw:.3f}, k_y={k_y:.3f}, cost={cost:.3f}\n")

    k_yaw, k_y = agent.learn([k_yaw, k_y], -cost)


k_yaw, k_y, rew = agent.predict()
print("#### Optimal Control ####")
print(f"k_yaw={k_yaw:.3f}, k_y={k_y:.3f}, cost={-rew:.3f}")

if args.save == "true":
    with open(f"./logs/{LOG_NAME}.pkl", "wb") as file:
        pickle.dump(runs, file)

# build gif
if args.animate:
    with imageio.get_writer(f"./gifs/{args.animate}.gif", mode="I") as writer:
        for i in range(i_fig):
            image = imageio.imread(f"./gifs/tmp/{i}.png")
            writer.append_data(image)
