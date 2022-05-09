import argparse

import numpy as np
import matplotlib.pyplot as plt
import imageio

from src.systems import Pendulum
from src.envs import PendulumBangEnv
from src.mpc import PendulumFilter
from src.gaussian_process_regression import BangBangPolicy

from utils.args import check_bool_arg

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num-iterations", type=str, nargs="?", default="1")
parser.add_argument("-lr", "--learning-rate", type=str, nargs="?", default="1e-1")
parser.add_argument("-s", "--simulation", type=str, nargs="?", default="false")
parser.add_argument("-d", "--plot-dynamics", type=str, nargs="?", default="false")
parser.add_argument("-a", "--animate", type=str, nargs="?", default="")
args = parser.parse_args()

check_bool_arg(args.simulation)
check_bool_arg(args.plot_dynamics)

sys = Pendulum()
filter = PendulumFilter(sys, 50, 1, 1e6)
env = PendulumBangEnv(sys, filter)
agent = BangBangPolicy()

if args.simulation == "true":
    sys.initialize_figure()

if args.plot_dynamics == "true":
    filter.initialize_figure()

# pre-train dynamics regressor
np.random.seed(2048)
N_rand = 10
lim = np.array([np.pi / 6, np.pi / 6, 0.7])
pre_data = np.random.uniform(low=-lim, high=lim, size=(N_rand, 3))
pre_x, pre_u = pre_data[:, :-1], pre_data[:, -1]
for x, u in zip(pre_x, pre_u):
    filter.learn(x, u, sys.update(x, u))

if args.plot_dynamics == "true":
    filter.animate_error()

i_fig = 0
for _ in range(int(args.num_iterations)):
    x = env.reset()
    k1, k2 = 30, 120
    print(f"Switch time: {k1}, {k2}")

    # run simulation
    cost = 0
    xs, us, fs = [], [], []
    for i in range(env.EPISODE_LENGTH):
        if i <= k1:
            ul = -env.INPUT_LIM
        elif i <= k2:
            ul = env.INPUT_LIM
        else:
            ul = 0

        x_pred, u_pred = filter.optimize(x, ul)

        if args.simulation == "true":
            sys.animate(x_pred, u_pred, ul)

        if args.plot_dynamics == "true":
            filter.animate_pred(x_pred)

        u = u_pred[0]
        x_ = sys.update(x, u)
        xs.append(x), us.append(u), fs.append(x_)
        x = x_

        cost += (env.REF - x[0]) ** 2 + (ul - u) ** 2

        if args.animate:
            plt.savefig(f"./figures/{i_fig}.png")
            i_fig += 1
    cost /= env.EPISODE_LENGTH

    # train agent
    k1, k2 = agent.learn([k1, k2], -cost)
    print(f"Cost: {cost:.3f}\n")

    # learn dynamics in filter
    xs = np.array(xs).reshape(-1, 2)
    us = np.array(us).reshape(-1, 1)
    fs = np.array(fs)
    for (x, u, f) in zip(xs, us, fs):
        filter.learn(x, u, f)

    if args.plot_dynamics == "true":
        filter.animate_error()

print(filter.params)
print(sys.PARAMS)
print(f"Parameter error norm: {np.linalg.norm(filter.params - sys.PARAMS)}")

# build gif
if args.animate:
    with imageio.get_writer(f"./gifs/{args.animate}.gif", mode="I") as writer:
        for i in range(i_fig):
            image = imageio.imread(f"./figures/{i}.png")
            writer.append_data(image)
