import argparse

import numpy as np
import matplotlib.pyplot as plt

from src.systems import Pendulum
from src.envs import PendulumBangEnv
from src.mpc import SafetyFilter
from src.gaussian_process_regression import BangBangPolicy

from utils.args import check_bool_arg

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num-iterations", type=str, nargs="?", default="1")
# parser.add_argument("-lr", "--learning-rate", type=str, nargs="?", default="1e-1")
# parser.add_argument("-s", "--simulation", type=str, nargs="?", default="false")
# parser.add_argument("-d", "--plot-dynamics", type=str, nargs="?", default="false")
# parser.add_argument("-a", "--animate", type=str, nargs="?", default="false")
args = parser.parse_args()

# check_bool_arg(args.simulation)
# check_bool_arg(args.plot_dynamics)
# check_bool_arg(args.animate)

sys = Pendulum()
filter = SafetyFilter(sys, 50)
env = PendulumBangEnv(sys, filter)
agent = BangBangPolicy()

runs = []
k1, k2 = 4, 150
for _ in range(int(args.num_iterations)):
    x = env.reset()
    print(f"Switch time: {k1}, {k2}")

    # run simulation
    state_cost, input_cost = 0, 0
    xs = [x]
    for i in range(env.EPISODE_LENGTH):

        if i <= k1:
            ul = -env.INPUT_LIM
        elif i <= k2:
            ul = env.INPUT_LIM
        else:
            ul = 0

        _, u_pred = filter.optimize(x, ul)

        u = u_pred[0]
        x = sys.update(x, u)
        xs.append(x)

        state_cost += (x[0] - env.REF) ** 2
        input_cost += (u - ul) ** 2
    state_cost /= env.EPISODE_LENGTH
    input_cost /= env.EPISODE_LENGTH
    total_cost = state_cost + input_cost

    # train agent
    k1, k2 = agent.learn([k1, k2], -total_cost)
    print(f"State cost: {state_cost:.3f}")
    print(f"Input cost: {input_cost:.3f}")
    print(f"Total cost: {total_cost:.3f}\n")

    runs.append(xs)

agent.plot()

k1, k2 = agent.predict()
print(f"Best results: k1={k1:.0f}, k2={k2:.0f}")

# plot results
plt.figure(figsize=(10, 3))
t = sys.DT * np.arange(0, env.EPISODE_LENGTH + 1)
for r in runs:
    theta = (180 / np.pi) * np.array(r)[:, 0]
    plt.plot(t, theta, "b", linewidth=0.1)
plt.axhline(-90, color="k", linestyle="--", linewidth=1)
plt.axhline(190, color="k", linestyle="--", linewidth=1)
plt.xlabel("k1")
plt.ylabel("k2")
plt.xlim(0, 3)
plt.ylim(-100, 200)
plt.grid()
plt.show()
