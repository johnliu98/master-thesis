import argparse
import stable_baselines3 as sb
import numpy as np

from src.systems import Pendulum
from src.envs import PendulumEnv
from utils.args import check_bool_arg

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, nargs="?", default="safe_rl")
parser.add_argument("--swing-up-pend", type=str, nargs="?", default="true")
parser.add_argument("--random-init", type=str, nargs="?", default="false")
args = parser.parse_args()

check_bool_arg(args.swing_up_pend)
check_bool_arg(args.random_init)

sys = Pendulum()
env = PendulumEnv(
    sys,
    swing_up_pend=args.swing_up_pend == "true",
    random_init=args.random_init == "true",
)
model = sb.DQN.load(args.model, env=env)

sys.initialize_figure()
x = env.reset()
th_prev = x[0]

total_cost = 0
while True:
    action, _ = model.predict(x, deterministic=True)

    ul = env.ACTIONS[action]

    # x, u = filter.optimize(x, ul)
    # x, u = x.full(), u.full()

    if th_prev - x[0] > 1:
        x[0] += 2 * np.pi
    th_prev = x[0]

    sys.animate(x, ul)
    x, r, d, _ = env.step(action)
    total_cost += r

    if d:
        print(f"Total cost: {-total_cost:.3f}")
        sys.initialize_figure()
        x = env.reset()
