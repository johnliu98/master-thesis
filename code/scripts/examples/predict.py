import argparse
import stable_baselines3 as sb
import numpy as np

from src.systems import Pendulum
from src.envs import PendulumEnv
from src.mpc import SafetyFilter
from utils.args import check_bool_arg

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, nargs="?", default="safe_rl")
parser.add_argument("--swing-up-pend", type=str, nargs="?", default="true")
parser.add_argument("--fixed-init", type=str, nargs="?", default="true")
parser.add_argument("--safety-filter", type=str, nargs="?", default="true")
args = parser.parse_args()

check_bool_arg(args.swing_up_pend)
check_bool_arg(args.fixed_init)
check_bool_arg(args.safety_filter)

sys = Pendulum()
env = PendulumEnv(
    sys,
    swing_up_pend=args.swing_up_pend=="true",
    fixed_init=args.fixed_init=="true",
    safety_filter=args.safety_filter=="true",
)
filter = SafetyFilter(sys, 10)
model = sb.DQN.load(args.model, env=env)

sys.initialize_figure()
x = env.reset()
th_prev = x[0]

total_cost = 0
while True:
    action, _ = model.predict(x, deterministic=True)

    ul = env.ACTIONS[action]
    xs, us = filter.optimize(x, ul)

    if th_prev - x[0] > 1:
        x[0] += 2 * np.pi
    th_prev = x[0]

    sys.animate(xs.full(), us.full(), ul)
    x, r, d, _ = env.step(action)
    total_cost += r

    if d:
        print(f"Total cost: {-total_cost:.3f}")
        sys.initialize_figure()
        x = env.reset()
