import argparse
import stable_baselines3 as sb
from gym import spaces

from src.systems import Pendulum
from src.envs import PendulumEnv, SafePendulumEnv
from src.mpc import SafetyFilter

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load", dest="model", type=str, nargs="?", default="safe_rl")
args = parser.parse_args()

sys = Pendulum()
env = SafePendulumEnv(sys, sys_type="discrete", horizon=25)
filter = SafetyFilter(sys, 25)
model = sb.DQN.load(args.model, env=env)

sys.initialize_figure()
x = env.reset(deterministic=False)

total_cost = 0
while True:
    action, _ = model.predict(x, deterministic=True)

    if isinstance(env.action_space, spaces.Discrete):
        ul = env.ACTIONS[action]
    else:
        ul = action

    x, u = filter.optimize(x, ul)
    x, u = x.full(), u.full()

    sys.animate(x, u, ul)
    x, r, d, _ = env.step(action)
    total_cost += r

    if d:
        print(f"Total cost: {-total_cost:.3f}")
        sys.initialize_figure()
        x = env.reset()
