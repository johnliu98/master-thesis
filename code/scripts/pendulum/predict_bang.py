import argparse
import stable_baselines3 as sb
import numpy as np

from src.systems import Pendulum
from src.envs import PendulumBangEnv

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, nargs="?", default="safe_rl")
args = parser.parse_args()

sys = Pendulum()
env = PendulumBangEnv(sys)
model = sb.DQN.load(args.model, env=env)

sys.initialize_figure()
x = env.reset()
switch_time, _ = model.predict(x, deterministic=True)
print(switch_time)
switch_time=5

for i in range(env.STEP_LENGTH):
    if i <= switch_time:
        u = -np.asarray(env.INPUT_LIM)
    else:
        u = np.asarray(env.INPUT_LIM)

    sys.animate(x, u)
    x = np.array(sys.update(x, u))
