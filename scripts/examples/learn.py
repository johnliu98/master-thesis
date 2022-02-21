import argparse
import stable_baselines3 as sb

from src.systems import Pendulum
from src.envs import PendulumEnv, SafePendulumEnv

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load", dest="model", type=str, nargs="?", default="")
parser.add_argument("-lr", "--learning-rate", dest="learning_rate", type=str, nargs="?", default="1e-3")
parser.add_argument("-ts", "--timesteps", dest="timesteps", type=str, nargs="?", default="2e5")
args = parser.parse_args()

sys = Pendulum()
env = SafePendulumEnv(sys, sys_type="discrete", horizon=25)

if args.model:
    model = sb.DQN.load(args.model)
    model.set_env(env)
else:
    model = sb.DQN(
        "MlpPolicy",
        env,
        learning_rate=float(args.learning_rate),
        verbose=1,
        exploration_fraction=0.9,
        tensorboard_log="./pendulum_log/",
    )

model.learn(total_timesteps=int(float(args.timesteps)))
model.save("safe_rl")
