import argparse
import stable_baselines3 as sb

from src.systems import Pendulum
from src.envs import PendulumEnv
from utils.args import check_bool_arg

parser = argparse.ArgumentParser()
parser.add_argument("--timesteps", type=str, nargs="?", default="2e5")
parser.add_argument("--learning-rate", type=str, nargs="?", default="1e-3")
parser.add_argument("--swing-up-pend", type=str, nargs="?", default="true")
parser.add_argument("--fixed-init", type=str, nargs="?", default="true")
parser.add_argument("--safety-filter", type=str, nargs="?", default="true")
parser.add_argument("--model-name", type=str, nargs="?", default="safe_rl")
parser.add_argument("--retrain", dest="model", type=str, nargs="?", default="")
args = parser.parse_args()

check_bool_arg(args.swing_up_pend)
check_bool_arg(args.fixed_init)
check_bool_arg(args.safety_filter)

SAVE_DIR = "./models/safe/"

sys = Pendulum()
env = PendulumEnv(
    sys,
    swing_up_pend=args.swing_up_pend=="true",
    fixed_init=args.fixed_init=="true",
    safety_filter=args.safety_filter=="true",
)

if args.model:
    model = sb.DQN.load(args.model)
    model.set_env(env)
    model.learning_rate = float(args.learning_rate)
else:
    model = sb.PPO(
        "MlpPolicy",
        env,
        learning_rate=float(args.learning_rate),
        verbose=1,
        tensorboard_log="./pendulum_log/",
    )

model.learn(total_timesteps=int(float(args.timesteps)))
model.save(SAVE_DIR + args.model_name + f"_{args.timesteps}" + f"_{args.learning_rate}" + f"_{env.TIME_LIMIT}")
