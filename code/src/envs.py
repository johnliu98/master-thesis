from locale import normalize
import numpy as np

import gym
from gym import spaces

from src.mpc import SafetyFilter
from utils.angle import normalize_angle


class PendulumEnv(gym.Env):
    """Pendulum Environment that follows gym interface"""

    INPUT_LIM = 0.6
    N_DISCRETE = 3

    TIME_LIMIT = 5
    STATE_REF = np.array([[np.pi], [0]])
    ACTIONS = np.linspace(-INPUT_LIM, INPUT_LIM, N_DISCRETE)

    BOUNDS = np.array([2 * np.pi, 6])
    RESET_BOUNDS = np.array([np.pi / 3, 6])

    metadata = {"render.modes": ["human"]}

    def __init__(self, sys, sys_type="discrete"):
        self.sys = sys

        if sys_type == "discrete":
            self.action_space = spaces.Discrete(self.N_DISCRETE)
            self._is_discrete = True
        elif sys_type == "continous":
            self.action_space = spaces.Box(
                low=-self.INPUT_LIM,
                high=self.INPUT_LIM,
                shape=(self.sys.NU, 1),
                dtype=np.float32,
            )
            self._is_discrete = False
        else:
            raise ValueError(f"Invalid system type: {sys_type}")

        self.observation_space = spaces.Box(
            low=-self.BOUNDS,
            high=self.BOUNDS,
            shape=(self.sys.NX,),
            dtype=np.float32,
        )

    def reset(self, deterministic=False):
        self.sys.t = 0

        if deterministic:
            self.x = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            self.x = np.random.uniform(
                low=-self.RESET_BOUNDS,
                high=self.RESET_BOUNDS,
            ).reshape(self.observation_space.shape)

        self.x[0] = normalize_angle(self.x[0])

        return self.x

    def step(self, action):

        # convert action to input
        if isinstance(self.action_space, spaces.Discrete):
            u = self.ACTIONS[action]
        else:
            u = action

        # update system
        self.x = np.array(self.sys.update(self.x, u), dtype=np.float32)
        self.x = self.x.reshape(self.observation_space.shape)

        # compute cost
        theta, _ = self.x
        cost = normalize_angle(np.pi - theta) ** 2 * self.sys.DT / self.TIME_LIMIT

        done = self.sys.t >= self.TIME_LIMIT

        return self.x, -cost, done, {}


class SafePendulumEnv(PendulumEnv):
    """Pendulum Environment with safety guarantees that follows gym interface"""

    def __init__(self, sys, sys_type="discrete", horizon=50):
        PendulumEnv.__init__(self, sys, sys_type=sys_type)

        self.filter = SafetyFilter(sys, horizon)

    def step(self, action):

        # convert action to input
        if self._is_discrete:
            ul = self.ACTIONS[action]
        else:
            ul = action

        _, us = self.filter.optimize(self.x, ul)
        u = float(us.full()[:, 0])

        # update system
        self.x = np.array(self.sys.update(self.x, u), dtype=np.float32)

        # check if done
        done = self.sys.t > self.TIME_LIMIT

        # compute cost
        theta, _ = self.x
        cost = normalize_angle(np.pi - theta) ** 2 * self.sys.DT / self.TIME_LIMIT

        return self.x, -cost, done, {}
