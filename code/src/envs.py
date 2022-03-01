import numpy as np

import gym
from gym import spaces

from src.mpc import SafetyFilter
from utils.angle import normalize_angle


class PendulumEnv(gym.Env):
    """Pendulum Environment that follows gym interface"""

    INPUT_LIM = 0.7
    N_DISCRETE = 3

    TIME_LIMIT = 100
    STATE_REF = np.array([[np.pi], [0]])
    ACTIONS = np.linspace(-INPUT_LIM, INPUT_LIM, N_DISCRETE)

    BOUNDS = np.array([np.pi, 4])
    RESET_BOUNDS = np.array([np.pi / 3, 4])

    def __init__(self, sys, swing_up_pend=True, random_init=False):
        self.sys = sys
        self.action_space = spaces.Discrete(self.N_DISCRETE)
        self.observation_space = spaces.Box(
            low=self.BOUNDS,
            high=self.BOUNDS,
            shape=(self.sys.NX,),
            dtype=np.float32,
        )

        self.swing_up_pend = swing_up_pend
        self.random_init = random_init

    def reset(self):
        self.sys.t = 0

        if self.swing_up_pend:
            self.x = np.zeros(self.observation_space.shape, np.float32)
        else:
            self.x = np.array([np.pi, 0])

        if self.random_init:
            self.x += np.random.uniform(
                low=-self.RESET_BOUNDS,
                high=self.RESET_BOUNDS,
            ).reshape(self.observation_space.shape)

        self.x[0] = normalize_angle(self.x[0])

        if not self.swing_up_pend and self.random_init:
            if self.x[0] > 0:
                self.x[1] = abs(self.x[1])
            else:
                self.x[1] = -abs(self.x[1])

        return self.x

    def _cost(self, x):
        th, dth = x
        # th_err = np.abs(normalize_angle(np.pi - th))
        #
        # c = lambda t, dt: (t ** 2 + 1e-2 * dt ** 2)
        # if th_err < np.pi * 2 / 3:
        #     cost = c(th_err, dth)
        # else:
        #     cost = c(np.pi * 2 / 3, 4) + 1e-1 * (4 - np.abs(dth)) ** 2
        cost = normalize_angle(np.pi - th) ** 2

        return cost

    def step(self, action):

        # convert action to input
        u = self.ACTIONS[action]

        # update system
        self.x = np.array(self.sys.update(self.x, u), dtype=np.float32)
        self.x = self.x.reshape(self.observation_space.shape)

        # compute cost
        cost = self._cost(self.x)

        # is done
        done = self.sys.t >= self.TIME_LIMIT

        return self.x, -cost, done, {}


class SafePendulumEnv(PendulumEnv):
    """Pendulum Environment with safety guarantees that follows gym interface"""

    def __init__(self, sys, sys_type="discrete", horizon=50):
        PendulumEnv.__init__(self, sys, sys_type=sys_type)

        self.filter = SafetyFilter(sys, horizon)

    def step(self, action):

        # convert action to input
        ul = self.ACTIONS[action]

        # compute safe input signal
        _, us = self.filter.optimize(self.x, ul)
        u = float(us.full()[:, 0])

        # update system
        self.x = np.array(self.sys.update(self.x, u), dtype=np.float32)

        # check if done
        done = self.sys.t > self.TIME_LIMIT

        # compute cost
        theta, _ = self.x
        cost = normalize_angle(np.pi - theta) ** 2 + (ul - u) ** 2
        cost *= self.sys.DT / self.TIME_LIMIT

        return self.x, -cost, done, {}
