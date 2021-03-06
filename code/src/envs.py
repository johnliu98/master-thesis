import numpy as np

import gym
from gym import spaces

from src.systems import Pendulum
from src.mpc import PendulumFilter
from utils.angle import minus_pi_to_pi


class PendulumBangEnv(gym.Env):
    """Pendulum Environment with a bang-bang controller"""

    INPUT_LIM: float = 0.7
    STATE_LIM: np.ndarray = np.array([0, 0])
    EPISODE_LENGTH: int = 150
    REF: float = np.pi

    def __init__(self, sys: Pendulum, filter: PendulumFilter) -> None:
        self.sys = sys

        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(
            -self.STATE_LIM, self.STATE_LIM, shape=(2,), dtype=np.float32
        )

        self.filter = filter

    def reset(self) -> np.ndarray:
        self.sys.t = 0
        self.x = np.zeros(self.observation_space.shape, dtype=np.float32)
        return self.x

    def step(self, switch_time):

        cost = (self.REF - self.x[0]) ** 2
        for i in range(self.EPISODE_LENGTH):
            if i <= switch_time:
                ul = -self.INPUT_LIM
            else:
                ul = self.INPUT_LIM

            _, us = self.filter.optimize(self.x, ul)
            u = us[0]
            self.x = np.array(self.sys.update(self.x, u))

            cost += (self.REF - self.x[0]) ** 2
        cost /= self.EPISODE_LENGTH

        return self.x, -cost, True, {}

    def render(self, mode="human"):
        return super().render(mode)


class PendulumEnv(gym.Env):
    """Pendulum Environment that follows gym interface"""

    INPUT_LIM = 0.6
    N_DISCRETE = 3

    TIME_LIMIT = 3.5
    STATE_REF = np.array([[np.pi], [0]])
    ACTIONS = np.linspace(-INPUT_LIM, INPUT_LIM, N_DISCRETE)

    BOUNDS = np.array([np.pi, 4])
    RESET_BOUNDS = np.array([np.pi / 6, 4])

    def __init__(self, sys, swing_up_pend=True, fixed_init=True, safety_filter=True):
        self.sys = sys
        self.filter = PendulumFilter(sys, 10)

        self.action_space = spaces.Discrete(self.N_DISCRETE)
        self.observation_space = spaces.Box(
            low=self.BOUNDS,
            high=self.BOUNDS,
            shape=(self.sys.NX,),
            dtype=np.float32,
        )

        self.swing_up_pend = swing_up_pend
        self.fixed_init = fixed_init
        self.safety_filter = safety_filter

    def reset(self):
        self.sys.t = 0

        if self.swing_up_pend:
            self.x = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            self.x = np.array([np.pi, 0], dtype=np.float32)

        if not self.fixed_init:
            self.x += np.random.uniform(
                low=-self.RESET_BOUNDS,
                high=self.RESET_BOUNDS,
            ).reshape(self.observation_space.shape)

        self.x[0] = minus_pi_to_pi(self.x[0])

        if not self.swing_up_pend and not self.fixed_init:
            if self.x[0] > 0:
                self.x[1] = abs(self.x[1])
            else:
                self.x[1] = -abs(self.x[1])

        return self.x

    def _cost(self, x):
        th, dth = x
        th_err = np.abs(minus_pi_to_pi(np.pi - th))

        c = lambda t, dt: (t**2 + 1e-2 * dt**2)
        if th_err < np.pi * 2 / 3:
            cost = c(th_err, dth)
        else:
            cost = c(np.pi * 2 / 3, 4) + 1e-1 * (4 - np.abs(dth)) ** 2
        cost = minus_pi_to_pi(np.pi - th) ** 2

        return cost

    def _theta_cost(self, x):
        th, _ = x
        cost = minus_pi_to_pi(np.pi - th) ** 2
        return cost

    def _safe_cost(self, dth, du):
        return minus_pi_to_pi(dth) ** 2 + du**2

    def step(self, action):

        # convert action to input
        ul = self.ACTIONS[action]

        # check input safety
        if self.safety_filter:
            _, us = self.filter.optimize(self.x, ul)
            u = float(us[:, 0])
        else:
            u = ul

        # update system
        self.x = np.array(self.sys.update(self.x, u), dtype=np.float32)
        self.x = self.x.reshape(self.observation_space.shape)

        # compute cost
        cost = self._safe_cost(np.pi - self.x[0], ul - u)

        # is done
        done = self.sys.t >= self.TIME_LIMIT

        return self.x, -cost, done, {}


class SafePendulumEnv(PendulumEnv):
    """Pendulum Environment with safety guarantees that follows gym interface"""

    def __init__(self, sys, sys_type="discrete", horizon=50):
        PendulumEnv.__init__(self, sys, sys_type=sys_type)

        self.filter = PendulumFilter(sys, horizon)

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
        cost = minus_pi_to_pi(np.pi - theta) ** 2 + (ul - u) ** 2
        cost *= self.sys.DT / self.TIME_LIMIT

        return self.x, -cost, done, {}
