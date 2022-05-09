from typing import Protocol

import numpy as np


class Controller(Protocol):
    def compute_control(self, t: float, err: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Pid:
    def __init__(self, K, Ti=np.inf, Td=0) -> None:
        self.K = K
        self.Ti = Ti
        self.Td = Td

        self.t_prev = 0
        self.err_prev = 0
        self.int_prev = 0

    def compute_control(self, t: float, err: float) -> float:

        dt = t - self.t_prev
        err_int = self.int_prev + dt * err
        err_der = (err - self.err_prev) / dt if dt >= 1e-2 else 0

        self.t_prev = t
        self.err_prev = err
        self.int_prev = err_int

        return self.K * (err + 1 / self.Ti * err_int + self.Td * err_der)


class Stanley:
    def __init__(self, k_y: float, k_yaw: float) -> None:
        self.k_y = k_y
        self.k_yaw = k_yaw


    def compute_control(self, vel_x, err_y, err_yaw):
        return -self.k_yaw * err_yaw - np.arctan(self.k_y * err_y / vel_x)
