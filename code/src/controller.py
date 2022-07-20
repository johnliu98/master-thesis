from typing import Protocol

import numpy as np


class Controller(Protocol):
    def compute_control(self, t: float, err: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class Pid:
    def __init__(self, k, i=np.inf, d=0) -> None:
        self.k = k
        self.i = i
        self.d = d

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

        return self.k * (err + 1 / self.i * err_int + self.d * err_der)


class Stanley:
    def __init__(self, k_y: float, k_yaw: float = 1) -> None:
        self.k_y = k_y
        self.k_yaw = k_yaw


    def compute_control(self, err_y, err_yaw, vel_x):
        # return -self.k_yaw * err_yaw - np.arctan(self.k_y * err_y / vel_x)
        return -self.k_yaw * err_yaw - self.k_y * err_y
