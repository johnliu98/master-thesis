import os

import numpy as np
import casadi as ca

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker

from utils.angle import minus_pi_to_pi
from utils.plot import add_to_plot, CircleSector

import warnings
warnings.filterwarnings("ignore")

SOLVER_OPTS = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}


class Pendulum:

    # Dimensions
    NX: int = 2
    NU: int = 1

    # System parameters
    G: float = 9.81
    L: float = 0.5
    M: float = 0.15
    ETA: float = 0.1
    DT: float = 2e-2

    # Animation parameters
    WIDTH: float = 0.07
    HEIGHT: float = 1.0

    def __init__(self) -> None:
        self.t = 0.0
        self.t_plot = 0.0

        self._update = self._build_update()

    def update(self, x, u, increment_time=True):
        if increment_time:
            self.t += self.DT
        x_next = self._update(x, u).full().squeeze()
        x_next[0] = minus_pi_to_pi(x_next[0])
        return x_next

    def _build_update(self):
        x = ca.SX.sym("x", self.NX)
        u = ca.SX.sym("u", self.NU)

        ode = ca.vertcat(
            x[0] + self.DT * x[1],
            x[1] + self.DT * (
                    - self.G / self.L * ca.sin(x[0])
                    - self.ETA / self.M / self.L ** 2 * x[1]
                    + 1 / self.M / self.L ** 2 * u[0] 
                )
        )

        return ca.Function("ode", [x, u], [ode])  
    
    def initialize_figure(self):
        plt.close()
        plt.figure(figsize=(10, 5))
        plt.title("Pendulum")
        plt.ion()

        self.ax_pendulum = plt.subplot2grid((3, 2), (0, 0), rowspan=3, aspect="equal")

        # bar
        self.bar_plot = patches.Rectangle(
            xy=(0, 0),
            width=self.WIDTH,
            height=self.HEIGHT,
            angle=0,
            zorder=9,
        )

        # outer circle
        circle = patches.Circle(
            xy=(0, 0),
            radius=1.05,
            fill=False,
        )

        # center point
        center = patches.Circle(
            xy=(0, 0),
            radius=self.WIDTH / 2,
            fill=True,
            color="k",
            zorder=10,
        )

        # dangerous domain
        red_sector = CircleSector(
            (0, 0),
            0.95,
            theta1=-60,
            theta2=85,
            fill=True,
            facecolor="red",
            edgecolor=None,
            alpha=0.2,
        )

        # safe domain
        green_sector = CircleSector(
            (0, 0),
            0.95,
            theta1=150,
            theta2=300,
            fill=True,
            facecolor="green",
            edgecolor=None,
            alpha=0.2,
        )

        # reference line
        self.ax_pendulum.plot([0, 0], [0, 1.05], "--k", linewidth=1)

        self.ax_pendulum.add_artist(self.bar_plot)
        self.ax_pendulum.add_artist(circle)
        self.ax_pendulum.add_artist(center)
        self.ax_pendulum.add_artist(red_sector)
        self.ax_pendulum.add_artist(green_sector)

        self.ax_pendulum.set_xlim(-1.5, 1.5)
        self.ax_pendulum.set_ylim(-1.5, 1.5)
        self.ax_pendulum.set_xticks([])
        self.ax_pendulum.set_yticks([])

        # graph yaw
        self.ax_theta = plt.subplot2grid((3, 2), (0, 1), aspect="auto")

        self.theta_plot = self.ax_theta.plot([], [], color="#1f77b4", label=r"$\theta$")
        self.theta_pred_plot = self.ax_theta.plot([], [], color="#1f77b4", linestyle="--")

        self.ax_theta.axhline(y=np.pi, color="k", label=r"$\theta_{ref}$", linestyle="--")
        self.ax_theta.axhline(y=-np.pi / 2, color="r", linestyle="--")
        self.ax_theta.axhline(y=185 / 180 * np.pi, color="r", linestyle="--")
        self.ax_theta.legend(loc="upper left")
        self.ax_theta.grid()
        self.ax_theta.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

        # graph yaw rate
        self.ax_thetadot = plt.subplot2grid((3, 2), (1, 1), aspect="auto")
        
        self.thetadot_plot = self.ax_thetadot.plot([], [], color="#ff7f0e", label=r"$\dot{\theta}$")
        self.thetadot_pred_plot = self.ax_thetadot.plot([], [], color="#ff7f0e", linestyle="--")

        self.ax_thetadot.legend(loc="upper left")
        self.ax_thetadot.grid()
        self.ax_thetadot.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

        # graph inputs
        self.ax_input = plt.subplot2grid((3, 2), (2, 1), aspect="auto")

        self.ctrl_plot = self.ax_input.step([], [], color="green", label=r"$u$")
        self.ctrl_pred_plot = self.ax_input.step([], [], color="green", linestyle="--")
        self.ctrl_rl_plot = self.ax_input.step([], [], color="red", alpha=0.5, linestyle="--", label=r"$u_L$")

        self.ax_input.set_xlabel("time [s]")
        self.ax_input.set_ylim(-0.8, 0.8)
        self.ax_input.legend(loc="upper left")
        self.ax_input.grid()


    def animate(self, x, u, ul=None):
        self.t_plot += self.DT

        if not isinstance(x, np.ndarray):
            raise TypeError("x is not a numpy array")
        x = x.reshape(self.NX, -1)
        u = u.reshape(self.NU, -1)

        # animate bar
        angle = ca.pi - x[0, 0]
        self.bar_plot.set_x(-self.WIDTH / 2 * ca.cos(angle))
        self.bar_plot.set_y(-self.WIDTH / 2 * ca.sin(angle))
        self.bar_plot.set_angle(angle / ca.pi * 180)

        # add data to plots
        theta = np.array([self.t_plot, x[0, 0]])
        thetadot = np.array([self.t_plot, x[1, 0]])
        ctrl = np.array([self.t_plot, u[:, 0]])
        ctrl_rl = np.array([self.t_plot, ul])

        add_to_plot(self.theta_plot, theta)
        add_to_plot(self.thetadot_plot, thetadot)
        add_to_plot(self.ctrl_plot, ctrl)
        add_to_plot(self.ctrl_rl_plot, ctrl_rl)

        # plot predictions
        t_steps = x.shape[1]
        if t_steps > 1:
            t_pred = np.linspace(self.t_plot, self.t_plot + self.DT * (t_steps - 1), t_steps)
            self.theta_pred_plot[0].set_data(np.vstack((t_pred, x[0])))
            self.thetadot_pred_plot[0].set_data(np.vstack((t_pred, x[1, :])))
            self.ctrl_pred_plot[0].set_data(np.vstack((t_pred[:-1], u[0, :])))

        for ax in [self.ax_theta, self.ax_thetadot, self.ax_input]:
            ax.relim()
            ax.autoscale_view()

        plt.show()
        plt.pause(0.001)


class Bruce:

    # Dimensions
    NX: int = 2
    NU: int = 2

    # System parameters
    G: float = 9.81
    # M: float = 0.15

    def update(self, x, u, dt):

        return ca.vertcat()
