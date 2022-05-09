import numpy as np
import casadi as ca

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker

from utils.plot import add_to_plot, CircleSector
from src.trajectory import create_trajectory, distance, Pose

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
    DT: float = 1e-1

    PARAMS = np.array(
        [
            [1, DT, 0, 0],
            [0, 1 - DT * ETA / (M * L**2), -DT * G / L, DT / (M * L**2)],
        ]
    ).T

    # Animation parameters
    WIDTH: float = 0.07
    HEIGHT: float = 1.0
    LOWER_BOUND: float = -90
    UPPER_BOUND: float = 190

    def __init__(self) -> None:
        self.t = 0.0
        self.t_plot = 0.0

        self._update = self._build_update()

    def update(self, x, u, increment_time=True):
        if increment_time:
            self.t += self.DT
        x_next = self._update(x, u).full().squeeze()
        return x_next

    def _build_update(self):
        x = ca.SX.sym("x", self.NX)
        u = ca.SX.sym("u", self.NU)

        ode = ca.vertcat(
            x[0] + self.DT * x[1],
            x[1]
            + self.DT
            * (
                -self.G / self.L * ca.sin(x[0])
                - self.ETA / self.M / self.L**2 * x[1]
                + 1 / self.M / self.L**2 * u[0]
            ),
        )

        return ca.Function("ode", [x, u], [ode])

    def initialize_figure(self):
        plt.figure(figsize=(10, 5), num="pendulum")
        plt.title("Pendulum")
        plt.ion()

        self.ax_pendulum = plt.subplot2grid((3, 2), (0, 0), rowspan=3, aspect="equal")

        # bar
        self.bar_plot = patches.Rectangle(
            xy=(self.WIDTH / 2, 0),
            width=self.WIDTH,
            height=self.HEIGHT,
            angle=180,
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
            theta1=-self.LOWER_BOUND - 90,
            theta2=-self.UPPER_BOUND + 270,
            fill=True,
            facecolor="red",
            edgecolor=None,
            alpha=0.2,
        )

        # safe domain
        green_sector = CircleSector(
            (0, 0),
            0.95,
            theta1=240,
            theta2=300,
            fill=True,
            facecolor="green",
            edgecolor=None,
            alpha=0.2,
        )

        # pendulum down hanging posision
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

        self.theta_plot = self.ax_theta.plot(
            [], [], color="#1f77b4", linewidth=1, label=r"$\theta$"
        )
        self.theta_pred_plot = self.ax_theta.plot(
            [], [], color="#1f77b4", linewidth=1, linestyle="--"
        )

        # lower bound
        self.ax_theta.axhline(
            y=self.LOWER_BOUND / 180 * np.pi, color="r", linestyle="--"
        )

        # upper bound
        self.ax_theta.axhline(
            y=self.UPPER_BOUND / 180 * np.pi, color="r", linestyle="--"
        )

        # plot sweetners
        self.ax_theta.legend(loc="upper left")
        self.ax_theta.grid()
        self.ax_theta.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

        # graph yaw rate
        self.ax_thetadot = plt.subplot2grid((3, 2), (1, 1), aspect="auto")

        self.thetadot_plot = self.ax_thetadot.plot(
            [], [], color="#ff7f0e", linewidth=1, label=r"$\dot{\theta}$"
        )
        self.thetadot_pred_plot = self.ax_thetadot.plot(
            [], [], color="#ff7f0e", linewidth=1, linestyle="--"
        )

        self.ax_thetadot.legend(loc="upper left")
        self.ax_thetadot.grid()
        self.ax_thetadot.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

        # graph inputs
        self.ax_input = plt.subplot2grid((3, 2), (2, 1), aspect="auto")

        self.ctrl_plot = self.ax_input.step(
            [], [], color="green", linewidth=1, label=r"$u$"
        )
        self.ctrl_pred_plot = self.ax_input.step(
            [], [], color="green", linewidth=1, linestyle="--"
        )
        self.ctrl_rl_plot = self.ax_input.step(
            [], [], color="red", alpha=0.5, linestyle="--", label=r"$u_L$"
        )

        self.ax_input.set_xlabel("time [s]")
        self.ax_input.set_ylim(-0.8, 0.8)
        self.ax_input.legend(loc="upper left")
        self.ax_input.grid()

    def animate(self, x, u, ul=None):
        self.t_plot += self.DT

        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(u, np.ndarray):
            u = np.array(u)

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
            t_pred = np.linspace(
                self.t_plot, self.t_plot + self.DT * (t_steps - 1), t_steps
            )
            self.theta_pred_plot[0].set_data(np.vstack((t_pred, x[0])))
            self.thetadot_pred_plot[0].set_data(np.vstack((t_pred, x[1, :])))
            self.ctrl_pred_plot[0].set_data(np.vstack((t_pred[:-1], u[0, :])))

        for ax in [self.ax_theta, self.ax_thetadot, self.ax_input]:
            ax.relim()
            ax.autoscale_view()

        plt.show()
        plt.pause(0.001)


class Bruce:

    """
    t: time
    p: absolute position and heading
    x: relative velocity and yaw rate
    """

    # Dimensions
    NX: int = 5
    NU: int = 1

    # Physical parameters
    M: float = 8975  # total mass
    IZ: float = 37632  # moment of ineria in z-direction

    # Distance from wheel to COG
    LF: float = 2.162  # front
    LR: float = 1.288  # rear
    LT: float = 2.638  # tag

    # Cornering stiffness
    CF: float = 2e5  # front
    CR: float = 5e5  # rear
    CT: float = 2e5  # tag

    # Longitudinal velocity
    VELX: float = 15 / 3.6  # m/s

    # Input contraints
    MAX_STEER: float = np.pi / 6  # rad
    MAX_STEER_CHANGE: float = 0.5  # 0.12  # rad/s

    FRONT_TAG_RATIO: float = -0.4

    # Time interval
    DT: float = 3e-2  # s

    def __init__(self, k) -> None:

        self._t = 0
        self._u = 0
        self._u_prev = 0
        self._state = np.zeros((self.NX,))
        self._frame = np.zeros((3,))
        self._pose = np.zeros((3,))

        self.k = k
        self._update_state, self._update_pose = self._build_update()

        s_max = 40
        s_num = 500
        s_d = np.linspace(0, s_max, s_num)
        curvatures = k(s_d).full().squeeze()
        self.traj = create_trajectory(curvatures, s_max / s_num)

    def update(self, u):

        # update time
        self._t += self.DT

        # steering change contraints
        u = self._u_prev + np.clip(
            u - self._u_prev,
            -self.MAX_STEER_CHANGE * self.DT,
            self.MAX_STEER_CHANGE * self.DT,
        )

        # steering contraints
        u = np.clip(u, -self.MAX_STEER, self.MAX_STEER)

        self._u = u
        self._u_prev = self._u

        pose_next = self._update_pose(self._pose, self._state)
        state_next = self._update_state(self._state, self._u)

        self._state = state_next.full().squeeze()
        self._pose = pose_next.full().squeeze()

        return self._state

    def _build_update(self):

        # state = [s, n, a, vel_y, yaw_rate]^T
        state = ca.SX.sym("state", self.NX)
        s = state[0]
        n = state[1]
        a = state[2]
        vel_y = state[3]
        yaw_rate = state[4]

        # pose = [x, y, yaw]
        pose = ca.SX.sym("pose", 3)
        yaw = pose[2]

        # u = d_f
        u = ca.SX.sym("u")
        d_f = u
        d_t = self.FRONT_TAG_RATIO * d_f

        # pose dynamics
        f_x = self.VELX * ca.cos(yaw) - vel_y * ca.sin(yaw)
        f_y = self.VELX * ca.sin(yaw) + vel_y * ca.cos(yaw)
        f_yaw = yaw_rate

        # state dynamics
        f_vel_y = (
            -1
            / self.M
            * (
                self.CF
                * (ca.atan2(vel_y + self.LF * yaw_rate, self.VELX) - d_f)
                * ca.cos(d_f)
                + self.CR * ca.atan2(vel_y - self.LR * yaw_rate, self.VELX)
                + self.CT
                * (ca.atan2(vel_y - self.LT * yaw_rate, self.VELX) - d_t)
                * ca.cos(d_t)
            )
            - self.VELX * yaw_rate
        )

        f_yaw_rate = (
            1
            / self.IZ
            * (
                -self.CF
                * self.LF
                * (ca.atan2(vel_y + self.LF * yaw_rate, self.VELX) - d_f)
                * ca.cos(d_f)
                + self.CR * self.LR * ca.atan2(vel_y - self.LR * yaw_rate, self.VELX)
                + self.CT
                * self.LT
                * (ca.atan2(vel_y - self.LT * yaw_rate, self.VELX) - d_t)
                * ca.cos(d_t)
            )
        )

        f_s = 1 / (1 + n * self.k(s)) * (self.VELX * ca.cos(a) - vel_y * ca.sin(a))
        f_n = self.VELX * ca.sin(a) + vel_y * ca.cos(a)
        f_a = yaw_rate - self.k(s) * f_s

        state_next = state + self.DT * ca.vertcat(f_s, f_n, f_a, f_vel_y, f_yaw_rate)
        pose_next = pose + self.DT * ca.vertcat(f_x, f_y, f_yaw)

        return (
            ca.Function("state_next", [state, d_f], [state_next]),
            ca.Function("frame_next", [pose, state], [pose_next]),
        )

    def initialize_figure(self):
        plt.figure(figsize=(12, 6), num="pod")
        plt.title("Pod")
        plt.ion()

        self.ax_pod = plt.subplot2grid((2, 3), (0, 0), colspan=3, aspect="equal")
        self.traj.plot(self.ax_pod)
        self.path = self.ax_pod.plot(
            [],
            [],
            color="black",
            linewidth=1,
            label="path",
            zorder=10,
        )
        self.frenet = self.ax_pod.scatter(
            [], [], s=5, color="r", label="frenet origin", zorder=9
        )
        self.ax_pod.set_ylim(-2, 8)
        self.ax_pod.set_xlabel("x [m]")
        self.ax_pod.set_ylabel("y [m]")

        self.ax_vely = plt.subplot2grid((2, 3), (1, 0), aspect="auto")
        self.vely = self.ax_vely.plot(
            [],
            [],
            color="#1f77b4",
            linewidth=1,
        )
        self.ax_vely.set_xlabel("time [s]")
        self.ax_vely.set_ylabel(r"$v_y$ [m/s]")
        self.ax_vely.set_ylim(-1.0, 1.0)

        self.ax_yawrate = plt.subplot2grid((2, 3), (1, 1), aspect="auto")
        self.yawrate = self.ax_yawrate.plot(
            [],
            [],
            color="#ff7f0e",
            linewidth=1,
        )
        self.ax_yawrate.set_xlabel("time [s]")
        self.ax_yawrate.set_ylabel(r"$\varphi$ [rad/s]")
        self.ax_yawrate.set_ylim(-1.0, 1.0)

        self.ax_steer = plt.subplot2grid((2, 3), (1, 2), aspect="auto")
        self.steer = self.ax_steer.step(
            [],
            [],
            color="green",
            linewidth=1,
        )
        self.ax_steer.set_xlabel("time [s]")
        self.ax_steer.set_ylabel(r"$\delta_f$ [rad]")
        self.ax_steer.set_ylim(-0.6, 0.6)

        self.axs = [
            self.ax_vely,
            self.ax_yawrate,
            self.ax_steer,
        ]

        # plot sweetners
        for ax in self.axs:
            ax.grid()
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        self.ax_pod.grid()
        self.ax_pod.legend(loc="upper left")

    def animate(self):

        x = self.pose[0]
        y = self.pose[1]
        vel_y = self._state[3]
        yaw_rate = self._state[4]

        # add data to plots
        vel_ys = np.array([self._t, vel_y])
        yaw_rates = np.array([self._t, yaw_rate])
        d_fs = np.array([self._t, self._u])

        self.frenet.set_offsets(self.traj.pose(self._state[0]).to_array()[:2])
        add_to_plot(self.path, [x, y])
        add_to_plot(self.vely, vel_ys)
        add_to_plot(self.yawrate, yaw_rates)
        add_to_plot(self.steer, d_fs)

        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()
        self.ax_pod.relim()
        self.ax_pod.autoscale_view()

        plt.tight_layout()
        plt.show()
        plt.pause(1e-3)

    @property
    def time(self):
        return self._t

    @property
    def state(self):
        return self._state

    @property
    def err(self):
        return self._state[1:3]

    @property
    def pose(self):
        state = self._state
        n = state[1]
        a = state[2]

        frenet = self.traj.pose(self._state[0])
        x = frenet.x - n * np.sin(frenet.yaw)
        y = frenet.y + n * np.cos(frenet.yaw)
        yaw = frenet.yaw + a
        return np.array([x, y, yaw])

    @property
    def dist_to_end(self):
        return distance(Pose(*self.pose), self.traj.poses[-1])
