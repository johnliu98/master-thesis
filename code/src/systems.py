import copy
from matplotlib.transforms import BboxTransformFrom

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


class Vehicle:

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

    # State constraints
    MAX_VEL_X: float = 20 / 3.6  # m/s
    MIN_VEL_X: float = 5 / 3.6  # m/s

    # Input contraints
    MAX_STEER: float = np.pi / 6  # rad
    MAX_DSTEER: float = 0.5  # rad/s
    MAX_THROTTLE: float = 2  # m/s^2
    MIN_THROTTLE: float = -4  # m/s^2
    MAX_DTHROTTLE: float = 10  # 2  # m/s^3

    # Trajectory constraints
    MAX_S: float = 60
    DS: float = 0.005

    # Tag angle
    FRONT_TAG_RATIO: float = -0.4

    # Time interval
    DT: float = 0.025  # s

    def __init__(self, k) -> None:

        self._t = 0
        self._u_prev = np.zeros((self.NU,))
        self._state = self.STATE0
        self._frame = np.zeros((3,))
        self._pose = np.zeros((3,))

        self.k = k
        self._update = self._build_update()

        s_d = np.linspace(0, self.MAX_S, int(self.MAX_S / self.DS))
        curvatures = k(s_d).full().squeeze()
        self.traj = create_trajectory(curvatures, self.DS)

    def reset(self):
        self._t = 0
        self._u_prev = np.zeros((self.NU,))
        self._state = self.STATE0
        self._frame = np.zeros((3,))
        self._pose = np.zeros((3,))

    def update(self, u):

        u = self.check_input_constraints(u, self._u_prev)
        self._u_prev = copy.copy(u)

        # update time
        self._t += self.DT

        # update state
        state_next = self._update(self._state, u, self.k(self._state[0]))
        self._state = state_next.full().squeeze()

        return self._state

    def initialize_figure(self):
        plt.figure(figsize=(10, 10), num="pod")
        plt.title("Pod")
        plt.ion()

        t = self._t
        x = self.pose[0]
        y = self.pose[1]
        n = self._state[1]
        alpha = self._state[2]
        vel_x = self._state[3]
        # vel_y = self._state[4]
        # yaw_rate = self._state[5]
        d_f = self._u_prev[0]
        tau = self._u_prev[1]

        # pod trajectory
        self.ax_pod = plt.subplot2grid((6, 1), (0, 0), rowspan=1, aspect="equal")
        self.traj.plot(self.ax_pod)
        self.path = self.ax_pod.plot(
            [x],
            [y],
            color="blue",
            linewidth=1,
            label="path",
            zorder=10,
        )
        self.path_pred = self.ax_pod.plot(
            [],
            [],
            color="blue",
            linewidth=1,
            linestyle="--",
            label="pred",
        )
        self.frenet = self.ax_pod.scatter(
            [0], [0], s=5, color="r", label="frenet origin", zorder=9
        )
        # self.ax_pod.set_ylim(-1, 6)
        self.ax_pod.set_xlabel("x [m]")
        self.ax_pod.set_ylabel("y [m]")

        # Lateral error
        self.ax_n = plt.subplot2grid((6, 1), (1, 0), aspect="auto")
        self.n = self.ax_n.plot(
            [t],
            [n],
            color="#1f77b4",
            linewidth=1,
        )
        self.n_pred = self.ax_n.plot(
            [],
            [],
            color="#1f77b4",
            linewidth=1,
            linestyle="--",
        )
        self.ax_n.set_ylabel(r"$n$ [m]")
        self.ax_n.set_ylim(-3.5, 3.5)

        # Angular error
        self.ax_a = plt.subplot2grid((6, 1), (2, 0), aspect="auto")
        self.a = self.ax_a.plot(
            [t],
            [alpha],
            color="#1f77b4",
            linewidth=1,
        )
        self.a_pred = self.ax_a.plot(
            [],
            [],
            color="#1f77b4",
            linewidth=1,
            linestyle="--",
        )
        self.ax_a.set_ylabel(r"$\alpha$ [rad]")
        self.ax_a.set_ylim(-np.pi / 3, np.pi / 3)

        # Longitudinal velocity
        self.ax_velx = plt.subplot2grid((6, 1), (3, 0), aspect="auto")
        self.velx = self.ax_velx.plot(
            [t],
            [vel_x],
            color="#1f77b4",
            linewidth=1,
        )
        self.velx_pred = self.ax_velx.plot(
            [],
            [],
            color="#1f77b4",
            linewidth=1,
            linestyle="--",
        )
        self.ax_velx.set_ylabel(r"$v_x$ [m/s]")
        self.ax_velx.set_ylim(0, 6)

        """
            # lateral velocity
            self.ax_vely = plt.subplot2grid((6, 1), (2, 0), aspect="auto")
            self.vely = self.ax_vely.plot(
                [],
                [],
                color="#1f77b4",
                linewidth=1,
            )
            self.ax_vely.set_ylabel(r"$v_y$ [m/s]")
            self.ax_vely.set_ylim(-1.0, 1.0)

            # yaw rate
            self.ax_yawrate = plt.subplot2grid((6, 1), (3, 0), aspect="auto")
            self.yawrate = self.ax_yawrate.plot(
                [],
                [],
                color="#ff7f0e",
                linewidth=1,
            )
            self.ax_yawrate.set_ylabel(r"$\varphi$ [rad/s]")
            self.ax_yawrate.set_ylim(-1.0, 1.0)
        """

        # steering angle
        self.ax_steer = plt.subplot2grid((6, 1), (4, 0), aspect="auto")
        self.steer = self.ax_steer.step(
            [t],
            [d_f],
            color="green",
            linewidth=1,
        )
        self.steer_pred = self.ax_steer.step(
            [],
            [],
            color="green",
            linewidth=1,
            linestyle="--",
        )
        self.steer_learn = self.ax_steer.step(
            [t],
            [d_f],
            color="red",
            linewidth=1,
        )
        self.ax_steer.set_ylabel(r"$\delta_f$ [rad]")
        self.ax_steer.set_ylim(-0.6, 0.6)

        # throttle
        self.ax_throttle = plt.subplot2grid((6, 1), (5, 0), aspect="auto")
        self.throttle = self.ax_throttle.step(
            [t],
            [tau],
            color="green",
            linewidth=1,
        )
        self.throttle_pred = self.ax_throttle.step(
            [],
            [],
            color="green",
            linewidth=1,
            linestyle="--",
        )
        self.throttle_learn = self.ax_throttle.step(
            [t],
            [tau],
            color="red",
            linewidth=1,
        )
        self.ax_throttle.set_xlabel("time [s]")
        self.ax_throttle.set_ylabel(r"$\tau$ [m/s$^2$]")
        self.ax_throttle.set_ylim(-4.5, 2.5)

        self.axs = [
            self.ax_n,
            self.ax_a,
            self.ax_velx,
            # self.ax_vely,
            # self.ax_yawrate,
            self.ax_steer,
            self.ax_throttle,
        ]

        # plot sweetners
        for ax in self.axs:
            ax.grid()
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        self.ax_pod.grid()
        # self.ax_pod.legend(loc="upper left")

        plt.tight_layout()

    def animate(self, ul, x_pred=None, u_pred=None):

        x = self.pose[0]
        y = self.pose[1]
        n = self._state[1]
        alpha = self._state[2]
        vel_x = self._state[3]
        # vel_y = self._state[4]
        # yaw_rate = self._state[5]
        d_f = self._u_prev[0]
        tau = self._u_prev[1]
        d_f_learn = ul[0]
        tau_learn = ul[1]

        # add data to plots
        ns = np.array([self._t, n])
        alphas = np.array([self._t, alpha])
        vel_xs = np.array([self._t, vel_x])
        # vel_ys = np.array([self._t, vel_y])
        # yaw_rates = np.array([self._t, yaw_rate])
        d_fs = np.array([self._t, d_f])
        taus = np.array([self._t, tau])
        d_f_learns = np.array([self._t, d_f_learn])
        tau_learns = np.array([self._t, tau_learn])

        self.frenet.set_offsets(self.traj.pose(self._state[0]).to_array()[:2])
        add_to_plot(self.path, [x, y])
        add_to_plot(self.n, ns)
        add_to_plot(self.a, alphas)
        add_to_plot(self.velx, vel_xs)
        # add_to_plot(self.vely, vel_ys)
        # add_to_plot(self.yawrate, yaw_rates)
        add_to_plot(self.steer, d_fs)
        add_to_plot(self.throttle, taus)
        add_to_plot(self.steer_learn, d_f_learns)
        add_to_plot(self.throttle_learn, tau_learns)

        if x_pred is not None:
            N = x_pred.shape[1] - 1
            t_pred = np.linspace(self._t, self._t + N * self.DT, N + 1)
            pose_pred = self.frenet_to_cartesian(x_pred)
            self.path_pred[0].set_data(np.vstack((pose_pred[0, :], pose_pred[1, :])))
            self.n_pred[0].set_data(np.vstack((t_pred, x_pred[1, :])))
            self.a_pred[0].set_data(np.vstack((t_pred, x_pred[2, :])))
            self.velx_pred[0].set_data(np.vstack((t_pred, x_pred[3, :])))
            self.steer_pred[0].set_data(np.vstack((t_pred[:-1], u_pred[0, :])))
            self.throttle_pred[0].set_data(np.vstack((t_pred[:-1], u_pred[1, :])))

        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()
        self.ax_pod.relim()
        self.ax_pod.autoscale_view()

        plt.pause(1e-3)
        plt.show()

    def check_input_constraints(self, u, u_prev):

        # steering change contraints
        u[0] = u_prev[0] + np.clip(
            u[0] - u_prev[0],
            -self.DT * self.MAX_DSTEER,
            self.DT * self.MAX_DSTEER,
        )

        # throttle change constraints
        u[1] = u_prev[1] + np.clip(
            u[1] - u_prev[1],
            -self.MAX_DTHROTTLE * self.DT,
            self.MAX_DTHROTTLE * self.DT,
        )

        # steering contraints
        u[0] = np.clip(u[0], -self.MAX_STEER, self.MAX_STEER)

        # throttle constraints
        u[1] = np.clip(u[1], self.MIN_THROTTLE, self.MAX_THROTTLE)

        return u

    def frenet_to_cartesian(self, state):
        n = state[1, :]
        a = state[2, :]
        v = state[3, :]

        frenet = np.array([self.traj.pose(s).to_array() for s in state[0]]).T
        x_f, y_f, yaw_f = frenet

        x = x_f - n * np.sin(yaw_f)
        y = y_f + n * np.cos(yaw_f)
        yaw = yaw_f + a
        return np.vstack([x, y, yaw, v])

    @property
    def time(self):
        return self._t

    @property
    def state(self):
        return self._state

    @property
    def err_front_axis(self):
        n = self._state[1]
        a = self._state[2]
        return np.array([n + self.LF * np.sin(a), a])

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
    def u_prev(self):
        return self._u_prev

    @property
    def desired_vel_x(self):
        n, a = self.err_front_axis
        vel = self.MAX_VEL_X - abs(n) - 5 * abs(a)
        vel = np.clip(vel, self.MIN_VEL_X, self.MAX_VEL_X)
        return vel

    @property
    def dist_to_end(self):
        return distance(Pose(*self.pose), self.traj.poses[-1])


class Kinematic(Vehicle):

    # Dimensions
    NX: int = 4
    NU: int = 2

    # Initial state
    STATE0 = np.array([0, 0, 0, Vehicle.MAX_VEL_X])

    def _build_update(self):

        # state = [s, n, a, vel_y, yaw_rate]^T
        state = ca.MX.sym("state", self.NX)
        # s = state[0]
        n = state[1]
        a = state[2]
        vel_x = state[3]
        # vel_y = state[4]
        # yaw_rate = state[5]

        # u = d_f
        u = ca.MX.sym("u", self.NU)
        d_f = u[0]
        tau = u[1]
        # d_t = self.FRONT_TAG_RATIO * d_f

        curv = ca.MX.sym("curv")

        beta = ca.atan(self.LR / (self.LF + self.LR) * ca.tan(d_f))
        f_s = 1 / (1 + n * curv) * vel_x * ca.cos(a + beta)
        f_n = vel_x * ca.sin(a + beta)
        f_a = vel_x * ca.tan(d_f) * ca.cos(beta) / (self.LF + self.LR) - curv * f_s

        f_vel_x = tau

        # state_next = state + self.DT * ca.vertcat(
        #     f_s, f_n, f_a, f_vel_x, f_vel_y, f_yaw_rate
        # )
        state_next = state + self.DT * ca.vertcat(f_s, f_n, f_a, f_vel_x)

        return ca.Function("state_next", [state, u, curv], [state_next])


class Dynamic(Vehicle):

    # Dimensions
    NX: int = 6
    NU: int = 2

    # Initial state
    STATE0 = np.array([0, 0, 0, Vehicle.MAX_VEL_X, 0, 0])

    def _build_update(self):

        # state = [s, n, a, vel_y, yaw_rate]^T
        state = ca.MX.sym("state", self.NX)
        s = state[0]
        n = state[1]
        a = state[2]
        vel_x = state[3]
        vel_y = state[4]
        yaw_rate = state[5]

        u = ca.MX.sym("u", self.NU)
        d_f = u[0]
        tau = u[1]
        d_t = self.FRONT_TAG_RATIO * d_f

        curv = ca.MX.sym("curv")

        # state dynamics
        a_f = ca.atan((vel_y + self.LF * yaw_rate) / (vel_x + 1e-6)) - d_f
        a_r = ca.atan((vel_y - self.LR * yaw_rate) / (vel_x + 1e-6))
        a_t = ca.atan((vel_y - self.LT * yaw_rate) / (vel_x + 1e-6)) - d_t

        f_vel_x = tau
        f_vel_y = (
            1
            / self.M
            * (
                -self.CF * a_f * ca.cos(d_f)
                - self.CR * a_r
                - self.CT * a_t * ca.cos(d_t)
            )
            - vel_x * yaw_rate
        )
        f_yaw_rate = (
            1
            / self.IZ
            * (
                -self.CF * self.LF * a_f * ca.cos(d_f)
                + self.CR * self.LR * a_r
                + self.CT * self.LT * a_t * ca.cos(d_t)
            )
        )

        f_s = 1 / (1 + n * self.k(s)) * (vel_x * ca.cos(a) - vel_y * ca.sin(a))
        f_n = vel_x * ca.sin(a) + vel_y * ca.cos(a)
        f_a = yaw_rate - self.k(s) * f_s

        state_next = state + self.DT * ca.vertcat(
            f_s, f_n, f_a, f_vel_x, f_vel_y, f_yaw_rate
        )

        return ca.Function("state_next", [state, u, curv], [state_next])

    @property
    def kinematic_state(self):
        return self._state[:4]
