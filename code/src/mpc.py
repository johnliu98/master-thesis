import scipy.stats as stats

import casadi as ca

from utils.angle import normalize_from
from src.bayesian_optimization import PendulumDynamicsRegression

IPOPT_OPTS = {
    "ipopt.print_level": 0,
    "ipopt.sb": "yes",
    "print_time": 0,
    "ipopt.max_iter": 1000,
    "ipopt.acceptable_constr_viol_tol": 1e-10,
    "ipopt.constr_viol_tol": 1e-10,
}


class MPC:

    INPUT_CONS = {"low": -0.7, "high": 0.7}
    STATE_CONS = {"low": -90 / 180 * ca.pi, "high": 190 / 180 * ca.pi}
    TERMINAL_CONS = {"low": -ca.pi / 6, "high": ca.pi / 6}

    EPS: float = 0.005
    RHO: float = 0.9

    def __init__(self, sys, N) -> None:
        self.N = N
        self.sys = sys

        self.eps = [
            self.EPS * (1 - ca.sqrt(self.RHO) ** i) / (1 - ca.sqrt(self.RHO))
            for i in range(N + 1)
        ]

        self._optimize = self.build_optimize()

    def optimize(self, x0, xref):
        x0[0] = normalize_from(x0[0], -100 / 180 * ca.pi)
        (
            x,
            u,
        ) = self._optimize(x0, xref)

        x = x.full()
        u = u.full().squeeze()

        return x, u

    def build_optimize(self):
        """Casadi wrapper method to build MPC function

        return: function that maps initial state and reference state to optimal input signal
        """

        # define variables and parameters
        opti = ca.Opti()
        x = opti.variable(self.sys.NX, self.N + 1)
        u = opti.variable(self.sys.NU, self.N)
        x0 = opti.parameter(self.sys.NX)
        xref = opti.parameter(self.sys.NX)

        # define cost function to minimize
        opti.minimize(ca.sumsqr(xref[0] - x[0, :]))

        # enforce system dynamics
        for k in range(self.N):
            opti.subject_to(x[:, k + 1] == self.sys._update(x[:, k], u[:, k]))

        for k in range(self.N):
            opti.subject_to(
                [
                    # state constraints
                    opti.bounded(
                        self.STATE_CONS["low"] * (1 - self.eps[k]),
                        x[0, k],
                        self.STATE_CONS["high"] * (1 - self.eps[k]),
                    ),
                    # input contraints
                    opti.bounded(
                        self.INPUT_CONS["low"] * (1 - self.eps[k]),
                        u[0, k],
                        self.INPUT_CONS["high"] * (1 - self.eps[k]),
                    ),
                ]
            )

        # problem constraints
        opti.subject_to(
            [
                # terminal contraints
                opti.bounded(
                    self.TERMINAL_CONS["low"], x[:, -1], self.TERMINAL_CONS["high"]
                ),
                # initial state
                x[:, 0] == x0,
            ]
        )

        # set default values
        opti.set_value(x0, ca.DM([0, 0]))
        opti.set_value(xref, ca.DM([ca.pi, 0]))

        # solve optimization problem
        opti.solver("ipopt", IPOPT_OPTS)
        opti.solve()

        return opti.to_function("optimize", [x0, xref], [x, u])


class PendulumFilter(MPC, PendulumDynamicsRegression):

    GAMMA: float = 0.02
    SAFETY_PROB: float = 0.99

    def __init__(self, sys, N, alpha, beta) -> None:

        PendulumDynamicsRegression.__init__(self, alpha, beta)
        MPC.__init__(self, sys, N)

    def optimize(self, x0, ul):
        x0[0] = normalize_from(x0[0], -100 / 180 * ca.pi)
        x, u = self._optimize(x0, ul, self.params, self.covariance)

        x = x.full()
        u = u.full().squeeze()

        return x, u

    def build_optimize(self):
        """Wrapper method to create a safety filter

        return: function that maps initial state and reference state to safe input signal
        """

        # define variables and parameters
        opti = ca.Opti()
        x = opti.variable(self.sys.NX, self.N + 1)
        u = opti.variable(self.sys.NU, self.N)
        x0 = opti.parameter(self.sys.NX)
        ul = opti.parameter(self.sys.NU)
        cov = opti.parameter(self.n_features, self.n_features)
        par = opti.parameter(self.n_features, self.n_outputs)

        # define cost function to minimize
        opti.minimize((ul - u[:, 0]) ** 2)

        # model undercenty contraints
        feat = ca.MX(self.n_features, self.N)
        feat[0, :] = x[0, :-1]
        feat[1, :] = x[1, :-1]
        feat[2, :] = ca.sin(x[0, :-1])
        feat[3, :] = u[0, :]
        feat = feat.T

        var = 1 / self.beta + ca.sum2(feat @ cov * feat)

        for k in range(self.N):
            opti.subject_to(
                [
                    # system dynamics
                    # x[:, k + 1] == self.sys._update(x[:, k], u[:, k]),
                    x[:, k + 1] == (feat[k, :] @ par).T,
                    # state constraints
                    opti.bounded(
                        self.STATE_CONS["low"] * (1 - self.eps[k]),
                        x[0, k],
                        self.STATE_CONS["high"] * (1 - self.eps[k]),
                    ),
                    # input constraints
                    opti.bounded(
                        self.INPUT_CONS["low"] * (1 - self.eps[k]),
                        u[0, k],
                        self.INPUT_CONS["high"] * (1 - self.eps[k]),
                    ),
                    # error constraints
                    ca.sqrt(var[k] * stats.chi2.ppf(self.SAFETY_PROB, 2) ** 2)
                    <= self.GAMMA * (1 - self.eps[k]),
                ]
            )

        # terminal contraints
        opti.subject_to(
            opti.bounded(
                self.TERMINAL_CONS["low"] * (1 - self.eps[-1]),
                x[:, -1],
                self.TERMINAL_CONS["high"] * (1 - self.eps[-1]),
            )
        )

        # initial state
        opti.subject_to(x[:, 0] == x0)

        # set default values
        opti.set_value(x0, 0)
        opti.set_value(ul, 0)
        opti.set_value(par, 0)
        opti.set_value(cov, 0)

        # solve optimization problem
        opti.solver("ipopt", IPOPT_OPTS)
        opti.solve()

        return opti.to_function("optimize", [x0, ul, par, cov], [x, u])


class VehicleFilter(MPC):

    MAX_Y_ERR: float = 2
    TERM_Y_ERR: float = 0.5
    TERM_YAW_ERR: float = 5 / 180 * ca.pi

    def __init__(self, sys, N) -> None:
        self.sys = sys
        self.N = N

        self._compute_control = self._build_compute_control()

    def compute_control(self, x0, ul, u_prev):
        x, u = self._compute_control(x0, ul, u_prev)

        x = x.full()
        u = u.full()

        return x, u

    def _build_compute_control(self):

        # define variables and parameters
        opti = ca.Opti()

        x = opti.variable(self.sys.NX, self.N + 1)
        u = opti.variable(self.sys.NU, self.N)
        dd_f = ca.diff(u[0, :])
        dtau = ca.diff(u[1, :])

        x0 = opti.parameter(self.sys.NX)
        ul = opti.parameter(self.sys.NU)
        u_prev = opti.parameter(self.sys.NU)

        beta = ca.atan(self.sys.LR / (self.sys.LF + self.sys.LR) * ca.tan(u_prev[0]))
        ds = (
            self.sys.DT / (1 + x0[1] * self.sys.k(x0[0])) * x0[3] * ca.cos(x0[2] + beta)
        )

        # define cost function to minimize
        opti.minimize(
            ((ul[0] - u[0, 0]) / self.sys.MAX_STEER) ** 2
            + ((ul[1] - u[1, 0]) / self.sys.MAX_THROTTLE) ** 2
        )
        # opti.minimize(
        #     ca.sum2(((self.sys.MAX_S - x[0, :]) / self.sys.MAX_S) ** 2)
        #     + ca.sum2(x[1, :] ** 2)
        #     + ca.sum2(x[2, :] ** 2)
        # )

        # system dynamics
        for k in range(self.N):
            opti.subject_to(
                x[:, k + 1]
                == self.sys._update(x[:, k], u[:, k], self.sys.k(x[0, 0] + k * ds))
            )

        # state constraints
        opti.subject_to(opti.bounded(-self.MAX_Y_ERR, x[1, :-1], self.MAX_Y_ERR))
        opti.subject_to(opti.bounded(self.sys.MIN_VEL_X, x[3, :], self.sys.MAX_VEL_X))

        # input constraints
        opti.subject_to(opti.bounded(-self.sys.MAX_STEER, u[0, :], self.sys.MAX_STEER))
        opti.subject_to(
            opti.bounded(self.sys.MIN_THROTTLE, u[1, :], self.sys.MAX_THROTTLE)
        )

        # terminal state constraints
        opti.subject_to(opti.bounded(-self.TERM_Y_ERR, x[1, -1], self.TERM_Y_ERR))
        opti.subject_to(opti.bounded(-self.TERM_YAW_ERR, x[2, -1], self.TERM_YAW_ERR))
        # opti.subject_to(
        #     ca.fabs(x[1, -1]) / self.TERM_Y_ERR + ca.fabs(x[2, -1]) / self.TERM_YAW_ERR
        #     <= 1
        # )

        # input change constraints
        opti.subject_to(
            opti.bounded(
                -self.sys.DT * self.sys.MAX_DSTEER,
                u[0, 0] - u_prev[0],
                self.sys.DT * self.sys.MAX_DSTEER,
            )
        )
        opti.subject_to(
            opti.bounded(
                -self.sys.DT * self.sys.MAX_DSTEER,
                dd_f,
                self.sys.DT * self.sys.MAX_DSTEER,
            )
        )
        opti.subject_to(
            opti.bounded(
                -self.sys.DT * self.sys.MAX_DTHROTTLE,
                u[1, 0] - u_prev[1],
                self.sys.DT * self.sys.MAX_DTHROTTLE,
            )
        )
        opti.subject_to(
            opti.bounded(
                -self.sys.DT * self.sys.MAX_DTHROTTLE,
                dtau,
                self.sys.DT * self.sys.MAX_DTHROTTLE,
            )
        )

        # initial state
        opti.subject_to(x[:, 0] == x0)

        # set default values
        opti.set_value(x0, [0, 0, 0, self.sys.MAX_VEL_X])
        opti.set_value(ul, [0, 0])
        opti.set_value(u_prev, [0, 0])

        # solve optimization problem
        opti.solver("ipopt", IPOPT_OPTS)
        opti.solve()

        return opti.to_function("optimize", [x0, ul, u_prev], [x, u])
