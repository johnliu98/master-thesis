import time
import casadi as ca

IPOPT_OPTS = {
    "ipopt.print_level": 0,
    "ipopt.sb": "yes",
    "print_time": 0,
}
SQP_OPTS = {
    "qpsol": "qrqp",
    "print_header": False,
    "print_iteration": False,
    "print_time": False,
    "qpsol_options": {
        "print_iter": False,
        "print_header": False,
        "print_info": False,
        "error_on_fail": False,
    },
}


class MPC:

    INPUT_CONS = {"low": -0.6, "high": 0.6}
    STATE_CONS = {"low": -ca.pi / 3, "high": 185 / 180 * ca.pi}
    TERMINAL_CONS = {"low": -ca.pi / 6, "high": ca.pi / 6}

    def __init__(self, sys, N) -> None:
        self.N = N
        self.sys = sys

        self._optimize = self.build_optimize()

    def optimize(self, x0, xref, verbose=0):
        start = time.time()
        x, u = self._optimize(x0, xref)
        end = time.time()

        if verbose:
            print(f"MPC optimization time: {1000*(end-start):.0f} ms")

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

        # problem constraints
        opti.subject_to([
            # input contraints
            opti.bounded(self.INPUT_CONS["low"], u, self.INPUT_CONS["high"]),
            # state constraints
            opti.bounded(self.STATE_CONS["low"], x[0, :], self.STATE_CONS["high"]),
            # terminal contraints
            opti.bounded(self.TERMINAL_CONS["low"], x[:, -1], self.TERMINAL_CONS["high"]),
            # initial state
            x[:, 0] == x0,
        ])

        # set default values
        opti.set_value(x0, ca.DM([0, 0]))
        opti.set_value(xref, ca.DM([ca.pi, 0]))

        # solve optimization problem
        opti.solver("ipopt", IPOPT_OPTS)
        opti.solve()

        return opti.to_function("optimize", [x0, xref], [x, u])


class SafetyFilter(MPC):
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

        # define cost function to minimize
        opti.minimize((ul - u[:, 0]) ** 2)

        # enforce system dynamics
        for k in range(self.N):
            opti.subject_to(x[:, k + 1] == self.sys._update(x[:, k], u[:, k]))

        # problem constraints
        opti.subject_to([
            # input contraints
            opti.bounded(self.INPUT_CONS["low"], u[0, :], self.INPUT_CONS["high"]),
            # state constraints
            opti.bounded(self.STATE_CONS["low"], x[0, :], self.STATE_CONS["high"]),
            # terminal contraints
            opti.bounded(self.TERMINAL_CONS["low"], x[:, -1], self.TERMINAL_CONS["high"]),
            # initial state
            x[:, 0] == x0,
        ])

        # set default values
        opti.set_value(x0, ca.DM.zeros(self.sys.NX,))
        opti.set_value(ul, ca.DM.zeros(self.sys.NU,))

        # solve optimization problem
        opti.solver("ipopt", IPOPT_OPTS)
        opti.solve()

        return opti.to_function("optimize", [x0, ul], [x, u])
