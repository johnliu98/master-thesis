import casadi as ca

from utils.angle import minus_pi_to_pi

class BangBangEnv:

    EPISODE_LENGTH: int = 120
    INPUT_LIM: float = 0.7
    REF: float = ca.pi

    def __init__(self, sys, learning_rate):
        self.sys = sys
        self.learning_rate = learning_rate

        self._grad, self._cost = self.build_grad_k()

    def reset(self):
        return ca.DM.zeros(2, 1)
    
    def next_k(self, k):
        return k - self._grad(k) * self.learning_rate

    def input_signal(self,i, k):
        return ca.tanh(i - k) * self.INPUT_LIM

    def build_grad_k(self):
        k = ca.SX.sym("k")

        x = ca.SX.zeros(self.sys.NX, 1)

        for i in range(self.EPISODE_LENGTH):
            u = self.input_signal(i, k)
            x = self.sys._update(x, u)
        err = minus_pi_to_pi(x[0] - self.REF)
        cost = ca.fabs(x[0] - self.REF)

        grad = ca.jacobian(cost, k)
        return ca.Function("grad", [k], [grad]), ca.Function("cost", [k], [cost])

