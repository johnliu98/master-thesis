import math
import tensorflow as tf


class GradientDescent:

    REF: tf.float32 = tf.constant(math.pi)

    def __init__(self, sys, learning_rate, verbose=0) -> None:
        self.sys = sys
        self.inertia = 1 / sys.M / sys.L**2
        self.learning_rate = learning_rate
        self.verbose = verbose

        self.sol = {"x": None, "f": None}

    def sys_update(self, x, u):
        x_next = tf.zeros(shape=(self.sys.NX,), dtype=tf.float32)
        x_next[0] = x[0] + self.sys.DT * x[1]
        x_next[1] = x[1] + self.sys.DT * (
            -self.sys.G / self.sys.L * tf.sin(x[0])
            - self.sys.ETA * self.inertia * x[1]
            + self.inertia * u
        )

    def learn(self, timesteps):
        x = tf.Variable(
            tf.zeros(shape=(self.sys.NX,), dtype=tf.float32, trainable=False),
            dtype=tf.float32,
        )
        k = tf.Variable(50, dtype=tf.float32, trainable=True)

        with tf.GradientTape() as tape:
            for i in range(timesteps):
                if i < k:
                    u = tf.constant(-self.sys.INPUT_LIM, dtype=tf.float32)
                else:
                    u = tf.constant(self.sys.INPUT_LIM, dtype=tf.float32)

                x = self.sys_update(x, u)

            cost = tf.square(x[0] - self.REF)

        grad = tape.gradient(cost, k)
        k.assign(k - self.learning_rate * grad.numpy())

        self.sol["x"] = k.numpy()

    def predict(self):
        return self.sol["x"]

    def _callback(self):
        pass

    def save(path):
        pass

    def load(path):
        pass
