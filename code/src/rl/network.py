from typing import Tuple

import tensorflow as tf


class Network(tf.keras.Model):
    def __init__(
        self, n_states: int, n_actions: int, hidden: Tuple[int, ...] = (8, 8, 8, 8)
    ) -> None:
        super().__init__()

        self.n_states = n_states
        self.n_actions = n_actions
        self.hidden = hidden

        self.net = self.bulid_net()

    def bulid_net(self) -> list:
        net = [tf.keras.layers.InputLayer(self.n_states)]
        for h in self.hidden:
            net.append(tf.keras.layers.Dense(h, activation="relu"))
        net.append(tf.keras.layers.Dense(self.n_actions, activation="softmax"))
        return net

    def call(self, state) -> tf.Tensor:
        x = tf.convert_to_tensor(state)
        for layer in self.net:
            x = layer(x)
        return x
