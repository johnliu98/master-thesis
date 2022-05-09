import tensorflow as tf
import tensorflow_probability as tfp

from src.rl.network import Network


class PolicyGradient:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 1e-3,
        hidden=(8, 8, 8, 8),
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions

        self.learning_rate = learning_rate
        self.net = Network(n_states, n_actions, hidden=hidden)
        self.opt = tf.keras.optimizers.Adam(learning_rate)

    def sample_action(self, state, deterministic=False) -> int:

        prob = self.net(state.reshape(-1, self.n_states))

        if deterministic:
            action = tf.math.argmax(prob, axis=1)
            return int(action.numpy()[0])

        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def train(self, state, action, reward) -> None:
        with tf.GradientTape() as tape:
            prob = self.net(state.reshape(-1, self.n_states), training=True)
            loss = self.loss(prob, action, reward)
        grads = tape.gradient(loss, self.net.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.net.trainable_variables))

    def loss(self, prob, action, reward) -> float:
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob * reward
        return loss
