import gym

from stable_baselines3 import DQN

env = gym.make("LunarLander-v2")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1e6)

input("Press <Enter> to continue")

obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
