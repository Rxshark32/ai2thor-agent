import envs.stove_env
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

env = gym.make("stove-turnoff-v0")
env = Monitor(env)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_stove_tensorboard/"
)

model.learn(total_timesteps=100_000)
model.save("models/ppo_stove_turnoff")
