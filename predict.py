import gymnasium as gym
from stable_baselines3 import PPO
import envs.stove_env

env = gym.make("stove-turnoff-v0", render_mode="human")
model = PPO.load("models/ppo_stove_turnoff_final_v4")

MAXSTEPS = 150
obs, info = env.reset()
total_reward = 0
terminated = False
truncated = False

for step in range(MAXSTEPS):
    if terminated or truncated:
        break
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs)
    total_reward += reward

print("Total Reward:", total_reward)

