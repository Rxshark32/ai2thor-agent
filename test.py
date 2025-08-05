import gymnasium as gym
import envs.stove_env
from gymnasium.utils.env_checker import check_env

# env = gym.make("stove-turnoff-v0")

# try:
#     check_env(env)
#     print("Environment passes all checks!")
# except Exception as e:
#     print(f"Environment has issues: {e}")

##### Observation Tester #####

env = gym.make("stove-turnoff-v0")
obs, info = env.reset()
print("First observation:", obs)

action = env.action_space.sample()
print("Taking action:", action)

obs, reward, terminated, truncated, info = env.step(action)

print("Second observation:", obs)
print("Reward:", reward)
print("Terminated:", terminated, "Truncated:", truncated)