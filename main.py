import gymnasium as gym
import envs.stove_env   # This line just ensures registration runs

env = gym.make("stove-turnoff-v0")
obs, _ = env.reset()
done = False
from gymnasium.utils.env_checker import check_env
check_env(env)

# env = gym.make("stove-turnoff-v0")
# obs, _ = env.reset()
# done = False
# while not done:
#     action = env.action_space.sample()
#     obs, reward, done, _, _ = env.step(action)
#     env.render()

