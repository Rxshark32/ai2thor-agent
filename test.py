import gymnasium as gym
import envs.stove_env   # This line just ensures registration runs

env = gym.make("stove-turnoff-v0")
obs, _ = env.reset()
print("observation: ", obs)
done = False

# Enviroment Checker!
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
input("Press any key to close")
env.close()

# while not done:


#     action = env.action_space.sample()
#     obs, reward, done, _, _ = env.step(action)
#     env.render()

