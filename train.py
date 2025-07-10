import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv

# Register your env here (or import from a module that does)
from envs.stove_env import StoveEnv
gym.register(
    id="stove-turnoff-v0",
    entry_point="envs.stove_env:StoveEnv",
)

def make_env(rank):
    def _init():
        return gym.make("stove-turnoff-v0")
    return _init

if __name__ == "__main__":
    import envs.stove_env
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    import os

    class TensorboardFlushAndCheckpointCallback(BaseCallback):
        def __init__(self, flush_freq=2048, save_path="models/ppo_stove_turnoff", verbose=0):
            super().__init__(verbose)
            self.flush_freq = flush_freq
            self.save_path = save_path
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        def _on_step(self) -> bool:
            if self.num_timesteps % self.flush_freq == 0:
                self.logger.dump(self.num_timesteps)
                self.model.save(self.save_path)
                if self.verbose > 0:
                    print(f"Flushed TensorBoard and saved model at step {self.num_timesteps}")
            return True

    def make_env(seed):
        def _init():
            env = gym.make("stove-turnoff-v0")
            env = Monitor(env)
            env.reset(seed=seed)
            return env
        return _init

    num_envs = 4
    envs = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # model = PPO(
    #     "MlpPolicy",
    #     envs,
    #     verbose=1,
    #     tensorboard_log="./ppo_stove_tensorboard/",
    #     n_steps=2048,
    #     batch_size=512
    # )

    model = PPO.load("models/ppo_stove_turnoff", env=envs)

    callback = TensorboardFlushAndCheckpointCallback(
        flush_freq=2048,
        save_path="models/ppo_stove_turnoff",
        verbose=1
    )

    model.learn(total_timesteps=1_000_000, callback=callback)
    model.save("models/ppo_stove_turnoff_final")
