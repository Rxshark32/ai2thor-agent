import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

from agent.StoveAgent import StoveAgent
import numpy as np

# Registering module as gymenv
register(
    id="stove-turnoff-v0",
    entry_point="envs.stove_env:StoveEnv",
)


#Creating custom class
class StoveEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        # Create the AI2-THOR controller/agent
        self.agent = StoveAgent()

        # Define a discrete action space
        self.delta = 0.05
        self.action_space = spaces.Discrete(7)

        self.actions = {
            0: (1, 0, 0),
            1: (-1, 0, 0),
            2: (0, 1, 0),
            3: (0, -1, 0),
            4: (0, 0, 1),
            5: (0, 0, -1),
            6: "toggle"
        }


        # Define an observation space (example: stove status + hand position)
        # You can define more detailed obs later
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(800, 800, 3),
            dtype=np.uint8
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Close the existing controller if it exists
        if hasattr(self, 'agent') and self.agent and hasattr(self.agent, 'controller'):
            self.agent.controller.stop()

        # Reinitialize the agent and env
        self.agent = StoveAgent()
        self.agent.navigate_to_stove()
        self.agent.turn_on_stove()

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        if action in range(6):  # movement
            dx, dy, dz = self.actions[action]
            self.agent.move_arm(dx * 0.05, dy * 0.05, dz * 0.05)  # scaled deltas
        elif action == 6:  # toggle
            self.agent.toggle_object()

        obs = self._get_obs()
        reward, terminated = self._compute_reward()
        info = {}
        return obs, reward, terminated, False, info

    def _get_obs(self):
        """Return visual frame as numpy array."""
        frame = self.agent.controller.last_event.frame  # PIL.Image
        return np.array(frame)

    def _compute_reward(self):
        metadata = self.agent.controller.last_event.metadata
        hand = metadata["arm"]["handSphereCenter"]

        # Find closest visible stove knob
        knobs = [obj for obj in metadata["objects"]
                 if obj["objectType"] == "StoveKnob" and obj["visible"]]
        
        if knobs:
            closest = min(knobs, key=lambda k: np.linalg.norm([
                k["position"]["x"] - hand["x"],
                k["position"]["y"] - hand["y"],
                k["position"]["z"] - hand["z"]
            ]))
            dist = np.linalg.norm([
                closest["position"]["x"] - hand["x"],
                closest["position"]["y"] - hand["y"],
                closest["position"]["z"] - hand["z"]
            ])
        else:
            dist = 1.0  # fallback distance

        # Stove burners state
        stove_off = all(
            not obj["isToggled"]
            for obj in metadata["objects"]
            if obj["objectType"] == "StoveBurner"
        )

        reward = (1.0 - min(dist, 1.0)) + (1.0 if stove_off else 0.0)
        return reward, stove_off

    def render(self):
        if self.render_mode == "human":
            print("Rendering frame...")
