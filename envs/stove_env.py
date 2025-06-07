import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
        
        # REAL TIME LIM!
        self.max_duration = 20  # seconds
        self.start_time = None

        # Define a discrete action space
        self.delta = 0.05
        self.action_space = spaces.MultiDiscrete([7, 7, 7, 7, 7])

        self.actions = {
            0: (0.1, 0, 0.5),   # Move arm +x
            1: (-0.1, 0, 0.5),  # Move arm -x
            2: (0, 0.1, 0.5),   # Move arm +y
            3: (0, -0.1, 0.5),  # Move arm -y
            4: (0, 0, 0.6),     # Move arm +z
            5: (0, 0, 0.4),     # Move arm -z
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

        # Close the existing controller if it exists (otherwise windows pile up!)
        if hasattr(self, 'agent') and self.agent and hasattr(self.agent, 'controller'):
            self.agent.controller.stop()

        # Reinitialize the agent and env
        self.agent = StoveAgent()
        self.agent.navigate_to_stove()
        self.agent.turn_on_stove()
        self.start_time = time.time()


        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):

        action_meanings = {
            0: "Move arm +x",
            1: "Move arm -x",
            2: "Move arm +y",
            3: "Move arm -y",
            4: "Move arm +z",
            5: "Move arm -z",
            6: "Toggle stove knob"
        }

        if np.isscalar(action):
            actions = [int(action)]
        else:
            actions = action

        for a in actions:
            print(f"Agent action: {a} -> {action_meanings.get(a, 'Unknown action')}")
            if a in range(6):
                dx, dy, dz = self.actions[a]
                self.agent.move_arm(dx, dy, dz)
            elif a == 6:
                self.agent.toggle_object()

        obs = self._get_obs()
        reward, terminated = self._compute_reward()
        elapsed_time = time.time() - self.start_time
        truncated = elapsed_time >= self.max_duration

        print(f"Reward obtained: {reward}")
        info = {"elapsed_time": elapsed_time}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """Return visual frame as numpy array."""
        frame = self.agent.controller.last_event.frame
        return np.array(frame)

    def _compute_reward(self):
        metadata = self.agent.controller.last_event.metadata
        hand = metadata["arm"]["handSphereCenter"]

        knobs = [obj for obj in metadata["objects"]
                if obj["objectType"] == "StoveKnob" and obj["visible"]]

        if knobs:
            distances = []
            for k in knobs:
                dx = k["position"]["x"] - hand["x"]
                dy = k["position"]["y"] - hand["y"]
                dz = k["position"]["z"] - hand["z"]
                dist = (dx**2 + dy**2 + dz**2) ** 0.5
                distances.append(dist)
            closest_dist = min(distances)
        else:
            closest_dist = 1.0

        norm_dist = min(closest_dist, 1.0)
        reward = 1.0 - 2 * norm_dist

        # Check stove knob toggle states
        toggled_states = [
            obj.get("isToggled", False)
            for obj in metadata["objects"]
            if obj["objectType"] == "StoveKnob" and obj["visible"]
        ]
        if any(not state for state in toggled_states):
            return reward + 0.5, True
        
        reward = max(-1.0, min(reward, 1.0))

        terminated = False  # just keep running until tha thing finally turns off the knob....

        return reward, terminated

    def render(self):
        if self.render_mode == "human":
            print("Rendering frame...")
