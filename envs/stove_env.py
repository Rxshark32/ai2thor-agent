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
        self.max_duration = 10  # seconds
        self.start_time = None

        # arm base init
        self.arm_base_y_pos = 0.6

        # Define a discrete action space
        self.delta = 0.05
        self.action_space = spaces.Discrete(9)

        self.actions = {
            0: (0.1, 0, 0.0),    # Move arm +x
            1: (-0.1, 0, 0.0),   # Move arm -x
            2: (0, 0.1, 0.0),    # Move arm +y
            3: (0, -0.1, 0.0),   # Move arm -y
            4: (0, 0, 0.1),      # Move arm +z
            5: (0, 0, -0.1),     # Move arm -z
            6: 0.1,              # Move arm base +y
            7: -0.1              # Move arm base -y
            # 8 is handled separately
        }

        # Agent arm gets stuck too much
        self.stuck_counter = 0
        self.prev_hand_pos = None

        self.observation_space = spaces.Box(
            low=np.array([-np.inf]*24, dtype=np.float32),
            high=np.array([np.inf]*24, dtype=np.float32),
            shape=(24,),
            dtype=np.float32
        )

        self.curr_hand_pos = self.agent.controller.last_event.metadata["arm"]["handSphereCenter"]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not self.agent:
            self.agent = StoveAgent()
        else:
            self.agent.env_randomizer()
            self.agent.man_teleport()
            self.agent.turn_on_stove()

        self.curr_hand_pos = {"x": 0.0, "y": 0.0, "z": 0.5}
        self.agent.move_arm(
            dx=self.curr_hand_pos["x"],
            dy=self.curr_hand_pos["y"],
            dz=self.curr_hand_pos["z"]
        )

        # Initialize arm base y pos here to the one from teleport or your default
        self.arm_base_y_pos = 0.6

        self.stuck_counter = 0
        self.prev_hand_pos = self.curr_hand_pos.copy()
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
            6: "Move arm base +y",
            7: "Move arm base -y",
            8: "Toggle stove knob"
        }

        if np.isscalar(action):
            actions = [int(action)]
        else:
            actions = action

        for a in actions:
            print(f"Agent action: {a} -> {action_meanings.get(a, 'Unknown action')}")
            if a in range(6):
                dx, dy, dz = self.actions[a]
                self.curr_hand_pos['x'] += dx
                self.curr_hand_pos['y'] += dy
                self.curr_hand_pos['z'] += dz
                self.agent.move_arm(
                    dx=self.curr_hand_pos["x"],
                    dy=self.curr_hand_pos["y"],
                    dz=self.curr_hand_pos["z"]
                )
            elif a in [6, 7]:
                dy = self.actions[a]
                self.arm_base_y_pos += dy
                self.arm_base_y_pos = np.clip(self.arm_base_y_pos, 0.0, 1.0)
                print(f"Attempting to move arm base to Y = {self.arm_base_y_pos}")
                success = self.agent.move_arm_base(self.arm_base_y_pos)
                print(f"MoveArmBase success: {success}")
            elif a == 8:
                self.agent.toggle_object()
            self.agent.controller.step('Pass')

        # Check if hand is stuck (position not changing)
        curr_pos = self.agent.controller.last_event.metadata["arm"]["handSphereCenter"]
        prev_pos = self.prev_hand_pos

        if all(abs(curr_pos[axis] - prev_pos[axis]) < 1e-4 for axis in ['x','y','z']):
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        self.prev_hand_pos = curr_pos

        if self.stuck_counter >= 20:
            reward = -3.0
            terminated = True
            truncated = False
            obs = self._get_obs()
            info = {"stuck_termination": True}
            print("Agent stuck for 20 steps. Terminating episode with penalty reward.")
            return obs, reward, terminated, truncated, info

        obs = self._get_obs()
        reward, terminated = self._compute_reward()
        elapsed_time = time.time() - self.start_time
        truncated = elapsed_time >= self.max_duration

        print(f"Reward obtained: {reward}")
        info = {"elapsed_time": elapsed_time}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        MAX_KNOBS = 6
        obs = []
        metadata = self.agent.controller.last_event.metadata['objects']

        stoveknob_list = [obj for obj in metadata if 'StoveKnob' in obj['name']]

        event = self.agent.controller.last_event
        hand_pos = event.metadata["arm"]['handSphereCenter']
        obs.extend([hand_pos['x'], hand_pos['y'], hand_pos['z']])

        knob_pos = [obj['position'] for obj in stoveknob_list]

        for i in range(MAX_KNOBS):
            if i < len(knob_pos):
                k = knob_pos[i]
                obs.extend([
                    abs(k['x'] - hand_pos['x']),
                    abs(k['y'] - hand_pos['y']),
                    abs(k['z'] - hand_pos['z'])
                ])
            else:
                obs.extend([0.0, 0.0, 0.0])

        tot_dist = [((abs(k['x']-hand_pos['x']))**2+(abs(k['y']-hand_pos['y']))**2+(abs(k['z']-hand_pos['z']))**2)**0.5 for k in knob_pos]
        obs.append(min(tot_dist) if tot_dist else 0.0)

        stove_tog = [obj['isToggled'] for obj in stoveknob_list]
        amt = sum(not v for v in stove_tog)
        obs.append(amt)

        obs.append(self.arm_base_y_pos)  # include arm base position

        return np.array(obs, dtype=np.float32)

    def _compute_reward(self):
        metadata = self.agent.controller.last_event.metadata
        hand_pos = metadata["arm"]["handSphereCenter"]
        knobs = [obj for obj in metadata["objects"] if obj["objectType"] == "StoveKnob" and obj["visible"]]

        if not knobs:
            return -0.2, False

        distances = [
            ((k["position"]["x"] - hand_pos["x"])**2 +
            (k["position"]["y"] - hand_pos["y"])**2 +
            (k["position"]["z"] - hand_pos["z"])**2) ** 0.5
            for k in knobs
        ]

        closest_dist = min(distances)
        norm_dist = min(closest_dist, 1.0)
        distance_reward = np.exp(-5 * norm_dist)

        knobs_on = [obj.get("isToggled", False) for obj in knobs]
        num_on = sum(knobs_on)
        num_knobs = len(knobs)

        toggle_reward = 1.0 - 2.0 * (num_on / num_knobs)
        success = (num_on == 0)

        reward = 0.4 * toggle_reward + 0.6 * distance_reward
        reward -= 0.1

        if success:
            reward += 2.0

        elapsed_time = time.time() - self.start_time
        reward -= 0.001 * elapsed_time

        reward = max(-2.0, min(2.0, reward))

        return reward, success

    def render(self):
        if self.render_mode == "human":
            print("Rendering frame...")
