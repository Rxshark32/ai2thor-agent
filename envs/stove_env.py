import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.StoveAgent import StoveAgent
import numpy as np

# Registering module as gymenv
register(
    id="stove-turnoff-v0",
    entry_point="envs.stove_env:StoveEnv",
    max_episode_steps=100
)

#Creating custom class
class StoveEnv(gym.Env):
    def __init__(self, render_mode=None, arm_base_y_pos = 0.6):
        self.render_mode = render_mode
        self.agent = StoveAgent()
        self.arm_base_y_pos = arm_base_y_pos
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
        }

        # Agent arm gets stuck too much
        self.stuck_counter = 0
        self.prev_hand_pos = None

        self.step_count = 0
        self.max_steps = 150
        # trackers
        self.knob_tracker = {}

        self.observation_space = spaces.Box(
            low=np.array([-np.inf]*6, dtype=np.float32),
            high=np.array([np.inf]*6, dtype=np.float32),
            shape=(6,),
            dtype=np.float32
        )

        self.curr_hand_pos = self.agent.controller.last_event.metadata["arm"]["handSphereCenter"]

    def _get_obs(self,success = True):
        event = self.agent.controller.last_event.metadata
        obs = []
        hand_pos = event['arm']['handSphereCenter']
        obs.append(hand_pos)

        # Needs to be both off and toggled
        stoveknob_list = [obj['position'] for obj in event['objects'] 
                        if 'StoveKnob' in obj['name']
                        and obj['visible'] == True
                        and obj['isToggled'] == True
                        ]

        hand_vec = np.array([hand_pos['x'], hand_pos['y'], hand_pos['z']])

        def euclidean_distance(knob):
            knob_vec = np.array([knob['x'], knob['y'], knob['z']])
            return np.linalg.norm(hand_vec - knob_vec)

        # closest_knob = min(stoveknob_list, key=euclidean_distance)
        # min_dist = euclidean_distance(closest_knob)

        # print('Hand position', hand_pos)
        # print("Min distance:", min_dist)
        # print("Closest knob position:", closest_knob)
        # print("Untoggled knobs:", len(stoveknob_list))

        # OBSERVATAION SPACE RN:
        # closest_knob_dx, closest_knob_dy, closest_knob_dz,
        # min_euclid_dist, amount_of_untoggled_knobs
        # last_tog_success

        if stoveknob_list:
            closest_knob = min(stoveknob_list, key=euclidean_distance)
            min_dist = euclidean_distance(closest_knob)
            knob_dx, knob_dy, knob_dz = closest_knob['x'] - hand_pos['x'], closest_knob['y'] - hand_pos['y'], closest_knob['z'] - hand_pos['z']
            knob_count = len(stoveknob_list)
        else:
            knob_dx, knob_dy, knob_dz = 0.0, 0.0, 0.0
            min_dist = 10.0
            knob_count = 0

        obs = np.array([
            float(knob_dx), float(knob_dy), float(knob_dz),
            float(min_dist),
            float(knob_count),
            float(success)
        ], dtype=np.float32)

        return obs

    def _get_info(self):
        # Debugging Stuff
        return {"Hello": 67}

    def _get_reward(self):
        base_penalty = -0.02
        reward = base_penalty
        terminated = False

        metadata_objects = self.agent.controller.last_event.metadata["objects"]
        hand_pos = self.agent.controller.last_event.metadata["arm"]["handSphereCenter"]

        all_knobs_off = True  # start assuming all are off
        min_dist_to_any_knob = float('inf')

        for obj in metadata_objects:
            if "stove" in obj["objectId"].lower() and "knob" in obj["objectId"].lower():
                obj_id = obj["objectId"]

                # Initialize tracking if not done yet
                if obj_id not in self.knob_tracker:
                    self.knob_tracker[obj_id] = {"touched": False, "toggled": False}

                # If any knob is still toggled ON, mark all_knobs_off = False
                if obj.get("isToggled", False):
                    all_knobs_off = False

                dist = sum(
                    (hand_pos[axis] - obj["position"][axis]) ** 2 for axis in ["x", "y", "z"]
                ) ** 0.5
                min_dist_to_any_knob = min(min_dist_to_any_knob, dist)

                if dist < 0.08 and not self.knob_tracker[obj_id]["touched"]:
                    reward += 0.5
                    self.knob_tracker[obj_id]["touched"] = True
                    print(f"🟦TOUCHED STOVE ({hand_pos['x']:.3f}, {hand_pos['y']:.3f}, {hand_pos['z']:.3f})")


                if not obj.get("isToggled", True) and not self.knob_tracker[obj_id]["toggled"]:
                    reward += 5.0
                    self.knob_tracker[obj_id]["toggled"] = True
                    print("🟥TOGGLE")

        # Add final reward and terminate episode if all knobs off
        if all_knobs_off and not getattr(self, "episode_completed", False):
            reward += 5.0
            terminated = True
            self.episode_completed = True
            print("🟦🟦🟦🟦🟦🟦🟦🟦🟦ALL STOVES OFF🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥")

        # Your existing penalty adjustment logic (optional)
        if min_dist_to_any_knob < 0.10:
            reward = max(reward, +0.01)
        elif min_dist_to_any_knob < 0.20:
            reward = max(reward, -0.01)
        else:
            reward = max(reward, -0.02)

        return reward, terminated

    def step(self, action):

        # Placeholders will implement later!!!
        terminated = False
        truncated = False
        info = self._get_info()

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

        if isinstance(action, (int, np.integer)):
            actions = [int(action)]
        elif isinstance(action, np.ndarray) and action.ndim == 0:
            actions = [int(action.item())]
        else:
            actions = list(action)

        for a in actions:
            print(f"Agent action: {a} -> {action_meanings.get(a, 'Unknown action')}")
            if a in range(6):
                dx, dy, dz = self.actions[a]
                new_pos = {
                    'x': self.curr_hand_pos['x'] + dx,
                    'y': self.curr_hand_pos['y'] + dy,
                    'z': self.curr_hand_pos['z'] + dz,
                }
                success = self.agent.move_arm(
                    dx=new_pos['x'],
                    dy=new_pos['y'],
                    dz=new_pos['z']
                )
                if success:
                    self.curr_hand_pos = new_pos
            elif a in [6, 7]:
                dy = self.actions[a]
                self.arm_base_y_pos += dy
                self.arm_base_y_pos = max((min(self.arm_base_y_pos, 1.0)), 0.0)
                print(f"Attempting to move arm base to Y = {self.arm_base_y_pos}")
                success = self.agent.move_arm_base(self.arm_base_y_pos)
                if (success == False):
                    self.arm_base_y_pos -= dy
                print(f"MoveArmBase success: {success}")
            elif a == 8:
                success = self.agent.toggle_object()

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
            obs = self._get_obs()
            info = {"stuck_termination": True}
            print("Agent stuck for 20 steps. Terminating episode with penalty reward.")
            return obs, reward, terminated, truncated, info

        obs = self._get_obs(success)
        reward, terminated = self._get_reward()

        # Last Action Success Penalty!
        if success == False:
            print("❌ FAILED LAST ACTION! -0.1 ❌")
            reward += -0.1

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
            print("STEP LIMIT REACHED. 150 Truncating episode.")

        print(f"Reward obtained: {reward}")
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super_return = super().reset(seed=seed)  # Gymnasium expects you call this

        if not self.agent:
            self.agent = StoveAgent()
        else:
            self.agent.env_randomizer(seed=seed)
            self.agent.man_teleport()
            self.agent.turn_on_stove()

        self.episode_completed = False
        self.curr_hand_pos = {"x": 0.0, "y": 0.0, "z": 0.5}
        self.arm_base_y_pos = 0.6
        self.stuck_counter = 0
        self.prev_hand_pos = {
            "x": float(self.curr_hand_pos["x"]),
            "y": float(self.curr_hand_pos["y"]),
            "z": float(self.curr_hand_pos["z"])
        }
        self.step_count = 0
        self.knob_tracker = {}

        # Gymnasium reset returns obs, info
        obs = self._get_obs()
        info = {}

        # If super().reset() returns obs, info, unpack and replace if needed
        if isinstance(super_return, tuple) and len(super_return) == 2:
            obs, info = super_return

        return obs, info

    def render(self):
        if self.render_mode == "human":
            print("Rendering frame...")
