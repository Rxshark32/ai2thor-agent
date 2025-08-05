# Import Directory Fix (Since this is in a subdirectory)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Imports
import numpy as np
import gymnasium as gym
from agents.StoveAgent import StoveAgent

# Registers the env
gym.register(
    id = "stove-turnoff-v1",
    entry_point="envs.stove_env_v1:stove_env",
    max_episode_steps = 100
)

# Creating the Stove Enviroment
class stove_env(gym.Env):
    def __init__(self, arm_base_pos = 0.6):
        # Initialize StoveAgent
        self.agent = StoveAgent()
        self.agent_arm_base_pos = arm_base_pos
        self.agent_hand_sphere_pos = {"x":0.0, "y":0.0, "z":0.5}
        
        # Observation Space
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf]*8, dtype=np.float32),
            high=np.array([np.inf]*8, dtype=np.float32),
            shape=(8,),
            dtype=np.float32
        )

        # Action Space & Values
        self.action_space = gym.spaces.Discrete(7)

        self.action_value = {
            0: np.array([0.1, 0, 0]),   # Move arm left
            1: np.array([-0.1, 0, 0]),  # Move arm right
            2: np.array([0, 0.1, 0]),  # Move arm up
            3: np.array([0, -0.1, 0]),  # Move arm down
            4: np.array([0, 0, 0.1]),  # Move arm forward
            5: np.array([0, 0, -0.1]),  # Move arm back
            6: "toggle"
        }
        
    def _get_obs(self):
        event = self.agent.controller.last_event.metadata
        obs = []
        hand_pos = event['arm']['handSphereCenter']
        obs.append(hand_pos)

        # Needs to be both off and untoggled
        stoveknob_list = [obj['position'] for obj in event['objects'] 
                        if 'StoveKnob' in obj['name']
                        and obj['visible'] == True
                        and obj['isToggled'] == False
                        ]

        hand_vec = np.array([hand_pos['x'], hand_pos['y'], hand_pos['z']])

        def euclidean_distance(knob):
            knob_vec = np.array([knob['x'], knob['y'], knob['z']])
            return np.linalg.norm(hand_vec - knob_vec)

        closest_knob = min(stoveknob_list, key=euclidean_distance)
        min_dist = euclidean_distance(closest_knob)

        # print('Hand position', hand_pos)
        # print("Min distance:", min_dist)
        # print("Closest knob position:", closest_knob)
        # print("Untoggled knobs:", len(stoveknob_list))

        # OBSERVATAION SPACE RN:
        # hand_pos_x, hand_pos_y, hand_pos_z,
        # closest_knob_x, closest_knob_y, closest_knob_z,
        # min_euclid_dist, amount_of_untoggled_knobs
        # last_tog_success

        obs = np.array([
            hand_pos['x'], hand_pos['y'], hand_pos['z'],
            closest_knob['x'], closest_knob['y'], closest_knob['z'],
            min_dist,
            len(stoveknob_list)
        ], dtype=np.float32)

        return obs
        
    def _get_info(self):
        # Debugging Stuff
        return {"Hello": 67}
    
    def _get_reward(self):
        base_penalty = -0.03
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

                if dist < 0.09 and not self.knob_tracker[obj_id]["touched"]:
                    reward += 0.5
                    self.knob_tracker[obj_id]["touched"] = True
                    print(f"🟦TOUCHED STOVE ({hand_pos['x']:.3f}, {hand_pos['y']:.3f}, {hand_pos['z']:.3f})")


                if not obj.get("isToggled", True) and not self.knob_tracker[obj_id]["toggled"]:
                    reward += 5.0
                    self.knob_tracker[obj_id]["toggled"] = True
                    print("🟥TOGGLE")

        # Add final reward and terminate episode if all knobs off
        if all_knobs_off and not getattr(self, "episode_completed", False):
            reward += 30.0
            terminated = True
            self.episode_completed = True
            print("🟦🟦🟦🟦🟦🟦🟦🟦🟦ALL STOVES OFF🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥")

        # Your existing penalty adjustment logic (optional)
        if min_dist_to_any_knob < 0.10:
            reward = max(reward, -0.01)
        elif min_dist_to_any_knob < 0.20:
            reward = max(reward, -0.02)
        else:
            reward = max(reward, -0.03)

        return reward, terminated
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent.env_randomizer()
        self.agent.man_teleport()
        self.agent.turn_on_stove()
        
        # Robot Internal States Resets
        self.agent_hand_sphere_pos = {"x": 0.0, "y": 0.0, "z": 0.5}
        self.arm_base_pos = 0.6

        # Stuck Counter
        self.stuck_counter = 0
        self.prev_hand_pos = self.agent_hand_sphere_pos.copy()
        
        obs = self._get_obs()
        info = {}
        return obs, info
        
    def step(self, action):
        movement = self.action_value[action]
        print(f'Movement{movement}')
            
        terminated = False
        truncated = False

        reward = self._get_reward()
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info