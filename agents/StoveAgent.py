from ai2thor.controller import Controller
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from collections import deque
import random


# StoveAgent class for OOP
class StoveAgent:
    def __init__(self, scene="FloorPlan1", width=800, height=800):
        self.controller = Controller(
            agentMode="arm",
            massThreshold=None,
            scene=scene,
            visibilityDistance=1.5,
            gridSize=0.25,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            width=width,
            height=height,
            fieldOfView=60
        )

        hand_pos = self.controller.last_event.metadata["arm"]["handSphereCenter"]
    
    def navigate_to_stove(self):
        self.controller.step("RotateLeft")
        for _ in range(10):
            self.controller.step("MoveAhead")
        self.controller.step("RotateLeft")
        for _ in range(3):
            self.controller.step("MoveAhead")
        self.controller.step("RotateRight")
        self.controller.step("LookDown")
        self.controller.step("MoveAhead")
        self.controller.step("MoveAhead")
        self.controller.step("Pass")

    def get_closest_toggleable_object(self, radius=0.08):
        hand_pos = self.controller.last_event.metadata["arm"]["handSphereCenter"]
        
        def dist(obj):
            obj_pos = obj["position"]
            return np.linalg.norm([
                obj_pos["x"] - hand_pos["x"],
                obj_pos["y"] - hand_pos["y"],
                obj_pos["z"] - hand_pos["z"]
            ])

        toggleable_objects = [
            obj for obj in self.controller.last_event.metadata["objects"]
            if obj["visible"] and obj["toggleable"]
        ]

        if not toggleable_objects:
            return None

        closest_obj = min(toggleable_objects, key=dist)

        if dist(closest_obj) < radius:
            return closest_obj["objectId"]
        else:
            return None
    
    def turn_on_stove(self):
        event = self.controller.step("Pass")
        for obj in event.metadata["objects"]:
            if obj["visible"] and obj["objectType"] == "StoveKnob":
                self.controller.step("ToggleObjectOn", objectId=obj["objectId"])
                self.controller.step("Pass")

    def show_map(self):
        positions = self.controller.step("GetReachablePositions")
        reachable = positions.metadata["actionReturn"]

        x_vals = [p['x'] for p in reachable]
        z_vals = [p['z'] for p in reachable]

        objects = positions.metadata["objects"]
        x_obj = [obj["position"]["x"] for obj in objects]
        z_obj = [obj["position"]["z"] for obj in objects]
        labels = [obj["objectType"] for obj in objects]

        plt.figure(figsize=(8, 8))
        plt.scatter(x_vals, z_vals, c='blue', marker='s', label="Reachable")
        for i, label in enumerate(labels):
            plt.annotate(label, (x_obj[i], z_obj[i]), textcoords="offset points", xytext=(5, 5), ha='left')
        plt.title("Reachable Positions (x-z grid)")
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.show()

    def move_arm(self, dx = 0, dy = 0, dz = 0):
        # Absolute Target Position
        new_pos = {
            'x': dx,
            'y': dy,
            'z': dz
        }
        
        event = self.controller.step(
            action = "MoveArm",
            position = new_pos,
            coordinateSpace = "armBase",
            restrictMovement = True,
            speed = 1,
            returnToStart = False,
            fixedDeltaTime = 0.02
        )

        self.controller.step("Pass")

        return event.metadata["lastActionSuccess"]
    
    def move_arm_base(self, y=0.5):
        print(f"[MoveArmBase] Moving to y = {y}")
        event = self.controller.step(
            action="MoveArmBase",
            y=y,
            speed=1,
            returnToStart=False,
            fixedDeltaTime=0.02
        )
        self.controller.step("Pass")
        return event.metadata.get("lastActionSuccess", False)

    def toggle_object(self):
        
        # Toggles the closest toggleable object within hand_radius

        obj_id = self.get_closest_toggleable_object()
        if obj_id:
            event = self.controller.step(action="ToggleObjectOff", objectId=obj_id)
            self.controller.step("Pass")
            return event.metadata["lastActionSuccess"]
        else:
            print("No toggleable object close enough to toggle.")
            return False

    def print_handloc(self):
        self.controller.step("Pass")
        metadata = self.controller.last_event.metadata
        handpos = metadata["arm"]["handSphereCenter"]
        print('Hand Pos:' + str(handpos))

    def auto_navigate(self):
        self.controller.step(action="MoveArmBase", y=0.5, speed=1, returnToStart=True, fixedDeltaTime=0.02)
        self.controller.step(action="MoveArm", position={'x': 0.0, 'y': 0.4, 'z': 0.35}, coordinateSpace="armBase", restrictMovement=False, speed=0.01, returnToStart=False, fixedDeltaTime=0.02)
        self.controller.step('Pass')

        def snap_to_grid(pos, grid_size=0.25):
            return {'x': round(pos['x'] / grid_size) * grid_size, 'y': pos['y'], 'z': round(pos['z'] / grid_size) * grid_size}

        def find_closest_y(x, z, reachable_positions):
            for p in reachable_positions:
                if abs(p['x'] - x) < 1e-3 and abs(p['z'] - z) < 1e-3:
                    return p['y']
            return None

        def pos_check(target_pos, positions):
            return any(abs(pos['x'] - target_pos['x']) < 1e-3 and abs(pos['z'] - target_pos['z']) < 1e-3 for pos in positions)

        def pos_key(pos):
            return (round(pos['x'], 3), round(pos['z'], 3))

        def bfs_around_target(target_pos, positions, grid_size=0.25):
            visited = set()
            q = deque([target_pos])
            directions = [
                {'x': -grid_size, 'z': 0}, {'x': grid_size, 'z': 0},
                {'x': 0, 'z': grid_size}, {'x': 0, 'z': -grid_size}
            ]
            while q:
                curr = q.popleft()
                for d in directions:
                    next_pos = {'x': round(curr['x'] + d['x'], 3), 'y': target_pos['y'], 'z': round(curr['z'] + d['z'], 3)}
                    next_next_pos = {'x': round(curr['x'] + 2 * d['x'], 3), 'y': target_pos['y'], 'z': round(curr['z'] + 2 * d['z'], 3)}
                    if pos_key(next_next_pos) not in visited:
                        visited.add(pos_key(next_next_pos))
                        q.append(next_pos)
                        if pos_check(next_pos, positions) and pos_check(next_next_pos, positions):
                            return next_next_pos
            return None

        reachable = self.controller.step("GetReachablePositions").metadata["actionReturn"]
        target_obj = next(obj for obj in self.controller.last_event.metadata['objects'] if "Stove" in obj['objectType'])
        target_obj['position'] = snap_to_grid(target_obj['position'])
        target_obj['rotation']['y'] -= 180

        nearest_valid = bfs_around_target(target_obj["position"], reachable)

        if nearest_valid:
            closest_y = find_closest_y(nearest_valid['x'], nearest_valid['z'], reachable)
            if closest_y is not None:
                nearest_valid['y'] = closest_y
            rotation_y = round((target_obj['rotation']['y'] + 360) % 360 / 90) * 90 % 360
            event = self.controller.step(
                action="Teleport",
                position=nearest_valid,
                rotation={"x": 0, "y": rotation_y, "z": 0}
            )
            if not event.metadata["lastActionSuccess"]:
                print("False teleport.")
        else:
            print("No valid teleport location.")

        self.controller.step('Pass')
        event = self.controller.step("Pass")
        stove_knob = next((obj for obj in event.metadata["objects"] if obj["objectType"] == "StoveKnob"), None)

        if stove_knob:
            dx = stove_knob["position"]["x"] - event.metadata["agent"]["position"]["x"]
            dz = stove_knob["position"]["z"] - event.metadata["agent"]["position"]["z"]
            angle_rad = math.atan2(dx, dz)
            angle_deg = (math.degrees(angle_rad) + 360) % 360
            rounded_angle = round(angle_deg / 45) * 45 % 360
            self.controller.step(action="Teleport", rotation={"x": 0, "y": rounded_angle, "z": 0})
        else:
            print("No stove knob found in metadata.")

        self.controller.step("LookDown")
        self.controller.step("Pass")

    def env_randomizer(self, seed=None):
        if seed is not None:
            random.seed(seed)
        # Only pick from FloorPlan7, FloorPlan10, FloorPlan11
        possible_scenes = ["FloorPlan7", "FloorPlan10", "FloorPlan11"]
        scene_name = random.choice(possible_scenes)
        print(scene_name)
        self.controller.reset(scene=scene_name)
    
    def man_teleport(self):
        perfect_agent_poses = {
            "FloorPlan1_physics": {
                "position": {'x': -0.25, 'y': 0.901, 'z': -1.5},
                "rotation": {'x': 0, 'y': 0, 'z': 0},
                "arm_base_y": 0.3
            },
            "FloorPlan2_physics": {
                "position": {'x': 1.0, 'y': 0.901, 'z': 0.5},
                "rotation": {'x': 0, 'y': 270, 'z': 0},
                "arm_base_y": 0.4
            },
            "FloorPlan3_physics": {
                "position": {'x': -0.5, 'y': 1.123, 'z': -2.0},
                "rotation": {'x': 0, 'y': 180, 'z': 0},
                "arm_base_y": 0.6
            },
            "FloorPlan4_physics": {
                "position": {'x': -3.25, 'y': 0.901, 'z': 1.5},
                "rotation": {'x': 0, 'y': 180, 'z': 0},
                "arm_base_y": 0.4
            },
            "FloorPlan5_physics": {
                "position": {'x': -0.25, 'y': 0.901, 'z': -1.0},
                "rotation": {'x': 0, 'y': 180, 'z': 0},
                "arm_base_y": 0.4
            },
            # Testing Floorplans
            "FloorPlan7_physics": {
                "position": {'x': 1.25, 'y': 0.901, 'z': -0.5},
                "rotation": {'x': 0, 'y': 180, 'z': 0},
                "arm_base_y": 0.3
            },
            "FloorPlan10_physics": {
                "position": {'x': 0.0, 'y': 0.901, 'z': -1.25},
                "rotation": {'x': 0, 'y': 0, 'z': 0},
                "arm_base_y": 0.4
            },
            "FloorPlan11_physics": {
                "position": {'x': 1.25, 'y': 0.901, 'z': -0.75},
                "rotation": {'x': 0, 'y': 180, 'z': 0},
                "arm_base_y": 0.4
            }
        }

        floorplan_name = self.controller.last_event.metadata['sceneName']
        if floorplan_name in perfect_agent_poses:
            pose = perfect_agent_poses[floorplan_name]
            self.controller.step(action="MoveArmBase", y=pose['arm_base_y'], speed=1, returnToStart=True, fixedDeltaTime=0.02)
            event = self.controller.step(
                action="Teleport",
                position=pose['position'],
                rotation=pose['rotation']
            )
            if event.metadata["lastActionSuccess"]:
                print(f"Teleported successfully to {floorplan_name}")
            else:
                print(f"Teleport failed on {floorplan_name}")
        else:
            print(f"No hardcoded pose found for {floorplan_name}")

        event = self.controller.step("Pass")
        stove_knob = next((obj for obj in event.metadata["objects"] if obj["objectType"] == "StoveKnob"), None)

        if stove_knob:
            dx = stove_knob["position"]["x"] - event.metadata["agent"]["position"]["x"]
            dz = stove_knob["position"]["z"] - event.metadata["agent"]["position"]["z"]
            angle_rad = math.atan2(dx, dz)
            angle_deg = (math.degrees(angle_rad) + 360) % 360
            rounded_angle = round(angle_deg / 90) * 90 % 360
            print(f"Rotating to face stove knob approx. {rounded_angle}°")
            
            self.controller.step(
                action="Teleport",
                rotation={"x": 0, "y": rounded_angle, "z": 0}
            )
        else:
            print("No stove knob found in metadata.")

        self.controller.step("LookDown")
        self.controller.step("Pass")
        