from ai2thor.controller import Controller
import numpy as np
import matplotlib.pyplot as plt
import time

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

    def get_closest_toggleable_object(self, radius=0.05):
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
                print(obj["objectId"], obj["objectType"], obj["position"])
                self.controller.step("ToggleObjectOn", objectId=obj["objectId"])
                self.controller.step("Pass")

    def turn_off_stove(self):
        event = self.controller.last_event
        print("Initial hand position:", event.metadata["arm"]["handSphereCenter"])

        self.controller.step(
            action="MoveArm",
            position={'x': 0, 'y': 0, 'z': 0.5},
            coordinateSpace="armBase",
            restrictMovement=False,
            speed=1,
            returnToStart=True,
            fixedDeltaTime=0.02
        )
        self.controller.step("Pass")

        self.controller.step(
            action="MoveArm",
            position={'x': -0.1, 'y': 0, 'z': 0.5},
            coordinateSpace="armBase",
            restrictMovement=False,
            speed=1.0,
            returnToStart=False,
            fixedDeltaTime=0.02
        )
        self.controller.step("Pass")

        obj_id = self.get_closest_toggleable_object()
        if obj_id:
            event = self.controller.step(action="ToggleObjectOff", objectId=obj_id)
            print("Toggled", obj_id, "Success?", event.metadata["lastActionSuccess"])
        else:
            print("No toggleable object close enough to hand.")
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

    def move_arm(self, dx=0, dy=0, dz=0, speed=1.0):
        # Get current hand position
        new_pos = {
            'x': dx,  # interpret dx, dy, dz as absolute target position offsets now
            'y': dy,
            'z': dz
        }

        # Perform the move arm step with fixedDeltaTime for smooth small movement
        event = self.controller.step(
            action="MoveArm",
            position=new_pos,
            coordinateSpace="armBase",
            restrictMovement=False,
            speed=speed,
            returnToStart=False,
            fixedDeltaTime=0.02
        )

        # Pass step to process action
        self.controller.step("Pass")

        # Return success status of the action
        return event.metadata["lastActionSuccess"]

    def toggle_object(self):
        """
        Toggles the closest toggleable object within radius.
        """
        obj_id = self.get_closest_toggleable_object()
        if obj_id:
            event = self.controller.step(action="ToggleObjectOff", objectId=obj_id)
            self.controller.step("Pass")
            return event.metadata["lastActionSuccess"]
        else:
            print("No toggleable object close enough to toggle.")
            return False

