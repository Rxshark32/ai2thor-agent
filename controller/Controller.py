from ai2thor.controller import Controller
from ai2thor.util.metrics import get_shortest_path_to_object_type
import numpy as np
import math
import matplotlib.pyplot as plt
import time


def init_Agent():
    controller = Controller(
        agentMode="arm",
        massThreshold=None,
        scene="FloorPlan1",
        visibilityDistance=1.5,
        gridSize=0.25,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width=800,
        height=800,
        fieldOfView=60
    )
    controller.reset("FloorPlan1")
    return controller

def navStove(controller):
    controller.step("RotateLeft")
    for i in range(10):
        controller.step("MoveAhead")
    controller.step("RotateLeft")
    for i in range(3):
        controller.step("MoveAhead")
    controller.step("RotateRight")
    controller.step("LookDown")
    controller.step("MoveAhead")
    controller.step("MoveAhead")
    controller.step('Pass')

def get_closest_toggleable_object(controller, radius=0.05):
    hand_pos = controller.last_event.metadata["arm"]["handSphereCenter"]
    
    def dist(obj):
        obj_pos = obj["position"]
        return np.linalg.norm([
            obj_pos["x"] - hand_pos["x"],
            obj_pos["y"] - hand_pos["y"],
            obj_pos["z"] - hand_pos["z"]
        ])

    toggleable_objects = [
        obj for obj in controller.last_event.metadata["objects"]
        if obj["visible"] and obj["toggleable"]
    ]

    if not toggleable_objects:
        return None

    # Sort by distance to hand
    closest_obj = min(toggleable_objects, key=dist)

    if dist(closest_obj) < radius:
        return closest_obj["objectId"]
    else:
        return None

def turnOnStove(controller):
    event = controller.step(action="Pass")
    for obj in event.metadata["objects"]:
        if (obj["visible"] and (obj["objectType"] == "StoveKnob")):
            print(obj["objectId"], obj["objectType"], obj["position"])
            controller.step(action="ToggleObjectOn", objectId=obj["objectId"])
            controller.step('Pass')

def turnOffStove(controller):
    event = controller.last_event
    print(event.metadata["arm"]["handSphereCenter"])
    controller.step(
        action="MoveArm",
        position={'x': 0, 'y': 0, 'z': 0.5},
        coordinateSpace="armBase",
        restrictMovement=False,
        speed=1,
        returnToStart=True,
        fixedDeltaTime=0.02
    )
    controller.step('Pass')
    event = controller.last_event
    print(event.metadata["arm"]["handSphereCenter"])
    controller.step(
        action="MoveArm",
        position=dict(x=-0.1, y=0, z=0.5),
        coordinateSpace="armBase",
        restrictMovement=False,
        speed=1.0,
        returnToStart=False,
        fixedDeltaTime=0.02
    )
    controller.step('Pass')
    event = controller.last_event
    print(event.metadata["arm"]["handSphereCenter"])
    obj_id = get_closest_toggleable_object(controller)
    if obj_id:
        event = controller.step(action="ToggleObjectOff", objectId=obj_id)
        controller.step('Pass')
        print("Toggled", obj_id, "Success?", event.metadata["lastActionSuccess"])
    else:
        print("No toggleable object close enough to hand.")
    controller.step('Pass')

def showMap(controller):
    positions = controller.step(action="GetReachablePositions")
    x_vals = [pos['x'] for pos in positions.metadata["actionReturn"]]
    z_vals = [pos['z'] for pos in positions.metadata["actionReturn"]]

    # Made a temp list to store objects labels
    objlist = []
    for obj in positions.metadata['objects']:
        objlist.append((obj['objectType'], obj['position']))

    x_values = [obj[1]['x'] for obj in objlist]
    z_values = [obj[1]['z'] for obj in objlist]
    labels = [obj[0] for obj in objlist]

    # Plot maker
    plt.figure(figsize=(8, 8))
    plt.xlim(-2.5, max(x_values) + 1)
    plt.ylim(min(z_values)-0.5, max(z_values)+0.5)  
    plt.scatter(x_vals, z_vals, c='blue', marker='s')
    for i, label in enumerate(labels):
        plt.annotate(label, (x_values[i], z_values[i]), textcoords="offset points", xytext=(5,5), ha='left')
    plt.title("Reachable Positions (x-z grid)")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


#TESTING AREA!! CAUTION EVERYTHING BREAKS (T⌓T)
# controller = init_Agent()
# navStove(controller)
# turnOnStove(controller)
# controller.step('Pass')
# turnOffStove(controller)
# controller.step('Pass')
# controller.step('Pass')

# input("any key to stop")
# controller.stop()

