from ai2thor.controller import Controller

import matplotlib.pyplot as plt
import time

controller = Controller(
    agentMode="default",
    visibilityDistance=1.5,
    scene="FloorPlan1",

    # step sizes
    gridSize=0.25,
    snapToGrid=True,
    rotateStepDegrees=90,

    # image modalities
    renderDepthImage=False,
    renderInstanceSegmentation=False,

    # camera properties
    width=800,
    height=800,
    fieldOfView=90
)

def exploreRoom():
    controller.step("RotateLeft")
    time.sleep(1)
    for i in range(10):
        controller.step("MoveAhead")
        time.sleep(1)
    controller.step("RotateLeft")
    time.sleep(1)
    for i in range(8):
        controller.step("MoveAhead")
        time.sleep(1)
    controller.step("RotateLeft")
    controller.step("MoveAhead")
    controller.step("RotateRight")
    for i in range(2):
        controller.step("MoveAhead")
    controller.step("RotateLeft")
    for i in range(10):
        controller.step("MoveAhead")
    controller.step("RotateLeft")
    for i in range(10):
        controller.step("MoveAhead")

def navStove():
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
    if not controller.step("MoveAhead").metadata["lastActionSuccess"]:
        print("Final MoveAhead failed")
        controller.step("MoveAhead")

def turnOnStove():
    event = controller.step(action="Pass")
    for obj in event.metadata["objects"]:
        if (obj["visible"] and (obj["objectType"] == "StoveKnob")):
            print(obj["objectId"], obj["objectType"], obj["position"])
            time.sleep(2)
            controller.step(action="ToggleObjectOn", objectId=obj["objectId"])
            time.sleep(2)

def showMap():
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


# exploreRoom()
# navStove()
# turnOnStove()
showMap()

