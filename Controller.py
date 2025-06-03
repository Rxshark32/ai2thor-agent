from ai2thor.controller import Controller
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


# exploreRoom()
navStove()
turnOnStove()


input("Press Enter to break.")
controller.stop()