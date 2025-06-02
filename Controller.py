from ai2thor.controller import Controller

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
    for i in range(10):
        controller.step("MoveAhead")
    controller.step("RotateLeft")
    for i in range(8):
        controller.step("MoveAhead")
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

exploreRoom()

input("Press Enter to exit.")

controller.stop()