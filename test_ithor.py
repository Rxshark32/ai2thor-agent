from ai2thor.controller import Controller

# Initialize the controller with a specific scene
controller = Controller(scene="FloorPlan1", renderDepthImage=False, renderInstanceSegmentation=False)

# Wait until the scene is ready before acting
controller.reset("FloorPlan1")
event = controller.step(action="MoveAhead")

print("Agent position after moving ahead:")
print(event.metadata["agent"]["position"])

controller.stop()

