from controller.Controller import init_Agent, navStove,turnOnStove, turnOffStove

def main():
    controller = init_Agent()
    navStove(controller)
    turnOnStove(controller)
    controller.step('Pass')
    turnOffStove(controller)
    controller.step('Pass')
    controller.step('Pass')

    input("any key to stop")
    controller.stop()

if __name__ == "__main__":
    main()
