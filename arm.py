from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import math
# The robot object is what you use to control the robot



robot = InterbotixManipulatorXS("px100", "arm", "gripper")
mode = 'h'
# Let the user select the position
print(robot.arm.group_info)
while mode != 'q':
    mode=input("[h]ome, [s]leep, [r]elease, [g]rasp, [j]oint, [q]uit ")
    if mode == "h":
        robot.arm.go_to_home_pose()
    elif mode == "s":
        robot.arm.go_to_sleep_pose()
    elif mode == 'r':
        robot.gripper.release()
    elif mode == 'g':
        robot.gripper.grasp()
    elif mode == 'j':
        print("Welcome to Single Joint Manipulation!")
        joint = input("['waist, 'shoulder', 'elbow', 'wrist_angle']")
        degrees = input("Give a Degrees to move")
        radians = float(degrees) * (math.pi/180)
        robot.arm.set_single_joint_position(joint, radians)
    elif mode == 'm':
        print("Now we are movin' and groovin'")
        # robot.arm.set_joint_positions()