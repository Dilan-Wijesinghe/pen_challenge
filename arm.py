from asyncio import set_event_loop_policy
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import modern_robotics as mr
import math
import numpy as np
# The robot object is what you use to control the robot


class RobotMovement:
    def __init__(self):
        self.robot = InterbotixManipulatorXS("px100", "arm", "gripper")
        self.mode = 'h'

    # def RunCommands(self):
    # # Let the user select the position
    #     print(f"Home position of robot is: {self.robot.arm.robot_des.M}\n")
    #     print(self.robot.arm.group_info)
    #     while mode != 'q':
    #       mode=input("[h]ome, [s]leep, [r]elease, [g]rasp, [j]oint,, [i]nfo, [q]uit ")
            # if mode == "h":
            #     self.robot.arm.go_to_home_pose()
            # elif mode == "s":
            #     self.robot.arm.go_to_sleep_pose()
            # elif mode == 'r':
            #     self.robot.gripper.release()
            # elif mode == 'g':
            #     self.robot.gripper.grasp()
            # elif mode == 'j':
            #     print("Welcome to Single Joint Manipulation!")
            #     joint = input("['waist, 'shoulder', 'elbow', 'wrist_angle']")
            #     degrees = input("Give a Degrees to move")
            #     radians = float(degrees) * (math.pi/180)
            #     self.robot.arm.set_single_joint_position(joint, radians)
            # elif mode == 'm':
            #     print("Now we are movin' and groovin'")
            #     # robot.arm.set_joint_positions()
            # elif mode == 'i':
            #     print("Now getting End Effector Info")
            #     joints = self.robot.arm.get_joint_commands()
            #     T = mr.FKinSpace(self.robot.arm.robot_des.M, self.robot.arm.robot_des.Slist, joints)
            #     [R,p] = mr.TransToRp(T)
            #     print(R,p)
            # elif mode == "c":
            #     print("setting cartesian trajectory") 
            #     self.robot.arm.set_ee_cartesian_trajectory(0,0,1, yaw=(float(20) * math.pi/180), moving_time=2)

    def move(self, phi):
        if -3.141582727432251 < phi and phi < 3.141582727432251:
            self.robot.arm.set_single_joint_position('waist', -phi)
    
    def GoSleep(self):
        self.robot.arm.go_to_sleep_pose()
    
    def GoHome(self):
        self.robot.arm.go_to_home_pose()
    
    def Release(self):
        self.robot.gripper.release()

    def Grasp(self):
        self.robot.gripper.grasp()
    
    def SingleJointMove(self):
        mode = 'h'
        while mode != q:
            joint = input("['waist, 'shoulder', 'elbow', 'wrist_angle']\n")
            degrees = input ("Give a Degrees to move")
            radians = float(degrees) * (math.pi/180)
            self.robot.arm.set_single_joint_position(joint, radians)

    def GetEEInfo(self):
        # print("Now getting End Effector Info")
        joints = self.robot.arm.get_joint_commands()
        T = mr.FKinSpace(self.robot.arm.robot_des.M, self.robot.arm.robot_des.Slist, joints)
        [R,p] = mr.TransToRp(T)
        print(p)
        return p
    
    def ExtendArm(self, radius_camera):
        p = self.GetEEInfo()
        robot_radius = np.sqrt((p[0]**2)+ (p[1]**2))
        print(f"robo_rad {robot_radius}\n", f"cam_rad is {radius_camera}")
        displacement = radius_camera - robot_radius
        # displacement = robot_radius - radius_camera
        print(displacement)
        self.robot.arm.set_ee_cartesian_trajectory(x=displacement)
    
    def TestArmExtend(self, disp):
        self.robot.arm.set_ee_cartesian_trajectory(x=disp)




# RoboMoveStart = RobotMovement()
# RoboMoveStart.GoSleep()

# mode = 'h'
# while mode != 'q':
#     mode=input("[h]ome, [s]leep, [r]elease, [g]rasp, [j]oint,, [i]nfo, [q]uit\n")
#     if mode == "h":
#         RoboMoveStart.GoHome()
#     elif mode == "s":
#         RoboMoveStart.GoSleep()
#     elif mode == 'r':
#         RoboMoveStart.Release()
#     elif mode == 'g':
#         RoboMoveStart.Grasp()
#     elif mode == 'j':
#         RoboMoveStart.SingleJointMove()
#     elif mode == 'i':
#         RoboMoveStart.GetEEInfo()
#     elif mode == 'm':
#         d = input("Please give a displacement")
#         RoboMoveStart.TestArmExtend(float(d))