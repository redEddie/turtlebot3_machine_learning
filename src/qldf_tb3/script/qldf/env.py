#!/usr/bin/python3

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from .respawnGoal import Respawn
from numpy import tanh


class Env:
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher("cmd_vel", Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber("odom", Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy("gazebo/reset_simulation", Empty)
        self.unpause_proxy = rospy.ServiceProxy("gazebo/unpause_physics", Empty)
        self.pause_proxy = rospy.ServiceProxy("gazebo/pause_physics", Empty)
        self.respawn_goal = Respawn()

        # self.reset_proxy()
        # self.respawn_goal.deleteModel()

    def getGoalDistace(self):
        goal_distance = round(
            math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2
        )
        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        goal_angle = math.atan2(
            self.goal_y - self.position.y, self.goal_x - self.position.x
        )

        heading = goal_angle - yaw  # angle error

        # -pi ~ pi로 제한
        if heading > pi:
            heading -= 2 * pi
        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def getState(self, scan, current_angular_z):
        scan_range = []
        linear_x = 0.15

        heading = self.heading
        min_range = 0.14
        collision = False

        # adds the scan data to the scan_range list(origianally 360 but reduced to 24)
        # https://emanual.robotis.com/docs/en/platform/turtlebot3/machine_learning/#set-state
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float("Inf"):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)

        if min_range > min(scan_range) > 0:
            collision = True

        current_distance = round(
            math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2
        )
        if current_distance < 0.2:
            self.get_goalbox = True

        return (
            scan_range
            + [
                heading,
                current_distance,
                obstacle_min_range,
                obstacle_angle,
                linear_x,
                current_angular_z,
            ],
            collision,
        )

    def setReward(self, state, collision, action):
        scan_range = state[:-6]
        current_distance = state[-5]  # distance from goal
        heading = state[-6]  # heading error

        action = action.item()
        angle = heading + (pi / 8 * (action - 2))
        if angle > pi:
            angle = angle - 2 * pi
        elif angle < -pi:
            angle = angle + 2 * pi
        normalize_error = 1 - 2 * np.abs(angle) / pi  # -1 ~ 0

        x = 2 * self.goal_distance - current_distance
        x = 1.5 * x / self.goal_distance  # 1 ~ ...

        y = min(scan_range)
        y_ref = 0.14 + 0.3
        y = 3 * (1 - math.exp(y_ref - y))  # 1-exp(0.3) = -1.35

        reward = min(normalize_error, x, y)

        if collision:
            reward = -500
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        if self.get_goalbox:
            reward = 500
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward

    def step(self, action):
        max_angular_vel = 1.5
        ang_vel = (action - 2) * (max_angular_vel * 0.5) * (-1)
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        current_ang_z = None
        while data is None:
            try:
                data = rospy.wait_for_message("scan", LaserScan, timeout=5)
            except:
                pass

        while current_ang_z is None:
            try:
                current_ang_z = rospy.wait_for_message("odom", Odometry, timeout=5)
                current_ang_z = current_ang_z.twist.twist.angular.z
            except:
                pass

        state, collision = self.getState(data, current_ang_z)
        reward = self.setReward(state, collision, action)

        return np.asarray(state), reward, collision

    def reset(self):
        rospy.wait_for_service("gazebo/reset_simulation")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message("scan", LaserScan, timeout=5)
            except:
                pass

        current_ang_z = None
        while current_ang_z is None:
            try:
                current_ang_z = rospy.wait_for_message("odom", Odometry, timeout=5)
                current_ang_z = current_ang_z.twist.twist.angular.z
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        self.goal_distance = self.getGoalDistace()
        state, collision = self.getState(data, current_ang_z)

        return np.asarray(state)
