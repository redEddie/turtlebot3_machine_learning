#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from .respawnGoal import Respawn  # no "." causes ImportError


class Env:
    def __init__(self, action_size):
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        self.previous_distance = 0

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
        _, _, yaw = euler_from_quaternion(orientation_list)  # roll, pitch, yaw

        goal_angle = math.atan2(
            self.goal_y - self.position.y, self.goal_x - self.position.x
        )

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def getState(self, scan):
        scan_range = []
        heading = self.heading
        min_range = 0.14
        collision = False

        # current_linear_x = 0.15

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float("Inf"):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        # obstacle_min_range = round(min(scan_range), 2)
        # obstacle_angle = np.argmin(scan_range)
        if min_range > min(scan_range) > 0:
            done = True

        current_distance = round(
            math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2
        )
        if current_distance < 0.2:
            self.get_goalbox = True

        # collision
        if min_range > min(scan_range) > 0:
            collision = True

        # print("current_linear_x: ", current_linear_x)
        # if current_linear_x == None:
        #     current_linear_x = 0
        # if current_angular_z == None:
        #     current_angular_z = 0

        return (
            scan_range
            + [
                heading,
                current_distance,
                # current_linear_x,
                # current_angular_z,
            ],
            collision,
        )  # state = 24 + 2

    def setReward(
        self, state, collision, linear_vel, ang_vel, steps, num_steps_per_iter
    ):
        scan_range = state[:-2]
        heading = state[-2]
        current_distance = state[-1]
        # self.goal_distance = self.getGoalDistace()
        # distance = self.goal_distance - current_distance

        angle = heading + ang_vel * 0.1
        if angle > 2 * pi:
            angle -= 2 * pi
        elif angle < -2 * pi:
            angle += 2 * pi

        print("distance: {:4f}, angle: {:4f}".format(current_distance, abs(angle)))
        reward = 3 * (current_distance) * (abs(angle))
        reward = 1 - round(reward, 2)

        if collision:
            print("[Learning] Collision!!")
            reward = -500
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        if self.get_goalbox:
            print("[Learning] Goal!!")
            reward = 1000
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        if (steps + 1) == num_steps_per_iter:
            print("[Learning] Time out!!")
            reward = -100
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward

    def step(self, action, steps, num_steps_per_iter):
        max_linear_vel = 0.22
        max_angular_vel = 2.84

        linear_vel = action[0] * max_linear_vel
        ang_vel = action[1] * max_angular_vel

        linear_vel = max(min(linear_vel, max_linear_vel), -max_linear_vel)
        ang_vel = max(min(ang_vel, max_angular_vel), -max_angular_vel)

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        # current_linear_x = None
        # current_ang_z = None
        while data is None:
            try:
                data = rospy.wait_for_message("scan", LaserScan, timeout=5)
            except:
                pass

        # while current_linear_x and current_ang_z is None:
        #     try:
        #         odom = rospy.wait_for_message("odom", Odometry, timeout=5)
        #         current_linear_x = odom.twist.twist.linear.x
        #         current_ang_z = odom.twist.twist.angular.z
        #     except:
        #         pass

        state, collision = self.getState(data)
        reward = self.setReward(
            state, collision, linear_vel, ang_vel, steps, num_steps_per_iter
        )

        return np.asarray(state), reward, collision

    def reset(self):
        rospy.wait_for_service("gazebo/reset_simulation")

        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        # current_linear_x = None
        # current_ang_z = None
        while data is None:
            try:
                data = rospy.wait_for_message("scan", LaserScan, timeout=5)
            except:
                pass

        # while current_linear_x and current_ang_z is None:
        #     try:
        #         odom = rospy.wait_for_message("odom", Odometry, timeout=5)
        #         current_linear_x = odom.twist.twist.linear.x
        #         # current_linear_y = odom.twist.twist.linear.y
        #         current_ang_z = odom.twist.twist.angular.z
        #     except:
        #         pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        self.goal_distance = self.getGoalDistace()
        state, collision = self.getState(data)

        return np.asarray(state)
