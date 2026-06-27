import math
import os
import random
import subprocess
import time
from os import path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import LaserScan

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.2 
TIME_DELTA = 0.1
MAX_ANGULAR = 1.0
MAX_LINEAR = 1.0 
MAX_RANGE = 3.0
reCL_STAGE = 1 #1 for constrained navigation

# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goal_ok = True

    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goal_ok = False

    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goal_ok = False

    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goal_ok = False

    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goal_ok = False

    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goal_ok = False

    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goal_ok = False

    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goal_ok = False

    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goal_ok = False

    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goal_ok = False

    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goal_ok = False

    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False

    return goal_ok

class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim

        self.pos_change = True
        self.recl_stage = 0 #0 for randomized

        #for reward function
        self.repeated_visits_count_0 = 0
        self.repeated_visits_count_1 = 0

        #robot0
        self.odom_x_0 = 0
        self.odom_y_0 = 0
        self.goal_x_0 = 1
        self.goal_y_0 = 0.0
        self.laserscan_data_0 = np.ones(self.environment_dim) * MAX_RANGE
        self.last_odom_0 = None

        self.set_self_state_0 = ModelState()
        self.set_self_state_0.model_name = "tb3_0"
        self.set_self_state_0.pose.position.x = 0.0
        self.set_self_state_0.pose.position.y = 0.0
        self.set_self_state_0.pose.position.z = 0.0
        self.set_self_state_0.pose.orientation.x = 0.0
        self.set_self_state_0.pose.orientation.y = 0.0
        self.set_self_state_0.pose.orientation.z = 0.0
        self.set_self_state_0.pose.orientation.w = 1.0
        
        #robot1
        self.odom_x_1 = 0
        self.odom_y_1 = 0
        self.goal_x_1 = 1
        self.goal_y_1 = 0.0
        self.laserscan_data_1 = np.ones(self.environment_dim) * MAX_RANGE
        self.last_odom_1 = None

        self.set_self_state_1 = ModelState()
        self.set_self_state_1.model_name = "tb3_1"
        self.set_self_state_1.pose.position.x = 0.0
        self.set_self_state_1.pose.position.y = 0.0
        self.set_self_state_1.pose.position.z = 0.0
        self.set_self_state_1.pose.orientation.x = 0.0
        self.set_self_state_1.pose.orientation.y = 0.0
        self.set_self_state_1.pose.orientation.z = 0.0
        self.set_self_state_1.pose.orientation.w = 1.0

        self.upper = 5.0
        self.lower = -5.0

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03
        
        port = "11311"
        
        env = os.environ.copy()
        env["ROS_MASTER_URI"] = "http://localhost:11311"
        env["ROS_HOSTNAME"] = "localhost"
        env["ROS_IP"] = "127.0.0.1"
        
        # Launch roscore and redirect stderr to /dev/null
        subprocess.Popen(["roscore", "-p", "11311"], env=env, stderr=subprocess.DEVNULL)

        print("Roscore launched!")

        # Launch the simulation with the given launchfile name
        rospy.init_node("gym", anonymous=True)
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")
            
        subprocess.Popen(["roslaunch", "-p", port, fullpath])

        #subprocess.Popen(["roslaunch", "-p", port, fullpath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Gazebo launched!")

        # Set up publishers and subscribers for tb3_0
        self.vel_pub_0 = rospy.Publisher("tb3_0/cmd_vel", Twist, queue_size=1)
        self.set_state_0 = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)
        self.laserscan_subscriber_0 = rospy.Subscriber("tb3_0/scan", LaserScan, self.laserscan_callback_0, queue_size=1)
        self.odom_subscriber_0 = rospy.Subscriber("tb3_0/odom", Odometry, self.odom_callback_0, queue_size=1)

        # Set up publishers and subscribers for tb3_1
        self.vel_pub_1 = rospy.Publisher("tb3_1/cmd_vel", Twist, queue_size=1)
        self.set_state_1 = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)
        self.laserscan_subscriber_1 = rospy.Subscriber("tb3_1/scan", LaserScan, self.laserscan_callback_1, queue_size=1)
        self.odom_subscriber_1 = rospy.Subscriber("tb3_1/odom", Odometry, self.odom_callback_1, queue_size=1)

        # Set up the ROS publishers and subscribers
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher4 = rospy.Publisher("goal_point2", MarkerArray, queue_size=3)

    def laserscan_callback_0(self, scan):
        # Initialize the laser scan data with max range values (environment_dim = 20 bins)
        self.laserscan_data_0 = np.ones(self.environment_dim) * scan.range_max
        
        # Calculate bin size based on the number of readings and environment dimension
        bin_size = max(1, len(scan.ranges) // self.environment_dim)  # Avoid division by zero

        # Process each reading in the front half of the scan
        for i in range(len(scan.ranges)):
            dist = scan.ranges[i]

        # Ignore invalid measurements (NaN or out of sensor's valid range)
            if np.isnan(dist) or dist < scan.range_min or dist > scan.range_max:
                continue

        # Calculate the bin index for the current reading
            bin_index = i // bin_size  # Determine the bin this reading belongs to

        # Store the minimum distance_0 for the bin
            if bin_index < self.environment_dim:
                self.laserscan_data_0[bin_index] = min(self.laserscan_data_0[bin_index], dist)

        # Print the processed laser scan data for debugging
        #print("Processed LaserScan Data:", self.laserscan_data_0)

    def laserscan_callback_1(self, scan):
        # Initialize the laser scan data with max range values (environment_dim = 20 bins)
        self.laserscan_data_1 = np.ones(self.environment_dim) * scan.range_max
        
        # Calculate bin size based on the number of readings and environment dimension
        bin_size = max(1, len(scan.ranges) // self.environment_dim)  # Avoid division by zero

        # Process each reading in the front half of the scan
        for i in range(len(scan.ranges)):
            dist = scan.ranges[i]

        # Ignore invalid measurements (NaN or out of sensor's valid range)
            if np.isnan(dist) or dist < scan.range_min or dist > scan.range_max:
                continue

        # Calculate the bin index for the current reading
            bin_index = i // bin_size  # Determine the bin this reading belongs to

        # Store the minimum distance_1 for the bin
            if bin_index < self.environment_dim:
                self.laserscan_data_1[bin_index] = min(self.laserscan_data_1[bin_index], dist)

        # Print the processed laser scan data for debugging
        #print("Processed LaserScan Data:", self.laserscan_data_1)

    def odom_callback_0(self, od_data):
        """Callback for odometry data of tb3_0."""
        self.last_odom_0 = od_data

    def odom_callback_1(self, od_data):
        """Callback for odometry data of tb3_1."""
        self.last_odom_1 = od_data

    # Perform an action and read a new state
    def step_0(self, action_0, timeout=False):
        target_0 = False

        # Publish the robot0 action
        vel_cmd_0 = Twist()
        vel_cmd_0.linear.x = action_0[0] * MAX_LINEAR
        vel_cmd_0.angular.z = action_0[1] * MAX_ANGULAR
        self.vel_pub_0.publish(vel_cmd_0)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # read laser state
        done_0, collision_0, min_laser_0 = self.observe_collision(self.laserscan_data_0)
        
        #if collision_0:
            #print("Collision detected! Min laser distance: {:.2f}".format(min_laser_0))
        
        v_state = []
        v_state[:] = self.laserscan_data_0[:]
        laser_state = [v_state]

        # Calculate robot heading from odometry data
        current_odom_0_x = self.last_odom_0.pose.pose.position.x
        current_odom_0_y = self.last_odom_0.pose.pose.position.y
        quaternion_0 = Quaternion(
            self.last_odom_0.pose.pose.orientation.w,
            self.last_odom_0.pose.pose.orientation.x,
            self.last_odom_0.pose.pose.orientation.y,
            self.last_odom_0.pose.pose.orientation.z,
        )
        euler = quaternion_0.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Calculate distance to the goal
        distance_0 = np.linalg.norm(
            [current_odom_0_x - self.goal_x_0, current_odom_0_y - self.goal_y_0]
        )

        # Calculate odometry distance
        distance_odom_0 = np.linalg.norm(
            [current_odom_0_x - self.odom_x_0, current_odom_0_y - self.odom_y_0]
        )

        # Update last odometry position
        self.odom_x_0, self.odom_y_0 = current_odom_0_x, current_odom_0_y

        # Calculate the percentage of distance traveled
        progress_0 = (distance_0 / self.initial_distance_0) * 100 # multiply by 100 so that will be percentage

        # Track current position
        current_pos_0 = (round(current_odom_0_x, 1), round(current_odom_0_y, 1))

        # Add current position to visited positions
        if current_pos_0 not in self.visited_positions_0:
            self.visited_positions_0.add(current_pos_0)
        else:
            # Increment the repeated visits counter
            self.repeated_visits_count_0 += 1

        # Calculate the relative angle to the goal
        skew_x = self.goal_x_0 - current_odom_0_x
        skew_y = self.goal_y_0 - current_odom_0_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            beta = -beta if skew_x < 0 else 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta -= 2 * np.pi
        if theta < -np.pi:
            theta += 2 * np.pi

        # Detect if the goal has been reached and give a large positive reward
        if distance_0 < GOAL_REACHED_DIST:
            target_0 = True
            done_0 = True
            #print("Goal reached!")
        
        rel_x = self.last_odom_1.pose.pose.position.x - current_odom_0_x
        rel_y = self.last_odom_1.pose.pose.position.y - current_odom_0_y
        rel_dist = math.hypot(rel_x, rel_y)
        rel_angle = math.atan2(rel_y, rel_x) - angle

        # Normalize
        if rel_angle > np.pi:
            rel_angle -= 2 * np.pi
        if rel_angle < -np.pi:
            rel_angle += 2 * np.pi
        

        robot_state = [distance_0, theta, action_0[0] * MAX_LINEAR, action_0[1] * MAX_ANGULAR]
        relative_state = [rel_dist, rel_angle]
        state_0 = np.append(laser_state, robot_state + relative_state)
        reward_0 = self.get_reward(
            target_0, collision_0, action_0, min_laser_0, progress_0,
            self.repeated_visits_count_0, rel_dist=rel_dist, timeout=timeout
        )

        #print("Reward R0: {:.2f}".format(reward_0))
        #print(f"Initial: {self.initial_distance_0:.2f}")
        #print(f"Progress R0: {progress_0:.2f}%")
        return state_0, reward_0, done_0, target_0
    
    def step_1(self, action_1, timeout=False):
        target_1 = False

        # Publish the robot0 action
        vel_cmd_1 = Twist()
        vel_cmd_1.linear.x = action_1[0] * MAX_LINEAR
        vel_cmd_1.angular.z = action_1[1] * MAX_ANGULAR
        self.vel_pub_1.publish(vel_cmd_1)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # read laser state
        done_1, collision_1, min_laser_1 = self.observe_collision(self.laserscan_data_1)
        
        #if collision:
            #print("Collision detected! Min laser distance: {:.2f}".format(min_laser))
        
        v_state = []
        v_state[:] = self.laserscan_data_1[:]
        laser_state = [v_state]

        # Calculate robot heading from odometry data
        current_odom_1_x = self.last_odom_1.pose.pose.position.x
        current_odom_1_y = self.last_odom_1.pose.pose.position.y
        quaternion_1 = Quaternion(
            self.last_odom_1.pose.pose.orientation.w,
            self.last_odom_1.pose.pose.orientation.x,
            self.last_odom_1.pose.pose.orientation.y,
            self.last_odom_1.pose.pose.orientation.z,
        )
        euler = quaternion_1.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Calculate distance to the goal
        distance_1 = np.linalg.norm(
            [current_odom_1_x - self.goal_x_1, current_odom_1_y - self.goal_y_1]
        )

        # Calculate odometry distance
        distance_odom_1 = np.linalg.norm(
            [current_odom_1_x - self.odom_x_1, current_odom_1_y - self.odom_y_1]
        )

        # Update last odometry position
        self.odom_x_1, self.odom_y_1 = current_odom_1_x, current_odom_1_y

        # Calculate the percentage of distance traveled
        progress_1 = (distance_1 / self.initial_distance_1) * 100 # multiply by 100 so that will be percentage

        # Track current position
        current_pos_1 = (round(current_odom_1_x, 1), round(current_odom_1_y, 1))

        # Add current position to visited positions
        if current_pos_1 not in self.visited_positions_1:
            self.visited_positions_1.add(current_pos_1)
        else:
            # Increment the repeated visits counter
            self.repeated_visits_count_1 += 1

        # Calculate the relative angle to the goal
        skew_x = self.goal_x_1 - current_odom_1_x
        skew_y = self.goal_y_1 - current_odom_1_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            beta = -beta if skew_x < 0 else 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta -= 2 * np.pi
        if theta < -np.pi:
            theta += 2 * np.pi

        # Detect if the goal has been reached and give a large positive reward
        if distance_1 < GOAL_REACHED_DIST:
            target_1 = True
            done_1 = True
            #print("Goal reached!")
        
        rel_x = self.last_odom_0.pose.pose.position.x - current_odom_1_x
        rel_y = self.last_odom_0.pose.pose.position.y - current_odom_1_y
        rel_dist = math.hypot(rel_x, rel_y)
        rel_angle = math.atan2(rel_y, rel_x) - angle

        # Normalize
        if rel_angle > np.pi:
            rel_angle -= 2 * np.pi
        if rel_angle < -np.pi:
            rel_angle += 2 * np.pi
        

        robot_state = [distance_1, theta, action_1[0] * MAX_LINEAR, action_1[1] * MAX_ANGULAR]
        relative_state = [rel_dist, rel_angle]
        
        state_1 = np.append(laser_state, robot_state + relative_state)
        reward_1 = self.get_reward(
            target_1, collision_1, action_1, min_laser_1, progress_1,
            self.repeated_visits_count_1, rel_dist=rel_dist, timeout=timeout
        )

        #print("Reward R1: {:.2f}".format(reward_1))
        #print(f"Initial: {self.initial_distance_0:.2f}")
        #print(f"Progress R0: {progress_1:.2f}%")
        return state_1, reward_1, done_1, target_1

    def reset(self):
        """
        Resets the state of the environment and returns initial observations for both robots.
        """
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        # Reset robot 0
        state_0 = self._reset_single_robot(
            self.set_self_state_0, self.laserscan_data_0, 0
        )
        
        # Reset robot 1
        state_1 = self._reset_single_robot(
            self.set_self_state_1, self.laserscan_data_1, 1
        )

        # Record the initial distance to the goal for progress calculation
        self.initial_distance_0 = np.linalg.norm(
            [self.odom_x_0 - self.goal_x_0, self.odom_y_0 - self.goal_y_0]
        )

        self.initial_distance_1 = np.linalg.norm(
            [self.odom_x_1 - self.goal_x_1, self.odom_y_1 - self.goal_y_1]
        )

        # Reset visited positions for robot 0
        self.visited_positions_0 = set()
        self.repeated_visits_count_0 = 0

        # Reset visited positions for robot 1
        self.visited_positions_1 = set()
        self.repeated_visits_count_1 = 0

        self.pos_change = not self.pos_change

        return state_0, state_1

    def _reset_single_robot(self, object_state, laserscan_data, robot_id):
        """
        Resets a single robot and returns its state.
        """
        if self.recl_stage == 0:
            angle = np.random.uniform(-np.pi, np.pi)
            quaternion = Quaternion.from_euler(0.0, 0.0, angle)

            x, y = 0, 0
            position_ok = False
            while not position_ok:
                x = np.random.uniform(-4.5, 4.5)
                y = np.random.uniform(-4.5, 4.5)
                position_ok = check_pos(x, y)

            object_state.pose.position.x = x
            object_state.pose.position.y = y
            object_state.pose.orientation.x = quaternion.x
            object_state.pose.orientation.y = quaternion.y
            object_state.pose.orientation.z = quaternion.z
            object_state.pose.orientation.w = quaternion.w

            if robot_id == 0:
                self.set_state_0.publish(object_state)
                self.odom_x_0, self.odom_y_0 = x, y
            else:
                self.set_state_1.publish(object_state)
                self.odom_x_1, self.odom_y_1 = x, y

            self.change_goal()
            self.random_box()
            self.publish_markers([0.0, 0.0])
        
        elif self.recl_stage == 1:
            if robot_id == 0:
                if self.pos_change:
                    x = 0.0
                    y = 2.0
                    angle = -1.57
                else:
                    x = 0.0
                    y = 2.0
                    angle = -1.57
                quaternion = Quaternion.from_euler(0.0, 0.0, angle)
                object_state.pose.position.x = x
                object_state.pose.position.y = y
                object_state.pose.orientation.x = quaternion.x
                object_state.pose.orientation.y = quaternion.y
                object_state.pose.orientation.z = quaternion.z
                object_state.pose.orientation.w = quaternion.w
                self.set_state_0.publish(object_state)
                self.odom_x_0, self.odom_y_0 = x, y
            else:
                if self.pos_change:
                    x = 0.0
                    y = -2.0
                    angle = 1.57 
                else:
                    x = 0.0
                    y = -2.0
                    angle = 1.57 
                quaternion = Quaternion.from_euler(0.0, 0.0, angle)
                object_state.pose.position.x = x
                object_state.pose.position.y = y
                object_state.pose.orientation.x = quaternion.x
                object_state.pose.orientation.y = quaternion.y
                object_state.pose.orientation.z = quaternion.z
                object_state.pose.orientation.w = quaternion.w
                self.set_state_1.publish(object_state)
                self.odom_x_1, self.odom_y_1 = x, y
            
            self.change_goal()
            self.publish_markers([0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        v_state = []
        v_state[:] = laserscan_data[:]
        laser_state = [v_state]

        goal_x, goal_y = (self.goal_x_0, self.goal_y_0) if robot_id == 0 else (self.goal_x_1, self.goal_y_1)
        distance = np.linalg.norm([x - goal_x, y - goal_y])

        skew_x = goal_x - x
        skew_y = goal_y - y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0]

        # Calculate rel_dist and rel_angle for initial state
        if robot_id == 0:
            rel_x = self.odom_x_1 - self.odom_x_0
            rel_y = self.odom_y_1 - self.odom_y_0
        else:
            rel_x = self.odom_x_0 - self.odom_x_1
            rel_y = self.odom_y_0 - self.odom_y_1

        rel_dist = math.hypot(rel_x, rel_y)
        rel_angle = math.atan2(rel_y, rel_x) - angle

        if rel_angle > np.pi:
            rel_angle -= 2 * np.pi
        if rel_angle < -np.pi:
            rel_angle += 2 * np.pi

        relative_state = [rel_dist, rel_angle]
        state = np.append(laser_state, robot_state + relative_state)
        #state = np.append(laser_state, robot_state)
        return state

    def change_goal(self):
        # Place a new goal and check if its location is not on one of the obstacles
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = False

        if self.recl_stage == 0:
            while not goal_ok:
                self.goal_x_0 = self.odom_x_0 + random.uniform(self.upper, self.lower)
                self.goal_y_0 = self.odom_y_0 + random.uniform(self.upper, self.lower)
                self.goal_x_1 = self.odom_x_1 + random.uniform(self.upper, self.lower)
                self.goal_y_1 = self.odom_y_1 + random.uniform(self.upper, self.lower)

                goal_ok = check_pos(self.goal_x_0, self.goal_y_0) and check_pos(self.goal_x_1, self.goal_y_1)
        
        elif self.recl_stage == 1:
            if self.pos_change:
                self.goal_x_0 = 0.0
                self.goal_y_0 = -2.0
                self.goal_x_1 = 0.0
                self.goal_y_1 = 2.0
            else:
                self.goal_x_0 = 0.0
                self.goal_y_0 = -2.0
                self.goal_x_1 = 0.0
                self.goal_y_1 = 2.0

    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training environment
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x_0, y - self.odom_y_0])
                distance_to_goal = np.linalg.norm([x - self.goal_x_0, y - self.goal_y_0])
                distance_to_robot1 = np.linalg.norm([x - self.odom_x_1, y - self.odom_y_1])
                distance_to_goal1 = np.linalg.norm([x - self.goal_x_1, y - self.goal_y_1])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5 or distance_to_robot1 < 1.5 or distance_to_goal1 < 1.5:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state_0.publish(box_state)

    def publish_markers(self, action_0):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x_0
        marker.pose.position.y = self.goal_y_0
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray4= MarkerArray()
        marker4 = Marker()
        marker4.header.frame_id = "odom"
        marker4.type = marker.CYLINDER
        marker4.action = marker.ADD
        marker4.scale.x = 0.1
        marker4.scale.y = 0.1
        marker4.scale.z = 0.01
        marker4.color.a = 1.0
        marker4.color.r = 0.0
        marker4.color.g = 0.0
        marker4.color.b = 1.0
        marker4.pose.orientation.w = 1.0
        marker4.pose.position.x = self.goal_x_1
        marker4.pose.position.y = self.goal_y_1
        marker4.pose.position.z = 0

        markerArray4.markers.append(marker4)
        self.publisher4.publish(markerArray4)

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser, progress, num_visited_positions,
                rel_dist=None, rel_dist_threshold=2.5, timeout=False):
        
        # ------------------------------
        # 1. Navigation Reward Component
        # ------------------------------
        if target:
            navigation_reward = 100.0  # Goal reached
        elif progress > 140:
            overshoot_penalty = -(progress - 100) * 0.1
            navigation_reward = max(overshoot_penalty, -1)
        elif progress < 40:
            navigation_reward = (40 - progress) * 0.1  # Currently zero slope region
        else:
            navigation_reward = 0.0  # Zero reward in the middle range

        # --------------------------------------
        # 2. Collision Avoidance Reward Component
        # --------------------------------------
        if collision:
            return -100.0  # Immediate collision penalty
        if timeout:
            return 0.0  # Timeout is treated as failure

        def r3(x):  # Proximity to obstacle penalty
            return (COLLISION_DIST * 5) - x if x < (COLLISION_DIST * 5) else 0.0

        ca_reward = (
            action[0] * 2.5              # Encouraging forward velocity
            - 0.5 * abs(action[1])       # Penalizing angular velocity
            - r3(min_laser) * 2        # Penalty for being too close to obstacles
        )

        # -----------------------------
        # 3. Stagnation Penalty Component
        # -----------------------------
        speed_penalty = 0.0
        exploration_penalty = 0.0

        if action[0] < 1e-2 and abs(action[1]) < 1e-2:
            speed_penalty = -1.0  # Robot is stuck
        elif num_visited_positions > 0:
            exploration_penalty = -0.05 * num_visited_positions
            exploration_penalty = max(exploration_penalty, -5)

        stagnation_penalty = speed_penalty + exploration_penalty

        # ------------------------------
        # 4. Robot Distance Penalty Component
        # ------------------------------
        inter_robot_penalty = 0.0
        if rel_dist is not None and rel_dist < rel_dist_threshold:
            inter_robot_penalty = -2.0 * (rel_dist_threshold - rel_dist)

        # ------------------------------
        # Total reward
        # ------------------------------
        total_reward = (
            navigation_reward +
            ca_reward +
            stagnation_penalty +
            inter_robot_penalty
        )
        return total_reward


