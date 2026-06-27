import math
import os
import random
import subprocess
import time
from os import path
import functools

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

def check_pos(y, x):
    goal_ok = True
    obs_pad = 0.1

    if (x > -7.25 - obs_pad and x < 7.25 + obs_pad) and (y > -10.00 - obs_pad and y < -8.00 + obs_pad):
        goal_ok = False

    if (x > -7.25 - obs_pad and x < 7.25 + obs_pad) and (y > 8.00 - obs_pad and y < 10.00 + obs_pad):
        goal_ok = False
    
    if (x > -1.85 - obs_pad and x < -1.65 + obs_pad) and (y > -6.15 - obs_pad and y < -1.15 + obs_pad):
        goal_ok = False
    
    if (x > 1.65 - obs_pad and x < 1.85 + obs_pad) and (y > -6.15 - obs_pad and y < -1.15 + obs_pad):
        goal_ok = False
    
    if (x > -1.85 - obs_pad and x < -1.65 + obs_pad) and (y > 1.15 - obs_pad and y < 6.15 + obs_pad):
        goal_ok = False
    
    if (x > 1.65 - obs_pad and x < 1.85 + obs_pad) and (y > 1.15 - obs_pad and y < 6.15 + obs_pad):
        goal_ok = False
    
    if (x > 1.65 - obs_pad and x < 1.85 + obs_pad) and (y > 2.00 - obs_pad and y < 7.00 + obs_pad):
        goal_ok = False
    
    if (x > 1.65 - obs_pad and x < 1.85 + obs_pad) and (y > -7.00 - obs_pad and y < -2.00 + obs_pad):
        goal_ok = False

    if (x > -1.85 - obs_pad and x < -1.65 + obs_pad) and (y > 2.00 - obs_pad and y < 7.00 + obs_pad):
        goal_ok = False
    
    if (x > -1.85 - obs_pad and x < -1.65 + obs_pad) and (y > -7.00 - obs_pad and y < -2.00 + obs_pad):
        goal_ok = False
    
    if (x > -7.25 - obs_pad and x < -1.00 + obs_pad) and (y > -6.50 - obs_pad and y < -0.5 + obs_pad):
        goal_ok = False
    
    if (x > 1.00 - obs_pad and x < 7.25 + obs_pad) and (y > -6.50 - obs_pad and y < -0.5 + obs_pad):
        goal_ok = False
    
    if (x > -7.25 - obs_pad and x < -1.00 + obs_pad) and (y > 0.5 - obs_pad and y < 6.50 + obs_pad):
        goal_ok = False
    
    if (x > 1.00 - obs_pad and x < 7.25 + obs_pad) and (y > 0.5 - obs_pad and y < 6.50 + obs_pad):
        goal_ok = False

    if x > 9.0 or x < -9.0 or y > 9.0 or y < -9.0:
        goal_ok = False

    return goal_ok

class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim
        
        self.max_ep = 1000
        self.num_robots = 8  # Set number of robots

        self.timestep = [0] * self.num_robots
        self.total_event = [0] * self.num_robots
        self.repeated_visits_count = [0] * self.num_robots
        self.initial_distance = [0] * self.num_robots  # Initialize for all robots
        self.visited_positions = [set() for _ in range(self.num_robots)]  # Initialize an empty set for each robot
        self.flag_done = [False] * self.num_robots

        # Define event counters for all robots
        self.goal_counter = [0] * self.num_robots
        self.collision_counter = [0] * self.num_robots
        self.optima_counter = [0] * self.num_robots
        self.prev_goal_x = [None] * self.num_robots
        self.prev_goal_y = [None] * self.num_robots

        self.odom_x = [0] * self.num_robots
        self.odom_y = [0] * self.num_robots
        self.goal_x = [0] * self.num_robots
        self.goal_y = [0] * self.num_robots
        self.laserscan_data = [np.ones(self.environment_dim) * MAX_RANGE for _ in range(self.num_robots)]
        self.last_odom = [None] * self.num_robots

        self.prev_scan = [None] * self.num_robots  # Previous scan data
        self.prev_time = [None] * self.num_robots  # Previous scan timestamp
        self.velocity = [np.zeros(self.environment_dim) for _ in range(self.num_robots)]  # Velocity storage
        self.scan_count = [0] * self.num_robots  # Track scans per robot

        self.robot_collision_counter = [0] * self.num_robots

        # Initialize ModelState for each robot
        self.set_self_state = []
        for i in range(self.num_robots):
            state = ModelState()
            state.model_name = f"tb3_{i}"
            state.pose.position.x = 0.0
            state.pose.position.y = 0.0
            state.pose.position.z = 0.0
            state.pose.orientation.x = 0.0
            state.pose.orientation.y = 0.0
            state.pose.orientation.z = 0.0
            state.pose.orientation.w = 1.0
            self.set_self_state.append(state)

        self.upper = 10.0
        self.lower = -10.0

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

        print("Gazebo launched!")

        self.vel_pub = []
        self.set_state = []
        self.laserscan_subscriber = []
        self.odom_subscriber = []

        for i in range(self.num_robots):
            # Publishers
            self.vel_pub.append(rospy.Publisher(f"tb3_{i}/cmd_vel", Twist, queue_size=1))
            self.set_state.append(rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10))

            # Subscribers (using functools.partial for safety)
            self.laserscan_subscriber.append(rospy.Subscriber(f"tb3_{i}/scan", LaserScan, functools.partial(self.laserscan_callback, robot_id=i), queue_size=1))
            self.odom_subscriber.append(rospy.Subscriber(f"tb3_{i}/odom", Odometry, functools.partial(self.odom_callback, robot_id=i), queue_size=1))
        
        self.moving_objects = {i: np.zeros(self.environment_dim, dtype=bool) for i in range(self.num_robots)}
        self.reference_scan = {i: None for i in range(self.num_robots)}
        self.reference_time = {i: None for i in range(self.num_robots)}
        self.moving_detected = False

        # Set up the ROS publishers and subscribers
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("goal_point2", MarkerArray, queue_size=3)
        self.publisher3 = rospy.Publisher("goal_point3", MarkerArray, queue_size=3)
        self.publisher4 = rospy.Publisher("goal_point4", MarkerArray, queue_size=3)
        self.publisher5 = rospy.Publisher("goal_point5", MarkerArray, queue_size=3)
        self.publisher6 = rospy.Publisher("goal_point6", MarkerArray, queue_size=3)
        self.publisher7 = rospy.Publisher("goal_point7", MarkerArray, queue_size=3)
        self.publisher8 = rospy.Publisher("goal_point8", MarkerArray, queue_size=3)

    def laserscan_callback(self, scan, robot_id):
        """Process laser scan data for a given robot and compute velocity."""
        current_time = rospy.get_time()  # Get the current timestamp
        bin_size = max(1, len(scan.ranges) // self.environment_dim)  # Bin size

        # Store current scan data
        self.laserscan_data[robot_id] = np.ones(self.environment_dim) * scan.range_max
        for i, dist in enumerate(scan.ranges):
            if np.isnan(dist) or dist < scan.range_min or dist > scan.range_max:
                continue
            bin_index = i // bin_size
            if bin_index < self.environment_dim:
                self.laserscan_data[robot_id][bin_index] = min(self.laserscan_data[robot_id][bin_index], dist)

        # Store moving object flags
        self.moving_objects[robot_id] = np.zeros(self.environment_dim, dtype=bool)

        # Identify moving objects
        self.moving_detected = False

        if self.reference_scan[robot_id] is None:
            self.reference_scan[robot_id] = np.copy(self.laserscan_data[robot_id])
            self.reference_time[robot_id] = current_time
            return

        dt = current_time - self.reference_time[robot_id]
        if dt < 0.1:
            return  # Wait until enough time elapsed

        for i in range(self.environment_dim):
            delta_d = self.reference_scan[robot_id][i] - self.laserscan_data[robot_id][i]
            v = delta_d / dt if abs(delta_d) > 0.01 else 0.0

            if v > 0.5:
                self.moving_objects[robot_id][i] = True
            else:
                self.moving_objects[robot_id][i] = False

        # Optionally refresh reference every few seconds to track changes
        if dt > 2.0:
            self.reference_scan[robot_id] = np.copy(self.laserscan_data[robot_id])
            self.reference_time[robot_id] = current_time

    def odom_callback(self, od_data, robot_id):
        """Process odometry data for a given robot."""
        self.last_odom[robot_id] = od_data

    # Perform an action and read a new state
    def step(self, action, robot_id, timeout=False):
        """Generalized step function for any robot using robot_id dynamically"""

        target = False

        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0] * MAX_LINEAR
        vel_cmd.angular.z = action[1] * MAX_ANGULAR
        self.vel_pub[robot_id].publish(vel_cmd)

        # Unpause Gazebo physics
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")

        # Propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        # Pause Gazebo physics
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")

        # Read laser state
        robot_collision = self.detect_robot_collisions(robot_id)
        collision = False
        min_laser = min(self.laserscan_data[robot_id])

        # Only count obstacle collision if it's NOT robot-to-robot
        if not robot_collision and min_laser < COLLISION_DIST:
            collision = True
            done = True
        else:
            done = robot_collision  # still ends episode if robot-to-robot
        
        laser_state = [self.laserscan_data[robot_id].copy()]

        # Calculate robot heading from odometry data
        current_odom_x = self.last_odom[robot_id].pose.pose.position.x
        current_odom_y = self.last_odom[robot_id].pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom[robot_id].pose.pose.orientation.w,
            self.last_odom[robot_id].pose.pose.orientation.x,
            self.last_odom[robot_id].pose.pose.orientation.y,
            self.last_odom[robot_id].pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Calculate distance to the goal
        distance = np.linalg.norm(
            [current_odom_x - self.goal_x[robot_id], current_odom_y - self.goal_y[robot_id]]
        )

        # Calculate odometry distance
        distance_odom = np.linalg.norm(
            [current_odom_x - self.odom_x[robot_id], current_odom_y - self.odom_y[robot_id]]
        )

        # Update last odometry position
        self.odom_x[robot_id], self.odom_y[robot_id] = current_odom_x, current_odom_y

        # Calculate the percentage of distance traveled
        progress = (distance / self.initial_distance[robot_id]) * 100  # Convert to percentage

        # Track visited positions
        current_pos = (round(current_odom_x, 2), round(current_odom_y, 2))
        if current_pos not in self.visited_positions[robot_id]:
            self.visited_positions[robot_id].add(current_pos)
        else:
            self.repeated_visits_count[robot_id] += 1

        # Calculate relative angle to goal
        skew_x = self.goal_x[robot_id] - self.odom_x[robot_id]
        skew_y = self.goal_y[robot_id] - self.odom_y[robot_id]
        dot = skew_x * 1 + skew_y * 0
        mag1 = np.linalg.norm([skew_x, skew_y])
        mag2 = np.linalg.norm([1, 0])
        beta = math.acos(dot / (mag1 * mag2))
        beta = -beta if skew_y < 0 else beta
        theta = beta - angle
        theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi  # Normalize theta between -π and π

        # Check goal or collision conditions
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True
            if not self.flag_done[robot_id]:  # Prevents double counting before reset
                self.goal_counter[robot_id] += 1
                self.prev_goal_x[robot_id] = self.goal_x[robot_id]
                self.prev_goal_y[robot_id] = self.goal_y[robot_id]
                self.timestep[robot_id] = 0
                self.flag_done[robot_id] = True  # Mark as finished until reset
        
        elif robot_collision:
            done = True
            if not self.flag_done[robot_id]:
                self.robot_collision_counter[robot_id] += 1
                self.prev_goal_x[robot_id] = self.goal_x[robot_id]
                self.prev_goal_y[robot_id] = self.goal_y[robot_id]
                self.timestep[robot_id] = 0
                self.flag_done[robot_id] = True

        elif collision:
            done = True
            if not self.flag_done[robot_id]:  # Prevents counting multiple collisions before reset
                self.collision_counter[robot_id] += 1
                self.prev_goal_x[robot_id] = self.goal_x[robot_id]
                self.prev_goal_y[robot_id] = self.goal_y[robot_id]
                self.timestep[robot_id] = 0
                self.flag_done[robot_id] = True  # Mark as finished until reset

        else:
            if self.prev_goal_x[robot_id] != self.goal_x[robot_id] or self.prev_goal_y[robot_id] != self.goal_y[robot_id]:
                self.timestep[robot_id] += 1

                if self.timestep[robot_id] > self.max_ep - 10:
                    self.optima_counter[robot_id] += 1
                    done = True
                    self.timestep[robot_id] = 0
                    self.prev_goal_x[robot_id] = self.goal_x[robot_id]
                    self.prev_goal_y[robot_id] = self.goal_y[robot_id]

        # Stop counting for robots that have already completed an attempt
        self.total_event[robot_id] = self.goal_counter[robot_id] + self.collision_counter[robot_id] + self.optima_counter[robot_id]

        robot_state = [distance, theta, action[0] * MAX_LINEAR, action[1] * MAX_ANGULAR]
        relative_state = [3, 0]
        state = np.append(laser_state, robot_state + relative_state)

        return state, done, target

    def reset(self):
        """
        Resets the state of the environment and returns initial observations for all robots dynamically.
        """
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException:
            print("/gazebo/reset_simulation service call failed")

        states = []

        for i in range(self.num_robots):
            self.scan_count[i] = 0  # Reset scan counter
            self.prev_scan[i] = None  # Clear previous scan data
            self.prev_time[i] = None  # Clear previous timestamp
        #rospy.logwarn("All robots have been reset. Scan count cleared.")

        print("---------------------------------------------------------------")
        
        for i in range(self.num_robots):
            state = self._reset_single_robot(self.set_self_state[i], self.laserscan_data[i], i)
            states.append(state)

            # Record the initial distance to the goal for progress calculation
            self.initial_distance[i] = np.linalg.norm(
                [self.odom_x[i] - self.goal_x[i], self.odom_y[i] - self.goal_y[i]]
            )

            # Reset visited positions
            self.visited_positions[i] = set()
            self.repeated_visits_count[i] = 0

            if self.flag_done[i]:  # Only reset if the robot was finished
                self.flag_done[i] = False  # Allow counting again after reset

            # Track total events per robot
            self.total_event[i] = self.goal_counter[i] + self.collision_counter[i] + self.optima_counter[i] +self.robot_collision_counter[i]

            # Print event stats if under the threshold
            if self.total_event[i] < 101:
                print(f"Total Events R{i}: {self.total_event[i]} (Goals: {self.goal_counter[i]}, "
                    f"Obstacle Collisions: {self.collision_counter[i]}, "
                    f"Robot Collisions: {self.robot_collision_counter[i]}, "
                    f"Stuck Optima: {self.optima_counter[i]})")
            else:
                self.total_event[i] = 0

        return states

    def _reset_single_robot(self, object_state, laserscan_data, robot_id):
        """
        Resets a single robot and returns its state dynamically.
        Ensures robots do not spawn too close to each other (< 1 meter).
        """
        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)

        x, y = 0, 0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-10.0, 10.0)
            y = np.random.uniform(-10.0, 10.0)
            position_ok = check_pos(x, y)

            # Ensure no other robot is within 1 meter of this robot
            for j in range(self.num_robots):
                if j != robot_id:  # Don't compare to itself
                    distance_to_robot = np.linalg.norm([x - self.odom_x[j], y - self.odom_y[j]])
                    if distance_to_robot < 3.5:  # If too close, reject position
                        position_ok = False
                        break  # Exit loop and retry position

        # Set robot's position and orientation
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w

        # Publish the state for the robot
        self.set_state[robot_id].publish(object_state)
        self.odom_x[robot_id], self.odom_y[robot_id] = x, y

        # Reset goals and objects in the environment
        self.change_goal()
        #self.random_box()
        self.publish_markers([0.0, 0.0])

        # Unpause Gazebo physics
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        # Pause Gazebo physics
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")

        # Process laser scan data
        laser_state = [laserscan_data.copy()]

        # Compute initial goal distance
        goal_x, goal_y = self.goal_x[robot_id], self.goal_y[robot_id]
        distance = np.linalg.norm([x - goal_x, y - goal_y])

        # Calculate relative angle to the goal
        skew_x = goal_x - x
        skew_y = goal_y - y
        dot = skew_x * 1 + skew_y * 0
        mag1 = np.linalg.norm([skew_x, skew_y])
        mag2 = np.linalg.norm([1, 0])
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            beta = -beta if skew_x < 0 else -beta
        theta = beta - angle
        theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi  # Normalize theta

        # Initial robot state
        robot_state = [distance, theta, 0.0, 0.0]
        relative_state = [3, 0]
        state = np.append(laser_state, robot_state + relative_state)
        return state


    def change_goal(self):
        """Assign new goals efficiently — valid, obstacle-free, and at least 3 m from current position."""
        # Gradually expand sampling boundary
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        min_goal_dist = 0.0   # minimum distance (meters)
        max_attempts = 100    # prevent infinite loops

        for i in range(self.num_robots):
            found = False
            for _ in range(max_attempts):
                # Randomly sample a new goal near current odom position
                gx = self.odom_x[i] + random.uniform(self.lower, self.upper)
                gy = self.odom_y[i] + random.uniform(self.lower, self.upper)

                # Compute distance from current position
                dist_to_robot = ((gx - self.odom_x[i]) ** 2 + (gy - self.odom_y[i]) ** 2) ** 0.5

                # Check both map validity and distance constraint
                if check_pos(gx, gy) and dist_to_robot >= min_goal_dist:
                    self.goal_x[i], self.goal_y[i] = gx, gy
                    found = True
                    break

            if not found:
                print(f"[WARN] Robot {i} failed to find valid goal — keeping last one.")

    def publish_markers(self, action_0):
        # Publish visual Goal 1
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
        marker.pose.position.x = self.goal_x[0]
        marker.pose.position.y = self.goal_y[0]
        marker.pose.position.z = 0
        markerArray.markers.append(marker)
        self.publisher.publish(markerArray)

        # Publish visual Goal 2
        markerArray2= MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CYLINDER
        marker2.action = marker.ADD
        marker2.scale.x = 0.1
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 0.0
        marker2.color.g = 0.0
        marker2.color.b = 1.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = self.goal_x[1]
        marker2.pose.position.y = self.goal_y[1]
        marker2.pose.position.z = 0
        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        # Publish visual Goal 3
        markerArray3= MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CYLINDER
        marker3.action = marker.ADD
        marker3.scale.x = 0.1
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 1.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = self.goal_x[2]
        marker3.pose.position.y = self.goal_y[2]
        marker3.pose.position.z = 0
        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

        # Publish visual Goal 4
        markerArray4= MarkerArray()
        marker4 = Marker()
        marker4.header.frame_id = "odom"
        marker4.type = marker.CYLINDER
        marker4.action = marker.ADD
        marker4.scale.x = 0.1
        marker4.scale.y = 0.1
        marker4.scale.z = 0.01
        marker4.color.a = 1.0
        marker4.color.r = 0.5
        marker4.color.g = 0.0
        marker4.color.b = 1.0
        marker4.pose.orientation.w = 1.0
        marker4.pose.position.x = self.goal_x[3]
        marker4.pose.position.y = self.goal_y[3]
        marker4.pose.position.z = 0
        markerArray4.markers.append(marker4)
        self.publisher4.publish(markerArray4)

        # Publish visual Goal 5
        markerArray5= MarkerArray()
        marker5 = Marker()
        marker5.header.frame_id = "odom"
        marker5.type = marker.CYLINDER
        marker5.action = marker.ADD
        marker5.scale.x = 0.1
        marker5.scale.y = 0.1
        marker5.scale.z = 0.01
        marker5.color.a = 1.0
        marker5.color.r = 1.0
        marker5.color.g = 0.0
        marker5.color.b = 0.25
        marker5.pose.orientation.w = 1.0
        marker5.pose.position.x = self.goal_x[4]
        marker5.pose.position.y = self.goal_y[4]
        marker5.pose.position.z = 0
        markerArray5.markers.append(marker5)
        self.publisher5.publish(markerArray5)

        # Publish visual Goal 6
        markerArray6= MarkerArray()
        marker6 = Marker()
        marker6.header.frame_id = "odom"
        marker6.type = marker.CYLINDER
        marker6.action = marker.ADD
        marker6.scale.x = 0.1
        marker6.scale.y = 0.1
        marker6.scale.z = 0.01
        marker6.color.a = 1.0
        marker6.color.r = 0.25
        marker6.color.g = 0.0
        marker6.color.b = 1.0
        marker6.pose.orientation.w = 1.0
        marker6.pose.position.x = self.goal_x[5]
        marker6.pose.position.y = self.goal_y[5]
        marker6.pose.position.z = 0
        markerArray6.markers.append(marker6)
        self.publisher6.publish(markerArray6)

        # Publish visual Goal 7
        markerArray7 = MarkerArray()
        marker7 = Marker()
        marker7.header.frame_id = "odom"
        marker7.type = marker.CYLINDER
        marker7.action = marker.ADD
        marker7.scale.x = 0.1
        marker7.scale.y = 0.1
        marker7.scale.z = 0.01
        marker7.color.a = 1.0
        marker7.color.r = 1.0
        marker7.color.g = 0.0
        marker7.color.b = 0.0
        marker7.pose.orientation.w = 1.0
        marker7.pose.position.x = self.goal_x[6]
        marker7.pose.position.y = self.goal_y[6]
        marker7.pose.position.z = 0
        markerArray7.markers.append(marker7)
        self.publisher7.publish(markerArray7)

        # Publish visual Goal 8
        markerArray8= MarkerArray()
        marker8 = Marker()
        marker8.header.frame_id = "odom"
        marker8.type = marker.CYLINDER
        marker8.action = marker.ADD
        marker8.scale.x = 0.1
        marker8.scale.y = 0.1
        marker8.scale.z = 0.01
        marker8.color.a = 1.0
        marker8.color.r = 0.0
        marker8.color.g = 0.0
        marker8.color.b = 1.0
        marker8.pose.orientation.w = 1.0
        marker8.pose.position.x = self.goal_x[7]
        marker8.pose.position.y = self.goal_y[7]
        marker8.pose.position.z = 0
        markerArray8.markers.append(marker8)
        self.publisher8.publish(markerArray8)
    
    def detect_robot_collisions(self, robot_id):
        for i in range(self.num_robots):
            if i == robot_id:
                continue
            dist = np.linalg.norm([
                self.odom_x[robot_id] - self.odom_x[i],
                self.odom_y[robot_id] - self.odom_y[i]
            ])
            if dist < 0.25:  # Collision threshold between robots
                return True
        return False
