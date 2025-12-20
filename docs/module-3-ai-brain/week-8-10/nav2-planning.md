---
sidebar_position: 5
---

# Nav2 for Humanoid Robot Path Planning

## Introduction to Navigation in Physical AI

Navigation is a critical component of Physical AI systems, enabling robots to move autonomously from one location to another while avoiding obstacles and respecting environmental constraints. The Navigation2 (Nav2) project provides a comprehensive, production-ready framework for robot navigation built on ROS 2, offering sophisticated path planning and execution capabilities.

For humanoid robots, navigation presents unique challenges due to their complex kinematics, balance requirements, and anthropomorphic movement patterns. This section explores how Nav2 can be configured and extended to meet the specific needs of humanoid robots.

## Understanding Nav2 Architecture

### Core Components

The Nav2 stack consists of several interconnected components that work together to provide complete navigation capabilities:

```
Goal → Planner Server → Path Planner → Path Post-Processing → Controller Server → Local Planner → Robot
```

### Key Nav2 Components

1. **Planner Server**: Manages global path planning
2. **Controller Server**: Handles local path following and obstacle avoidance
3. **Recovery Server**: Provides recovery behaviors when navigation fails
4. **Lifecycle Manager**: Manages the state of navigation components
5. **BT Navigator**: Uses behavior trees for navigation decision-making

### Behavior Tree Navigation

Nav2 uses behavior trees to orchestrate navigation decisions:

```xml
<!-- Example behavior tree configuration -->
<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <PipelineSequence name="NavigateWithReplanning">
            <RateController hz="1.0">
                <RecoveryNode number_of_retries="6" name="ComputeAndSmoothPath">
                    <PipelineSequence name="ComputeAndSmooth">
                        <RecoveryNode number_of_retries="1" name="ComputePath">
                            <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
                        </RecoveryNode>
                        <SmoothPath input_path="{path}" output_path="{path}" smoother_id="SimpleSmoother"/>
                    </PipelineSequence>
                    <RecoveryNode number_of_retries="1" name="SmoothPath" path="{path}"/>
                </RecoveryNode>
            </RateController>
            <FollowPath path="{path}" controller_id="FollowPath"/>
        </PipelineSequence>
    </BehaviorTree>
</root>
```

## Nav2 Configuration for Humanoid Robots

### Parameter Configuration

Humanoid robots require specific navigation parameters due to their unique kinematics and movement patterns:

```yaml
# Example Nav2 configuration for humanoid robot
bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    default_nav_through_poses_bt_xml: "navigate_w_replanning_and_recovery.xml"
    default_nav_to_pose_bt_xml: "navigate_w_replanning_and_recovery.xml"

bt_navigator_navigate_through_poses:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20

bt_navigator_navigate_to_pose:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.001
    min_theta_velocity_threshold: 0.001
    progress_checker_plugins: ["progress_checker"]
    goal_checker_plugins: ["goal_checker"]
    controller_plugins: ["FollowPath"]
    
    # Humanoid-specific controller settings
    FollowPath:
      plugin: "nav2_mppi_controller::MPPICController"
      debug: False
      rate: 20
      transform_tolerance: 0.1
      robot_params:
        k_velocity: 0.8
        radius: 0.4  # Humanoid robot radius
        footprint_model:
          type: "polygon"
          points: [[ 0.4,  0.3], [ 0.4, -0.3], [-0.4, -0.3], [-0.4,  0.3]]
      costmap:
        enabled: true
        topic: "local_costmap/costmap_raw"
        global_frame: "odom"
        robot_base_frame: "base_link"
        update_frequency: 20.0
        width: 10
        height: 10
        resolution: 0.05
      motion_model:
        type: "Holonomic"
        acc_lim: [2.5, 2.5, 3.2]  # Accel limits for humanoid
        decel_lim: [-2.5, -2.5, -3.2]
        vel_lim: [1.0, 1.0, 1.5]  # Velocity limits for humanoid (x, y, theta)

local_costmap:
  ros__parameters:
    use_sim_time: True
    global_frame: odom
    robot_base_frame: base_link
    update_frequency: 5.0
    publish_frequency: 2.0
    width: 10
    height: 10
    resolution: 0.05
    origin_x: 0.0
    origin_y: 0.0
    rolling_window: true
    always_send_full_costmap: false
    footprint: "[ [0.4, 0.3], [0.4, -0.3], [-0.4, -0.3], [-0.4, 0.3] ]"
    footprint_padding: 0.01
    plugins: ["voxel_layer", "inflation_layer"]
    inflation_layer:
      plugin: "nav2_costmap_2d::InflationLayer"
      cost_scaling_factor: 3.0
      inflation_radius: 0.55
    voxel_layer:
      plugin: "nav2_costmap_2d::VoxelLayer"
      enabled: True
      publish_voxel_map: False
      origin_z: 0.0
      z_resolution: 0.2
      z_voxels: 8
      max_obstacle_height: 2.0
      mark_threshold: 0
      observation_sources: scan
      scan:
        topic: /scan
        sensor_frame: laser_link
        max_obstacle_height: 2.0
        clearing: True
        marking: True
        data_type: "LaserScan"
        raytrace_range: 3.0
        obstacle_range: 2.5
        transform_tolerance: 0.2
        inf_is_valid: True

global_costmap:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    update_frequency: 1.0
    static_map: True
    rolling_window: false
    width: 200
    height: 200
    resolution: 0.05
    origin_x: 0.0
    origin_y: 0.0
    footprint: "[ [0.4, 0.3], [0.4, -0.3], [-0.4, -0.3], [-0.4, 0.3] ]"
    footprint_padding: 0.01
    plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
    obstacle_layer:
      plugin: "nav2_costmap_2d::ObstacleLayer"
      enabled: True
      observation_sources: scan
      scan:
        topic: /scan
        sensor_frame: laser_link
        max_obstacle_height: 2.0
        clearing: True
        marking: True
        data_type: "LaserScan"
        raytrace_range: 3.0
        obstacle_range: 2.5
        transform_tolerance: 0.2
        inf_is_valid: True
    static_layer:
      plugin: "nav2_costmap_2d::StaticLayer"
      map_subscribe_transient_local: True
    inflation_layer:
      plugin: "nav2_costmap_2d::InflationLayer"
      cost_scaling_factor: 3.0
      inflation_radius: 0.55
```

## Path Planning Algorithms

### Global Path Planners

#### A* (A-star) Planner
The A* algorithm finds the shortest path considering obstacles using a heuristic approach:

```cpp
// Pseudo-code for A* path planning
#include <vector>
#include <queue>
#include <unordered_map>

struct Node {
    double x, y;
    double g_cost; // Cost from start
    double h_cost; // Heuristic cost to goal
    double f_cost; // g_cost + h_cost
    Node* parent;
    
    bool operator>(const Node& other) const {
        return f_cost > other.f_cost;
    }
};

std::vector<Node> a_star_path_planner(double start_x, double start_y, 
                                     double goal_x, double goal_y,
                                     const Costmap& costmap) {
    // Initialize open and closed sets
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open_set;
    std::unordered_map<std::string, Node> closed_set;
    
    // Add start node to open set
    Node start_node = {start_x, start_y, 0.0, 
                      heuristic(start_x, start_y, goal_x, goal_y), 
                      heuristic(start_x, start_y, goal_x, goal_y), 
                      nullptr};
    open_set.push(start_node);
    
    while (!open_set.empty()) {
        Node current = open_set.top();
        open_set.pop();
        
        // Check if goal is reached
        if (distance(current.x, current.y, goal_x, goal_y) < 0.1) {
            // Reconstruct path
            return reconstruct_path(&current);
        }
        
        // Add to closed set
        std::string key = std::to_string(current.x) + "," + std::to_string(current.y);
        closed_set[key] = current;
        
        // Explore neighbors
        std::vector<Node> neighbors = get_neighbors(current, costmap);
        for (auto& neighbor : neighbors) {
            std::string neighbor_key = std::to_string(neighbor.x) + "," + std::to_string(neighbor.y);
            
            if (closed_set.find(neighbor_key) != closed_set.end()) {
                continue; // Already evaluated
            }
            
            // Calculate tentative g_cost
            double tentative_g = current.g_cost + distance(current, neighbor);
            
            // If this path to neighbor is better than previous one
            if (tentative_g < neighbor.g_cost) {
                neighbor.parent = &current;
                neighbor.g_cost = tentative_g;
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost;
                
                open_set.push(neighbor);
            }
        }
    }
    
    return {}; // No path found
}
```

#### Dijkstra Algorithm
Dijkstra's algorithm finds the shortest path without using heuristics:

```python
import heapq
import numpy as np

def dijkstra_path_planning(start, goal, costmap):
    """
    Dijkstra path planning implementation
    """
    # Initialize distances and previous nodes
    distances = {}
    previous = {}
    unvisited = []
    
    # Set all distances to infinity and initialize start
    for row in range(costmap.shape[0]):
        for col in range(costmap.shape[1]):
            if costmap[row, col] >= 99:  # Consider occupied cells as infinity
                distances[(row, col)] = float('inf')
            else:
                distances[(row, col)] = float('inf')
            previous[(row, col)] = None
    
    # Set start distance to 0
    distances[start] = 0
    heapq.heappush(unvisited, (0, start))
    
    while unvisited:
        # Get node with smallest distance
        current_distance, current_node = heapq.heappop(unvisited)
        
        if current_node == goal:
            break  # Found the goal
        
        if current_distance > distances[current_node]:
            continue  # Skip if already processed with a shorter distance
        
        # Check all neighbors
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            neighbor = (current_node[0] + dr, current_node[1] + dc)
            
            # Check if neighbor is within bounds
            if (0 <= neighbor[0] < costmap.shape[0] and 
                0 <= neighbor[1] < costmap.shape[1] and
                costmap[neighbor[0], neighbor[1]] < 99):  # Not occupied
                
                # Calculate tentative distance
                move_cost = 1.0 if abs(dr) + abs(dc) == 1 else 1.414  # Manhattan vs diagonal
                tentative_distance = distances[current_node] + move_cost + costmap[neighbor[0], neighbor[1]]/100.0
                
                # If this path is shorter than any previous path to neighbor
                if tentative_distance < distances[neighbor]:
                    distances[neighbor] = tentative_distance
                    previous[neighbor] = current_node
                    heapq.heappush(unvisited, (tentative_distance, neighbor))
    
    # Reconstruct path
    path = []
    current = goal
    while current != start:
        if previous[current] is None:
            return []  # No path found
        path.append(current)
        current = previous[current]
    
    path.append(start)
    path.reverse()
    
    return path
```

### Humanoid-Specific Path Planning Considerations

Humanoid robots require special consideration in path planning:

1. **Balance Constraints**: The path must maintain the robot's balance and stability
2. **Step Constraints**: The robot can only step on certain surfaces
3. **Footstep Planning**: Generate valid footstep sequences
4. **Kinematic Constraints**: Respect joint limits and reachability

```python
class HumanoidPathPlanner:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.step_length_limit = 0.3  # Maximum step length
        self.step_height_limit = 0.15  # Maximum step height difference
        self.support_polygon = self.calculate_support_polygon()
    
    def plan_holonomic_path(self, start_pose, goal_pose, costmap):
        """
        Plan path considering humanoid-specific constraints
        """
        # Convert to grid coordinates
        start = self.world_to_grid(start_pose)
        goal = self.world_to_grid(goal_pose)
        
        # Use A* for initial path
        raw_path = self.a_star_search(start, goal, costmap)
        
        # Post-process for humanoid constraints
        humanoid_path = self.post_process_for_humanoid(raw_path)
        
        # Generate footstep plan
        footsteps = self.generate_footsteps(humanoid_path)
        
        return humanoid_path, footsteps
    
    def post_process_for_humanoid(self, raw_path):
        """
        Modify path to respect humanoid step constraints
        """
        processed_path = [raw_path[0]]
        
        i = 0
        while i < len(raw_path) - 1:
            j = i + 1
            
            # Find the furthest point reachable in one step
            while (j < len(raw_path) and 
                   self.calculate_distance(raw_path[i], raw_path[j]) < self.step_length_limit):
                j += 1
            
            # Add the furthest reachable point
            if j > i + 1:
                processed_path.append(raw_path[j-1])
                i = j - 1
            else:
                processed_path.append(raw_path[i+1])
                i += 1
        
        return processed_path
    
    def generate_footsteps(self, path):
        """
        Generate footstep sequence for the humanoid robot
        """
        footsteps = []
        left_support = True  # Start with left foot support
        
        for i in range(len(path) - 1):
            step = {
                'position': path[i+1],
                'orientation': self.calculate_step_orientation(path[i], path[i+1]),
                'step_type': 'left' if left_support else 'right',
                'timing': self.calculate_step_timing(path[i], path[i+1])
            }
            footsteps.append(step)
            left_support = not left_support  # Alternate feet
        
        return footsteps

# Example of footstep planning for humanoid robots
def plan_footsteps_for_humanoid(start_pos, goal_pos):
    """
    Generate footstep plan for humanoid robot navigation
    """
    # Calculate step sequence
    dx = goal_pos[0] - start_pos[0]
    dy = goal_pos[1] - start_pos[1]
    distance = np.sqrt(dx**2 + dy**2)
    
    # Estimate number of steps needed
    num_steps = max(int(distance / 0.3), 1)  # Assuming 0.3m step length
    
    footsteps = []
    for i in range(1, num_steps + 1):
        # Calculate intermediate position
        step_x = start_pos[0] + (dx / num_steps) * i
        step_y = start_pos[1] + (dy / num_steps) * i
        
        # Add footstep with alternating feet
        footstep = {
            'x': step_x,
            'y': step_y,
            'theta': np.arctan2(dy, dx),  # Heading direction
            'foot': 'left' if i % 2 == 0 else 'right',
            'step_count': i
        }
        footsteps.append(footstep)
    
    return footsteps
```

## Local Path Following and Control

### Controller Server Configuration

The controller server manages local path following and dynamic obstacle avoidance:

```python
# Example controller implementation
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import math

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        
        # Parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('goal_dist_threshold', 0.2),
                ('goal_yaw_threshold', 0.1),
                ('max_linear_vel', 0.5),
                ('max_angular_vel', 1.0),
                ('min_linear_vel', 0.1),
                ('min_angular_vel', 0.1),
                ('control_freq', 10.0),
                ('kp_linear', 1.0),
                ('kp_angular', 2.0)
            ]
        )
        
        # Get parameters
        self.goal_dist_threshold = self.get_parameter('goal_dist_threshold').value
        self.goal_yaw_threshold = self.get_parameter('goal_yaw_threshold').value
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        self.min_linear_vel = self.get_parameter('min_linear_vel').value
        self.min_angular_vel = self.get_parameter('min_angular_vel').value
        self.control_freq = self.get_parameter('control_freq').value
        self.kp_linear = self.get_parameter('kp_linear').value
        self.kp_angular = self.get_parameter('kp_angular').value
        
        # Subscribers and publishers
        self.path_sub = self.create_subscription(
            Path, 'global_plan', self.path_callback, 10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Path following variables
        self.current_path = None
        self.current_waypoint_idx = 0
        
        # Control timer
        self.control_timer = self.create_timer(1.0/self.control_freq, self.control_loop)
        
        self.get_logger().info('Humanoid Controller initialized')

    def path_callback(self, msg):
        """Receive and store path to follow"""
        self.current_path = msg
        self.current_waypoint_idx = 0
        self.get_logger().info(f'Received path with {len(msg.poses)} waypoints')

    def control_loop(self):
        """Main control loop for path following"""
        if self.current_path is None or len(self.current_path.poses) == 0:
            return
        
        # Get current robot pose
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time()
            )
            current_x = transform.transform.translation.x
            current_y = transform.transform.translation.y
            current_yaw = self.quaternion_to_yaw(transform.transform.rotation)
        except TransformException as e:
            self.get_logger().error(f'Could not get transform: {e}')
            return
        
        # Get current target waypoint
        if self.current_waypoint_idx >= len(self.current_path.poses):
            # Reached goal
            self.stop_robot()
            return
        
        target_pose = self.current_path.poses[self.current_waypoint_idx]
        target_x = target_pose.pose.position.x
        target_y = target_pose.pose.position.y
        
        # Calculate distance and angle to target
        dist_to_target = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
        
        target_yaw = math.atan2(target_y - current_y, target_x - current_x)
        angle_to_target = self.normalize_angle(target_yaw - current_yaw)
        
        # Check if reached current waypoint
        if dist_to_target < self.goal_dist_threshold:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.current_path.poses):
                # Reached end of path
                self.stop_robot()
                return
            
            # Get next waypoint
            target_pose = self.current_path.poses[self.current_waypoint_idx]
            target_x = target_pose.pose.position.x
            target_y = target_pose.pose.position.y
            dist_to_target = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
            target_yaw = math.atan2(target_y - current_y, target_x - current_x)
            angle_to_target = self.normalize_angle(target_yaw - current_yaw)
        
        # Calculate velocities - with humanoid-specific constraints
        linear_vel = min(self.kp_linear * dist_to_target, self.max_linear_vel)
        angular_vel = min(self.kp_angular * abs(angle_to_target), self.max_angular_vel)
        
        # Adjust sign of angular velocity
        if angle_to_target < 0:
            angular_vel = -angular_vel
        
        # Create and publish command
        cmd_msg = Twist()
        cmd_msg.linear.x = max(linear_vel, self.min_linear_vel)  # Ensure minimum velocity for stability
        cmd_msg.angular.z = angular_vel
        
        self.cmd_vel_pub.publish(cmd_msg)
        
        # Log control information
        self.get_logger().debug(
            f'Control: linear={cmd_msg.linear.x:.2f}, angular={cmd_msg.angular.z:.2f}, '
            f'distance={dist_to_target:.2f}, angle={math.degrees(angle_to_target):.1f}°'
        )

    def stop_robot(self):
        """Stop the robot movement"""
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0
        cmd_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_msg)
        self.get_logger().info('Robot stopped - reached goal')

    def quaternion_to_yaw(self, quat):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
```

## Nav2 Integration with Physical AI Systems

### Perception-Planning Integration

Integrating perception with navigation planning:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Bool
from builtin_interfaces.msg import Duration
import numpy as np
import tf2_ros

class PerceptionAwareNav(Node):
    def __init__(self):
        super().__init__('perception_aware_nav')
        
        # Create clients for navigation services
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Subscribers for perception data
        self.lidar_sub = self.create_subscription(
            LaserScan, 'scan', self.lidar_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )
        
        # Publishers for visualization
        self.obstacle_pub = self.create_publisher(MarkerArray, 'detected_obstacles', 10)
        
        # Perception processing
        self.obstacle_threshold = 1.0  # Minimum distance to consider obstacle
        self.robot_radius = 0.4  # Robot radius for collision checking
        
        # TF buffer for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.get_logger().info('Perception-aware navigation node initialized')

    def lidar_callback(self, msg):
        """Process LiDAR data for obstacle detection"""
        # Convert laser scan to obstacle markers
        obstacles = self.extract_obstacles_from_scan(msg)
        
        # Publish for visualization
        obstacle_markers = self.create_obstacle_markers(obstacles)
        self.obstacle_pub.publish(obstacle_markers)
        
        # Update local costmap with detected obstacles
        self.update_costmap_with_obstacles(obstacles)

    def extract_obstacles_from_scan(self, scan_msg):
        """Extract obstacle positions from laser scan"""
        obstacles = []
        angle = scan_msg.angle_min
        
        for range_val in scan_msg.ranges:
            if not np.isnan(range_val) and range_val < self.obstacle_threshold:
                # Convert polar to Cartesian coordinates
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)
                
                # Transform to map frame
                obstacle_map_frame = self.transform_to_map_frame(x, y, 0.0)
                
                if obstacle_map_frame is not None:
                    obstacles.append(obstacle_map_frame)
            
            angle += scan_msg.angle_increment
        
        return obstacles

    def navigate_with_obstacle_avoidance(self, goal_pose):
        """Navigate to goal with dynamic obstacle avoidance"""
        # Send initial navigation goal
        goal = NavigateToPose.Goal()
        goal.pose = goal_pose
        
        # Wait for result with feedback
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self.navigation_result_callback)
        
        # Continuously monitor environment for new obstacles
        self.obstacle_monitor_timer = self.create_timer(
            0.5,  # Check for obstacles every 0.5 seconds
            self.check_environment
        )

    def check_environment(self):
        """Check environment for new obstacles that might require replanning"""
        # Implement logic to detect if current path is blocked
        if self.is_path_blocked():
            # Cancel current navigation and replan
            self.nav_client.cancel_goal_async(self.current_goal_handle)
            self.replan_path()

    def is_path_blocked(self):
        """Check if current planned path is blocked by obstacles"""
        # Implementation would check current path against latest sensor data
        # This is a simplified version
        return False

    def replan_path(self):
        """Replan path considering new obstacles"""
        # Get current robot pose
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time()
            )
            current_pose = PoseStamped()
            current_pose.pose.position.x = transform.transform.translation.x
            current_pose.pose.position.y = transform.transform.translation.y
            current_pose.pose.position.z = transform.transform.translation.z
            current_pose.pose.orientation = transform.transform.rotation
        except tf2_ros.TransformException:
            self.get_logger().error('Could not get robot pose')
            return
        
        # Get goal pose (maintaining the same goal)
        # In practice, you'd store the original goal
        # For this example, assume we have a stored goal
        if hasattr(self, 'stored_goal'):
            self.navigate_with_obstacle_avoidance(self.stored_goal)
```

## Advanced Navigation Strategies

### Multi-Robot Navigation

For coordinating multiple humanoid robots:

```python
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from std_msgs.msg import String
import json

class MultiRobotNav(Node):
    def __init__(self):
        super().__init__('multi_robot_nav')
        
        # Communication channel for coordination
        self.coordination_pub = self.create_publisher(
            String, 'navigation_coordination', 10
        )
        self.coordination_sub = self.create_subscription(
            String, 'navigation_coordination', self.coordination_callback, 10
        )
        
        # Robot ID for coordination
        self.robot_id = self.declare_parameter('robot_id', 'robot_0').value
        
        # Active robots and their goals
        self.active_robots = {}
        
        self.get_logger().info(f'Multi-robot navigation node {self.robot_id} initialized')

    def coordination_callback(self, msg):
        """Handle coordination messages from other robots"""
        try:
            coord_data = json.loads(msg.data)
            sender_id = coord_data['robot_id']
            
            if sender_id != self.robot_id:
                self.active_robots[sender_id] = coord_data['current_goal']
        except Exception as e:
            self.get_logger().error(f'Error parsing coordination message: {e}')

    def request_path_reservation(self, path):
        """Request reservation of path to avoid conflicts with other robots"""
        # Calculate path segments
        path_segments = self.discretize_path(path, resolution=0.5)
        
        # Check for conflicts with other robots
        conflicts = self.check_path_conflicts(path_segments)
        
        if not conflicts:
            # No conflicts, proceed with navigation
            return True
        else:
            # Plan alternative route or wait
            return self.handle_conflicts(conflicts)

    def discretize_path(self, path, resolution=0.5):
        """Discretize path into segments for conflict checking"""
        segments = []
        
        for i in range(len(path.poses) - 1):
            start = path.poses[i].pose.position
            end = path.poses[i+1].pose.position
            
            # Calculate number of intermediate points
            dist = ((end.x - start.x)**2 + (end.y - start.y)**2)**0.5
            num_points = int(dist / resolution)
            
            for j in range(num_points):
                t = j / num_points
                x = start.x + t * (end.x - start.x)
                y = start.y + t * (end.y - start.y)
                
                segments.append((x, y, i))  # Include original segment index
        
        return segments

    def check_path_conflicts(self, path_segments):
        """Check if planned path conflicts with other robots"""
        conflicts = []
        
        for other_robot_id, other_goal in self.active_robots.items():
            # Check if paths intersect in space and time
            if self.paths_conflict(path_segments, other_goal):
                conflicts.append(other_robot_id)
        
        return conflicts

    def paths_conflict(self, path1_segments, path2_goal):
        """Check if two paths conflict"""
        # Implementation would consider timing and spatial overlap
        # Simplified for this example
        return False

    def handle_conflicts(self, conflicts):
        """Handle path conflicts with other robots"""
        # Implement negotiation strategy
        # For now, just return False to indicate replanning needed
        return False
```

## Navigation Safety and Recovery

### Recovery Behaviors

Nav2 includes various recovery behaviors to handle navigation failures:

```python
# Example of custom recovery behavior
from nav2_behavior_tree import Condition

class ClearPathRecovery(Condition):
    def __init__(self, name, condition_params):
        super().__init__(name)
        self.recovery_count = 0
        self.max_recoveries = 3

    def condition(self):
        # Check if clear path is available
        local_costmap = self.context.get_local_costmap()
        
        # Check immediate area around robot
        robot_pos = self.context.get_robot_position()
        
        # Check if path is clear within robot's footprint
        if self.is_path_clear(robot_pos, local_costmap):
            self.recovery_count = 0  # Reset on success
            return True
        else:
            if self.recovery_count < self.max_recoveries:
                # Attempt to clear path by moving slightly
                self.execute_recovery()
                self.recovery_count += 1
                return False  # Still attempting recovery
            else:
                # Too many failures, give up
                return False

    def is_path_clear(self, robot_pos, costmap):
        """Check if immediate path is clear"""
        # Implementation would check costmap values around robot
        return True  # Simplified for example

    def execute_recovery(self):
        """Execute recovery action"""
        # Move robot slightly to clear immediate obstacles
        recovery_cmd = Twist()
        recovery_cmd.linear.x = 0.1  # Move forward slowly
        recovery_cmd.angular.z = 0.0
        # Publish command to robot
```

## Performance Optimization

### Computational Efficiency

Optimizing navigation for real-time performance:

```python
class EfficientPathPlanner:
    def __init__(self):
        self.path_cache = {}
        self.last_poses = {}
        
        # Use spatial data structures for efficient lookup
        from scipy.spatial import cKDTree
        self.cached_paths_kdtree = None
    
    def get_cached_path(self, start, goal, tolerance=0.5):
        """Get path from cache if start/end points are similar"""
        cache_key = (round(start[0], 1), round(start[1], 1), 
                     round(goal[0], 1), round(goal[1], 1))
        
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        return None
    
    def plan_path_with_optimization(self, start, goal, costmap):
        """Plan path with optimization techniques"""
        # First, check if path is available in cache
        cached_path = self.get_cached_path(start, goal)
        if cached_path:
            return cached_path
        
        # Simplified path planning with optimization
        # Use grid-based approach for efficiency
        grid_resolution = 0.5  # meters per grid cell
        
        # Convert world coordinates to grid coordinates
        start_grid = (int(start[0] / grid_resolution), int(start[1] / grid_resolution))
        goal_grid = (int(goal[0] / grid_resolution), int(goal[1] / grid_resolution))
        
        # Use A* for path planning
        path = self.a_star_path_planner(start_grid, goal_grid, costmap)
        
        # Convert back to world coordinates
        world_path = [(x * grid_resolution, y * grid_resolution) for x, y in path]
        
        # Cache the result
        cache_key = (round(start[0], 1), round(start[1], 1), 
                     round(goal[0], 1), round(goal[1], 1))
        self.path_cache[cache_key] = world_path
        
        return world_path
```

## Troubleshooting Common Issues

### Issue 1: Path Planning Fails
- **Symptoms**: Robot cannot find path to goal
- **Solutions**: Check costmap settings, increase inflation radius, validate map quality

### Issue 2: Navigation Oscillation
- **Symptoms**: Robot oscillates around obstacles
- **Solutions**: Adjust controller parameters, increase minimum velocities, tune PID gains

### Issue 3: Local Minima Problems
- **Symptoms**: Robot gets stuck in local obstacles
- **Solutions**: Implement better recovery behaviors, use more sophisticated planners

### Issue 4: Inconsistent Behavior
- **Symptoms**: Navigation works sometimes but not others
- **Solutions**: Check sensor reliability, validate TF trees, ensure proper synchronization

## Best Practices

1. **Parameter Tuning**: Carefully tune Nav2 parameters for humanoid robot characteristics
2. **Map Quality**: Ensure high-quality maps for reliable localization
3. **Sensor Validation**: Verify sensor data quality before navigation
4. **Safety Measures**: Implement appropriate safety checks and fallbacks
5. **Testing**: Extensively test navigation in various environments and scenarios
6. **Monitoring**: Implement comprehensive monitoring and logging for debugging

## Summary

Nav2 provides a robust framework for humanoid robot navigation, integrating path planning, obstacle avoidance, and dynamic replanning capabilities. By properly configuring Nav2 for humanoid-specific characteristics and constraints, robots can effectively navigate complex environments while maintaining balance and stability.

The integration of perception with navigation planning creates intelligent systems that can adapt to dynamic environments and make informed decisions about movement. These capabilities are essential for Physical AI systems that must operate in real-world environments, bridging the gap between digital AI models and physical robotic bodies.

Understanding the Nav2 architecture and its components enables the development of sophisticated navigation behaviors that allow humanoid robots to operate autonomously and safely in diverse environments.