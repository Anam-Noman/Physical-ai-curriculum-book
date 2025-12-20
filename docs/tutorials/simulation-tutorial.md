---
sidebar_position: 1
---

# Simulation Tutorial: Creating Your First Robot Simulation

## Introduction

This tutorial will guide you through creating your first complete robot simulation environment using Gazebo and ROS 2. You'll build a simple differential drive robot with sensors that can navigate around a virtual world, demonstrating the core concepts of Digital Twin technology for Physical AI systems.

By the end of this tutorial, you'll have:
- Created a robot model with proper URDF description
- Set up a Gazebo simulation environment
- Added realistic sensors (camera and LiDAR)
- Implemented basic robot control using ROS 2
- Validated the simulation through navigation tasks

## Prerequisites

Before starting this tutorial, ensure you have:
- ROS 2 Humble Hawksbill installed and configured
- Gazebo installed and verified
- Basic Python and ROS 2 knowledge
- Completed the basic ROS 2 tutorial

## Step 1: Setting Up Your Workspace

### Create a New ROS 2 Package

First, let's create a workspace and package for our simulation:

```bash
# Create a new workspace
mkdir -p ~/simulation_ws/src
cd ~/simulation_ws

# Create a new package for our robot simulation
ros2 pkg create --build-type ament_python robot_simulation_tutorial --dependencies rclpy geometry_msgs sensor_msgs std_msgs ros_gz_bridge
```

### Create Directory Structure

```bash
cd ~/simulation_ws/src/robot_simulation_tutorial
mkdir -p launch meshes urdf config worlds
```

## Step 2: Creating the Robot Model (URDF)

### Create the Basic Robot URDF

Create `urdf/turtlebot_example.urdf`:

```xml
<?xml version="1.0"?>
<robot name="turtlebot_example" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Base Link -->
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.1"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.02"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0.1"/>
      <geometry>
        <cylinder radius="0.15" length="0.1"/>
      </geometry>
      <material name="green">
        <color rgba="0 0.8 0 1"/>
      </material>
    </visual>
    
    <collision>
      <origin xyz="0 0 0.1"/>
      <geometry>
        <cylinder radius="0.15" length="0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Wheel -->
  <link name="wheel_left">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.02"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.02"/>
      </geometry>
    </collision>
  </link>

  <!-- Right Wheel -->
  <link name="wheel_right">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.02"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.02"/>
      </geometry>
    </collision>
  </link>

  <!-- Camera Link -->
  <link name="camera_link">
    <inertial>
      <mass value="0.01"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.02 0.08 0.04"/>
      </geometry>
      <material name="black"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.02 0.08 0.04"/>
      </geometry>
    </collision>
  </link>

  <!-- Joints -->
  <joint name="base_to_wheel_left" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_left"/>
    <origin xyz="0 0.15 0.05" rpy="-1.570796 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="base_to_wheel_right" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_right"/>
    <origin xyz="0 -0.15 0.05" rpy="-1.570796 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="base_to_camera" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.1 0 0.1" rpy="0 0 0"/>
  </joint>

</robot>
```

## Step 3: Adding Gazebo-Specific Extensions

### Create the Gazebo URDF with Plugins

Create `urdf/turtlebot_example.gazebo.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Gazebo Material -->
  <gazebo reference="base_link">
    <material>Gazebo/Green</material>
  </gazebo>

  <gazebo reference="wheel_left">
    <material>Gazebo/Black</material>
  </gazebo>

  <gazebo reference="wheel_right">
    <material>Gazebo/Black</material>
  </gazebo>

  <gazebo reference="camera_link">
    <material>Gazebo/Black</material>
  </gazebo>

  <!-- Gazebo Plugin for Differential Drive -->
  <gazebo>
    <plugin filename="libgazebo_ros_diff_drive.so" name="differential_drive">
      <left_joint>wheel_left_joint</left_joint>
      <right_joint>wheel_right_joint</right_joint>
      <wheel_separation>0.3</wheel_separation>
      <wheel_diameter>0.1</wheel_diameter>
      <max_wheel_torque>20</max_wheel_torque>
      <max_wheel_acceleration>10.0</max_wheel_acceleration>
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
      <publish_odom>true</publish_odom>
      <publish_odom_tf>true</publish_odom_tf>
      <publish_wheel_tf>true</publish_wheel_tf>
    </plugin>
  </gazebo>

  <!-- Camera Sensor -->
  <gazebo reference="camera_link">
    <sensor name="camera" type="camera">
      <update_rate>30</update_rate>
      <camera name="head">
        <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.05</near>
          <far>3</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>camera_link</frame_name>
        <topic_name>camera/image_raw</topic_name>
      </plugin>
    </sensor>
  </gazebo>

  <!-- LiDAR Sensor -->
  <gazebo reference="base_link">
    <sensor name="lidar" type="ray">
      <update_rate>10</update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>360</samples>
            <resolution>1</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>10</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
        <ros>
          <namespace>/</namespace>
          <remapping>~/out:=scan</remapping>
        </ros>
        <output_type>sensor_msgs/LaserScan</output_type>
      </plugin>
    </sensor>
  </gazebo>

</robot>
```

### Create Combined URDF

Create `urdf/turtlebot_example.urdf.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="turtlebot_example">
  
  <!-- Include the base robot -->
  <xacro:include filename="turtlebot_example.urdf"/>
  
  <!-- Include Gazebo-specific extensions -->
  <xacro:include filename="turtlebot_example.gazebo.xacro"/>

</robot>
```

## Step 4: Creating a Simulation World

Create `worlds/simple_room.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_room">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_update_rate>1000</real_time_update_rate>
      <real_time_factor>1</real_time_factor>
    </physics>

    <!-- Gravity -->
    <gravity>0 0 -9.8</gravity>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Sun light -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Room walls -->
    <model name="room_walls">
      <!-- Wall 1: Front wall -->
      <link name="wall_1">
        <pose>-3 0 1.5 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 6 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 6 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>

      <!-- Wall 2: Back wall -->
      <link name="wall_2">
        <pose>3 0 1.5 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 6 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 6 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>

      <!-- Wall 3: Left wall -->
      <link name="wall_3">
        <pose>0 -3 1.5 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>6 0.1 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>6 0.1 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>

      <!-- Wall 4: Right wall -->
      <link name="wall_4">
        <pose>0 3 1.5 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>6 0.1 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>6 0.1 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Obstacle -->
    <model name="obstacle">
      <pose>1 1 0.2 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.4 0.4 0.4</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.4 0.4 0.4</size>
            </box>
          </geometry>
          <material>
            <ambient>0.9 0.4 0.1 1</ambient>
            <diffuse>0.9 0.4 0.1 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.01</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.01</iyy>
            <iyz>0</iyz>
            <izz>0.01</izz>
          </inertia>
        </inertial>
      </link>
    </model>

  </world>
</sdf>
```

## Step 5: Creating Launch Files

Create `launch/simulation_launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            'world',
            default_value='simple_room.sdf',
            description='Choose one of the world files from `/robot_simulation_tutorial/worlds`'
        )
    )
    
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Whether to launch RViz'
        )
    )

    # Launch arguments
    world = LaunchConfiguration('world')
    use_rviz = LaunchConfiguration('use_rviz')

    # Get URDF via xacro
    robot_description_content = PathJoinSubstitution([
        FindPackageShare('robot_simulation_tutorial'),
        'urdf',
        'turtlebot_example.urdf.xacro'
    ])

    # Robot State Publisher node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description_content}],
    )

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={'gz_args': PathJoinSubstitution([
            FindPackageShare('robot_simulation_tutorial'),
            'worlds',
            world
        ])}.items()
    )

    # Spawn robot in Gazebo
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'turtlebot_example',
            '-topic', 'robot_description',
            '-x', '0',
            '-y', '0',
            '-z', '0.2'
        ],
        output='screen'
    )

    # RViz2 node
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        condition=IfCondition(use_rviz)
    )

    # Create launch description
    ld = LaunchDescription(declared_arguments)
    ld.add_action(gazebo)
    ld.add_action(spawn_robot)
    ld.add_action(robot_state_publisher_node)
    ld.add_action(rviz2_node)

    return ld
```

## Step 6: Creating a Simple Navigation Node

Create `robot_simulation_tutorial/navigation_example.py`:

```python
#!/usr/bin/env python3

"""
Simple navigation example for the simulated robot.
This node subscribes to laser scan data and publishes velocity commands
to navigate around obstacles.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from math import pi


class SimpleNavigation(Node):
    def __init__(self):
        super().__init__('simple_navigation')
        
        # Create publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Create subscriber for laser scan
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )
        
        # Timer for periodic control updates
        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz
        
        # Robot state
        self.obstacle_detected = False
        self.scan_data = None
        
        self.get_logger().info('Simple Navigation node initialized')

    def scan_callback(self, msg):
        """Process incoming laser scan data"""
        self.scan_data = msg
        # Check for obstacles in front of the robot
        min_range = float('inf')
        
        # Look at the front 30 degrees (15 degrees on each side of center)
        front_range_start = len(msg.ranges) // 2 - 15
        front_range_end = len(msg.ranges) // 2 + 15
        
        for i in range(front_range_start, front_range_end):
            if 0 <= i < len(msg.ranges):
                if msg.ranges[i] < min_range and not float('inf') == msg.ranges[i]:
                    min_range = msg.ranges[i]
        
        # Set obstacle flag if obstacle is closer than 1 meter
        self.obstacle_detected = min_range < 1.0

    def control_loop(self):
        """Main control loop"""
        if self.scan_data is None:
            return
            
        cmd_msg = Twist()
        
        # Simple obstacle avoidance behavior
        if self.obstacle_detected:
            # If obstacle detected, turn right
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = -0.5  # Turn right
        else:
            # Otherwise, move forward
            cmd_msg.linear.x = 0.5   # Move forward at 0.5 m/s
            cmd_msg.angular.z = 0.0  # No rotation
        
        # Publish the command
        self.cmd_vel_pub.publish(cmd_msg)


def main(args=None):
    rclpy.init(args=args)
    
    navigator = SimpleNavigation()
    
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Step 7: Creating Setup Scripts and Package Configuration

### Update setup.py

Edit `setup.py` to include the new Python script:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'robot_simulation_tutorial'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include all URDF files
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        # Include all world files
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Robot Simulation Tutorial Package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'navigation_example = robot_simulation_tutorial.navigation_example:main',
        ],
    },
)
```

## Step 8: Running the Simulation

### Building the Package

First, build your ROS 2 package:

```bash
cd ~/simulation_ws
colcon build --packages-select robot_simulation_tutorial
source install/setup.bash
```

### Launching the Simulation

Now you can launch the complete simulation:

```bash
# Launch the simulation with the simple room world
ros2 launch robot_simulation_tutorial simulation_launch.py

# In a new terminal, run the navigation node
ros2 run robot_simulation_tutorial navigation_example
```

### Testing the Simulation

1. Open Gazebo GUI to visualize the robot in the simulated environment
2. Use `rviz2` to visualize sensor data and robot state
3. Monitor topics with `ros2 topic echo /scan` and `ros2 topic echo /odom`
4. The robot should navigate autonomously, avoiding obstacles in its path

## Troubleshooting Common Issues

### Issue 1: Robot Doesn't Move
- **Symptom**: The robot remains stationary in Gazebo
- **Solution**: Check that the `cmd_vel` topic is properly connected:
  ```bash
  ros2 topic echo /cmd_vel
  ros2 topic list
  ```

### Issue 2: Sensor Data Not Available
- **Symptom**: No data on sensor topics like `/scan` or `/camera/image_raw`
- **Solution**: Verify that Gazebo plugins are properly loaded:
  ```bash
  gz topic -l
  ros2 run ros_gz_bridge parameter_bridge
  ```

### Issue 3: URDF Not Loading
- **Symptom**: Error when spawning the robot: "No valid robot description found"
- **Solution**: Check that the URDF file path is correct and accessible:
  ```bash
  ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="$(xacro path/to/urdf.xacro)"
  ```

## Extending the Tutorial

### Adding More Sensors

You can extend this tutorial by adding additional sensors. For example, to add an IMU:

```xml
<!-- Add to the base_link in your URDF -->
<sensor name="imu" type="imu">
  <update_rate>100</update_rate>
  <topic>imu/data</topic>
  <always_on>true</always_on>
  <visualize>false</visualize>
</sensor>
```

### Implementing More Complex Behaviors

Try implementing more advanced navigation behaviors:
- Goal-based navigation (using MoveIt2 or Nav2)
- Mapping (using SLAM packages)
- Object recognition (using computer vision)
- Multi-robot scenarios

### Performance Optimization

For larger simulations:
- Reduce the complexity of meshes
- Adjust update rates for sensors
- Use simpler collision geometries
- Limit the number of objects in the simulation

## Summary

This tutorial has walked you through creating a complete robot simulation environment using Gazebo and ROS 2. You've learned to:

- Create robot models with proper URDF descriptions
- Add realistic sensors and physics properties
- Set up simulation environments with obstacles
- Implement basic robot control algorithms
- Integrate everything with ROS 2

These skills form the foundation for more complex Physical AI applications, allowing you to safely develop and test robot behaviors in simulation before deploying to real hardware. The Digital Twin approach enables rapid iteration and validation of complex AI systems in a controlled environment.

The simulation provides a safe, cost-effective way to test and refine Physical AI algorithms before deploying to real robots, significantly reducing the risk and cost of development.