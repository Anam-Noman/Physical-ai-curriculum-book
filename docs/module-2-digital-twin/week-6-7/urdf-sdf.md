---
sidebar_position: 2
---

# Robot Description Formats: URDF and SDF

## Introduction

In the Physical AI and robotics ecosystem, accurately describing robot models is crucial for simulation, control, and visualization. The two primary formats for robot description in ROS/Gazebo environments are URDF (Unified Robot Description Format) and SDF (Simulation Description Format). Understanding both formats and when to use each is essential for effective Digital Twin development.

This section explores both formats, their use cases, and how they integrate with simulation environments like Gazebo to create accurate virtual representations of physical robots.

## Unified Robot Description Format (URDF)

URDF is the standard format for representing robot models in ROS. It's an XML-based format that describes a robot's kinematic and dynamic properties, visual appearance, and collision properties.

### URDF Structure

A URDF file contains:
- **Links**: Rigid body elements of the robot
- **Joints**: Kinematic relationships between links
- **Materials**: Visual properties and colors
- **Gazebo-specific elements**: Simulation-specific properties

### Example URDF

```xml
<?xml version="1.0"?>
<robot name="simple_robot">

  <!-- Base link with inertial, visual, and collision properties -->
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.2 0.2 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.2 0.2 0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- A simple wheel link -->
  <link name="wheel">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Joint connecting wheel to base -->
  <joint name="wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel"/>
    <origin xyz="0.15 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

</robot>
```

### URDF Limitations

While URDF is excellent for describing robot kinematics and basic properties, it has some limitations:
- Less suitable for complex simulation scenarios
- Limited support for multi-robot environments
- Not ideal for describing entire simulation worlds
- No support for plugins directly

## Simulation Description Format (SDF)

SDF (Simulation Description Format) is the native format for Gazebo simulation. It's more comprehensive than URDF and designed specifically for simulation environments, though it can also describe robots.

### SDF Structure

An SDF file typically contains:
- **World elements**: Complete simulation environments
- **Model elements**: Individual robot or object descriptions  
- **Physics engines**: Simulation parameters
- **Light sources**: Lighting in the environment
- **Plugins**: Custom simulation behaviors

### Example SDF

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_world">
    <!-- Physics parameters -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Include a ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include a light source -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Define a robot model -->
    <model name="simple_robot">
      <pose>0 0 0.1 0 0 0</pose>
      
      <!-- Base link -->
      <link name="base_link">
        <pose>0 0 0 0 0 0</pose>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.01</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>0.01</iyy>
            <iyz>0.0</iyz>
            <izz>0.01</izz>
          </inertia>
        </inertial>
        
        <visual name="visual">
          <geometry>
            <box>
              <size>0.2 0.2 0.2</size>
            </box>
          </geometry>
          <material>
            <ambient>0 0 1 1</ambient>
            <diffuse>0 0 1 1</diffuse>
          </material>
        </visual>
        
        <collision name="collision">
          <geometry>
            <box>
              <size>0.2 0.2 0.2</size>
            </box>
          </geometry>
        </collision>
      </link>
      
      <!-- Wheel link -->
      <link name="wheel">
        <pose>0.15 0 0 0 0 0</pose>
        <inertial>
          <mass>0.1</mass>
          <inertia>
            <ixx>0.001</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>0.001</iyy>
            <iyz>0.0</iyz>
            <izz>0.001</izz>
          </inertia>
        </inertial>
        
        <visual name="wheel_visual">
          <geometry>
            <cylinder>
              <length>0.05</length>
              <radius>0.1</radius>
            </cylinder>
          </geometry>
          <material>
            <ambient>0 0 0 1</ambient>
            <diffuse>0 0 0 1</diffuse>
          </material>
        </visual>
        
        <collision name="wheel_collision">
          <geometry>
            <cylinder>
              <length>0.05</length>
              <radius>0.1</radius>
            </cylinder>
          </geometry>
        </collision>
      </link>
      
      <!-- Joint connecting wheel to base -->
      <joint name="wheel_joint" type="revolute">
        <parent>base_link</parent>
        <child>wheel</child>
        <axis>
          <xyz>0 1 0</xyz>
        </axis>
      </joint>
    </model>
  </world>
</sdf>
```

## URDF to SDF Conversion

In many cases, you'll have a URDF model but need to use it in Gazebo, which requires SDF. The `gz sdf` command can help with this:

```bash
# Convert URDF to SDF
gz sdf -p robot.urdf > robot.sdf

# Or using the older command
gz sdf -p robot.urdf > robot.sdf
```

### Using xacro with SDF

The xacro package, which provides macros for URDF, can also be used with SDF:

```xml
<!-- robot.xacro -->
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="my_robot">

  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_width" value="0.05" />
  
  <!-- Define a macro for wheels -->
  <xacro:macro name="wheel" params="prefix parent x y z">
    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${x} ${y} ${z}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
    </joint>
    
    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder length="${wheel_width}" radius="${wheel_radius}"/>
        </geometry>
      </visual>
      <collision>
        <geometry>
          <cylinder length="${wheel_width}" radius="${wheel_radius}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.1"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Robot base -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Create wheels using the macro -->
  <xacro:wheel prefix="front_left" parent="base_link" x="0.2" y="0.2" z="0"/>
  <xacro:wheel prefix="front_right" parent="base_link" x="0.2" y="-0.2" z="0"/>
  <xacro:wheel prefix="back_left" parent="base_link" x="-0.2" y="0.2" z="0"/>
  <xacro:wheel prefix="back_right" parent="base_link" x="-0.2" y="-0.2" z="0"/>

</robot>
```

Then convert with:
```bash
# Process xacro to URDF first
ros2 run xacro xacro robot.xacro > robot.urdf

# Convert URDF to SDF
gz sdf -p robot.urdf > robot.sdf
```

## Gazebo-Specific Extensions to URDF

When using URDF models in Gazebo, you can add Gazebo-specific tags to enhance the simulation:

```xml
<?xml version="1.0"?>
<robot name="gazebo_enhanced_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.2 0.2 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.2 0.2 0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Gazebo-specific extensions -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
    <mu1>0.9</mu1>
    <mu2>0.9</mu2>
    <self_collide>false</self_collide>
    <gravity>true</gravity>
  </gazebo>

  <!-- Gazebo plugin for ROS control -->
  <gazebo>
    <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
      <parameters>$(find my_robot_description)/config/robot_control.yaml</parameters>
    </plugin>
  </gazebo>

</robot>
```

## Model Organization and Best Practices

### Model Directory Structure

When organizing robot models for simulation, follow this structure:

```
my_robot_description/
├── urdf/
│   ├── my_robot.urdf
│   └── my_robot.xacro
├── meshes/
│   ├── base_link.stl
│   ├── wheel.dae
│   └── ...
├── config/
│   └── robot_control.yaml
└── launch/
    └── spawn_robot.launch.py
```

### URDF Best Practices

1. **Use consistent naming**: Apply a consistent naming scheme for links and joints
2. **Group related elements**: Organize related links and joints together
3. **Use xacro**: Utilize xacro macros for repetitive elements
4. **Include proper inertias**: Calculate accurate inertial properties for realistic simulation
5. **Separate visual and collision**: Use different geometries for visual and collision models when appropriate

### SDF Best Practices

1. **Define appropriate physics parameters**: Tune simulation parameters to match real-world behavior
2. **Use plugins wisely**: Add plugins only when necessary for simulation behavior
3. **Organize models**: Keep complex world files organized and modular
4. **Validate files**: Use validation tools to check for errors

## Using URDF/SDF in Gazebo with ROS 2

### Spawning Models in Gazebo

Once you have your URDF/SDF model, you can spawn it in Gazebo using ROS 2:

```python
# Spawn entity node
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        # Launch Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('ros_gz_sim'),
                    'launch',
                    'gz_sim.launch.py'
                ])
            ]),
            launch_arguments={'gz_args': '-r empty.sdf'}.items()
        ),
        
        # Spawn the robot
        Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-name', 'my_robot',
                '-topic', 'robot_description',
                '-x', '0',
                '-y', '0', 
                '-z', '0.1'
            ],
            output='screen'
        )
    ])
```

### Loading Robot Description

To load your robot description in ROS 2:

```python
# robot_state_publisher launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            'description_file',
            default_value='my_robot.urdf',
            description='URDF file name'
        )
    )

    # Get URDF via xacro
    robot_description = Command([
        PathJoinSubstitution([FindPackageShare('my_robot_description'), 'urdf', LaunchConfiguration('description_file')]),
    ])

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
    )

    return LaunchDescription(declared_arguments + [robot_state_publisher])
```

## Visualization and Debugging

### Checking Model Validity

Verify your URDF/SDF models before simulation:

```bash
# Check URDF validity
check_urdf my_robot.urdf

# Convert and check SDF
gz sdf -p my_robot.urdf

# Visualize URDF in Rviz
ros2 run rviz2 rviz2
```

### Debugging Tips

1. **Check joint limits**: Ensure joint limits in URDF match real robot capabilities
2. **Verify inertial properties**: Incorrect inertias can cause unrealistic simulation behavior
3. **Validate transforms**: Use `tf2_tools` to check robot transforms
4. **Compare simulation to real**: Validate that simulation behavior matches real robot characteristics

## Summary

URDF and SDF are fundamental formats for describing robots and simulation environments in the Physical AI pipeline. URDF is ideal for robot kinematic and dynamic description within the ROS ecosystem, while SDF provides comprehensive simulation capabilities in Gazebo.

Key takeaways:
- Use URDF for robot descriptions and convert to SDF for Gazebo simulation
- Apply Gazebo-specific extensions to URDF for enhanced simulation properties
- Follow best practices for model organization and naming
- Validate models before simulation to avoid issues during Physical AI development

These formats form the foundation for creating accurate Digital Twins that bridge the gap between digital AI models and physical robotic bodies. Understanding how to properly describe robots and environments is essential for effective Physical AI development and validation.

In the next sections, we'll explore how to implement physics simulation of gravity, collisions, and dynamics in these environments.