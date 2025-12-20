---
sidebar_position: 6
---

# URDF for Humanoid Robot Structure

## Understanding URDF

The Unified Robot Description Format (URDF) is the standard format for representing robot models in ROS. URDF is an XML-based format that describes a robot's physical and visual properties, including its kinematic structure, inertial properties, visual appearance, and collision properties.

URDF serves as the bridge between digital robot models and their physical counterparts, allowing simulation environments and controllers to reason about the robot's structure and capabilities. For humanoid robots, URDF is particularly important as it defines the complex kinematic structure of limbs, joints, and sensors.

### What URDF Describes

- **Kinematic structure**: The arrangement of links and joints
- **Inertial properties**: Mass, center of mass, and moments of inertia for each link
- **Visual representation**: Meshes, colors, and textures for visualization
- **Collision properties**: Shapes used for collision detection
- **Sensors**: Locations and types of various sensors on the robot

## URDF Structure

A URDF file is composed of several main elements:

### Robot Element
The root element of a URDF file is the `<robot>` tag, which contains the entire robot definition:

```xml
<robot name="my_robot">
  <!-- Links and joints go here -->
</robot>
```

### Links
Links represent the rigid bodies of the robot. Each link contains information about its inertial properties, visual representation, and collision properties:

```xml
<link name="link_name">
  <inertial>
    <mass value="1.0"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
  </inertial>
  
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <cylinder length="0.1" radius="0.05"/>
    </geometry>
    <material name="blue">
      <color rgba="0 0 1 1"/>
    </material>
  </visual>
  
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <cylinder length="0.1" radius="0.05"/>
    </geometry>
  </collision>
</link>
```

### Joints
Joints connect links together and define the kinematic relationships between them:

```xml
<joint name="joint_name" type="revolute">
  <parent link="parent_link_name"/>
  <child link="child_link_name"/>
  <origin xyz="0.1 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
</joint>
```

## Joint Types

URDF supports several joint types that define the degrees of freedom between connected links:

### Fixed Joint
A fixed joint constrains all motion between two links, effectively making them a single rigid body:

```xml
<joint name="fixed_joint" type="fixed">
  <parent link="link1"/>
  <child link="link2"/>
</joint>
```

### Revolute Joint
A revolute joint allows rotation around a single axis with limited range:

```xml
<joint name="hinge_joint" type="revolute">
  <parent link="upper_arm"/>
  <child link="forearm"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
</joint>
```

### Continuous Joint
Similar to a revolute joint but with unlimited rotation:

```xml
<joint name="continuous_joint" type="continuous">
  <parent link="base"/>
  <child link="rotating_part"/>
  <axis xyz="0 0 1"/>
</joint>
```

### Prismatic Joint
Allows linear motion along a single axis:

```xml
<joint name="slider_joint" type="prismatic">
  <parent link="base"/>
  <child link="slider"/>
  <axis xyz="1 0 0"/>
  <limit lower="0" upper="0.5" effort="100" velocity="1"/>
</joint>
```

## Complete URDF Example: Simple Humanoid Robot

Here's a simplified URDF for a basic humanoid robot:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">

  <!-- Base/Body Link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0.5"/>
      <geometry>
        <box size="0.5 0.3 1.0"/>
      </geometry>
      <material name="body_color">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    
    <collision>
      <origin xyz="0 0 0.5"/>
      <geometry>
        <box size="0.5 0.3 1.0"/>
      </geometry>
    </collision>
  </link>

  <!-- Head Link -->
  <link name="head">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
      <material name="head_color">
        <color rgba="1.0 0.8 0.6 1.0"/>
      </material>
    </visual>
    
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </collision>
  </link>

  <!-- Joint connecting head to body -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 1.0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
  </joint>

  <!-- Left Arm Example -->
  <link name="left_upper_arm">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.15"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 -0.15"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="arm_color">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    
    <collision>
      <origin xyz="0 0 -0.15"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
  </link>

  <!-- Joint connecting left arm to body -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0.2 0.5"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>

</robot>
```

## URDF and ROS 2 Integration

### Loading URDF in ROS 2
In ROS 2, URDFs are typically loaded into the parameter server and accessed through the `robot_state_publisher` node:

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
import math

class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')
        
        # Create a QoS profile for the joint state subscriber
        qos_profile = QoSProfile(depth=10)
        
        # Subscribe to joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            qos_profile
        )
        
        # Create transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
    def joint_state_callback(self, msg):
        # Process joint states and publish transforms
        # This is simplified - a real implementation would be more complex
        pass
```

### Using xacro
For complex robots, URDFs can become very large and repetitive. The XML Macro (xacro) package allows you to create more maintainable URDFs:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_xacro">

  <!-- Define a property -->
  <xacro:property name="M_PI" value="3.1415926535897931" />

  <!-- Define a macro for repeated elements -->
  <xacro:macro name="simple_arm" params="side">
    <link name="${side}_upper_arm">
      <visual>
        <geometry>
          <cylinder length="0.3" radius="0.05"/>
        </geometry>
      </visual>
    </link>
    
    <joint name="${side}_shoulder_joint" type="revolute">
      <parent link="base_link"/>
      <child link="${side}_upper_arm"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
    </joint>
  </xacro:macro>

  <!-- Use the macro to create both arms -->
  <xacro:simple_arm side="left"/>
  <xacro:simple_arm side="right"/>

</robot>
```

## URDF Best Practices

### Naming Conventions
- Use consistent naming schemes (e.g., snake_case)
- Include descriptive names that indicate functionality
- Use prefixes for related components (e.g., left_arm, right_arm)

### Inertial Properties
- Accurate inertial properties are crucial for physics simulation
- Use CAD tools to calculate exact inertial properties
- For real robots, determine properties through measurement or CAD models

### Visual vs. Collision Models
- Use detailed visual meshes for rendering
- Use simpler collision geometries for performance
- Align both visual and collision frames properly

## Humanoid-Specific Considerations

### Kinematic Chains
Humanoid robots have multiple kinematic chains (legs, arms) that must be properly connected to represent walking and manipulation capabilities.

### Degrees of Freedom
Humanoid robots typically have many degrees of freedom, requiring careful joint limit configuration to represent human-like ranges of motion.

### Balance and Stability
Proper center of mass calculation is crucial for humanoid robot simulation, especially for bipedal locomotion.

## Validation and Debugging

### URDF Validation
Use the `check_urdf` tool to validate your URDF:

```bash
# Assuming your URDF is in a package called my_robot_description
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:=$(ros2 pkg prefix my_robot_description)/share/my_robot_description/urdf/my_robot.urdf
```

### Visualization
Use RViz2 to visualize your robot model and verify it matches your expectations:

```bash
ros2 run rviz2 rviz2
```

## Advanced URDF Features

### Gazebo-Specific Elements
URDF can include Gazebo-specific elements for simulation:

```xml
<gazebo reference="link_name">
  <material>Gazebo/Blue</material>
  <mu1>0.9</mu1>
  <mu2>0.9</mu2>
</gazebo>
```

### Transmission Elements
Define how joints are controlled:

```xml
<transmission name="left_wheel_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_wheel_joint">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_wheel_motor">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## Summary

URDF is a fundamental component of the ROS ecosystem, allowing for the description of robot models that bridge the gap between digital representations and physical robot bodies. For humanoid robots, proper URDF modeling is essential for accurate simulation and effective control.

Understanding the structure and components of URDFs, along with best practices for their creation, is crucial for creating embodied intelligence systems that can effectively interact with the physical world through humanoid robot platforms.

In the next sections, we'll look at how to validate these models and use them in simulation and control systems.