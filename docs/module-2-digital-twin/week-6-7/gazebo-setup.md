---
sidebar_position: 1
---

# Gazebo Simulation Environment Setup

## Introduction to Gazebo

Gazebo is a powerful physics-based simulation environment that provides realistic rendering, accurate physics simulation, and sensor emulation for robotics applications. It's an essential tool in the Physical AI pipeline, allowing developers to test and validate robot behaviors before deploying to real hardware.

Gazebo simulates complex environments with accurate physics properties, making it ideal for testing Physical AI systems that must interact with the physical world. It supports integration with ROS 2 through the Gazebo ROS packages, enabling seamless communication between simulated robots and ROS 2 nodes.

## Installing Gazebo

### Prerequisites

Before installing Gazebo, ensure you have:
- A compatible Linux distribution (Ubuntu 22.04+ recommended for Gazebo Garden/Harmonic)
- ROS 2 Humble Hawksbill installed
- Sufficient hardware resources (dedicated GPU recommended for rendering)

### Installation Methods

#### Method 1: Installing Gazebo from OSRF Repository (Recommended)

For Ubuntu 22.04 with Gazebo Garden:

```bash
# Add the OSRF apt repository
sudo apt update && sudo apt install -y wget
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# Update and install Gazebo
sudo apt update
sudo apt install gazebo-garden
```

For the latest version (Gazebo Harmonic):
```bash
# Add the OSRF apt repository for the latest
sudo apt update && sudo apt install -y wget
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-latest $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-latest.list > /dev/null

sudo apt update
sudo apt install gazebo-harmonic
```

#### Method 2: Installing Gazebo via Snap

```bash
sudo snap install gazebo-sim --classic
```

#### Method 3: Building from Source (Advanced)

For the latest features or development purposes:

```bash
# Install build dependencies
sudo apt install -y cmake pkg-config python3-pip
pip3 install colcon-common-extensions vcstool

# Create a workspace for Gazebo
mkdir -p ~/gz_ws/src
cd ~/gz_ws

# Download the latest source
wget https://raw.githubusercontent.com/gazebo-tooling/gazebodistro/master/index.yaml
vcs import src < index.yaml

# Build Gazebo
colcon build
```

### Installing Gazebo ROS Packages

To use Gazebo with ROS 2, you need the Gazebo ROS packages:

```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Install Gazebo ROS packages
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control ros-humble-gazebo-dev
```

## Verifying Installation

After installation, verify that Gazebo is properly installed:

```bash
# Test basic Gazebo functionality
gz sim --version

# Launch the basic Gazebo GUI
gz sim

# If you have ROS 2 installed, you can also test:
ros2 run gazebo_ros gazebo
```

## Gazebo Components

### Core Components

1. **Gazebo Server (gz sim)**: Runs the physics simulation
2. **Gazebo GUI (gz sim -g)**: Provides the visual interface
3. **Gazebo Client (gz client)**: Allows command-line interaction

### Key Features

- **Physics Engines**: Support for ODE, Bullet, and DART physics engines
- **Sensor Simulation**: Cameras, LiDAR, IMUs, GPS, contact sensors, etc.
- **Realistic Rendering**: Advanced graphics rendering with support for lighting
- **Plugin System**: Extensible functionality through plugins
- **ROS 2 Integration**: Direct communication between Gazebo and ROS 2

## Basic Gazebo Usage

### Launching Gazebo

```bash
# Launch Gazebo with the default empty world
gz sim

# Launch with a specific world file
gz sim -r simple_room.sdf

# Launch with GUI
gz sim -g

# Launch with a specific world and run in real-time
gz sim -r -v 1 empty.sdf
```

### Basic Commands in Gazebo

- **Right-click** and drag: Move the camera view
- **Middle-click** and drag: Pan the camera
- **Mouse wheel**: Zoom in/out
- **Ctrl + Left-click**: Select objects
- **Spacebar**: Pause/unpause simulation
- **R**: Reset simulation
- **S**: Take a screenshot

## Gazebo and ROS 2 Integration

### Launching Gazebo with ROS 2

To use Gazebo with ROS 2, you typically launch it using a ROS 2 launch file:

```python
# Example launch file content
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
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
        )
    ])
```

### Gazebo ROS 2 Packages

Key ROS 2 packages for Gazebo integration:

- **`gazebo_ros_pkgs`**: Core ROS 2 plugins for Gazebo communication
- **`gazebo_dev`**: Development tools and headers
- **`ros_gz_bridge`**: Bridges between ROS 2 and Gazebo transport
- **`ros_gz_sim`**: Launch files and utilities for running Gazebo with ROS 2

## Creating a Basic Simulation Environment

### Step 1: Create a Simple World File

Create a basic world file named `simple_room.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_room">
    <!-- Include a default ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Include a default light source -->
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Create a simple room -->
    <model name="room_walls">
      <pose>0 0 1.5 0 0 0</pose>
      <link name="wall_1">
        <pose>-5 0 1.5 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 10 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 10 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
      <!-- Additional walls would be defined similarly -->
    </model>
  </world>
</sdf>
```

### Step 2: Launch Gazebo with Your World

```bash
# Launch with your custom world
gz sim -r simple_room.sdf
```

## Configuring GPU Acceleration

For optimal performance, ensure GPU acceleration is properly configured:

```bash
# Check if GPU is detected
lspci | grep -E "VGA|3D"

# Verify OpenGL support
glxinfo | grep -i "direct rendering"
glxinfo | grep -i "opengl"

# Run Gazebo with dedicated GPU (if using Optimus/Bumblebee)
optirun gz sim  # For NVIDIA Optimus systems
```

## Troubleshooting Common Issues

### Issue 1: Graphics/Rendering Problems
- **Symptom**: Black screens, poor rendering, or crashes
- **Solution**: Update graphics drivers and ensure OpenGL 3.3+ is supported

### Issue 2: Performance Issues
- **Symptom**: Slow simulation or low frame rates
- **Solution**: Reduce visual quality, use simpler models, or increase hardware resources

### Issue 3: ROS 2 Communication Failures
- **Symptom**: Nodes can't communicate with Gazebo
- **Solution**: Check that Gazebo and ROS 2 are using the same RMW implementation

```bash
# Check RMW implementation
printenv | grep RMW
```

### Issue 4: Port Conflicts
- **Symptom**: Error about ports already in use
- **Solution**: Kill existing Gazebo processes or change ports

```bash
# Kill existing Gazebo processes
pkill gz
pkill gazebo
```

## Best Practices

1. **Start Simple**: Begin with basic worlds and gradually add complexity
2. **Use Proper Models**: Use simple geometric shapes for initial testing before complex meshes
3. **Validate Physics**: Test physics parameters to ensure they match real-world behavior
4. **Monitor Performance**: Keep an eye on CPU/GPU usage and simulation step time
5. **Document Configurations**: Maintain notes on working configurations for reproducibility

## Summary

Setting up Gazebo is a critical step in developing Physical AI systems. A properly configured simulation environment allows for safe, efficient testing of robot behaviors before deployment to real hardware. This foundation enables more advanced simulation techniques including physics simulation, sensor integration, and robot-world interaction that we'll explore in the following sections.

With Gazebo properly installed and configured, you're ready to explore robot description formats (URDF/SDF) that define the robots you'll simulate in the next section.