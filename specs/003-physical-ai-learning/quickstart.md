# Quickstart Guide: Physical AI & Humanoid Robotics

## Overview
This guide will help you set up your environment to follow along with the Physical AI & Humanoid Robotics technical book. The curriculum is designed around a simulation-first approach using industry-standard tools.

## Prerequisites

### System Requirements
- Operating System: Ubuntu 22.04 LTS (recommended) or Windows 10/11 with WSL2
- RAM: 16GB minimum, 32GB recommended
- Storage: 50GB free space
- GPU: NVIDIA GPU with CUDA support (for Isaac Sim, optional but recommended)

### Software Prerequisites
- Git
- Python 3.10 or higher
- Docker (optional, for containerized environments)
- Node.js 18+ and npm (for local documentation)

## Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/[your-organization]/physical-ai-curriculum.git
cd physical-ai-curriculum
```

### 2. Install ROS 2 Humble Hawksbill
Follow the official installation guide at https://docs.ros.org/en/humble/Installation.html

For Ubuntu:
```bash
# Set locale
locale  # verify LANG is en_US.UTF-8
sudo locale-gen en_US.UTF-8

# Add ROS 2 apt repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop
sudo apt install -y python3-rosdep2
sudo apt install -y python3-colcon-common-extensions

# Source the ROS 2 environment
source /opt/ros/humble/setup.bash
```

### 3. Install Gazebo Simulation
```bash
sudo apt install ros-humble-gazebo-*
sudo apt install gazebo
```

### 4. Set up NVIDIA Isaac Sim (Optional but Recommended)
1. Download Isaac Sim from NVIDIA Developer portal
2. Follow installation instructions in the Isaac Sim documentation
3. Ensure CUDA and compatible GPU drivers are installed

### 5. Install Documentation Environment
```bash
# Install Node.js if not already installed
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install Docusaurus dependencies
npm install
```

## Running the Documentation Locally

### 1. Navigate to the Documentation Directory
```bash
cd physical-ai-curriculum
```

### 2. Start the Documentation Server
```bash
npm run start
```

### 3. Access the Documentation
Open your browser and go to `http://localhost:3000` to view the documentation.

## Example: Running Your First Simulation

### 1. Source ROS 2 Environment
```bash
source /opt/ros/humble/setup.bash
```

### 2. Create a Workspace
```bash
mkdir -p ~/physical_ai_ws/src
cd ~/physical_ai_ws
colcon build
source install/setup.bash
```

### 3. Launch a Basic Simulation (Week 3-5 content)
```bash
# Launch a simple ROS 2 + Gazebo simulation
ros2 launch gazebo_ros empty_world.launch.py
```

## Troubleshooting

### Common Issues

#### ROS 2 Environment Not Found
If you get "command not found" errors, ensure you've sourced the ROS 2 environment:
```bash
source /opt/ros/humble/setup.bash
```

#### Gazebo Won't Start
Check that you have a display server running:
```bash
echo $DISPLAY
```

#### Isaac Sim Installation Issues
Ensure your GPU drivers are properly installed and CUDA is working:
```bash
nvidia-smi
nvcc --version
```

## Next Steps

1. Begin with **Module 1, Week 1** content: Physical AI fundamentals
2. Follow the weekly breakdown as outlined in the curriculum
3. Complete each week's practical exercises before proceeding
4. Use the simulation environments to experiment with concepts
5. Join the discussion forums for additional support

## Getting Help

- Check the official documentation for ROS 2, Gazebo, and Isaac Sim
- Use the issue tracker in the repository for curriculum-specific questions
- Join our community Discord/Slack channel for real-time help