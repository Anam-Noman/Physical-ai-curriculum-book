---
sidebar_position: 3
---

# Environment Setup

This guide will help you set up your development environment for the Physical AI and Humanoid Robotics curriculum. We'll cover both basic setup for curriculum access and advanced setup for simulation and development.

## Basic Setup (Curriculum Access Only)

If you only want to access the curriculum content without running simulations:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-organization/physical-ai-curriculum-book.git
   cd physical-ai-curriculum-book
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Start the documentation server:**
   ```bash
   npm start
   ```

4. Open your browser to `http://localhost:3000` to access the curriculum.

## Full Development Setup

For the complete experience including simulation and hands-on exercises:

### 1. Install Core Dependencies

- **Node.js 18+**: Download from [nodejs.org](https://nodejs.org/)
- **Python 3.8+**: Download from [python.org](https://www.python.org/)
- **Git**: Download from [git-scm.com](https://git-scm.com/)

### 2. Install ROS 2 (Humble Hawksbill)

#### Ubuntu:
```bash
# Set locale
sudo locale-gen en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8

# Add ROS 2 apt repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 packages
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-rosdep2
sudo rosdep init
rosdep update
```

#### Windows:
Use WSL2 with Ubuntu, then follow Ubuntu instructions above.

### 3. Install Simulation Environments (Optional)

#### Gazebo Installation:
```bash
# After ROS 2 installation
sudo apt install ros-humble-gazebo-*
```

#### NVIDIA Isaac Sim Setup (Requires NVIDIA GPU):
1. Download Isaac Sim from [NVIDIA Developer Portal](https://developer.nvidia.com/isaac-sim)
2. Follow installation instructions for your platform
3. Ensure CUDA-compatible NVIDIA GPU is installed

### 4. Verify Setup

Create a test script to verify your installation:

```bash
# Create a test directory
mkdir ~/physical_ai_test
cd ~/physical_ai_test

# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Create a new ROS 2 workspace
mkdir -p ~/physical_ai_ws/src
cd ~/physical_ai_ws

# Build the workspace
colcon build
source install/setup.bash
```

### 5. Clone and Setup Curriculum Repository

```bash
# Clone the repository
git clone https://github.com/your-organization/physical-ai-curriculum-book.git
cd physical-ai-curriculum-book

# Install dependencies
npm install

# Start the curriculum server
npm start
```

## Docker Setup (Alternative)

If you prefer using Docker for a consistent environment:

1. **Install Docker** from [docker.com](https://www.docker.com/)

2. **Build the curriculum environment:**
   ```bash
   # Create a Dockerfile
   cat > Dockerfile << EOF
   FROM osrf/ros:humble-desktop
   RUN apt-get update && apt-get install -y \
       nodejs \
       npm \
       python3-pip \
       && rm -rf /var/lib/apt/lists/*
   
   # Install Node.js and create working directory
   RUN npm install -g n
   RUN n 18
   RUN mkdir -p /usr/src/app
   WORKDIR /usr/src/app
   EOF
   ```

3. **Build and run the container:**
   ```bash
   docker build -t physical-ai-curriculum .
   docker run -it -p 3000:3000 -v $(pwd):/usr/src/app physical-ai-curriculum
   ```

## Troubleshooting

### Common Issues:

1. **Port 3000 already in use:**
   ```bash
   # Find the process using port 3000
   lsof -i :3000
   # Kill the process or use a different port
   npm start -- --port 3001
   ```

2. **Permission errors with ROS 2:**
   ```bash
   # Make sure to source ROS 2 environment
   source /opt/ros/humble/setup.bash
   ```

3. **Node.js version issues:**
   ```bash
   # Use nvm to manage Node.js versions
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
   nvm install 18
   nvm use 18
   ```

## Next Steps

Once your environment is set up, proceed to Module 1 where you'll learn about Physical AI foundations and ROS 2 architecture.