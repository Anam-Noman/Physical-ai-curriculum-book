---
sidebar_position: 1
---

# Basic ROS 2 Tutorial

## Introduction

This tutorial will guide you through creating your first ROS 2 package and nodes. You'll learn the fundamental concepts of ROS 2 by building a simple but complete robot application that demonstrates the key communication patterns: topics, services, and actions.

## Prerequisites

Before starting this tutorial, ensure you have:
- ROS 2 Humble Hawksbill installed (or your preferred ROS 2 distribution)
- Basic Python knowledge
- Familiarity with the command line

## Creating a ROS 2 Package

### Step 1: Set up your workspace

First, create a workspace directory for your ROS 2 packages:

```bash
# Create the workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
```

### Step 2: Create a new package

ROS 2 packages group related functionality together. Let's create a package for our tutorial:

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python py_robot_tutorial --dependencies rclpy std_msgs geometry_msgs
```

This command creates:
- A new package named `py_robot_tutorial`
- With Python build system (`ament_python`)
- Dependencies on `rclpy`, `std_msgs`, and `geometry_msgs`

### Step 3: Explore the package structure

After creating the package, you'll see:

```
py_robot_tutorial/
├── py_robot_tutorial/
│   ├── __init__.py
│   └── py_robot_tutorial
├── setup.cfg
├── setup.py
├── package.xml
└── test/
    ├── __init__.py
    └── test_copyright.py
    └── test_flake8.py
    └── test_pep257.py
```

## Basic Publisher Example

### Step 4: Create a simple publisher

Let's create a simple node that publishes messages to a topic. Create a new file `~/ros2_ws/src/py_robot_tutorial/py_robot_tutorial/talker.py`:

```python
#!/usr/bin/env python3

"""
Simple talker node that publishes messages to a topic.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Talker(Node):
    def __init__(self):
        # Initialize the node with the name 'talker'
        super().__init__('talker')
        
        # Create a publisher on the 'chatter' topic with String message type
        self.publisher = self.create_publisher(String, 'chatter', 10)
        
        # Create a timer to publish messages every 0.5 seconds
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # Counter for messages
        self.i = 0
        
        self.get_logger().info('Talker node initialized')

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    
    node = Talker()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 5: Make the file executable

```bash
chmod +x ~/ros2_ws/src/py_robot_tutorial/py_robot_tutorial/talker.py
```

## Basic Subscriber Example

### Step 6: Create a simple subscriber

Now let's create a node that subscribes to the messages. Create `~/ros2_ws/src/py_robot_tutorial/py_robot_tutorial/listener.py`:

```python
#!/usr/bin/env python3

"""
Simple listener node that subscribes to messages from a topic.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Listener(Node):
    def __init__(self):
        # Initialize the node with the name 'listener'
        super().__init__('listener')
        
        # Create a subscription to the 'chatter' topic
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        
        # Prevent unused variable warning
        self.subscription  # type: ignore
        
        self.get_logger().info('Listener node initialized')

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    rclpy.init(args=args)
    
    node = Listener()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 7: Make the file executable

```bash
chmod +x ~/ros2_ws/src/py_robot_tutorial/py_robot_tutorial/listener.py
```

## Setting up the Entry Points

### Step 8: Update setup.py

Edit `~/ros2_ws/src/py_robot_tutorial/setup.py` to include command-line entry points:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'py_robot_tutorial'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Basic ROS 2 Python tutorial package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = py_robot_tutorial.talker:main',
            'listener = py_robot_tutorial.listener:main',
        ],
    },
)
```

## Building and Running the Package

### Step 9: Build the package

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash  # or your ROS 2 distribution
colcon build --packages-select py_robot_tutorial
```

### Step 10: Source the workspace

```bash
source ~/ros2_ws/install/setup.bash
```

### Step 11: Run the publisher and subscriber

Open two terminals and run:

Terminal 1:
```bash
ros2 run py_robot_tutorial talker
```

Terminal 2:
```bash
ros2 run py_robot_tutorial listener
```

You should see the talker publishing messages and the listener receiving them.

## Adding a Service

### Step 12: Create a service server and client

Let's add a simple service to our package. First, create a service definition file `srv/AddTwoInts.srv` in the py_robot_tutorial directory:

```srv
int64 a
int64 b
---
int64 sum
```

Then create a service server `~/ros2_ws/src/py_robot_tutorial/py_robot_tutorial/service_server.py`:

```python
#!/usr/bin/env python3

"""
Simple service server that adds two integers.
"""

import sys
import rclpy
from rclpy.node import Node
from py_robot_tutorial.srv import AddTwoInts


class ServiceServer(Node):
    def __init__(self):
        super().__init__('add_two_ints_server')
        self.srv = self.create_service(
            AddTwoInts, 
            'add_two_ints', 
            self.add_two_ints_callback
        )
        self.get_logger().info('Service server initialized')

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {request.a} + {request.b} = {response.sum}')
        return response


def main(args=None):
    rclpy.init(args=args)

    service_server = ServiceServer()

    try:
        rclpy.spin(service_server)
    except KeyboardInterrupt:
        print('Server stopped')
    finally:
        service_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

And a service client `~/ros2_ws/src/py_robot_tutorial/py_robot_tutorial/service_client.py`:

```python
#!/usr/bin/env python3

"""
Simple service client that calls the add_two_ints service.
"""

import sys
import rclpy
from rclpy.node import Node
from py_robot_tutorial.srv import AddTwoInts


class ServiceClient(Node):
    def __init__(self):
        super().__init__('add_two_ints_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main(args=None):
    rclpy.init(args=args)

    client = ServiceClient()
    response = client.send_request(2, 3)
    
    if response is not None:
        print(f'Result of add_two_ints: {response.sum}')
    else:
        print('Service call failed')

    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

Update setup.py to include these new executables:

```python
entry_points={
    'console_scripts': [
        'talker = py_robot_tutorial.talker:main',
        'listener = py_robot_tutorial.listener:main',
        'service_server = py_robot_tutorial.service_server:main',
        'service_client = py_robot_tutorial.service_client:main',
    ],
},
```

### Step 13: Rebuild the package

```bash
cd ~/ros2_ws
colcon build --packages-select py_robot_tutorial
source ~/ros2_ws/install/setup.bash
```

### Step 14: Run the service example

Terminal 1:
```bash
ros2 run py_robot_tutorial service_server
```

Terminal 2:
```bash
ros2 run py_robot_tutorial service_client
```

## Creating a Launch File

### Step 15: Create a launch file to run multiple nodes

Create a directory for launch files:

```bash
mkdir -p ~/ros2_ws/src/py_robot_tutorial/launch
```

Create a launch file `~/ros2_ws/src/py_robot_tutorial/launch/talker_listener.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='py_robot_tutorial',
            executable='talker',
            name='talker_node'
        ),
        Node(
            package='py_robot_tutorial',
            executable='listener',
            name='listener_node'
        )
    ])
```

### Step 16: Run using launch file

```bash
ros2 launch py_robot_tutorial talker_listener.launch.py
```

## Quality of Service (QoS) Settings

### Step 17: Understanding QoS

Quality of Service settings allow you to control the behavior of your topic communication. Let's modify the talker to use specific QoS settings. Create `~/ros2_ws/src/py_robot_tutorial/py_robot_tutorial/qos_talker.py`:

```python
#!/usr/bin/env python3

"""
Talker node with Quality of Service settings.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String


class QoSTalker(Node):
    def __init__(self):
        super().__init__('qos_talker')
        
        # Create a QoS profile
        qos_profile = QoSProfile(
            depth=10,  # History depth
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Reliability
            history=HistoryPolicy.KEEP_LAST  # History behavior
        )
        
        # Create publisher with QoS
        self.publisher = self.create_publisher(String, 'qos_chatter', qos_profile)
        
        # Create timer
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'QoS Hello: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'QoS Publishing: "{msg.data}"')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    node = QoSTalker()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

Don't forget to add it to setup.py and make it executable:

```bash
chmod +x ~/ros2_ws/src/py_robot_tutorial/py_robot_tutorial/qos_talker.py
```

## Understanding ROS 2 Concepts Through This Tutorial

### Nodes
A node is a process that performs computation. In our tutorial:
- `talker` node: publishes messages
- `listener` node: subscribes to messages
- `service_server` node: provides a service
- `service_client` node: uses a service

### Topics
Topics enable one-way communication using a publish-subscribe pattern:
- Publishers send messages to topics
- Subscribers receive messages from topics
- Multiple subscribers can receive the same message
- Publishers and subscribers don't know about each other

### Services
Services provide request-response communication:
- A client sends a request to a service
- The service processes the request and returns a response
- This is synchronous communication

### Quality of Service (QoS)
QoS parameters allow you to control communication behavior:
- `Reliability`: Best effort vs reliable delivery
- `History`: How many messages to keep
- `Durability`: Keep messages for late joiners or not

## Next Steps

This basic tutorial has introduced you to:
- Creating ROS 2 packages
- Writing publisher and subscriber nodes
- Implementing services
- Using launch files
- Applying Quality of Service settings

These foundational concepts form the basis for more complex Physical AI systems that connect digital AI models with physical robotic bodies. In the next tutorials, you'll learn to:
- Create more complex robot models
- Integrate AI libraries with ROS 2
- Simulate robot behavior
- Plan and execute robot actions

## Troubleshooting Tips

### Common Issues

1. **Package Not Found**: Make sure to source your workspace after building: `source ~/ros2_ws/install/setup.bash`

2. **Import Errors**: Ensure your setup.py includes all dependencies and entry points

3. **Permission Errors**: Make sure your Python files have execute permissions (`chmod +x`)

4. **Service Not Available**: Make sure the service server is running before the client tries to call it

### Useful Commands

- `ros2 node list`: List all active nodes
- `ros2 topic list`: List all active topics
- `ros2 service list`: List all available services
- `ros2 run <package> <executable>`: Run a specific executable
- `ros2 launch <package> <launch_file>`: Run a launch file

## Summary

This tutorial has covered the fundamentals of ROS 2 programming with Python. You've learned to create nodes, implement publishers/subscribers, and use services. These concepts are essential for connecting AI agents to robot controllers in Physical AI systems.

The skills you've developed form the foundation for more advanced applications that bridge digital AI models with physical robotic bodies, which you'll explore in the subsequent curriculum sections.