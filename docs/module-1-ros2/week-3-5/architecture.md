---
sidebar_position: 3
---

# ROS 2 Architecture

## Introduction to ROS 2

The Robot Operating System 2 (ROS 2) is not an operating system in the traditional sense, but rather a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms, hardware architectures, and applications.

ROS 2 is the successor to ROS 1, designed to address the limitations of the original system and to enable deployment in production environments. It provides the infrastructure for distributed computing, hardware abstraction, device drivers, libraries for implementing common robot functionality, and visualization tools.

### Key Improvements in ROS 2

#### Real-time Support
ROS 2 is designed with real-time applications in mind, providing better support for time-constrained applications where precise timing is critical for robot behavior.

#### Security
ROS 2 includes built-in security mechanisms, including authentication, authorization, and encryption, making it suitable for deployment in production environments.

#### Improved Architecture
ROS 2 uses a more robust communication architecture based on Data Distribution Service (DDS), which provides better reliability and performance for real-time systems.

#### Support for Multiple Platforms
ROS 2 runs on multiple operating systems including Linux, macOS, and Windows, and supports various CPU architectures.

## Core Concepts of ROS 2 Architecture

### Nodes
A node is a process that performs computation. ROS 2 is designed to be a distributed system of multiple nodes working together. Nodes are typically organized into a package, which contains the source code, configuration, and other resources needed to run the node.

Nodes in ROS 2 can:
- Subscribe to topics to receive data
- Publish data to topics
- Provide services
- Call services
- Create and act on parameters

```python
# Example ROS 2 node structure
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1
```

### Topics and Messages
Topics are named buses over which nodes exchange messages. Messages are the data that travels between nodes. The ROS 2 messaging system is designed to be as agnostic as possible about the content of messages, supporting various data types including primitive types and nested structures.

The communication is publish-subscribe based, where publishers send messages to topics without knowing who (if anyone) is subscribed. Subscribers receive messages from topics without knowing who (if anyone) is publishing.

### Services
Services provide a request-response communication pattern. A service client sends a request to a service server, which processes the request and returns a response. This is useful for operations that should return a result immediately or that require more complex interactions than topics provide.

### Actions
Actions are a more advanced communication pattern for long-running tasks that include feedback during execution and the ability to cancel the task. Actions are appropriate for goals that take a significant amount of time to complete, such as navigation goals.

### Parameters
Parameters are named, typed values that belong to a specific node. They allow nodes to be configured without recompilation and are typically used for configuration values that don't change during runtime.

## Data Distribution Service (DDS)

ROS 2's communication layer is built on DDS (Data Distribution Service), a middleware standard for distributed real-time systems. DDS provides:

### Quality of Service (QoS) Settings
QoS settings allow publishers and subscribers to specify their communication preferences and requirements. These include:
- **Reliability**: Best effort or reliable delivery
- **Durability**: Volatile or transient local durability
- **History**: Keep all or keep last N messages
- **Deadline**: Maximum time between consecutive messages
- **Liveliness**: How to determine if a publisher is alive

### Middleware Independence
ROS 2 can work with different DDS implementations including Fast DDS, Cyclone DDS, and RTI Connext DDS, allowing users to choose the one that best fits their requirements.

## Client Library Layers

### rclcpp and rclpy
ROS 2 client libraries provide the standard interface to ROS 2 functionality. The two main client libraries are:
- **rclcpp**: The C++ client library
- **rclpy**: The Python client library

These libraries handle the communication with the DDS implementation and provide the standard ROS 2 API to users.

## Package Management

### Ament
ROS 2 uses Ament as its build system and package manager, replacing the Catkin build system from ROS 1. Ament is more compatible with standard CMake practices and supports packages written in different languages.

### Package Structure
A typical ROS 2 package contains:
- **CMakeLists.txt**: Build instructions for C++ packages
- **package.xml**: Package metadata and dependencies
- **launch/**: Launch files for starting multiple nodes
- **src/**: Source code files
- **test/**: Unit and integration test files
- **config/**: Configuration files and parameters

## Architecture Design Philosophy

### Distributed Computing
ROS 2 is designed to support distributed computing across multiple machines. This is essential for complex robots that may have multiple computers, or for multi-robot systems.

### Language Independence
ROS 2 supports multiple programming languages (initially C++ and Python, with additional languages like Rust and Java being developed). This allows developers to use the most appropriate language for specific tasks.

### Real-time Safety
ROS 2 is designed to be compatible with real-time systems (though real-time guarantees depend on the underlying OS). This makes it suitable for safety-critical robotic applications.

### Modularity
The architecture is designed to be modular, allowing components to be replaced or extended as needed. This includes the DDS implementation, build system, and client libraries.

## ROS 2 vs. ROS 1 Architecture

| Feature | ROS 1 | ROS 2 |
|---------|-------|-------|
| Communication | Custom TCP/UDP | DDS-based |
| Master | Required master node | No master, peer-to-peer |
| Security | None | Built-in security |
| Platforms | Linux primarily | Linux, Windows, macOS |
| Real-time | Limited | Better support |
| Build system | Catkin | Ament (CMake-based) |

## Implementation Considerations

### Node Design
When designing nodes in ROS 2, consider:
- Each node should have a single responsibility
- Use appropriate QoS settings for your use case
- Properly handle node lifecycle (startup, running, shutdown)
- Implement good error handling and logging

### Communication Patterns
Choose the appropriate communication pattern:
- **Topics**: For streaming data and loose coupling
- **Services**: For request-response interactions
- **Actions**: For long-running tasks with feedback and cancellation

## Summary

ROS 2 architecture provides a robust foundation for building complex robotic systems. Its distributed computing capabilities, security features, and real-time support make it suitable for both research and production environments. Understanding the core concepts of nodes, topics, services, and actions is essential for effective robot software development with ROS 2.

In the following sections, we'll explore these concepts in more detail and see how they enable the implementation of embodied intelligence systems that bridge digital AI models with physical robotic bodies.