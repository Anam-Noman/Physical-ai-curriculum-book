---
sidebar_position: 4
---

# Nodes, Topics, Services, and Actions

## Understanding ROS 2 Communication Patterns

ROS 2 provides several communication patterns to enable nodes to exchange information and coordinate robot behavior. Understanding these patterns is crucial for building effective robotic systems that connect AI agents to physical robotic bodies.

### Nodes: The Basic Computational Unit

A node is a single executable that uses ROS 2 to communicate with other nodes. Nodes are the fundamental building blocks of ROS 2 applications, each responsible for a specific task or capability.

#### Node Characteristics
- Each node runs in its own process
- Nodes can be written in different programming languages (C++, Python, etc.)
- Nodes communicate with each other through topics, services, actions, and parameters
- Nodes can be launched independently or grouped using launch files

#### Node Lifecycle
Nodes typically go through phases:
1. **Initialization**: Setting up communication interfaces, parameters, and internal state
2. **Running**: Processing data, communicating with other nodes
3. **Shutdown**: Cleaning up resources and terminating gracefully

```python
# Example of a complete ROS 2 node
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TalkerNode(Node):
    def __init__(self):
        # Initialize the node with the name 'talker'
        super().__init__('talker')
        
        # Create a publisher on the 'chatter' topic with String message type
        self.publisher = self.create_publisher(String, 'chatter', 10)
        
        # Create a timer to publish messages every 0.5 seconds
        timer_period = 0.5
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
    node = TalkerNode()
    
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

### Topics: Publish-Subscribe Communication

Topics enable one-way communication where publishers send messages to topics and subscribers receive messages from them. This pattern is ideal for streaming data like sensor readings or robot state information.

#### Topic Characteristics
- **Loose Coupling**: Publishers and subscribers don't need to know about each other
- **Broadcasting**: Multiple subscribers can receive the same message
- **Asynchronous**: Communication happens independently of publisher/subscriber processing speeds

#### Quality of Service (QoS) for Topics
QoS parameters allow you to control how messages are handled:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Create a QoS profile for reliable communication with keep-all history
qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_ALL,
    depth=10
)

publisher = self.create_publisher(String, 'topic_name', qos_profile)
```

#### Common QoS Settings
- **Reliability**: `RELIABLE` (all messages delivered) or `BEST_EFFORT` (best attempt)
- **Durability**: `TRANSIENT_LOCAL` (keep messages for late joiners) or `VOLATILE` (don't keep messages)
- **History**: `KEEP_ALL` (unlimited queue) or `KEEP_LAST` (fixed-size queue)

### Services: Request-Response Communication

Services provide synchronous request-response communication. A client sends a request to a service server, which processes the request and returns a response.

#### Service Characteristics
- **Synchronous**: The client waits for the response
- **One-to-one**: One client communicates with one server at a time
- **Stateless**: Each request-response cycle is independent

#### Defining a Service
Service definitions use `.srv` files with the format `request fields --- response fields`:

```srv
# Example service definition (AddTwoInts.srv)
int64 a
int64 b
---
int64 sum
```

#### Service Implementation
```python
from example_interfaces.srv import AddTwoInts

class ServerNode(Node):
    def __init__(self):
        super().__init__('server_node')
        # Create a service server
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning: {request.a} + {request.b} = {response.sum}')
        return response

class ClientNode(Node):
    def __init__(self):
        super().__init__('client_node')
        # Create a client for the service
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        
        # Wait for the service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        
        self.send_request_async()

    def send_request_async(self):
        request = AddTwoInts.Request()
        request.a = 42
        request.b = 3
        # Send the request asynchronously
        self.future = self.cli.call_async(request)
        self.future.add_done_callback(self.response_callback)

    def response_callback(self, future):
        result = future.result()
        self.get_logger().info(f'Result of add_two_ints: {result.sum}')
```

### Actions: Long-Running Tasks with Feedback

Actions are designed for long-running tasks that provide feedback during execution and support cancellation. They're ideal for goals like robot navigation or manipulation.

#### Action Characteristics
- **Long-running**: Tasks that take significant time to complete
- **Feedback**: Continuous updates on progress
- **Goal/Result**: Clear start and end states
- **Cancelability**: Tasks can be stopped before completion

#### Action Implementation
Action definitions use `.action` files with the format `Goal --- Result --- Feedback`:

```action
# Example action definition (Fibonacci.action)
int32 order
---
int32[] sequence
---
int32[] sequence
```

```python
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        
        # Create an action server
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup())

    def goal_callback(self, goal_request):
        # Accept or reject a client request to begin an action
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        # Accept or reject a client request to cancel an action
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        
        # Create feedback message
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]
        
        # Send initial feedback
        goal_handle.publish_feedback(feedback_msg)
        
        # Simulate the long-running process
        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()
            
            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])
            
            self.get_logger().info(f'Publishing feedback: {feedback_msg.sequence}')
            goal_handle.publish_feedback(feedback_msg)
        
        # Populate result message
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        
        goal_handle.succeed()
        self.get_logger().info(f'Returning result: {result.sequence}')
        
        return result
```

## Communication Pattern Selection Guide

### When to Use Each Pattern

| Pattern | Use Case | Example |
|---------|----------|---------|
| **Topics** | Streaming data, sensor readings | Camera images, laser scans, robot pose |
| **Services** | Request-response operations | Map saving, sensor calibration, parameter updates |
| **Actions** | Long-running tasks with feedback | Navigation, arm movements, object detection |

### Design Best Practices

#### Topics
- Use topics for data that changes continuously
- Be mindful of message frequency and bandwidth
- Use appropriate QoS settings based on your application needs
- Consider message size and frequency for real-time performance

#### Services
- Use services for operations with clear input and output
- Avoid using services for streaming data
- Design services to be stateless when possible
- Handle service failures gracefully

#### Actions
- Use actions for goals that take time to complete
- Provide meaningful feedback during execution
- Design tasks to be cancelable
- Return appropriate results when successful

## Practical Implementation: Connecting AI Agents to Robot Controllers

Let's look at how these communication patterns work together to connect AI agents to robot controllers:

```python
# AI Agent Node
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class AIAgentNode(Node):
    def __init__(self):
        super().__init__('ai_agent')
        
        # Subscribe to sensor data
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)
        
        # Publish commands to robot controller
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Publish high-level decisions
        self.decision_pub = self.create_publisher(String, 'decisions', 10)
        
        self.obstacle_detected = False

    def scan_callback(self, msg):
        # Process sensor data
        min_distance = min(msg.ranges)
        
        # Simple AI decision making
        cmd = Twist()
        if min_distance < 1.0:  # Obstacle within 1 meter
            self.obstacle_detected = True
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # Turn
            self.decision_pub.publish(String(data="Turning due to obstacle"))
        else:
            self.obstacle_detected = False
            cmd.linear.x = 0.5  # Move forward
            cmd.angular.z = 0.0
            self.decision_pub.publish(String(data="Moving forward"))
        
        # Send command to robot controller
        self.cmd_pub.publish(cmd)
```

## Summary

Understanding the different communication patterns in ROS 2 is essential for building effective robotic systems. Nodes provide the basic computational units, while topics, services, and actions define how information flows between them. By choosing the appropriate pattern for your use case, you can create robust and efficient robot software that effectively bridges digital AI models with physical robotic bodies.

These communication patterns form the foundation for connecting AI agents to robot controllers, enabling the implementation of complex behaviors that require both high-level intelligence and low-level control.