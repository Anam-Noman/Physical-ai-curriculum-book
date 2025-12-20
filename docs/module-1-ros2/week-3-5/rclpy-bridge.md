---
sidebar_position: 5
---

# Python AI Agents to Robot Controllers using rclpy

## Introduction to rclpy

The Robot Operating System 2 (ROS 2) Python client library, `rclpy`, provides the interface for creating ROS 2 nodes in Python. This is particularly important for connecting AI agents written in Python to robot controllers, enabling the integration of digital AI models with physical robotic bodies.

Python is the language of choice for many AI and machine learning applications, making `rclpy` a crucial bridge between the AI ecosystem and robotics. This module explores how to effectively use `rclpy` to implement the connection between Python-based AI systems and robot controllers.

### Why Python for AI Integration

Python is the dominant language in AI and machine learning for several reasons:
- Extensive libraries for AI/ML (TensorFlow, PyTorch, scikit-learn)
- Easy prototyping and experimentation
- Strong community support and documentation
- Good integration with scientific computing tools

By using `rclpy`, we can seamlessly integrate these AI capabilities with ROS 2's robotics infrastructure.

## Setting up rclpy

### Installation and Prerequisites

To use `rclpy`, you need to have ROS 2 installed on your system. The library is part of the ROS 2 distribution and will be available once ROS 2 is properly installed.

```bash
# Source ROS 2 environment (example for Humble Hawksbill on Ubuntu)
source /opt/ros/humble/setup.bash

# Create and build a workspace with a Python package
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python my_ai_robot_pkg
```

### Basic Node Structure

A typical `rclpy` node follows this structure:

```python
import rclpy
from rclpy.node import Node

class MyAIAgentNode(Node):
    def __init__(self):
        super().__init__('ai_agent_node')
        # Initialize publishers, subscribers, services, etc.
        self.get_logger().info('AI Agent Node initialized')

def main(args=None):
    rclpy.init(args=args)  # Initialize rclpy
    
    # Create and configure the node
    node = MyAIAgentNode()
    
    try:
        # Spin the node to process callbacks
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

## Connecting AI Models to Robot Controllers

### Basic Publisher Example

Here's how to create a publisher to send commands from an AI agent to a robot controller:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np  # Common AI library

class SimpleAIAgent(Node):
    def __init__(self):
        super().__init__('simple_ai_agent')
        
        # Publisher to send velocity commands to the robot
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Timer to periodically apply AI logic
        self.timer = self.create_timer(0.1, self.ai_callback)  # 10Hz
        
        self.get_logger().info('Simple AI Agent initialized')

    def ai_callback(self):
        # Simple AI logic to move robot
        cmd_msg = Twist()
        
        # Example AI decision (in reality, this might involve neural networks, etc.)
        cmd_msg.linear.x = 0.5  # Move forward at 0.5 m/s
        cmd_msg.angular.z = 0.0  # No turning
        
        self.cmd_publisher.publish(cmd_msg)
        self.get_logger().info(f'Published command: linear.x={cmd_msg.linear.x}, angular.z={cmd_msg.angular.z}')
```

### Subscriber Example

To receive sensor data for AI processing:

```python
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class ReactiveAIAgent(Node):
    def __init__(self):
        super().__init__('reactive_ai_agent')
        
        # Publisher for robot commands
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriber for sensor data
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        self.get_logger().info('Reactive AI Agent initialized')

    def scan_callback(self, msg):
        # Process sensor data to make AI decisions
        # Find minimum distance in front of robot
        front_scan = msg.ranges[len(msg.ranges)//2 - 90:len(msg.ranges)//2 + 90]
        min_distance = min(front_scan)
        
        # Simple reactive behavior
        cmd_msg = Twist()
        if min_distance > 1.0:  # No obstacle nearby
            cmd_msg.linear.x = 0.5  # Move forward
        else:
            cmd_msg.linear.x = 0.0  # Stop
            cmd_msg.angular.z = 0.5  # Turn right
            
        self.cmd_publisher.publish(cmd_msg)
        self.get_logger().info(f'Min distance: {min_distance:.2f}, Command: ({cmd_msg.linear.x}, {cmd_msg.angular.z})')
```

## Integrating with Machine Learning Libraries

### Using TensorFlow/Keras Models

Here's how to run a TensorFlow model in a ROS 2 node:

```python
import rclpy
from rclpy.node import Node
import tensorflow as tf
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class TFAIAgent(Node):
    def __init__(self):
        super().__init__('tf_ai_agent')
        
        # Publisher for robot commands
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriber for camera images
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Bridge for converting ROS images to OpenCV format
        self.bridge = CvBridge()
        
        # Load a pre-trained model
        try:
            self.model = tf.keras.models.load_model('/path/to/your/model.h5')
            self.get_logger().info('Model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            self.model = None
        
        self.get_logger().info('TensorFlow AI Agent initialized')

    def image_callback(self, msg):
        if self.model is None:
            return
            
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Preprocess the image for the model
            input_image = cv2.resize(cv_image, (224, 224)) / 255.0
            input_image = np.expand_dims(input_image, axis=0)
            
            # Run inference
            prediction = self.model.predict(input_image)
            
            # Use prediction to decide robot action
            cmd_msg = Twist()
            if prediction[0][0] > 0.5:  # If probability of "go straight" > 0.5
                cmd_msg.linear.x = 0.5
                cmd_msg.angular.z = 0.0
            else:  # Turn
                cmd_msg.linear.x = 0.0
                cmd_msg.angular.z = 0.5
                
            self.cmd_publisher.publish(cmd_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
```

### Using PyTorch Models

```python
import rclpy
from rclpy.node import Node
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np

class TorchAIAgent(Node):
    def __init__(self):
        super().__init__('torch_ai_agent')
        
        # Publisher for robot commands
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriber for camera images
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.bridge = CvBridge()
        
        # Load a pre-trained PyTorch model
        try:
            self.model = torch.load('/path/to/your/model.pth')
            self.model.eval()  # Set to evaluation mode
            self.get_logger().info('PyTorch model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load PyTorch model: {e}')
            self.model = None
            
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])

    def image_callback(self, msg):
        if self.model is None:
            return
            
        try:
            # Convert ROS image to OpenCV format, then to PIL for transforms
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            pil_image = PILImage.fromarray(cv_image)
            
            # Apply transforms
            tensor_image = self.transform(pil_image).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                prediction = self.model(tensor_image)
                probabilities = torch.nn.functional.softmax(prediction[0], dim=0)
                
            # Use prediction to decide robot action
            cmd_msg = Twist()
            if torch.argmax(probabilities) == 0:  # For example: class 0 = go straight
                cmd_msg.linear.x = 0.5
            else:  # Turn
                cmd_msg.angular.z = 0.5
                
            self.cmd_publisher.publish(cmd_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
```

## Advanced Patterns: Service Integration

For more complex AI behaviors that require request-response patterns:

```python
from example_interfaces.srv import Trigger
from std_msgs.msg import String

class ServiceBasedAIAgent(Node):
    def __init__(self):
        super().__init__('service_ai_agent')
        
        # Publisher for robot commands
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Service server for triggering AI behaviors
        self.ai_service = self.create_service(
            Trigger, 
            'execute_ai_behavior', 
            self.execute_behavior_callback
        )
        
        # Publisher for AI status updates
        self.status_publisher = self.create_publisher(String, '/ai_status', 10)
        
        self.current_behavior = "idle"
        
    def execute_behavior_callback(self, request, response):
        self.get_logger().info('AI behavior triggered')
        
        # Publish status
        status_msg = String()
        status_msg.data = "executing_behavior"
        self.status_publisher.publish(status_msg)
        
        # Execute AI logic here
        try:
            # Example: execute a complex AI behavior
            cmd_msg = Twist()
            cmd_msg.linear.x = 0.3
            cmd_msg.angular.z = 0.1
            self.cmd_publisher.publish(cmd_msg)
            
            # Simulate behavior execution time
            # In a real implementation, you might use an action server
            # for long-running tasks
            response.success = True
            response.message = "AI behavior executed successfully"
            
        except Exception as e:
            response.success = False
            response.message = f"Error executing behavior: {str(e)}"
        
        return response
```

## Performance Considerations

### Threading and Concurrency

When running AI models that may take significant time, consider using separate threads:

```python
import threading
from rclpy.qos import QoSProfile
from std_msgs.msg import Bool

class ThreadedAIAgent(Node):
    def __init__(self):
        super().__init__('threaded_ai_agent')
        
        # Publisher for robot commands
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Publisher to indicate AI busy state
        self.busy_publisher = self.create_publisher(Bool, '/ai_busy', 10)
        
        # Subscriber for sensor data
        self.sensor_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.sensor_callback,
            10
        )
        
        # Threading lock to ensure thread safety
        self.ai_lock = threading.Lock()
        
    def sensor_callback(self, msg):
        # Only start new AI processing if not already processing
        if self.ai_lock.acquire(blocking=False):
            # Publish busy status
            busy_msg = Bool()
            busy_msg.data = True
            self.busy_publisher.publish(busy_msg)
            
            # Run AI processing in a separate thread
            thread = threading.Thread(target=self.process_ai_logic, args=(msg,))
            thread.start()
        else:
            self.get_logger().info('AI is busy, skipping processing')

    def process_ai_logic(self, sensor_data):
        try:
            # Simulate AI processing time
            # In a real implementation, this might involve neural network inference
            # or other complex AI tasks
            time.sleep(2)  # Simulate processing
            
            # Make decision based on sensor data
            cmd_msg = Twist()
            cmd_msg.linear.x = 0.5  # Move forward after processing
            self.cmd_publisher.publish(cmd_msg)
            
        finally:
            # Release lock and publish not-busy status
            self.ai_lock.release()
            busy_msg = Bool()
            busy_msg.data = False
            self.busy_publisher.publish(busy_msg)
```

## Quality of Service (QoS) for AI Applications

Selecting appropriate QoS settings is important for robust AI-robot integration:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

class QoSAIAgent(Node):
    def __init__(self):
        super().__init__('qos_ai_agent')
        
        # Create a QoS profile for sensor data (recent data is more important)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10  # Keep only the 10 most recent messages
        )
        
        # Create a QoS profile for robot commands (reliable delivery important)
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publisher with specific QoS
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', cmd_qos)
        
        # Subscriber with specific QoS
        self.sensor_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            sensor_qos
        )
```

## Summary

The `rclpy` library provides a powerful interface for connecting Python-based AI agents to robot controllers. By understanding and effectively using publishers, subscribers, services, and actions, you can create sophisticated systems that bridge digital AI models with physical robotic bodies.

Key takeaways:
- Use appropriate ROS 2 communication patterns for your AI-robot integration needs
- Consider threading for computationally expensive AI operations
- Select appropriate QoS settings based on your application requirements
- Implement proper error handling and logging for debugging
- Design your AI nodes to be modular and reusable

These patterns form the foundation for creating embodied intelligence systems that can perform complex tasks by combining AI reasoning with physical robot action.