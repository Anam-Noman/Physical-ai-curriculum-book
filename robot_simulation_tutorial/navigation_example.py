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