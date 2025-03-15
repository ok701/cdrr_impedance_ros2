#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Vector3
import sys
import argparse

class LinearEnvVizNode(Node):
    def __init__(self, freq=10.0, frame_id='world'):
        super().__init__('linear_env_viz')
        self.freq = freq
        self.dt = 1.0 / freq
        self.frame_id = frame_id

        # Subscribers for measured and reference positions (in mm)
        self.mes_pos_sub = self.create_subscription(Int32, '/mes_position', self.mes_pos_callback, 10)
        self.ref_pos_sub = self.create_subscription(Int32, '/ref_position', self.ref_pos_callback, 10)

        # Publishers for RViz markers
        self.mes_marker_pub = self.create_publisher(Marker, '/mes_marker', 10)
        self.ref_marker_pub = self.create_publisher(Marker, '/ref_marker', 10)
        self.text_marker_pub = self.create_publisher(Marker, '/text_marker', 10)

        # Internal state for positions
        self.x_measured = None
        self.x_reference = None

        # Timer for updating markers
        self.timer = self.create_timer(self.dt, self.update_markers)
        self.get_logger().info("LinearEnvVizNode started.")

    def mes_pos_callback(self, msg: Int32):
        """Callback to update the measured position."""
        self.x_measured = msg.data

    def ref_pos_callback(self, msg: Int32):
        """Callback to update the reference position."""
        self.x_reference = msg.data

    def update_markers(self):
        """Publish RViz markers for the measured and reference positions."""
        if self.x_measured is None or self.x_reference is None:
            return

        # Create a marker for the measured position (red sphere)
        mes_marker = Marker()
        mes_marker.header.frame_id = self.frame_id
        mes_marker.header.stamp = self.get_clock().now().to_msg()
        mes_marker.ns = "measured"
        mes_marker.id = 0
        mes_marker.type = Marker.SPHERE
        mes_marker.action = Marker.ADD
        # Convert position from mm to meters
        mes_marker.pose.position.x = float(self.x_measured) / 1000.0
        mes_marker.pose.position.y = 0.0
        mes_marker.pose.position.z = 0.0
        mes_marker.pose.orientation.w = 1.0
        mes_marker.scale = Vector3(x=0.05, y=0.05, z=0.05)
        mes_marker.color.r = 1.0
        mes_marker.color.g = 0.0
        mes_marker.color.b = 0.0
        mes_marker.color.a = 1.0

        # Create a marker for the reference position (green sphere)
        ref_marker = Marker()
        ref_marker.header.frame_id = self.frame_id
        ref_marker.header.stamp = self.get_clock().now().to_msg()
        ref_marker.ns = "reference"
        ref_marker.id = 0
        ref_marker.type = Marker.SPHERE
        ref_marker.action = Marker.ADD
        ref_marker.pose.position.x = float(self.x_reference) / 1000.0
        ref_marker.pose.position.y = 0.0
        ref_marker.pose.position.z = 0.0
        ref_marker.pose.orientation.w = 1.0
        ref_marker.scale = Vector3(x=0.05, y=0.05, z=0.05)
        ref_marker.color.r = 0.0
        ref_marker.color.g = 1.0
        ref_marker.color.b = 0.0
        ref_marker.color.a = 1.0

        # Create a text marker to display position values
        text_marker = Marker()
        text_marker.header.frame_id = self.frame_id
        text_marker.header.stamp = self.get_clock().now().to_msg()
        text_marker.ns = "text"
        text_marker.id = 0
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        # Place text marker slightly above the markers
        text_marker.pose.position.x = 0.0
        text_marker.pose.position.y = 0.0
        text_marker.pose.position.z = 1.0
        text_marker.pose.orientation.w = 1.0
        text_marker.scale = Vector3(x=0.1, y=0.1, z=0.1)
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.text = (f"Measured: {self.x_measured} mm\n"
                            f"Reference: {self.x_reference} mm")

        # Publish the markers
        self.mes_marker_pub.publish(mes_marker)
        self.ref_marker_pub.publish(ref_marker)
        self.text_marker_pub.publish(text_marker)

def main(args=None):
    parser = argparse.ArgumentParser(description="RViz Visualization Node for 1D Linear Environment")
    parser.add_argument("--freq", type=float, default=50.0, help="Visualization update frequency")
    parser.add_argument("--frame-id", type=str, default="world", help="Frame ID for RViz markers")
    known_args, _ = parser.parse_known_args()

    rclpy.init(args=sys.argv)
    node = LinearEnvVizNode(freq=known_args.freq, frame_id=known_args.frame_id)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down LinearEnvVizNode.")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
