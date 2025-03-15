#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Int32MultiArray
import numpy as np
import argparse

class RBFController:
    def __init__(self, eta=0.0001, n_rbf=3, centers=None, sigma=150, forgetting_factor=1.0, initial_weights=None, train=True):
        """
        RBF Controller with online learning.
        
        :param eta: Learning rate.
        :param n_rbf: Number of RBF units.
        :param centers: Array of RBF centers; if None, set to n_rbf evenly spaced between 0 and 1.
        :param sigma: RBF width.
        :param forgetting_factor: Factor to scale old weights each update.
        :param initial_weights: Optionally set initial weights.
        :param train: If True, update weights on each control computation.
        """
        self.n_rbf = n_rbf
        if centers is None:
            centers = np.linspace(0, 500, n_rbf)
        self.centers = centers
        self.sigma = sigma
        self.eta = eta
        self.forgetting_factor = forgetting_factor
        self.train = train

        if initial_weights is not None:
            self.weights = np.copy(initial_weights)
        else:
            self.weights = np.zeros(n_rbf)

    def compute_control(self, x_measured, x_reference):
        """
        Compute the assist force using the RBF network and update the network weights
        based on the tracking error (x_reference - x) if training is enabled.
        
        :param x: Measured position.
        :param xdot: Measured velocity (unused here, so set to 0).
        :param x_reference: Reference position.
        :param dt: Time step (not used in this simple update rule).
        :param cycle: Cycle count (can be used for additional scheduling if needed).
        :return: Computed control (assist) force.
        """
        error = x_reference - x_measured
        phi = np.exp(-((x_reference - self.centers)**2) / (2 * self.sigma**2))
        f_rbfn = np.dot(self.weights, phi)
        # print("phi:", phi)
        
        # Online weight update only if training is enabled
        if self.train:
            self.weights = self.forgetting_factor * self.weights + self.eta * error * phi
        
        return f_rbfn

class AdaptiveLearningNode(Node):
    def __init__(self, freq=10.0):
        super().__init__('adaptive_learning')
        self.dt = 1.0 / freq
        self.rbf_controller = RBFController()  # default: 3 RBFs as per our specification
        self.cycle = 0

        self.end_call_count = 0

        # Subscribers: actual (mes) position, reference position, and trigger
        self.create_subscription(Int32, '/mes_position', self.mes_pos_callback, 10)
        self.create_subscription(Int32, '/ref_position', self.ref_pos_callback, 10)
        self.create_subscription(Int32, '/trigger', self.trigger_callback, 10)

        # Publisher for RBF weights as an integer array
        # self.weights_pub = self.create_publisher(Int32MultiArray, '/rbf_weights', 10)
        # self.ref_tension_pub = self.create_publisher(Int32, '/ref_tension', 10)  # for the simulation
        self.rbf_w1_pub = self.create_publisher(Int32, '/rbf_w1', 10)
        self.rbf_w2_pub = self.create_publisher(Int32, '/rbf_w2', 10)
        self.rbf_w3_pub = self.create_publisher(Int32, '/rbf_w3', 10)

        # Variables to store incoming messages
        self.x_measured = None
        self.x_reference = None
        self.trigger_state = 0

        self.timer = self.create_timer(self.dt, self.timer_callback)
        # self.get_logger().info(f"AdaptiveLearningNode started at {freq} Hz.")

    def mes_pos_callback(self, msg: Int32):
        self.x_measured = float(msg.data)

    def ref_pos_callback(self, msg: Int32):
        self.x_reference = float(msg.data)

    def trigger_callback(self, msg: Int32):
        self.trigger_state = msg.data

    def ref_tension_callback(self, msg: Int32):
        self.ref_tension = msg.data

    def timer_callback(self):
        # Ensure we have received both measured and reference positions.
        if self.x_measured is None or self.x_reference is None:
            return
        prev_train_state = self.rbf_controller.train
        # Enable online learning only when the trigger is active (== 1)
        if self.trigger_state == 1 and self.end_call_count < 5:
            self.rbf_controller.train = True
        else:
            self.rbf_controller.train = False

        if prev_train_state is True and self.rbf_controller.train is False:
            self.end_call_count += 1
            self.get_logger().info(f"Cycle end call count: {self.end_call_count}")

        # Compute the assist force (and update weights) using the RBF controller.
        ref_tension = self.rbf_controller.compute_control(self.x_measured, self.x_reference)
        # self.get_logger().info(f"{self.x_measured, self.x_reference}")
        # Log the current RBF weights.
        # self.get_logger().info(f"Current RBF weights: {self.rbf_controller.weights}")

        # Convert weights to integers and publish as an Int32MultiArray message.
        # msg = Int32MultiArray()
        # msg.data = [int(weight) for weight in self.rbf_controller.weights]
        # self.weights_pub.publish(msg)
        # self.ref_tension_pub.publish(Int32(data=int(ref_tension)))
        msg_w1 = Int32()
        msg_w2 = Int32()
        msg_w3 = Int32()
        msg_w1.data = int(self.rbf_controller.weights[0] * 100)
        msg_w2.data = int(self.rbf_controller.weights[1] * 100)
        msg_w3.data = int(self.rbf_controller.weights[2] * 100)
        self.rbf_w1_pub.publish(msg_w1)
        self.rbf_w2_pub.publish(msg_w2)
        self.rbf_w3_pub.publish(msg_w3)
        self.get_logger().info(f"Current RBF weights: {self.rbf_controller.weights}")

def main(args=None):
    parser = argparse.ArgumentParser(description="Adaptive Learning Node")
    parser.add_argument("--freq", type=float, default=50.0, help="Frequency of the node")
    known_args, _ = parser.parse_known_args()
    
    rclpy.init(args=args)
    node = AdaptiveLearningNode(freq=known_args.freq)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # node.get_logger().info("Shutting down AdaptiveLearningNode.")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
