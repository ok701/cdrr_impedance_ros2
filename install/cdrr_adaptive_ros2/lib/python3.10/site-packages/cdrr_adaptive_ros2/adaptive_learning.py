#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import numpy as np
import argparse

class RBFController:
    def __init__(self, eta=6.0, n_rbf=3, centers=None, sigma=0.3, forgetting_factor=1.0, initial_weights=None, train=True):
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
            centers = np.linspace(0, 1, n_rbf)
        self.centers = centers
        self.sigma = sigma
        self.eta = eta
        self.forgetting_factor = forgetting_factor
        self.train = train

        if initial_weights is not None:
            self.weights = np.copy(initial_weights)
        else:
            self.weights = np.zeros(n_rbf)

    def compute_rbf_outputs(self, x_ref):
        """Compute Gaussian RBF outputs for a given reference position."""
        return np.exp(-((x_ref - self.centers)**2) / (2 * self.sigma**2))

    def compute_control(self, x, xdot, x_ref, dt, cycle):
        """
        Compute the assist force using the RBF network and update the network weights
        based on the tracking error (x_ref - x) if training is enabled.
        
        :param x: Measured position.
        :param xdot: Measured velocity (unused here, so set to 0).
        :param x_ref: Reference position.
        :param dt: Time step (not used in this simple update rule).
        :param cycle: Cycle count (can be used for additional scheduling if needed).
        :return: Computed control (assist) force.
        """
        error = x_ref - x
        phi = self.compute_rbf_outputs(x_ref)
        f_rbfn = np.dot(self.weights, phi)
        
        # Online weight update only if training is enabled
        if self.train:
            self.weights = self.forgetting_factor * self.weights + self.eta * error * phi
        
        return f_rbfn

class AdaptiveLearningNode(Node):
    def __init__(self, freq=50.0):
        super().__init__('adaptive_learning')
        self.dt = 1.0 / freq
        self.rbf_controller = RBFController()  # default: 3 RBFs as per our specification
        self.cycle = 0

        # Subscribers: actual (mes) position, reference position, and trigger
        self.create_subscription(Int32, '/mes_pos', self.mes_pos_callback, 10)
        self.create_subscription(Int32, '/ref_pos', self.ref_pos_callback, 10)
        self.create_subscription(Int32, '/trigger', self.trigger_callback, 10)

        # Publisher for reference force; note: topic name should match the simulation node's usage
        self.ref_force_pub = self.create_publisher(Int32, '/ref_force', 10)

        # Variables to store incoming messages
        self.x_measured = None
        self.x_ref = None
        self.trigger_state = 0

        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.get_logger().info(f"AdaptiveLearningNode started at {freq} Hz.")

    def mes_pos_callback(self, msg: Int32):
        self.x_measured = float(msg.data)

    def ref_pos_callback(self, msg: Int32):
        self.x_ref = float(msg.data)

    def trigger_callback(self, msg: Int32):
        self.trigger_state = msg.data

    def timer_callback(self):
        # Ensure we have received both measured and reference positions.
        if self.x_measured is None or self.x_ref is None:
            return

        # Enable online learning only when the trigger is active (== 1)
        if self.trigger_state == 1:
            self.rbf_controller.train = True
        else:
            self.rbf_controller.train = False

        # Compute the assist force using the RBF controller.
        # Since we do not have a velocity measurement, xdot is set to 0.
        ref_force = self.rbf_controller.compute_control(self.x_measured, 0.0, self.x_ref, self.dt, self.cycle)
        self.cycle += 1

        self.get_logger().info(f"Computed ref_force: {ref_force:.2f}")

        # Publish the computed reference force as an Int32 message.
        msg = Int32()
        msg.data = int(ref_force)
        self.ref_force_pub.publish(msg)

def main(args=None):
    parser = argparse.ArgumentParser(description="Adaptive Learning Node with 3 RBFs")
    parser.add_argument("--freq", type=float, default=50.0, help="Frequency of the node")
    known_args, _ = parser.parse_known_args()
    
    rclpy.init(args=args)
    node = AdaptiveLearningNode(freq=known_args.freq)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down AdaptiveLearningNode.")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
