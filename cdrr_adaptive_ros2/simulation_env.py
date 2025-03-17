#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import numpy as np
import sys
import argparse

class LinearEnvSimNode(Node):
    def __init__(self, freq=10.0):
        super().__init__('linear_env_sim')
        
        # Simulation parameters
        self.mass = 1       # Mass in kg
        self.b = 0.01           # Damping coefficient in N/(mm/s)
        self.L = 500.0         # Maximum position in mm
        self.T = 10.0          # Total simulation time in seconds
        self.freq = freq
        self.dt = 1.0 / freq
        self.Kp = 0         # "Human" spring gain
        
        # RBF network parameters
        self.n_rbf = 3
        self.centers = np.linspace(0, 500, self.n_rbf)  # Evenly spaced centers
        self.sigma = 150.0       # RBF width
        
        # Subscriptions
        self.trigger_sub = self.create_subscription(Int32, '/trigger', self.trigger_callback, 10)
        self.rbf_w1_sub = self.create_subscription(Int32, '/rbf_w1', self.rbf_w1_callback, 10)
        self.rbf_w2_sub = self.create_subscription(Int32, '/rbf_w2', self.rbf_w2_callback, 10)
        self.rbf_w3_sub = self.create_subscription(Int32, '/rbf_w3', self.rbf_w3_callback, 10)
        
        # Publications for simulation positions
        self.mes_pos_pub = self.create_publisher(Int32, '/mes_position', 10)
        self.ref_pos_pub = self.create_publisher(Int32, '/ref_position', 10)
        
        # Internal simulation state
        self.trigger_state = 0
        self.simulation_active = False
        self.x = 0.0         # Current position in mm
        self.x_dot = 0.0     # Current velocity in mm/s
        self.time_sim = 0.0  # Simulation time in seconds
        
        # RBF weight values (received as scaled integers)
        self.rbf_w1 = 0.0
        self.rbf_w2 = 0.0
        self.rbf_w3 = 0.0
        
        # Timer for simulation update
        self.timer = self.create_timer(self.dt, self.update_simulation)
        self.get_logger().info(f"LinearEnvSimNode started at {freq} Hz.")

    def trigger_callback(self, msg: Int32):
        """
        Callback for the trigger topic.
        If the trigger goes from 0 to 1, start the simulation.
        If the trigger goes from 1 to 0, reset the simulation.
        """
        new_trigger = msg.data
        if new_trigger == 1 and self.trigger_state == 0:
            self.get_logger().info("Trigger activated: Starting simulation.")
            self.simulation_active = True
        elif new_trigger == 0 and self.trigger_state == 1:
            self.get_logger().info("Trigger deactivated: Resetting simulation.")
            self.reset_simulation()
        self.trigger_state = new_trigger

    def rbf_w1_callback(self, msg: Int32):
        """
        Callback for the rbf_w1 topic.
        Stores the received weight (scaled integer).
        """
        self.rbf_w1 = float(msg.data)
        self.get_logger().debug(f"Updated rbf_w1: {self.rbf_w1}")

    def rbf_w2_callback(self, msg: Int32):
        """
        Callback for the rbf_w2 topic.
        Stores the received weight (scaled integer).
        """
        self.rbf_w2 = float(msg.data)
        self.get_logger().debug(f"Updated rbf_w2: {self.rbf_w2}")

    def rbf_w3_callback(self, msg: Int32):
        """
        Callback for the rbf_w3 topic.
        Stores the received weight (scaled integer).
        """
        self.rbf_w3 = float(msg.data)
        self.get_logger().debug(f"Updated rbf_w3: {self.rbf_w3}")

    def reset_simulation(self):
        """
        Resets the simulation state.
        """
        self.x = 0.0
        self.x_dot = 0.0
        self.time_sim = 0.0
        self.simulation_active = False

    def target_trajectory(self, t):
        """
        Defines a smooth target trajectory from 0 to L over time T.
        Uses a polynomial profile.
        """
        if t >= self.T:
            return self.L
        tau = t / self.T
        return self.L * (10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5)

    def update_simulation(self):
        """
        Main simulation update loop.
        Computes the cable force using the RBF network,
        updates the system dynamics, and publishes the positions.
        """
        if not self.simulation_active:
            return

        # Compute reference position using the target trajectory
        x_ref = self.target_trajectory(self.time_sim)
        
        # Compute RBF activations based on the reference position
        phi = np.exp(-((x_ref - self.centers)**2) / (2 * self.sigma**2))
        
        # Recover weights from received values (scale back by dividing by 100)
        weights = np.array([self.rbf_w1, self.rbf_w2, self.rbf_w3]) / 100.0
        
        # Compute assist force using the RBF network (dot product)
        f_rbfn = np.dot(weights, phi)
        cable_force = f_rbfn
        self.get_logger().info(f"Computed cable force (f_rbfn): {cable_force:.2f} N")
        
        # Calculate the "human" spring force
        human_force = self.Kp * (x_ref - self.x)
        
        # Compute the net force (assist force minus damping plus human force)
        net_force = cable_force - (self.b * self.x_dot) + human_force
        
        # Compute acceleration (converted to mm/s^2)
        x_ddot = 1000.0 * (net_force / self.mass)
        
        # Integrate to update velocity and position
        self.x_dot += x_ddot * self.dt
        self.x += self.x_dot * self.dt
        
        # Clamp the position within the allowed limits
        if self.x < 0.0:
            self.x = 0.0
            self.x_dot = 0.0
        elif self.x > self.L:
            self.x = self.L
            self.x_dot = 0.0
        
        self.time_sim += self.dt
        
        # Stop simulation when total time is reached
        if self.time_sim >= self.T:
            self.simulation_active = False
            self.get_logger().info("Simulation complete. Waiting for next trigger.")
        
        # Publish the measured and reference positions
        self.mes_pos_pub.publish(Int32(data=int(self.x)))
        self.ref_pos_pub.publish(Int32(data=int(x_ref)))

def main(args=None):
    parser = argparse.ArgumentParser(description="1D Linear Environment Simulation Node")
    parser.add_argument("--freq", type=float, default=50.0, help="Simulation frequency")
    known_args, _ = parser.parse_known_args()

    rclpy.init(args=sys.argv)
    node = LinearEnvSimNode(freq=known_args.freq)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down LinearEnvSimNode.")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
