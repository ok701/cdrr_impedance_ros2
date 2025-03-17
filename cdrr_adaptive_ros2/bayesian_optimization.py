#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Adjust backend as needed (e.g., 'Qt5Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys
import argparse

class BayesOptResistNode(Node):
    """
    A ROS2 Node that performs Bayesian Optimization on a single parameter 'resist_level'
    using Bayesian Quadratic Regression.
    
    - Range: [-20, 0] (more negative values represent higher resistance)
    - Cost function: cost = 0.4 * (MSE/100) + 0.6 * resist_level
      (Lower cost is better.)
    - Steps:
        1) Initially measure two points: resist_level = 0 and resist_level = -10.
        2) For each resist_level, perform N repeats (default 3), remove outliers,
           and compute the average cost.
        3) Use Bayesian Quadratic Regression (with a quadratic model f(x)=w1*x² + w2*x + w3)
           to obtain the predictive mean and variance.
        4) Use the Expected Improvement acquisition function to propose the next resist_level.
        5) Finally, publish the result to /resist_offset (Int32).
    - If the --show-plot option is set, the predictive mean, uncertainty, and data are plotted.
    - The first --ignore-cycles cycles are completely ignored (no data is collected or processed).
      Optimization starts only after those initial cycles.
    """
    def __init__(self, repeats_per_level=3, max_runs=5, show_plot=True, ignore_cycles=5):
        super().__init__('bayes_opt_resist_node')

        # ---------------- Parameters ----------------
        self.repeats_per_level = repeats_per_level
        self.max_runs = max_runs
        self.show_plot = show_plot
        self.ignore_cycles = ignore_cycles  # Number of cycles to ignore (no processing)
        # We'll use run_count to decide when to start processing
        self.run_count = 0

        # ---------------- ROS Topics ----------------
        # Subscribe to /mes_pos, /ref_pos, /trigger topics
        self.mes_pos_sub = self.create_subscription(
            Int32,
            '/mes_pos',
            self.mes_pos_callback,
            10
        )
        self.ref_pos_sub = self.create_subscription(
            Int32,
            '/ref_pos',
            self.ref_pos_callback,
            10
        )
        self.trigger_sub = self.create_subscription(
            Int32,
            '/trigger',
            self.trigger_callback,
            10
        )
        # Publish to /resist_offset (Int32)
        self.resist_pub = self.create_publisher(
            Int32,
            '/resist_offset',
            10
        )

        # ---------------- Internal State ----------------
        self.trigger_state = 0
        self.collecting_data = False
        self.ee_error_list = []  # Data for the current run
        self.mes_pos = None
        self.ref_pos = None

        # Data for optimization (only used after ignore_cycles):
        self.X = []  # Tested resist_levels
        self.y = []  # Corresponding cost values

        # For repeating measurements at a given resist_level
        self.current_repeat_count = 0
        self.current_level_costs = []

        # Initial candidates: [0, -10]
        self.initial_candidates = [0, -10.0]
        self.next_resist_level = self.initial_candidates.pop(0)

        # Define search bounds for resist_level
        self.lower_bound = -20.0
        self.upper_bound = 0.0

        # ---------------- Plot Setup (Optional) ----------------
        if self.show_plot:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 4))
            self.fig.tight_layout(pad=3.0)
            self.fig.show()

        self.get_logger().info(
            f"BayesOptResistNode started with repeats={self.repeats_per_level}, "
            f"max_runs={self.max_runs}, show_plot={self.show_plot}, ignore_cycles={self.ignore_cycles}"
        )
        self.publish_resist_level(self.next_resist_level)

    # ---------------- Subscriber Callbacks ----------------
    def mes_pos_callback(self, msg: Int32):
        self.mes_pos = float(msg.data)
        self.compute_and_store_error()

    def ref_pos_callback(self, msg: Int32):
        self.ref_pos = float(msg.data)
        self.compute_and_store_error()

    def trigger_callback(self, msg: Int32):
        new_trigger = msg.data
        # We assume a trigger transition: 0->1 starts, 1->0 ends a cycle.
        if new_trigger == 1 and self.trigger_state == 0:
            self.start_recording()
        elif new_trigger == 0 and self.trigger_state == 1:
            self.stop_recording_and_optimize()
        self.trigger_state = new_trigger

    # ---------------- Data Recording ----------------
    def start_recording(self):
        self.run_count += 1  # Increase run_count at the start of a cycle
        self.get_logger().info(
            f"*** Trigger 0->1: Starting run #{self.run_count}, resist_level={self.next_resist_level:.2f}"
        )
        self.collecting_data = True
        self.ee_error_list = []

    def stop_recording_and_optimize(self):
        # If we are in the ignore period, simply ignore this cycle.
        if self.run_count <= self.ignore_cycles:
            self.get_logger().info(
                f"Cycle {self.run_count}/{self.ignore_cycles} ignored. No data processing."
            )
            self.collecting_data = False
            self.ee_error_list = []  # Clear any collected data (if any)
            self.publish_resist_level(self.next_resist_level)
            return

        self.get_logger().info(f"*** Trigger 1->0: Ending run #{self.run_count}. Processing data.")
        self.collecting_data = False
        tested_level = self.next_resist_level

        # Process collected data:
        cost_val = self.compute_cost_from_data(self.ee_error_list, tested_level)
        n_samples = len(self.ee_error_list)
        self.get_logger().info(f"   - Collected {n_samples} error samples. Cost = {cost_val:.5f}")

        # Increment repeat count and store cost
        self.current_repeat_count += 1
        self.current_level_costs.append(cost_val)
        self.get_logger().info(
            f"   - current_repeat_count = {self.current_repeat_count}/{self.repeats_per_level}"
        )

        # If not all repeats are done, republish the same resist_level and wait for next cycle
        if self.current_repeat_count < self.repeats_per_level:
            self.get_logger().info("   - Still repeating the same resist_level; no new optimization step.")
            self.publish_resist_level(self.next_resist_level)
            return

        self.get_logger().info(f"=== Repeats complete for resist_level={tested_level:.2f}. ===")
        cost_str = ", ".join(f"{c:.5f}" for c in self.current_level_costs)
        self.get_logger().info(f"   - All repeated costs: [{cost_str}]")

        # Remove outliers from the repeated costs
        filtered_costs = self.remove_outliers(self.current_level_costs, k=1.3)
        if len(filtered_costs) == 0:
            self.get_logger().warn("   - ALL repeats were outliers! Using raw data anyway.")
            filtered_costs = self.current_level_costs

        final_cost = float(np.mean(filtered_costs))
        self.get_logger().info(f"   - Average cost after outlier removal: {final_cost:.5f}")

        # Reset repeat counters for the next resist_level
        self.current_repeat_count = 0
        self.current_level_costs = []

        # Update the optimization dataset with the current resist_level and its cost
        self.X.append(tested_level)
        self.y.append(final_cost)

        # Decide on the next action (if there is enough data, run Bayesian quadratic regression)
        if len(self.X) < self.max_runs:
            if self.initial_candidates:
                self.next_resist_level = self.initial_candidates.pop(0)
                self.get_logger().info(f"   - Next is initial candidate: {self.next_resist_level:.2f}")
            else:
                self.run_bayesian_quad_and_suggest()
            self.publish_resist_level(self.next_resist_level)
        else:
            self.get_logger().info(f"Reached max_runs={self.max_runs}. No further optimization.")
            self.publish_resist_level(0.0)  # Or publish a default value

    # ---------------- Compute and Store Error ----------------
    def compute_and_store_error(self):
        if self.collecting_data and (self.mes_pos is not None) and (self.ref_pos is not None):
            error_val = self.ref_pos - self.mes_pos
            self.ee_error_list.append(error_val)

    # ---------------- Cost Computation ----------------
    def compute_cost_from_data(self, error_list, resist_level):
        """
        Compute the cost based on:
          cost = 0.4 * (MSE/100) + 0.6 * resist_level
        """
        if len(error_list) == 0:
            return 0.0
        mse = np.mean(np.square(error_list))
        mse_part = mse / 100.0
        cost_val = 0.4 * mse_part + 0.6 * resist_level
        return cost_val

    # ---------------- Remove Outliers ----------------
    def remove_outliers(self, cost_list, k=1.3):
        """
        Exclude values that are more than k standard deviations away from the mean.
        Returns a filtered list of costs.
        """
        if len(cost_list) < 2:
            return cost_list
        mean_val = np.mean(cost_list)
        std_val = np.std(cost_list)
        if std_val == 0:
            return cost_list
        filtered = [c for c in cost_list if abs(c - mean_val) <= k * std_val]
        return filtered

    # ---------------- Bayesian Quadratic Regression ----------------
    def bayesian_quad_predict(self, x, sigma_p=1.0, sigma_n=1.0):
        """
        Predictive mean and variance for a given x using Bayesian Quadratic Regression.
        The quadratic model is: f(x) = w1*x^2 + w2*x + w3.
        Using a prior: p(w) = N(0, sigma_p^2 * I) and noise variance sigma_n^2.
        """
        X_arr = np.array(self.X)
        y_arr = np.array(self.y)
        # Construct design matrix for observed data
        Phi = np.vstack([X_arr**2, X_arr, np.ones(len(X_arr))]).T  # shape (n, 3)
        prior_cov = sigma_p**2 * np.eye(3)
        Sigma_inv = (1/sigma_n**2) * (Phi.T @ Phi) + np.linalg.inv(prior_cov)
        Sigma = np.linalg.inv(Sigma_inv)
        mu_post = (1/sigma_n**2) * (Sigma @ (Phi.T @ y_arr))
        phi_x = np.array([x**2, x, 1]).reshape(3, 1)
        m = (phi_x.T @ mu_post).item()
        v = (phi_x.T @ Sigma @ phi_x).item() + sigma_n**2
        return m, v

    def run_bayesian_quad_and_suggest(self):
        """
        Use Bayesian Quadratic Regression to fit a model to (X, y),
        then use the Expected Improvement acquisition function to propose the next resist_level.
        """
        if len(self.X) < 3:
            self.get_logger().warn("Not enough data for Bayesian quadratic regression. Keeping current resist_level.")
            return

        sigma_p = 1.0  # Prior standard deviation
        sigma_n = 1.0  # Noise standard deviation
        candidates = np.linspace(self.lower_bound, self.upper_bound, 200)
        ei_values = []
        f_best = min(self.y)  # Best (minimum) observed cost
        xi = 0.001  # Exploration parameter
        for x in candidates:
            m, v = self.bayesian_quad_predict(x, sigma_p, sigma_n)
            sigma = np.sqrt(v)
            improvement = f_best - m - xi
            if sigma > 0:
                Z = improvement / sigma
                ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            else:
                ei = 0.0
            ei_values.append(ei)
        ei_values = np.array(ei_values)
        best_idx = np.argmax(ei_values)
        proposed = candidates[best_idx]
        self.next_resist_level = float(np.clip(proposed, self.lower_bound, self.upper_bound))
        self.get_logger().info(
            f"Bayesian quadratic regression suggests next resist_level: {self.next_resist_level:.2f}"
        )
        if self.show_plot:
            self.update_plot_bayesian(sigma_p, sigma_n)

    def update_plot_bayesian(self, sigma_p=1.0, sigma_n=1.0):
        self.ax.clear()
        self.ax.scatter(self.X, self.y, c='r', label='Data')
        x_vals = np.linspace(self.lower_bound, self.upper_bound, 200)
        means = []
        stds = []
        for x in x_vals:
            m, v = self.bayesian_quad_predict(x, sigma_p, sigma_n)
            means.append(m)
            stds.append(np.sqrt(v))
        means = np.array(means)
        stds = np.array(stds)
        self.ax.plot(x_vals, means, 'b-', label='Predictive Mean')
        self.ax.fill_between(x_vals, means - stds, means + stds, color='blue', alpha=0.2, label='Uncertainty')
        self.ax.set_title("Bayesian Quadratic Regression")
        self.ax.set_xlabel("resist_level")
        self.ax.set_ylabel("Cost (lower=better)")
        self.ax.legend()
        self.ax.grid(True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.2)

    # ---------------- Publish Resist Level ----------------
    def publish_resist_level(self, level):
        out_msg = Int32()
        out_msg.data = int(round(level))
        self.resist_pub.publish(out_msg)
        self.get_logger().info(f"Published resist_offset={out_msg.data} to /resist_offset.")

def main(args=None):
    parser = argparse.ArgumentParser(
        description="Bayesian Opt Resist Node using Bayesian Quadratic Regression"
    )
    parser.add_argument("--repeats", type=int, default=3, help="How many repeats per resist_level")
    parser.add_argument("--max-runs", type=int, default=5, help="Max number of runs (i.e. distinct resist_levels)")
    parser.add_argument("--show-plot", action='store_true', help="Show Bayesian quadratic regression plot after each iteration")
    parser.add_argument("--ignore-cycles", type=int, default=5, help="Number of cycles to ignore (do nothing) before starting optimization")
    known_args, _ = parser.parse_known_args()

    rclpy.init(args=sys.argv)
    node = BayesOptResistNode(
        repeats_per_level=known_args.repeats,
        max_runs=known_args.max_runs,
        show_plot=known_args.show_plot,
        ignore_cycles=known_args.ignore_cycles
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down BayesOptResistNode.")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
