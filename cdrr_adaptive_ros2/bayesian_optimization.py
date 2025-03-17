#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import numpy as np

# matplotlib plot은 옵션에 따라 사용
import matplotlib
matplotlib.use('TkAgg')  # 환경에 따라 'Qt5Agg' 등으로 조정 가능
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import norm

import sys
import argparse


class BayesOptResistNode(Node):
    """
    A ROS2 Node that performs Bayesian Optimization on a single parameter 'resist_level'.
    - Range: [-20, 0] (음수일수록 더 큰 저항)
    - Cost function: cost = 0.4 * (MSE/100) + 0.6 * resist_level
        => resist_level이 더 음수이면(cost 식에서 음수 기여), 최적화가 그쪽을 선호할 수 있음
    - 1) 초기에 (resist_level = 0) → (resist_level = -10) 두 번 측정
    - 2) 세 번째 측정부터 GP로 EI 최대점 탐색
    - 3) 각 resist_level 당 N번 반복 측정 (기본 3번), outlier 제거 후 평균 cost를 GP에 사용
    - 4) 최종 /resist_offset(Int32) 로 발행
    - 실시간 점수, 실시간 Publish 등은 생략
    - --show-plot 옵션 시, 각 iteration마다 GP Mean/Std + EI plot
    """

    def __init__(self, repeats_per_level=3, max_runs=10, show_plot=False):
        super().__init__('bayes_opt_resist_node')

        # ---------------- Parameters ----------------
        self.repeats_per_level = repeats_per_level
        self.max_runs = max_runs
        self.show_plot = show_plot

        # ---------------- ROS Topics ----------------
        # Subscribe to /mes_pos, /ref_pos, /trigger
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
        self.ee_error_list = []  # per-run data

        self.mes_pos = None
        self.ref_pos = None

        # For Bayesian Optimization:
        self.X = []  # tested resist_level
        self.y = []  # cost
        self.run_count = 0

        # Repeats (같은 resist_level로 N번 반복)
        self.current_repeat_count = 0
        self.current_level_costs = []

        # 초기 2점: [0, -10]
        self.initial_candidates = [0.0, -10.0]

        # GP 설정
        kernel = ConstantKernel(1.0) * RBF(length_scale=5.0, length_scale_bounds=(1.0, 20.0))
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=10,
            n_restarts_optimizer=5,
            random_state=42
        )

        # 탐색 범위: [-20, 0]
        self.lower_bound = -20.0
        self.upper_bound = 0.0

        # 현재(또는 다음) 테스트할 resist_level
        # 첫 번째는 initial_candidates[0] = 0
        self.next_resist_level = self.initial_candidates.pop(0)

        # (Optional) Plot setup
        if self.show_plot:
            plt.ion()
            self.fig, (self.ax_gp, self.ax_acq) = plt.subplots(2, 1, figsize=(6, 8))
            self.fig.tight_layout(pad=3.0)
            self.fig.show()

        # 시작 알림 및 첫 파라미터 Publish
        self.get_logger().info(
            f"BayesOptResistNode started with repeats={self.repeats_per_level}, max_runs={self.max_runs}, show_plot={self.show_plot}"
        )
        self.publish_resist_level(self.next_resist_level)

    # ------------------------------------------------------------------
    #                       Subscriber Callbacks
    # ------------------------------------------------------------------
    def mes_pos_callback(self, msg: Int32):
        self.mes_pos = float(msg.data)
        self.compute_and_store_error()

    def ref_pos_callback(self, msg: Int32):
        self.ref_pos = float(msg.data)
        self.compute_and_store_error()

    def trigger_callback(self, msg: Int32):
        new_trigger = msg.data
        if new_trigger == 1 and self.trigger_state == 0:
            # 0->1: start recording
            self.start_recording()
        elif new_trigger == 0 and self.trigger_state == 1:
            # 1->0: end recording -> optimize
            self.stop_recording_and_optimize()

        self.trigger_state = new_trigger

    # ------------------------------------------------------------------
    #                       Data Recording
    # ------------------------------------------------------------------
    def start_recording(self):
        self.run_count += 1  # 새 run
        self.get_logger().info(f"*** Trigger 0->1: Starting run #{self.run_count}, resist_level={self.next_resist_level:.2f}")
        self.collecting_data = True
        self.ee_error_list = []

    def stop_recording_and_optimize(self):
        self.get_logger().info(f"*** Trigger 1->0: Ending run #{self.run_count}. Processing data.")
        self.collecting_data = False

        tested_level = self.next_resist_level

        # 1) Compute cost from this single run
        cost_val = self.compute_cost_from_data(self.ee_error_list, tested_level)
        n_samples = len(self.ee_error_list)
        self.get_logger().info(f"   - Collected {n_samples} error samples. Cost = {cost_val:.5f}")

        # 2) Store cost in current_level_costs
        self.current_level_costs.append(cost_val)

        # 3) Increase the repeat count
        self.current_repeat_count += 1
        self.get_logger().info(f"   - current_repeat_count = {self.current_repeat_count}/{self.repeats_per_level}")

        # 4) Check if we need more repeats
        if self.current_repeat_count < self.repeats_per_level:
            # Re-publish the SAME resist_level
            self.get_logger().info("   - Still repeating the same resist_level; no new optimization step.")
            self.publish_resist_level(self.next_resist_level)
            return

        # ----------------------------------------------------------------
        # Repeats done for this resist_level
        # ----------------------------------------------------------------
        self.get_logger().info(f"=== Repeats complete for resist_level={tested_level:.2f}. ===")

        # (a) Log all repeated costs
        cost_str = ", ".join(f"{c:.5f}" for c in self.current_level_costs)
        self.get_logger().info(f"   - All repeated costs: [{cost_str}]")

        # (b) Remove outliers
        filtered_costs = self.remove_outliers(self.current_level_costs, k=1.3)
        outliers = list(set(self.current_level_costs) - set(filtered_costs))
        if len(outliers) > 0:
            outlier_str = ", ".join(f"{c:.5f}" for c in outliers)
            self.get_logger().warn(f"   - Outliers detected: [{outlier_str}]")
        else:
            self.get_logger().info("   - No outliers detected.")

        if len(filtered_costs) == 0:
            self.get_logger().warn("   - ALL repeats were outliers! Using raw data anyway.")
            filtered_costs = self.current_level_costs

        # (c) Average the valid costs
        final_cost_for_gp = float(np.mean(filtered_costs))
        self.get_logger().info(f"   - Average cost after outlier removal: {final_cost_for_gp:.5f}")

        # Clear for next
        self.current_repeat_count = 0
        self.current_level_costs = []

        # 5) Update GP dataset
        self.X.append(tested_level)
        self.y.append(final_cost_for_gp)

        # 6) Decide if we do further optimization
        if len(self.X) < self.max_runs:
            # (A) If there's still an initial candidate, use that next
            if self.initial_candidates:
                self.next_resist_level = self.initial_candidates.pop(0)
                self.get_logger().info(f"   - Next is initial candidate: {self.next_resist_level:.2f}")
            else:
                # (B) We do GP-based suggestion
                self.get_logger().info("   - Running Bayesian Optimization iteration ...")
                self.run_gp_and_suggest()

            # Publish new level
            self.publish_resist_level(self.next_resist_level)
        else:
            self.get_logger().info(f"Reached max_runs={self.max_runs}. No further optimization.")
            self.publish_resist_level(0.0)  # 혹은 -20.0 등의 기본값

    # ------------------------------------------------------------------
    #             Compute and Store Error Each Time We Get Data
    # ------------------------------------------------------------------
    def compute_and_store_error(self):
        if self.collecting_data and (self.mes_pos is not None) and (self.ref_pos is not None):
            error_val = self.ref_pos - self.mes_pos
            self.ee_error_list.append(error_val)

    # ------------------------------------------------------------------
    #                       Cost Computation
    # ------------------------------------------------------------------
    def compute_cost_from_data(self, error_list, resist_level):
        """
        cost = 0.4 * (MSE/100) + 0.6 * resist_level
        """
        if len(error_list) == 0:
            return 0.0

        mse = np.mean(np.square(error_list))
        # scale MSE by 100
        mse_part = mse / 100.0

        cost_val = 0.4 * mse_part + 0.6 * resist_level
        return cost_val

    # ------------------------------------------------------------------
    #                 Remove Outliers from a List
    # ------------------------------------------------------------------
    def remove_outliers(self, cost_list, k=1.3):
        """
        Exclude values that are more than k std dev from the mean.
        Returns a filtered list of costs.
        """
        if len(cost_list) < 2:
            return cost_list

        mean_val = np.mean(cost_list)
        std_val = np.std(cost_list)
        if std_val == 0:
            # all identical
            return cost_list

        filtered = [c for c in cost_list if abs(c - mean_val) <= k * std_val]
        return filtered

    # ------------------------------------------------------------------
    #                Bayesian Optimization Step
    # ------------------------------------------------------------------
    def run_gp_and_suggest(self):
        X_data = np.array(self.X).reshape(-1, 1)
        y_data = np.array(self.y)

        # Fit GP
        self.gp.fit(X_data, y_data)

        # Update plot if desired
        if self.show_plot:
            self.update_plot(X_data, y_data)

        # Propose next resist_level by EI
        self.next_resist_level = self.propose_next_resist(X_data, y_data)
        self.get_logger().info(f"   - Proposed next resist_level: {self.next_resist_level:.2f}")

    def propose_next_resist(self, X_data, y_data):
        # Grid in [lower_bound, upper_bound]
        candidates = np.linspace(self.lower_bound, self.upper_bound, 200).reshape(-1, 1)
        ei_values = self.expected_improvement(candidates, X_data, y_data, self.gp, xi=0.001)
        best_idx = np.argmax(ei_values)
        return float(candidates[best_idx, 0])

    def expected_improvement(self, X_new, X, y, model, xi=0.01):
        mu, sigma = model.predict(X_new, return_std=True)
        y_min = np.min(y)
        improvement = (y_min - mu) - xi
        Z = improvement / sigma
        # EI formula
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
        return ei

    # ------------------------------------------------------------------
    #                Publish Resist Level (Int32)
    # ------------------------------------------------------------------
    def publish_resist_level(self, level):
        # 음수 실수 → int 변환 시 주의. 여기서는 round 후 int.
        out_msg = Int32()
        out_msg.data = int(round(level))
        self.resist_pub.publish(out_msg)
        self.get_logger().info(f"Published resist_offset={out_msg.data} to /resist_offset.")

    # ------------------------------------------------------------------
    #                      Plotting (if show_plot)
    # ------------------------------------------------------------------
    def update_plot(self, X_data, y_data):
        self.ax_gp.clear()
        self.ax_acq.clear()

        # Plot range (maybe a bit wider than [-20,0] just for visualization)
        X_plot = np.linspace(-25, 5, 200).reshape(-1, 1)
        mu, std = self.gp.predict(X_plot, return_std=True)

        # GP mean & std
        self.ax_gp.plot(X_plot, mu, 'b-', label='GP Mean')
        self.ax_gp.fill_between(
            X_plot.ravel(), mu - std, mu + std,
            alpha=0.2, color='blue', label='GP ±1σ'
        )
        self.ax_gp.scatter(X_data, y_data, c='r', label='Data')
        self.ax_gp.set_title("Gaussian Process Model of the Cost")
        self.ax_gp.set_xlabel("resist_level")
        self.ax_gp.set_ylabel("Cost (lower=better)")
        self.ax_gp.legend()
        self.ax_gp.grid(True)

        # Acquisition
        ei_values = self.expected_improvement(X_plot, X_data, y_data, self.gp)
        self.ax_acq.plot(X_plot, ei_values, 'g-', label='EI')
        self.ax_acq.set_title("Acquisition Function (EI)")
        self.ax_acq.set_xlabel("resist_level")
        self.ax_acq.set_ylabel("EI")
        self.ax_acq.legend()
        self.ax_acq.grid(True)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.2)


def main(args=None):
    parser = argparse.ArgumentParser(description="Bayesian Opt Resist Node")
    parser.add_argument("--repeats", type=int, default=3, help="How many repeats per resist_level")
    parser.add_argument("--max-runs", type=int, default=10, help="Max number of runs (i.e. distinct resist_levels)")
    parser.add_argument("--show-plot", action='store_true', help="Show GP plot after each iteration")
    known_args, _ = parser.parse_known_args()

    rclpy.init(args=sys.argv)
    node = BayesOptResistNode(
        repeats_per_level=known_args.repeats,
        max_runs=known_args.max_runs,
        show_plot=known_args.show_plot
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
