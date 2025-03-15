#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Int32MultiArray
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel
from scipy.stats import norm

# Define a simple quadratic kernel (a degree-2 polynomial kernel)
class QuadraticKernel(Kernel):
    def __init__(self, constant=1.0):
        self.constant = constant

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        K = (np.dot(X, Y.T) + self.constant) ** 2
        if eval_gradient:
            raise NotImplementedError("Gradient not implemented for QuadraticKernel.")
        return K

    def diag(self, X):
        return np.diag(self.__call__(X))

    def is_stationary(self):
        return False

class BayesianRBFOptNode(Node):
    """
    A ROS2 node that fine-tunes one selected RBF weight using Bayesian optimization.
    
    Data recording starts when the trigger goes from 0 to 1 and stops when it falls from 1 to 0.
    During recording, the node:
      - Computes error as (ref - mes) for each sample.
      - Assigns each sample to the closest RBF center and accumulates the error per unit.
      - Stores each error sample for cost computation.
      
    When the trigger falls (1 → 0), the node:
      1. Selects the RBF unit (weight index) with the largest accumulated error.
      2. Computes a cost from the recorded error data and the new weight value.
      3. Uses Bayesian optimization (with a quadratic kernel) to propose a new δ (delta)
         value (i.e. the amount to subtract) within the allowed bounds.
      4. Updates the selected weight: new_weight = current_weight - δ.
      5. Publishes an update message containing [weight_index, δ].
    """
    def __init__(self):
        super().__init__('bayesian_rbf_opt_node')
        
        # ------------------ RBF Parameters ------------------
        self.n_rbf = 3
        self.centers = np.linspace(0, 500, self.n_rbf)  # Example: centers evenly spaced from 0 to 500
        self.error_sum = np.zeros(self.n_rbf)           # Error accumulator per RBF unit
        
        # Current RBF weights (for all units), initialized to some default value (예: 0)
        self.current_weights = np.zeros(self.n_rbf)
        
        # ------------------ Data Recording ------------------
        self.mes_pos = None
        self.ref_pos = None
        self.ee_error_list = []  # List of error samples (ref - mes)
        self.collecting_data = False
        
        # ------------------ Trigger Control ------------------
        self.trigger_state = 0
        self.end_call_count = 0  # 1→0 떨어질 때마다(즉 stop_recording_and_optimize 할 때마다) 증가
        
        # ------------------ Bayesian Optimization for Fine-Tuning ------------------
        # We now optimize the change amount δ (delta), i.e. how much to subtract from current weight.
        self.delta_history = []  # History of δ values applied
        self.cost_history = []   # Corresponding cost values
        # Define bounds for δ (the subtraction amount)
        self.delta_lower_bound = 0.0  # 최소 10N 낮춤
        self.delta_upper_bound = 20.0  # 최대 20N 낮춤
        # 초기에는 기본 δ값 (예: 10N)로 설정
        self.next_delta = 10.0
        
        # Set up the Gaussian Process regressor using a quadratic kernel.
        kernel = ConstantKernel(1.0) * QuadraticKernel(constant=1.0)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=10,
            n_restarts_optimizer=5,
            random_state=42
        )
        
        # ------------------ ROS Subscribers & Publisher ------------------
        self.create_subscription(Int32, '/mes_pos', self.mes_pos_callback, 10)
        self.create_subscription(Int32, '/ref_pos', self.ref_pos_callback, 10)
        self.create_subscription(Int32, '/trigger', self.trigger_callback, 10)
        # Publisher sends an update message: [weight_index, subtraction_amount (δ)]
        self.weight_update_pub = self.create_publisher(Int32MultiArray, '/rbf_weight_update', 10)
        
        self.get_logger().info("BayesianRBFOptNode for fine-tuning started. Waiting for trigger events.")
        
    def mes_pos_callback(self, msg: Int32):
        self.mes_pos = float(msg.data)
        self.process_sample()
    
    def ref_pos_callback(self, msg: Int32):
        self.ref_pos = float(msg.data)
        self.process_sample()
        
    def trigger_callback(self, msg: Int32):
        new_trigger = msg.data
        # Rising edge: 0 -> 1: start data recording
        if new_trigger == 1 and self.trigger_state == 0:
            self.start_recording()
        # Falling edge: 1 -> 0: stop recording and optimize weight
        elif new_trigger == 0 and self.trigger_state == 1:
            self.stop_recording_and_optimize()
        self.trigger_state = new_trigger
        
    def start_recording(self):
        self.get_logger().info("Trigger rising edge: Starting data recording for fine-tuning weight.")
        self.collecting_data = True
        self.ee_error_list = []                # Reset error samples
        self.error_sum = np.zeros(self.n_rbf)   # Reset RBF error accumulation
        
    def stop_recording_and_optimize(self):
        self.get_logger().info("Trigger falling edge: Stopping data recording.")
        self.collecting_data = False
        
        # 최적화 호출 횟수 카운트
        self.end_call_count += 1
        
        # 1~5번째에는 그냥 데이터만 수집하고 최적화는 스킵
        if self.end_call_count < 6:
            self.get_logger().info(f"({self.end_call_count}회차) 아직 최적화 실행 전 단계. 최적화를 스킵합니다.")
            
            # 다음 run을 위해 데이터 초기화.
            self.ee_error_list = []
            self.error_sum = np.zeros(self.n_rbf)
            self.mes_pos = None
            self.ref_pos = None
            return
        
        # ------------------ 6번째부터 최적화 ------------------
        self.get_logger().info(f"({self.end_call_count}회차) 최적화를 진행합니다.")
        
        # 1. Select the RBF unit (weight index) with the largest accumulated error.
        selected_index = int(np.argmax(self.error_sum))
        self.get_logger().info(f"Selected RBF unit index for fine-tuning: {selected_index}")
        
        # 2. 현재 선택된 웨이트 읽기
        current_weight = self.current_weights[selected_index]
        
        # 3. 새로운 웨이트 = current_weight - δ (이번 run에 사용할 self.next_delta)
        new_weight = current_weight - self.next_delta
        
        # 4. 측정된 데이터와 new_weight을 바탕으로 cost 계산 (예: mse + beta * new_weight)
        cost_val, _, _ = self.compute_cost_from_data(self.ee_error_list, new_weight)
        self.get_logger().info(f"Computed cost: {cost_val:.5f}")
        
        # 5. Bayesian 최적화를 위해, 이번 run에서 적용한 δ와 cost를 기록.
        self.delta_history.append(self.next_delta)
        self.cost_history.append(cost_val)
        
        # 6. 만약 기록된 데이터가 2개 이상이면 GP를 업데이트하고 새로운 δ를 제안
        if len(self.delta_history) > 1:
            X_data = np.array(self.delta_history).reshape(-1, 1)
            y_data = np.array(self.cost_history)
            self.gp.fit(X_data, y_data)
            self.next_delta = self.propose_next_delta(X_data, y_data)
            self.get_logger().info(f"Proposed next delta: {self.next_delta:.2f} N")
        else:
            self.get_logger().info("Not enough data for Bayesian optimization. Lowering weight by 10N.")
            self.next_delta = 10.0  # 기본값
            
        # 7. 실제 업데이트: 새 웨이트 = current_weight - (제안된 δ)
        new_weight = current_weight - self.next_delta
        self.current_weights[selected_index] = new_weight
        
        # 8. δ(변동값)를 업데이트 메시지로 발행: [selected_index, δ]
        self.publish_weight_update(selected_index, self.next_delta)
        
        # 9. 다음 run을 위해 데이터 초기화.
        self.ee_error_list = []
        self.error_sum = np.zeros(self.n_rbf)
        self.mes_pos = None
        self.ref_pos = None
        
    def process_sample(self):
        if not self.collecting_data or self.mes_pos is None or self.ref_pos is None:
            return
        error = self.ref_pos - self.mes_pos
        self.ee_error_list.append(error)
        # 각 샘플을 가장 가까운 RBF 중심에 할당하여 누적 오차를 계산.
        closest_index = int(np.argmin(np.abs(self.centers - self.ref_pos)))
        self.error_sum[closest_index] += error
        self.get_logger().debug(
            f"mes: {self.mes_pos:.2f}, ref: {self.ref_pos:.2f}, error: {error:.2f} added to index {closest_index}."
        )
        
    def compute_cost_from_data(self, error_list, weight_value):
        """
        Compute cost from the recorded error list and the new weight value.
        Here, cost is defined as MSE plus a penalty term.
        Since smaller weight is better, the penalty is linear.
        """
        if len(error_list) == 0:
            return 0.0, 0.0, 0.0
        mse = np.mean(np.square(error_list))
        beta = 0.1  # Penalty factor on the weight value
        cost_val = mse + beta * weight_value
        strength_score = weight_value  # (dummy mapping)
        accuracy_score = np.exp(-mse / 15000) * 100
        return cost_val, strength_score, accuracy_score
        
    def expected_improvement(self, X_new, X, y, model, xi=0.001):
        mu, sigma = model.predict(X_new, return_std=True)
        y_min = np.min(y)
        improvement = (y_min - mu) - xi
        with np.errstate(divide='warn'):
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei
        
    def propose_next_delta(self, X_data, y_data):
        grid = np.linspace(self.delta_lower_bound, self.delta_upper_bound, 200).reshape(-1, 1)
        ei = self.expected_improvement(grid, X_data, y_data, self.gp, xi=0.001)
        best_idx = np.argmax(ei)
        return float(grid[best_idx, 0])
        
    def publish_weight_update(self, weight_index, delta_value):
        msg = Int32MultiArray()
        # 메시지 형식: [업데이트할 웨이트 인덱스, 적용할 δ (변동값)]
        msg.data = [weight_index, int(delta_value)]
        self.weight_update_pub.publish(msg)
        self.get_logger().info(f"Published update: Weight index {weight_index}, lower by {delta_value:.2f} N")
        
def main(args=None):
    rclpy.init(args=args)
    node = BayesianRBFOptNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
    main()
