#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import numpy as np
import sys, argparse


class LinearEnvSimNode(Node):
    """
    1-D linear environment simulator that now publishes
    • /force_user     – instantaneous cable force      [mN]
    • /velocity_user  – instantaneous handle velocity  [mm/s]
    Raw samples are broadcast only while /trigger is 1.
    """

    def __init__(self, freq: float = 50.0):
        super().__init__("linear_env_sim")

        # ───── sim constants ────────────────────────────────────────
        self.mass   = 1.0          # kg
        self.b      = 0.01         # N / (mm/s)
        self.L      = 500.0        # mm
        self.T      = 10.0         # s
        self.dt     = 1.0 / freq
        self.Kp     = 0.0          # “human” spring gain

        # RBF assist parameters
        self.centers = np.linspace(0, 500, 3)
        self.sigma   = 150.0
        self.weights = np.zeros(3)  # decoded in N

        # ───── ROS I/O ──────────────────────────────────────────────
        qos = rclpy.qos.QoSProfile(depth=10)

        self.sub_trigger = self.create_subscription(
            Int32, "/trigger", self.cb_trigger, qos)
        self.sub_w1 = self.create_subscription(
            Int32, "/rbf_w1", lambda m: self._update_w(0, m), qos)
        self.sub_w2 = self.create_subscription(
            Int32, "/rbf_w2", lambda m: self._update_w(1, m), qos)
        self.sub_w3 = self.create_subscription(
            Int32, "/rbf_w3", lambda m: self._update_w(2, m), qos)

        self.pub_mes = self.create_publisher(Int32, "/mes_position",   10)
        self.pub_ref = self.create_publisher(Int32, "/ref_position",   10)
        self.pub_F   = self.create_publisher(Int32, "/force_user",     10)
        self.pub_v   = self.create_publisher(Int32, "/velocity_user",  10)

        # ───── state ────────────────────────────────────────────────
        self.trigger_state     = 0
        self.simulation_active = False
        self.x      = 0.0   # mm
        self.x_dot  = 0.0   # mm/s
        self.t_sim  = 0.0   # s

        self.create_timer(self.dt, self.update_sim)
        self.get_logger().info(f"LinearEnvSimNode running at {freq} Hz")

    # ───── callbacks ────────────────────────────────────────────────
    def cb_trigger(self, msg: Int32):
        trg = msg.data
        if trg == 1 and self.trigger_state == 0:
            self.simulation_active = True
            self.get_logger().info("trigger ↑  →  sim start")
        elif trg == 0 and self.trigger_state == 1:
            self._reset()
            self.get_logger().info("trigger ↓  →  sim reset")
        self.trigger_state = trg

    def _update_w(self, idx: int, msg: Int32):
        self.weights[idx] = msg.data / 100.0  # back-scale (mN → N)
        self.get_logger().debug(f"w{idx+1} = {self.weights[idx]:.3f} N")

    # ───── helpers ─────────────────────────────────────────────────
    def _reset(self):
        self.x = self.x_dot = self.t_sim = 0.0
        self.simulation_active = False

    def _x_ref(self, t: float) -> float:
        if t >= self.T:
            return self.L
        tau = t / self.T
        return self.L * (10*tau**3 - 15*tau**4 + 6*tau**5)

    # ───── main loop ───────────────────────────────────────────────
    def update_sim(self):
        if not self.simulation_active:
            return

        # reference trajectory
        x_ref = self._x_ref(self.t_sim)

        # RBF assist force (N)
        phi = np.exp(-((x_ref - self.centers) ** 2) / (2 * self.sigma ** 2))
        F_assist = float(np.dot(self.weights, phi))

        # “human” spring force (N)
        F_human = self.Kp * (x_ref - self.x)

        # net force (N)  (damping uses mm/s → convert)
        F_net = F_assist - self.b * self.x_dot + F_human

        # dynamics (mm/s²)
        x_ddot = 1000.0 * (F_net / self.mass)

        # integrate
        self.x_dot += x_ddot * self.dt
        self.x     += self.x_dot * self.dt
        self.x      = np.clip(self.x, 0.0, self.L)

        self.t_sim += self.dt
        if self.t_sim >= self.T:
            self.simulation_active = False
            self.get_logger().info("simulation finished; waiting for next trigger")

        # ─ publish raw samples (mN, mm/s) ─
        self.pub_F.publish(Int32(data=int(round(F_assist * 1000))))   # N→mN
        self.pub_v.publish(Int32(data=int(round(self.x_dot))))        # mm/s
        # publish positions for any visualiser
        self.pub_mes.publish(Int32(data=int(self.x)))
        self.pub_ref.publish(Int32(data=int(x_ref)))


# ───── main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", type=float, default=50.0,
                        help="simulation frequency [Hz]")
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = LinearEnvSimNode(freq=args.freq)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
