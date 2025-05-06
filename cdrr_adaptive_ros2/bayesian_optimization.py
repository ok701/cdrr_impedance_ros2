#!/usr/bin/env python3
"""
BOPowerNode — ROS 2 Safe‑BO (UCB, RBF) for online maximisation of
average mechanical power P = mean(F) × mean(v).

Key points
===========
* **Raw force/velocity samples are collected only while `/trigger`==1**
  and averaged at the falling edge.
* Optimisation variables: damping coefficient `c` [arbitrary units]
  and trajectory duration `T_tot` [ms].
* Gaussian Process with an **RBF kernel** and **UCB acquisition**.
* No built‑in physics demo or mass constant; connect this node to
  your *own* simulator/hardware that publishes `/force_user` and
  `/velocity_user`.
"""

# ════════════════════════════════════════════════════════════════════
#  ROS 2
# ════════════════════════════════════════════════════════════════════
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

# ════════════════════════════════════════════════════════════════════
#  Numerics
# ════════════════════════════════════════════════════════════════════
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# ════════════════════════════════════════════════════════════════════
class BOPowerNode(Node):
    """Safe‑BO node that maximises average user power in real time."""

    def __init__(self):
        super().__init__("bo_power_node")

        # ── Parameters ──────────────────────────────────────────────
        self.declare_parameter("beta",               2.0)
        self.declare_parameter("c_min",             100)   # [arb]
        self.declare_parameter("c_max",            1000)
        self.declare_parameter("T_min",            8000)   # [ms]
        self.declare_parameter("T_max",           30000)
        self.declare_parameter("repeats_per_param", 3)
        self.declare_parameter("warmup_cycles",      0)

        self.beta   = self.get_parameter("beta").value
        cmin, cmax  = self.get_parameter("c_min").value,  self.get_parameter("c_max").value
        tmin, tmax  = self.get_parameter("T_min").value,  self.get_parameter("T_max").value
        self.box    = np.array([[cmin, cmax], [tmin, tmax]])
        self.repeats_per_param = self.get_parameter("repeats_per_param").value
        self.warmup_cycles     = self.get_parameter("warmup_cycles").value

        # ── Gaussian Process ───────────────────────────────────────
        length_init = [(cmax-cmin)/2, (tmax-tmin)/2]
        kernel = RBF(length_scale=length_init, length_scale_bounds=(1e-2, 1e4))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                           normalize_y=True)
        self.X, self.Y = [], []   # collected data

        # ── Initial candidates ─────────────────────────────────────
        self.candidate_queue = [[cmin, (tmin+tmax)//2],
                                [(cmin+cmax)//2, (tmin+tmax)//2]]
        self.current_param   = self.candidate_queue.pop(0)
        self.repeats_done    = 0
        self.cycles_seen     = 0

        # ── ROS 2 I/O ──────────────────────────────────────────────
        qos = rclpy.qos.QoSProfile(depth=10)
        self.sub_trg  = self.create_subscription(Int32, "/trigger",       self.cb_trigger, qos)
        self.sub_F    = self.create_subscription(Int32, "/force_user",    self.cb_force,   qos)
        self.sub_v    = self.create_subscription(Int32, "/velocity_user", self.cb_velocity,qos)
        self.pub_c    = self.create_publisher   (Int32, "/damping",       10)
        self.pub_T    = self.create_publisher   (Int32, "/traj_time",     10)

        # ── Buffers for one cycle ──────────────────────────────────
        self.collecting   = False
        self.prev_trigger = 0
        self.force_buf    = []  # [N]
        self.vel_buf      = []  # [m/s]

        self.publish_param()
        self.get_logger().info(
            "BOPowerNode running   |   first params  c=%.0f  T=%d ms" %
            tuple(self.current_param))

    # ────────────────────────────────────────────────────────────────
    #  Callback: Trigger
    # ────────────────────────────────────────────────────────────────
    def cb_trigger(self, msg: Int32):
        trg = msg.data
        rising  = (self.prev_trigger == 0) and (trg == 1)
        falling = (self.prev_trigger == 1) and (trg == 0)
        self.prev_trigger = trg

        if rising:
            self.collecting = True
            self.force_buf.clear(); self.vel_buf.clear()
            self.get_logger().debug("cycle start — buffering raw samples")

        elif falling and self.collecting:
            self.collecting = False
            self.finish_cycle()

    # ────────────────────────────────────────────────────────────────
    #  Callbacks: raw data
    # ────────────────────────────────────────────────────────────────
    def cb_force(self, msg: Int32):
        if self.collecting:
            self.force_buf.append(msg.data / 1000.0)   # mN → N

    def cb_velocity(self, msg: Int32):
        if self.collecting:
            self.vel_buf.append(msg.data / 1000.0)     # mm/s → m/s

    # ────────────────────────────────────────────────────────────────
    #  End‑of‑cycle processing
    # ────────────────────────────────────────────────────────────────
    def finish_cycle(self):
        self.cycles_seen += 1
        if self.cycles_seen <= self.warmup_cycles:
            self.get_logger().info("cycle %d ignored (warm‑up)" % self.cycles_seen)
            return
        if not self.force_buf or not self.vel_buf:
            self.get_logger().warn("incomplete force/velocity buffers — skipped")
            return

        P = float(np.mean(self.force_buf) * np.mean(self.vel_buf))
        self.X.append(self.current_param)
        self.Y.append(P)
        self.repeats_done += 1
        self.get_logger().info(
            f"cycle {self.cycles_seen}: c={self.current_param[0]}, T={self.current_param[1]}, "
            f"P={P:.3f} W  (rep {self.repeats_done}/{self.repeats_per_param})")

        if self.repeats_done >= self.repeats_per_param:
            self.repeats_done = 0
            self.update_bo()

    # ────────────────────────────────────────────────────────────────
    #  Bayesian Optimisation step
    # ────────────────────────────────────────────────────────────────
    def update_bo(self):
        X, Y = np.array(self.X), np.array(self.Y)
        self.gp.fit(X, Y)

        # Grid‑based UCB
        c_lin = np.linspace(*self.box[0], 40)
        T_lin = np.linspace(*self.box[1], 40)
        grid  = np.array([[c, T] for c in c_lin for T in T_lin])
        mu, sig = self.gp.predict(grid, return_std=True)
        ucb = mu + self.beta * sig
        best = grid[np.argmax(ucb)]
        self.current_param = best.tolist()
        self.publish_param()
        self.get_logger().info("BO suggests  c=%.0f  T=%d ms" % tuple(best))

    # ────────────────────────────────────────────────────────────────
    #  Publish parameters
    # ────────────────────────────────────────────────────────────────
    def publish_param(self):
        c_msg, T_msg = Int32(), Int32()
        c_msg.data = int(round(self.current_param[0]))
        T_msg.data = int(round(self.current_param[1]))
        self.pub_c.publish(c_msg)
        self.pub_T.publish(T_msg)

# ════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════

def main():
    rclpy.init()
    node = BOPowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
