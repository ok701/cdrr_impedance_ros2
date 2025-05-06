#!/usr/bin/env python3
"""
BOResistNode ― 실시간 Safe-BO + 데모 시각화 (타임아웃 로직 제거판)
────────────────────────────────────────────────────────────────
* Δ[mm], K[N/m]를 퍼블리시하여 실험 장치 제어
* repeat_per_param 회 실험 후 GP(UCB) 업데이트
* BO 업데이트 때마다 데모와 동일한 3-pane 그래프(EI·궤적·힘/K)를 표시
* W_user 타임아웃( watchdog ) 기능은 **모두 비활성화**됨
"""

# ────── ROS ────────────────────────────────────────────────
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

# ────── 수치·시각화 ────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

# ────── 데모 시뮬레이션 상수 ───────────────────────────────
T, DT = 10.0, 0.001
TIME  = np.arange(0, T, DT)
F_STEP = 1.0
MASS   = 1.0
B_DAMP = 4.0

def _min_jerk(s):
    """minimum-jerk trajectory profile"""
    return 10*s**3 - 15*s**4 + 6*s**5


class BOResistNode(Node):
    """
    Safe-BO 노드 (trigger 버전 + 시각화, 타임아웃 제거)
    trigger = 1000(=1.0) 동안 한 세트 측정 → 세트 종료 직후 /w_user 값 하나 사용
    """

    # ────────── 초기화 ────────────────────────────────────
    def __init__(self):
        super().__init__("bo_resist_node")

        # ---------- 파라미터 ----------
        self.declare_parameter("beta",              2.0)
        self.declare_parameter("box_delta_min",   100)
        self.declare_parameter("box_delta_max",  1000)
        self.declare_parameter("box_k_min",          0)
        self.declare_parameter("box_k_max",         20)
        self.declare_parameter("repeats_per_param", 3)
        self.declare_parameter("ignore_cycles",     0)
        self.declare_parameter("enable_plot",    True)

        self.beta        = self.get_parameter("beta").value
        dm, dM           = (self.get_parameter("box_delta_min").value,
                            self.get_parameter("box_delta_max").value)
        km, kM           = (self.get_parameter("box_k_min").value,
                            self.get_parameter("box_k_max").value)
        self.box         = np.array([[dm, dM], [km, kM]])
        self.repeats_per_param = self.get_parameter("repeats_per_param").value
        self.ignore_cycles     = self.get_parameter("ignore_cycles").value
        self.enable_plot       = self.get_parameter("enable_plot").value

        # ---------- BO 구조 ----------
        self.X, self.Y = [], []              # Δ,K   /   cost(= –W_user)
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True
        )

        # 초기 두 점
        self.candidate_queue = [[dm, (km + kM) / 2],
                                [(dm + dM)//2, (km + kM)//2]]
        self.current_param = self.candidate_queue.pop(0)
        self.repeats_done  = 0
        self.cycles_seen   = 0

        # ---------- ROS 통신 ----------
        qos = rclpy.qos.QoSProfile(depth=10)
        self.sub_trigger = self.create_subscription(
            Int32, "/trigger", self.cb_trigger, qos)
        self.sub_w = self.create_subscription(
            Int32, "/w_user", self.cb_w, qos)

        self.pub_delta = self.create_publisher(Int32, "/lead_margin", 10)
        self.pub_k     = self.create_publisher(Int32, "/resist_k",    10)

        # ---------- 상태 ----------
        self.collecting       = False
        self.latest_w         = None
        self.prev_trigger_val = 0.0        # ← 상승·하강 에지 검출용

        # (watchdog 타이머 완전 제거)

        # 첫 파라미터 퍼블리시
        self.publish_param()
        self.get_logger().info("BOResistNode (trigger, viz, no-timeout) started.")
        self.get_logger().info(f"초기 Δ={self.current_param[0]}, "
                               f"K={self.current_param[1]}")

    # ────────── /trigger 콜백 ─────────────────────────────
    def cb_trigger(self, msg: Int32):
        trig = msg.data / 1000.0
        rising  = (self.prev_trigger_val != 1.0) and (trig == 1.0)
        falling = (self.prev_trigger_val == 1.0) and (trig != 1.0)
        self.prev_trigger_val = trig

        if rising:                                    # 세트 시작
            self.collecting = True
            self.latest_w = None
            self.get_logger().info(
                f"--- Cycle {self.cycles_seen + 1} START (trigger↑) ---")

        elif falling and self.collecting:             # 세트 종료
            self.collecting = False
            self.get_logger().info(
                f"--- Cycle {self.cycles_seen + 1} END (trigger↓) --- "
                "(대기: /w_user)")
            self.finish_cycle()

    # ────────── /w_user 콜백 ────────────────────────────
    def cb_w(self, msg: Int32):
        if not self.collecting and self.latest_w is None:
            self.latest_w = float(msg.data) / 1000.0  # mJ → J
            self.get_logger().info(f"    ↳ /w_user: {self.latest_w:.3f} J")

    # ────────── 세트 완료 처리 ──────────────────────────
    def finish_cycle(self):
        self.cycles_seen += 1

        if self.cycles_seen <= self.ignore_cycles:
            self.get_logger().info(
                f"Cycle {self.cycles_seen} 무시 (ignore_cycles)")
            return
        if self.latest_w is None:
            self.get_logger().warn("W_user 미수신 → 건너뜀")
            return

        cost = -self.latest_w
        self.X.append(self.current_param)
        self.Y.append(cost)
        self.repeats_done += 1
        self.get_logger().info(
            f"[cycle {self.cycles_seen}] Δ={self.current_param[0]}, "
            f"K={self.current_param[1]}, W={self.latest_w:.3f} J "
            f"(repeat {self.repeats_done}/{self.repeats_per_param})")

        if self.repeats_done >= self.repeats_per_param:
            self.repeats_done = 0
            self.get_logger().info("  ↳ repeat 완료 → BO 업데이트")
            self.run_bo_step()

    # ────────── BO 업데이트 + 시각화 ────────────────────
    def run_bo_step(self):
        X_arr, Y_arr = np.array(self.X), np.array(self.Y)
        self.gp.fit(X_arr, Y_arr)

        # UCB 그리드 탐색
        d_lin = np.linspace(*self.box[0], 40)
        k_lin = np.linspace(*self.box[1], 40)
        grid  = np.array([[d, k] for d in d_lin for k in k_lin])

        mu, sig = self.gp.predict(grid, return_std=True)
        ucb     = mu - self.beta * sig
        best    = grid[np.argmin(ucb)]
        self.current_param = best.tolist()
        self.publish_param()
        self.get_logger().info(
            f" → new param Δ={best[0]:.0f}, K={best[1]:.1f}")

        if self.enable_plot:
            try:
                self._plot_visuals(grid)
            except Exception as e:
                self.get_logger().warn(f"시각화 실패: {e}")

    # ────────── Expected Improvement ───────────────────
    def _expected_improv(self, cand):
        mu, sig = self.gp.predict(cand, return_std=True)
        sig = np.maximum(sig, 1e-9)
        imp = np.min(self.Y) - mu
        Z   = imp / sig
        ei  = imp * norm.cdf(Z) + sig * norm.pdf(Z)
        ei[sig == 0] = 0
        return ei

    # ────────── 시각화 루틴 ────────────────────────────
    def _plot_visuals(self, grid):
        d_lin = np.linspace(*self.box[0], 40)
        k_lin = np.linspace(*self.box[1], 40)
        D, K_mesh = np.meshgrid(d_lin, k_lin)
        EI = self._expected_improv(grid).reshape(D.shape)

        d_mm, k_val = self.current_param
        W_sim, x_ref, x_act, F_user, mask = self._simulate_demo(d_mm, k_val)

        fig = plt.figure(figsize=(11, 4.5))

        # (1) EI contour
        ax1 = fig.add_subplot(1, 3, 1)
        cs  = ax1.contourf(D, K_mesh, EI, 20, cmap=cm.viridis)
        ax1.scatter(*np.array(self.X).T, c='red')
        ax1.set(xlabel='Δ [mm]', ylabel='K [N/m]', title='Expected Improvement')
        fig.colorbar(cs, ax=ax1)

        # (2) Trajectory
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(TIME, x_ref, label='ref')
        ax2.plot(TIME, x_act, label='act')
        ax2.set(xlabel='time [s]', ylabel='pos',
                title=f'Trajectory  (cycle {self.cycles_seen})')
        ax2.legend()

        # (3) Force & K timeline
        ax3  = fig.add_subplot(1, 3, 3)
        ax3.plot(TIME, F_user, label='F_user')
        ax3.set(xlabel='time [s]', ylabel='F_user [N]',
                title='User force & K timeline')

        ax3b = ax3.twinx()
        K_vis = np.where(mask, k_val, 0.0)
        ax3b.plot(TIME, K_vis, 'r', lw=2, label='K_resist')
        ax3b.set_ylabel('K_resist [N/m]')
        ax3b.set_ylim(0, self.box[1, 1])

        ax3.legend(loc='upper left'); ax3b.legend(loc='upper right')
        plt.tight_layout()

        # head-less 환경이면 PNG 저장, 아니면 창 표시
        if not plt.get_backend().lower().startswith("qt"):
            fname = f"/tmp/bo_cycle_{self.cycles_seen:03d}.png"
            fig.savefig(fname, dpi=120)
            self.get_logger().info(f"그래프 저장: {fname}")
            plt.close(fig)
        else:
            plt.show()

    # ────────── 데모용 시뮬레이션 ───────────────────────
    def _simulate_demo(self, d_mm: float, k_val: float):
        Δ = d_mm / 1000.0
        x_ref = _min_jerk(TIME / T)
        v_ref = np.gradient(x_ref, DT)

        x_act = np.zeros_like(TIME)
        v_act = np.zeros_like(TIME)
        F_user = (x_ref > 0.5) * F_STEP
        F_res  = np.zeros_like(TIME)
        mask   = np.zeros_like(TIME, dtype=bool)

        resist_on = False
        for k in range(1, len(TIME)):
            lead = x_act[k-1] - x_ref[k-1]
            if not resist_on and lead > Δ:
                resist_on = True
            if resist_on:
                rel_vel  = v_act[k-1] - v_ref[k-1]
                F_res[k] = k_val * (lead - Δ) + B_DAMP * rel_vel
                mask[k]  = True
            a        = (F_user[k] - F_res[k]) / MASS
            v_act[k] = v_act[k-1] + a * DT
            x_act[k] = x_act[k-1] + v_act[k] * DT

        W_user = np.trapezoid(F_user * v_act, dx=DT)
        return W_user, x_ref, x_act, F_user, mask

    # ────────── 파라미터 퍼블리시 ──────────────────────
    def publish_param(self):
        d_msg, k_msg = Int32(), Int32()
        d_msg.data = int(round(self.current_param[0]))
        k_msg.data = int(round(self.current_param[1]))
        self.pub_delta.publish(d_msg)
        self.pub_k.publish(k_msg)


# ────────── main ───────────────────────────────────────
def main():
    rclpy.init()
    node = BOResistNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
