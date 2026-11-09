"""로봇에 매달리기: 튜브가 없는 익수자는 가까이 온 배를 붙잡고 올라타려 합니다 → 구조 실패.

공황 상태의 익수자는 손에 닿는 물체에 매달리고 올라타려 하기 때문에, 구조자는 부력 도구(튜브)를
사이에 두고 거리를 둡니다. 이 모듈은 로봇이 그 거리를 지키지 못했을 때의 실패를 흉내 냅니다.

단계 (state)
  CALM      평온 (배가 위험 거리 밖)
  NOTICED   배가 위험 거리(grab_zone) 안 → climb_reaction 초 지나면 달려듦
  LUNGING   배를 향해 헤엄침 (몸통에 lunge_force). 배가 물러나 멀어지면 CALM으로
  CLINGING  손이 뱃전에 닿아 매달림: 뱃전을 아래로 누르고(체중의 climb_force_ratio), 팔 길이로 붙잡음
            → climb_fail_time 초 지나면 구조 실패 (그 전에 배가 뒤집히면 '뒤집힘' 실패)

'거리'는 사람의 가까운 어깨 ~ 선체 상자의 가장 가까운 점입니다 (시뮬레이터가 실제 위치로 판정, 로봇 두뇌는 모름).
"""
from __future__ import annotations

import mujoco
import numpy as np

CALM, NOTICED, LUNGING, CLINGING = "CALM", "NOTICED", "LUNGING", "CLINGING"
STATE_KO = {CALM: "평온", NOTICED: "배를 봄", LUNGING: "배로 달려듦", CLINGING: "배에 매달림"}

RELEASE_MARGIN = 0.3     # 달려들다가 이만큼 더 멀어지면 포기 [m]
HOLD_K = 2000.0          # 매달린 손 ~ 뱃전 붙잡는 강성 [N/m]
HOLD_C = 150.0           # 붙잡는 감쇠 [N/(m/s)]


class BoardingRisk:
    def __init__(self, sim):
        self.sim = sim
        r = sim.cfg.robot
        self.half = np.array([r.length, r.width, r.height]) / 2
        self._v6 = np.zeros(6)
        self.reset()

    def reset(self):
        self.state = CALM
        self.t_state = 0.0
        self.timer = 0.0
        self.gap = float("inf")                 # 가까운 어깨 ~ 선체 [m] (화면 표시용)
        self.side = +1
        self.grip_local = np.zeros(3)           # 붙잡은 뱃전 위치 (로봇 몸체 좌표)

    def _go(self, state: str):
        self.state, self.t_state, self.timer = state, self.sim.time, 0.0

    # ---------------------------------------------------------------- 기하
    def _robot(self):
        d, b = self.sim.data, self.sim.robot_id
        return d.xpos[b], d.xmat[b].reshape(3, 3)

    def _shoulders(self):
        d, b = self.sim.data, self.sim.torso_id
        tp, tR = d.xpos[b], d.xmat[b].reshape(3, 3)
        return {side: tp + tR @ sh for side, sh in self.sim.handoff.puppet.shape.shoulder.items()}

    def measure(self) -> tuple[float, int, np.ndarray]:
        """(가까운 어깨 ~ 선체 거리, 그 어깨 쪽, 선체 위 가장 가까운 점(몸체 좌표))"""
        p, R = self._robot()
        best = (float("inf"), +1, np.zeros(3))
        for side, s in self._shoulders().items():
            loc = R.T @ (s - p)
            near = np.clip(loc, -self.half, self.half)
            dist = float(np.linalg.norm(loc - near))
            if dist < best[0]:
                best = (dist, side, near)
        return best

    def _point_vel(self, body: int, point) -> np.ndarray:
        """몸(body) 위 한 점의 월드 속도"""
        d = self.sim.data
        mujoco.mj_objectVelocity(self.sim.model, d, mujoco.mjtObj.mjOBJ_BODY, body, self._v6, 0)
        return self._v6[3:] + np.cross(self._v6[:3], point - d.xipos[body])

    # ---------------------------------------------------------------- 한 걸음 (조종 박자)
    def update(self, dt: float, has_tube: bool) -> str | None:
        """반환: "clinging"(매달림 시작) / "failed"(매달린 채 시간 초과) / None.
        has_tube: 달려들기 전에 튜브를 잡았으면 배에 달려들지 않음 (달려들기 시작하면 튜브 무시)."""
        h = self.sim.cfg.human
        self.gap, side, near = self.measure()
        if self.state == CLINGING:
            return "failed" if self.sim.time - self.t_state >= h.climb_fail_time else None
        if has_tube and self.state in (CALM, NOTICED):    # 이미 튜브를 잡았으면 배에 달려들지 않음
            if self.state != CALM:
                self._go(CALM)
            return None
        if self.state in (CALM, NOTICED):
            if self.gap < h.grab_zone:
                if self.state == CALM:
                    self._go(NOTICED)
                self.timer += dt
                if self.timer >= h.climb_reaction:
                    self._go(LUNGING)
            elif self.state == NOTICED:
                self._go(CALM)
        elif self.state == LUNGING:
            if self.gap > h.grab_zone + RELEASE_MARGIN:
                self._go(CALM)                              # 배가 물러나서 포기
            elif self.gap < h.arm_reach:
                self.side = side
                self.grip_local = near.copy()
                self.grip_local[2] = self.half[2]           # 뱃전(갑판 높이)을 붙잡음
                self._go(CLINGING)
                return "clinging"
        return None

    # ---------------------------------------------------------------- 힘 (물리 박자)
    def add_forces(self, wrench: np.ndarray, add_point_force):
        """달려들기·매달리기 힘을 wrench에 더함. add_point_force(wrench, body, force, point)."""
        sim, h = self.sim, self.sim.cfg.human
        if self.state not in (LUNGING, CLINGING):
            return
        rb, tb = sim.robot_id, sim.torso_id
        s = self._shoulders()[self.side if self.state == CLINGING else self.measure()[1]]
        p, R = self._robot()
        if self.state == LUNGING:
            _, _, near = self.measure()
            to = p + R @ near - s
            to[2] = 0.0                                     # 수평으로 헤엄침
            n = float(np.linalg.norm(to))
            if n > 1e-6:
                add_point_force(wrench, tb, h.lunge_force * to / n, sim.data.xipos[tb])
            return
        # CLINGING: 손이 뱃전을 놓지 않음 (팔 길이보다 멀어지면 끌어당김)
        g = p + R @ self.grip_local
        to = g - s
        n = float(np.linalg.norm(to))
        if n > h.arm_reach and n > 1e-6:
            u = to / n
            v_rel = (self._point_vel(rb, g) - self._point_vel(tb, s)) @ u
            f = max(0.0, HOLD_K * (n - h.arm_reach) + HOLD_C * v_rel) * u
            add_point_force(wrench, tb, f, s)
            add_point_force(wrench, rb, -f, g)
        # 올라타려고 뱃전을 아래로 누름 (사람은 그만큼 위로 떠받쳐짐)
        w = h.climb_force_ratio * (h.torso_mass + h.head_mass) * sim.cfg.ocean.g
        down = np.array([0.0, 0.0, -w])
        add_point_force(wrench, rb, down, g)
        add_point_force(wrench, tb, -down, s)

    # ---------------------------------------------------------------- 화면
    def apply_visuals(self):
        """달려들 때 가까운 팔을 배 쪽으로, 매달리면 두 손을 뱃전에."""
        if self.state not in (LUNGING, CLINGING):
            return
        sim = self.sim
        d, tb = sim.data, sim.torso_id
        tp, tR = d.xpos[tb], d.xmat[tb].reshape(3, 3)
        p, R = self._robot()
        if self.state == LUNGING:
            _, side, near = self.measure()
            targets = {side: p + R @ near}
        else:
            g = p + R @ self.grip_local
            along = R[:, 0] * 0.18                         # 두 손을 뱃전을 따라 조금 벌림
            targets = {self.side: g, -self.side: g + along}
        for side, tgt in targets.items():
            sim.handoff.puppet.set_hand(side, tR.T @ (tgt - tp), np.array([-0.2, side, 0.6]))
