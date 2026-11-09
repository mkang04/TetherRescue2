"""구명튜브 건네기: 로봇이 튜브를 내밀면 → 사람이 팔을 뻗어 잡고 → 가슴(겨드랑이 아래)에 끼움 → 줄 연결.

단계 (state)
  DECK      갑판 위에 실려 있음
  OFFERED   로봇이 뱃머리 앞 물 위에 튜브를 줄에 매달아 띄움 (줄 힘 없음).
            배는 거리를 지키고, 줄을 조금씩 풀어 튜브를 사람 쪽으로 보냄
  GRABBING  튜브가 손 닿는 곳에 있어서 사람이 팔을 뻗는 중
  DONNING   사람이 잡은 튜브를 가슴으로 끌어와 끼우는 중
  WORN      끼움 완료 → 줄 연결 (견인 가능)

'손이 닿는가'는 시뮬레이터가 실제 어깨 위치로 판정합니다 (로봇 두뇌는 모름).
팔·튜브 움직임은 보이는 모양만 바꾸며 물리에는 영향이 없습니다.
"""
from __future__ import annotations

import math

import numpy as np

from .scene import (IDLE_BEND, TUBE_CHEST_X, TUBE_MAJOR, TUBE_MINOR, HumanPuppet, idle_hand,
                    set_tube_display)

DECK, OFFERED, GRABBING, DONNING, WORN = "DECK", "OFFERED", "GRABBING", "DONNING", "WORN"
STATE_KO = {DECK: "갑판 위", OFFERED: "건네는 중", GRABBING: "사람이 잡는 중", DONNING: "사람이 끼우는 중",
            WORN: "착용 (줄 연결)"}

BOW_GAP = 0.05                                     # 뱃머리 끝과 튜브 바깥 사이 [m]
Q_CHEST = np.array([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])   # 튜브 고리를 가슴 둘레로 세움
GRAB_MISS_MARGIN = 0.15                            # 잡는 도중 이만큼 더 멀어지면 놓침 [m]


# ---------------------------------------------------------------- 작은 수학 도우미
def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _qmat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _slerp(a, b, s: float):
    a, b = np.asarray(a, float), np.asarray(b, float)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:
        q = a + s * (b - a)
        return q / np.linalg.norm(q)
    th = math.acos(dot)
    return (math.sin((1 - s) * th) * a + math.sin(s * th) * b) / math.sin(th)


def _smooth(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


def nearest_ring_point(center, R, p) -> np.ndarray:
    """고리(중심 center, 자세 R, 고리 면 = 로컬 x-y)에서 점 p에 가장 가까운 점."""
    local = R.T @ (np.asarray(p, float) - center)
    local[2] = 0.0
    n = float(np.linalg.norm(local))
    local = np.array([1.0, 0.0, 0.0]) if n < 1e-6 else local / n
    return center + R @ (TUBE_MAJOR * local)


class TubeHandoff:
    def __init__(self, sim):
        self.sim = sim
        self.puppet = HumanPuppet(sim.model, sim.cfg.human)
        self.tube_mocap = sim.model.body("tube_free").mocapid[0]
        self.reset(worn=True)

    def reset(self, worn: bool):
        self.state = WORN if worn else DECK
        self.t_state = 0.0
        self.reach_t = 0.0                  # 손 닿는 곳에 튜브가 있었던 시간 [s]
        self.side = +1                      # 먼저 뻗는 팔 (+1 왼쪽, -1 오른쪽)
        self.reach_dist = float("nan")      # 가까운 어깨 ~ 튜브 거리 [m] (화면 표시용)
        self.p_grab = np.zeros(3)           # 잡은 순간 튜브 위치·자세 (몸통 좌표)
        self.q_grab = np.array([1.0, 0.0, 0.0, 0.0])
        self.line = self.sim.cfg.mission.tube_line_start   # 뱃머리 ~ 튜브 중심 거리 (풀린 줄) [m]
        self.t_offer = 0.0

    # ---------------------------------------------------------------- 명령 (MissionSim이 부름)
    def offer(self):
        if self.state == DECK:
            self.line = self.sim.cfg.mission.tube_line_start
            self.t_offer = self.sim.time
            self._go(OFFERED)

    @property
    def has_tube(self) -> bool:
        """사람이 튜브를 잡았는가 (잡는 중 포함)"""
        return self.state in (GRABBING, DONNING, WORN)

    def retract(self):
        """사람이 아직 안 잡았으면 튜브를 다시 갑판으로."""
        if self.state in (OFFERED, GRABBING):
            self._go(DECK)

    def _go(self, state: str):
        self.state, self.t_state, self.reach_t = state, self.sim.time, 0.0

    # ---------------------------------------------------------------- 자세 계산
    def _torso(self):
        d = self.sim.data
        b = self.sim.torso_id
        return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy(), d.xquat[b].copy()

    def offered_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """로봇 뱃머리 앞 line 거리, 물 위에 평평하게 떠 있는 튜브 (월드 위치, 쿼터니언)."""
        sim, d = self.sim, self.sim.data
        rb = sim.robot_id
        R = d.xmat[rb].reshape(3, 3)
        bow = 0.70 * sim.cfg.robot.length / 1.4
        fwd = R[:, 0].copy()
        fwd[2] = 0.0
        fwd /= max(np.linalg.norm(fwd), 1e-9)
        c = d.xpos[rb].copy() + fwd * (bow + max(self.line, TUBE_MAJOR + TUBE_MINOR + BOW_GAP))
        c[2] = float(sim.waves.height(c[0], c[1], d.time))
        yaw = math.atan2(fwd[1], fwd[0])
        return c, np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])

    def _donning_pose(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        """잡은 자리 → 가슴, 몸통 좌표로 보간 (사람이 튜브를 끌어오는 모습)."""
        k = _smooth(s)
        p = (1 - k) * self.p_grab + k * np.array([TUBE_CHEST_X, 0.0, 0.0])
        return p, _slerp(self.q_grab, Q_CHEST, k)

    def _reach(self, c, q):
        """두 어깨 중 튜브에 가까운 쪽과 그 거리 (월드)."""
        tp, tR, _ = self._torso()
        R = _qmat(q)
        best = (float("inf"), +1)
        for side, sh in self.puppet.shape.shoulder.items():
            s_w = tp + tR @ sh
            dist = float(np.linalg.norm(nearest_ring_point(c, R, s_w) - s_w))
            best = min(best, (dist, side))
        return best

    # ---------------------------------------------------------------- 한 걸음 (조종 박자)
    def update(self, dt: float, robot_speed: float) -> str | None:
        """반환: "grabbed"(사람이 튜브를 잡음) / "worn"(끼움 완료 → 줄 연결) / None"""
        h, mc = self.sim.cfg.human, self.sim.cfg.mission
        t = self.sim.time
        if self.state == OFFERED and t - self.t_offer >= mc.tube_payout_delay:
            self.line = min(mc.tube_line_max, self.line + mc.tube_payout_speed * dt)   # 줄을 풀어 튜브를 보냄
        if self.state in (OFFERED, GRABBING):
            c, q = self.offered_pose()
            dist, side = self._reach(c, q)
            self.reach_dist = dist
            if self.state == OFFERED:
                ok = dist < h.arm_reach and abs(robot_speed) < mc.capture_speed
                self.reach_t = self.reach_t + dt if ok else 0.0
                if self.reach_t >= h.grab_reaction:
                    self.side = side
                    self._go(GRABBING)
            elif dist > h.arm_reach + GRAB_MISS_MARGIN:
                self._go(OFFERED)                          # 튜브가 멀어져서 놓침
            elif t - self.t_state >= h.grab_time:
                tp, tR, tq = self._torso()
                self.p_grab = tR.T @ (c - tp)
                self.q_grab = _qmul(_qconj(tq), q)
                self._go(DONNING)
                return "grabbed"
        elif self.state == DONNING and t - self.t_state >= h.don_time:
            self._go(WORN)
            return "worn"
        return None

    # ---------------------------------------------------------------- 화면 (매 조종 박자)
    def apply_visuals(self):
        sim, h = self.sim, self.sim.cfg.human
        t = sim.time
        tp, tR, tq = self._torso()
        shape = self.puppet.shape
        ring = None                                    # 튜브 고리 (몸통 좌표: 중심, 자세 행렬)

        if self.state in (OFFERED, GRABBING):
            c, q = self.offered_pose()
            ring = (tR.T @ (c - tp), _qmat(_qmul(_qconj(tq), q)))
        elif self.state == DONNING:
            p, ql = self._donning_pose((t - self.t_state) / max(h.don_time, 1e-6))
            c, q = tp + tR @ p, _qmul(tq, ql)
            ring = (p, _qmat(ql))
        if ring is not None:
            sim.data.mocap_pos[self.tube_mocap] = c
            sim.data.mocap_quat[self.tube_mocap] = q
        set_tube_display(sim.model, {DECK: "deck", WORN: "worn"}.get(self.state, "free"))

        for side in (+1, -1):
            rest = idle_hand(shape, side, t)
            bend = IDLE_BEND(side)
            if self.state == WORN:                     # 팔을 튜브 위로 걸치고 옆으로 벌림
                rest = np.array([TUBE_CHEST_X - 0.03, side * 0.72, -0.03])
                bend = np.array([0.0, 0.0, 1.0])
            elif ring is not None and (self.state == DONNING or (self.state == GRABBING and side == self.side)):
                grip = nearest_ring_point(ring[0], ring[1], shape.shoulder[side])
                if self.state == GRABBING:
                    s = _smooth((t - self.t_state) / max(h.grab_time, 1e-6))
                elif side == self.side:
                    s = 1.0
                else:                                  # 다른 손은 조금 늦게 튜브를 잡음
                    s = _smooth((t - self.t_state) / max(0.3 * h.don_time, 1e-6))
                rest = (1 - s) * rest + s * grip
                bend = np.array([-0.2, side, 0.6])
            self.puppet.set_hand(side, rest, bend)
