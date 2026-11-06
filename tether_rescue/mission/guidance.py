"""구조 임무 두뇌: 두 단계 유도 + 견인.

단계
  WAIT_GOAL  목표 좌표를 기다림 (지도 클릭 / rviz2 "2D Goal Pose" / 자동 신고)
  TRANSIT    멀리서: GPS로 좌표까지 이동 (정확도 수 m)
  SEARCH     좌표에 왔는데 아직 못 찾음 → 사각 나선으로 돌며 탐색
  APPROACH   가까이서: 열화상·소나로 직접 감지한 위치로 마무리 접근
  TOW        줄 연결 완료 → 기지로 끌고 옴
  DONE       끝

로봇 두뇌가 쓰는 정보는 'GPS 위치(오차 있음), 나침반 방향, 속도, 센서 탐지'뿐입니다.
실제 사람 위치는 모릅니다. (줄 연결 성공 여부와 기지 도착은 시뮬레이터가 판정)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..config import MissionConfig
from .sensors import Detection

WAIT_GOAL, TRANSIT, SEARCH, APPROACH, TOW, DONE = "WAIT_GOAL", "TRANSIT", "SEARCH", "APPROACH", "TOW", "DONE"
PHASE_KO = {WAIT_GOAL: "좌표 기다림", TRANSIT: "좌표로 이동", SEARCH: "주변 탐색", APPROACH: "직접 감지 접근",
            TOW: "견인", DONE: "끝"}


def heading_to(src, dst) -> float:
    d = np.asarray(dst, float) - np.asarray(src, float)
    return math.atan2(d[1], d[0])


def square_spiral(center, leg: float, n_legs: int = 16) -> list[np.ndarray]:
    """사각 나선 탐색 경로: 동 1칸, 북 1칸, 서 2칸, 남 2칸, 동 3칸, ... (칸 = leg)"""
    pts = []
    p = np.asarray(center, float).copy()
    dirs = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([-1.0, 0.0]), np.array([0.0, -1.0])]
    for i in range(n_legs):
        p = p + dirs[i % 4] * leg * (i // 2 + 1)
        pts.append(p.copy())
    return pts


@dataclass
class _Candidate:
    pos: np.ndarray
    t: float
    hits: int = 1


@dataclass
class TargetTracker:
    """센서 탐지를 모아 '사람이 여기 있다'는 추정 위치를 만듭니다 (GPS 좌표계 기준).

    - 소나 탐지 3번이 3m 안에서 겹치면 '확인' (가짜 탐지는 매번 다른 곳에 생기므로 걸러짐)
    - 열화상 방향과 소나 후보 방향이 6° 안으로 맞으면 바로 '확인' (두 센서가 동의)
    - 신고 좌표에서 너무 먼 탐지는 후보로 받지 않음 (gate)
    - 확인 후에도 후보를 계속 모으고, 지금 추정이 한동안 갱신 안 되는데 다른 후보가 확인 수준이 되면 갈아탐
    - 오래 감지가 없으면 '놓침'
    """
    lost_time: float = 8.0
    confirm_hits: int = 3
    gate_center: np.ndarray | None = None
    gate_radius: float = 1e9
    confirmed: bool = False
    est: np.ndarray | None = None
    last_t: float = -1e9
    first_confirm_t: float | None = None
    candidates: list = field(default_factory=list)
    thermal_bearing_world: float | None = None
    thermal_t: float = -1e9
    false_alarms: int = 0         # 분석용: 사람 아닌 탐지를 받은 횟수

    def reset(self):
        self.__init__(lost_time=self.lost_time, confirm_hits=self.confirm_hits)

    def drop(self):
        """지금 추정이 틀린 것 같을 때 버림 (다시 탐색)."""
        self.confirmed = False
        self.candidates.clear()

    def _in_gate(self, pos) -> bool:
        return self.gate_center is None or np.linalg.norm(pos - self.gate_center) < self.gate_radius

    def update(self, t: float, dets: list[Detection], robot_gps, robot_yaw: float):
        robot_gps = np.asarray(robot_gps, float)
        for d in dets:
            if not d.real:
                self.false_alarms += 1
            bw = robot_yaw + d.bearing
            if d.sensor == "thermal":
                self.thermal_bearing_world, self.thermal_t = bw, t
                if self.confirmed and self.est is not None:
                    r = float(np.linalg.norm(self.est - robot_gps))
                    proj = robot_gps + r * np.array([math.cos(bw), math.sin(bw)])
                    self.est = 0.6 * self.est + 0.4 * proj
                    self.last_t = t
                else:
                    for c in self.candidates:          # 소나 후보와 방향이 맞으면 확인
                        if t - c.t < 5.0 and abs(_wrap(heading_to(robot_gps, c.pos) - bw)) < math.radians(6):
                            self._confirm(c.pos, t)
                            break
            else:
                pos = robot_gps + d.range * np.array([math.cos(bw), math.sin(bw)])
                if self.confirmed and self.est is not None and np.linalg.norm(pos - self.est) < 4.0:
                    self.est = 0.6 * self.est + 0.4 * pos
                    self.last_t = t
                    continue
                if self._in_gate(pos):
                    self._add_candidate(pos, t)
        if self.confirmed and t - self.last_t > self.lost_time:
            self.drop()

    def _add_candidate(self, pos, t):
        self.candidates = [c for c in self.candidates if t - c.t < 5.0]
        for c in self.candidates:
            if np.linalg.norm(c.pos - pos) < 3.0:
                c.pos = 0.5 * (c.pos + pos)
                c.t, c.hits = t, c.hits + 1
                # 아직 확인 전이거나, 지금 추정이 3초 넘게 갱신이 없으면 이 후보로 갈아탐
                if c.hits >= self.confirm_hits and (not self.confirmed or t - self.last_t > 3.0):
                    self._confirm(c.pos, t)
                return
        self.candidates.append(_Candidate(pos.copy(), t))

    def _confirm(self, pos, t):
        self.confirmed, self.est, self.last_t = True, np.asarray(pos, float).copy(), t
        if self.first_confirm_t is None:
            self.first_confirm_t = t
        self.candidates.clear()

    def recent_thermal(self, t: float, within: float = 1.0) -> float | None:
        return self.thermal_bearing_world if t - self.thermal_t < within else None


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Command:
    speed_ref: float
    heading_ref: float
    want_attach: bool = False
    hold: bool = False           # True면 제자리 (추진력 0)


class MissionController:
    def __init__(self, cfg: MissionConfig, tow_speed_fn):
        """tow_speed_fn(t_since_attach) → 견인 목표 속도 (PID 설정의 목표 속도·가속 시간 사용)."""
        self.cfg = cfg
        self.tow_speed_fn = tow_speed_fn
        self.tracker = TargetTracker(lost_time=cfg.track_lost_time)
        self.reset()

    def reset(self):
        self.phase = WAIT_GOAL
        self.goal: np.ndarray | None = None
        self.goal_t: float | None = None
        self.search_path: list[np.ndarray] = []
        self.search_idx = 0
        self.search_start: float | None = None
        self.arrive_t: float | None = None
        self.attach_t: float | None = None
        self.fail_reason: str | None = None
        self.near_since: float | None = None
        self.tracker.reset()
        self.tracker.lost_time = self.cfg.track_lost_time

    # ---------------------------------------------------------------- 입력
    def set_goal(self, xy, t: float) -> bool:
        """새 목표 좌표 (지도 클릭 / rviz2 / 자동 신고). 견인 중이거나 끝났으면 무시."""
        if self.phase in (TOW, DONE):
            return False
        self.goal = np.asarray(xy, float)[:2].copy()
        self.goal_t = t
        # 신고 좌표에서 이 거리 밖의 소나 탐지는 후보로 안 받음 (가짜 탐지 거르기)
        self.tracker.gate_center = self.goal
        self.tracker.gate_radius = max(12.0, 3.0 * self.cfg.report_error + self.cfg.arrive_radius)
        self.search_path, self.search_idx, self.search_start = [], 0, None
        if self.phase != APPROACH:
            self.phase = TRANSIT
        return True

    def on_attached(self, t: float):
        self.phase, self.attach_t = TOW, t

    def finish(self):
        self.phase = DONE

    # ---------------------------------------------------------------- 결정
    def decide(self, t: float, gps_xy, yaw: float, speed: float, dets: list[Detection]) -> Command:
        c = self.cfg
        gps_xy = np.asarray(gps_xy, float)
        self.tracker.update(t, dets, gps_xy, yaw)
        base = np.asarray(c.base, float)

        if self.phase in (TRANSIT, SEARCH) and self.tracker.confirmed:
            self.phase = APPROACH                       # 직접 감지가 되면 GPS보다 감지를 믿음
        if self.phase == APPROACH and not self.tracker.confirmed:
            self.phase = SEARCH                         # 놓쳤으면 마지막 추정 위치 주변을 다시 탐색
            center = self.tracker.est if self.tracker.est is not None else self.goal
            self.search_path, self.search_idx = square_spiral(center, c.search_leg), 0
            self.search_start = t

        if self.phase == WAIT_GOAL or self.phase == DONE:
            return Command(0.0, yaw, hold=True)

        if self.phase == TRANSIT:
            dist = float(np.linalg.norm(self.goal - gps_xy))
            if dist < c.arrive_radius:
                self.phase, self.arrive_t = SEARCH, t
                self.search_path, self.search_idx = square_spiral(self.goal, c.search_leg), 0
                self.search_start = t
            else:
                return Command(min(c.transit_speed, 0.4 * dist + 0.4), heading_to(gps_xy, self.goal))

        if self.phase == SEARCH:
            if self.search_start is None:
                self.search_start = t
            if t - self.search_start > c.search_timeout:
                self.phase, self.fail_reason = DONE, "not_found"
                return Command(0.0, yaw, hold=True)
            tb = self.tracker.recent_thermal(t)
            if tb is not None:                          # 열화상에 뭔가 보이면 그쪽으로 고개를 돌림
                return Command(c.search_speed * 0.6, tb)
            if not self.search_path:
                self.search_path = square_spiral(self.goal if self.goal is not None else gps_xy, c.search_leg)
            wp = self.search_path[self.search_idx]
            if np.linalg.norm(wp - gps_xy) < 2.0:
                self.search_idx = (self.search_idx + 1) % len(self.search_path)
                wp = self.search_path[self.search_idx]
            return Command(c.search_speed, heading_to(gps_xy, wp))

        if self.phase == APPROACH:
            est = self.tracker.est
            r = float(np.linalg.norm(est - gps_xy))
            hdg = heading_to(gps_xy, est)
            tb = self.tracker.recent_thermal(t, 0.5)
            if tb is not None and r < 8.0:              # 가까우면 방향은 열화상이 더 정확
                hdg = tb
            want = r < c.capture_distance + 1.0
            v = float(np.clip(0.25 * (r - 1.0), 0.3, c.approach_speed_max))
            if r < 1.5:
                v = 0.0                                  # 너무 가까우면 멈춰서 부딪힘 방지
            # 추정 위치에 왔는데 한참 연결이 안 되면 추정이 틀린 것 → 버리고 다시 탐색
            if want:
                self.near_since = t if self.near_since is None else self.near_since
                if t - self.near_since > 8.0:
                    self.tracker.drop()
                    self.near_since = None
            else:
                self.near_since = None
            return Command(v, hdg, want_attach=want)

        if self.phase == TOW:
            d_home = float(np.linalg.norm(base - gps_xy))
            v = min(self.tow_speed_fn(t - self.attach_t), 0.3 * d_home + 0.3)
            return Command(v, heading_to(gps_xy, base))

        return Command(0.0, yaw, hold=True)
