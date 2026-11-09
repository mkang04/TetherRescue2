"""전체 구조 임무 시뮬레이션.

기지(0, 0)에서 출발 → 바다 어딘가의 익수자에게 좌표로 이동 → 직접 감지로 접근
→ 손 닿는 곳에 튜브 건네기 → 사람이 잡아서 끼우면 줄 연결 → 기지로 견인.
물리(파도·부력·저항·줄)는 TowSim을 그대로 씁니다. 처음에는 줄이 연결돼 있지 않습니다.

사용 예
    sim = MissionSim(load_config())
    sim.reset(seed=0)
    sim.set_goal(sim.sighting)          # 목격 지점을 목표로 (지도 클릭 대신)
    while not sim.done:
        sim.step_auto()
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from ..config import TowConfig
from ..tow.pid import TowPIDController
from ..tow_env import CAPSIZE_ANGLE, LOG_FIELDS, Outcome, TowSim
from .boarding import CLINGING, LUNGING, BoardingRisk
from .gps import GpsModel
from .guidance import APPROACH, DONE, HANDOFF, MissionController
from .sensors import Detection, Sonar, ThermalCamera


@dataclass
class MissionMetrics:
    reason: str
    success: bool
    total_time: float
    detect_time: float | None      # 처음 '확인' 감지까지 [s] (목표 받은 뒤부터)
    attach_time: float | None      # 줄 연결까지 [s] (목표 받은 뒤부터)
    goal_error: float | None       # 목표 좌표와 실제 사람 위치의 거리 (목표 받을 때) [m]
    gps_error_at_arrival: float | None
    tow_airway_ratio: float        # 견인 중 입이 물 밖에 있던 비율
    max_tension: float
    false_alarms: int


class MissionSim(TowSim):
    """mode="mission": 전체 구조 임무 / mode="tow": 기존 끌고 오기만 (같은 3D 모델을 공유, reset 때 전환)."""

    def __init__(self, cfg: TowConfig, mode: str = "mission"):
        self.mode = mode
        self.ctrl = TowPIDController(cfg.pid, cfg.robot.max_reverse)
        self.brain = MissionController(cfg.mission, self.ctrl.target_speed)
        self._goal_site_default = None
        super().__init__(cfg)

    # ---------------------------------------------------------------- 준비
    def _before_settle(self):
        """익수자를 시드에 따라 바다 어딘가에 놓고, 줄은 아직 연결 안 함."""
        mc = self.cfg.mission
        self.mrng = np.random.default_rng(10_000 + int(self.cfg.sim.seed))
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal")
        if self._goal_site_default is None:
            self._goal_site_default = (self.model.site_pos[gid].copy(), self.model.site_size[gid].copy(),
                                       self.model.site_rgba[gid].copy())
        if self.mode != "mission":                                  # 끌고 오기만: 원래 목표선 (세운 기둥)
            self.model.site_pos[gid], self.model.site_size[gid], self.model.site_rgba[gid] = self._goal_site_default
            return
        # 구조 임무: 기지를 물 위의 납작한 원판(구조 완료 반경)으로 표시 → 출발하는 배와 겹치지 않음
        self.model.site_size[gid] = [mc.home_radius, 0.02, 0.0]
        self.model.site_rgba[gid] = [0.2, 0.9, 0.3, 0.25]
        self.model.site_pos[gid][2] = 0.05
        dist = mc.victim_distance * self.mrng.uniform(0.8, 1.2)
        bearing = math.radians(self.mrng.uniform(-mc.victim_bearing, mc.victim_bearing))
        base = np.asarray(mc.base, float)
        self.victim_start = base + dist * np.array([math.cos(bearing), math.sin(bearing)])
        yaw = self.mrng.uniform(-math.pi, math.pi)
        adr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "torso_free")]
        q = self.data.qpos
        q[adr:adr + 2] = self.victim_start
        q[adr + 3:adr + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        radr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot_free")]
        q[radr:radr + 2] = base
        self.tether.reset(attached=False)
        self.model.site_pos[gid][:2] = base          # 기지 표시: '목표선' 표시를 기지 위치로 옮김

    def reset(self, seed: int | None = None) -> dict:
        self.boarding = None                       # 자리 잡기 동안은 매달리기 판정 안 함
        obs = super().reset(seed)
        mc = self.cfg.mission
        self.boarding = BoardingRisk(self)
        self.climbed = False                       # 사람이 배에 매달려 구조 실패
        self.gps = GpsModel(mc, self.mrng)
        self.thermal = ThermalCamera(self.cfg.thermal, self.mrng)
        self.sonar = Sonar(self.cfg.sonar, self.mrng)
        self.ctrl.reset()
        self.brain.reset()
        # 목격 지점 = 실제 위치 + 신고 오차. 지도에 원으로 보여 주고, 자동 신고면 이걸 목표로 보냄.
        self.sighting = self._torso_xy() + self.mrng.normal(0.0, mc.report_error, 2)
        self.recent_dets: list[Detection] = []
        self.goal_error: float | None = None
        self.goal_received_t: float | None = None
        self.gps_err_arrival: float | None = None
        self.attach_t: float | None = None
        self.grab_t: float | None = None
        self.command = None
        if mc.auto_report and self.mode == "mission":
            self.set_goal(self.sighting)
        self._last_obs = self._observe_mission(obs)
        return self._last_obs

    def _torso_xy(self) -> np.ndarray:
        return self.data.xipos[self.torso_id][:2].copy()

    def set_goal(self, xy) -> bool:
        if self.mode != "mission":
            return False
        ok = self.brain.set_goal(xy, self.time)
        if ok:
            self.goal_error = float(np.linalg.norm(np.asarray(xy, float)[:2] - self._torso_xy()))
            if self.goal_received_t is None:
                self.goal_received_t = self.time
        return ok

    # ---------------------------------------------------------------- 한 걸음
    def step_auto(self) -> dict:
        """두뇌가 결정 → 조종 → 물리 한 걸음."""
        if self.outcome.done:
            return self._last_obs
        dt = 1.0 / self.cfg.sim.control_hz
        t = self.time
        if self.mode != "mission":                   # 끌고 오기만: 기존 PID 기준선
            o = self._last_obs
            thr, rud = self.ctrl.act(t, dt, o["robot_speed"], o["yaw"], o["yaw_rate"])
            self.command = None
            return self.step(thr, rud, target_speed=self.ctrl.target_speed(t))
        robot_xy = self.data.xipos[self.robot_id][:2]
        yaw = self._last_obs["yaw"]
        self.gps.update(dt)
        gps_xy = self.gps.measure(robot_xy)

        hs = self.waves.wave_height
        head_visible = self._last_obs["mouth_clearance"] > -0.08     # 얼굴이 조금이라도 보이면
        dets = self.thermal.sense(t, robot_xy, yaw, self.data.xipos[self.head_id][:2], head_visible, hs)
        dets += self.sonar.sense(t, robot_xy, yaw, self._torso_xy(), hs)
        self.recent_dets = [d for d in self.recent_dets if t - d.t < 2.0] + dets

        prev_phase = self.brain.phase
        cmd = self.brain.decide(t, gps_xy, yaw, self._last_obs["robot_speed"], dets)
        self.command = cmd
        if prev_phase != self.brain.phase and self.brain.phase in ("SEARCH", APPROACH) \
                and self.gps_err_arrival is None:
            self.gps_err_arrival = float(np.linalg.norm(self.gps.error))

        # 튜브 건네기: 두뇌가 내밀라고 하면 뱃머리 앞에 내밂. 손이 닿는지는 실제 어깨 위치로 판정
        if cmd.offer:
            self.handoff.offer()
        elif self.brain.phase != HANDOFF:
            self.handoff.retract()                       # 건네기를 그만두면 (다시 탐색 등) 튜브 회수
        # 배로 달려들기 시작한 사람은 튜브 대신 배를 붙잡으려 함 (튜브 무시)
        clinging = self.boarding.state == CLINGING
        event = None if self.boarding.state in (LUNGING, CLINGING) else             self.handoff.update(dt, self._last_obs["robot_speed"])
        # 튜브가 없는 사람은 가까이 온 배를 붙잡으려 함 (실제 어깨 ~ 선체 거리로 판정)
        if not self.tether.attached and self.brain.phase in (APPROACH, HANDOFF) or clinging:
            if self.boarding.update(dt, self.handoff.has_tube) == "failed":
                self.climbed = True
        if event == "grabbed":
            self.grab_t = t
            self.brain.on_grabbed(t)
        elif event == "worn":                            # 가슴에 끼움 = 줄 연결
            p_r, p_h = self.data.site_xpos[self.tow_site], self.data.site_xpos[self.harness_site]
            self.tether.attach(float(np.linalg.norm(p_h - p_r)))
            self.attach_t = t
            self.brain.on_attached(t)
            self.ctrl.reset()

        if cmd.hold:
            thr, rud = 0.0, 0.0
        else:
            thr, rud = self.ctrl.track(dt, cmd.speed_ref, cmd.heading_ref, self._last_obs["robot_speed"], yaw,
                                       self._last_obs["yaw_rate"])
            if thr < 0:
                rud = -rud           # 뒤로 밀 때는 추진기 꺾는 방향이 반대로 작용
        obs = self.step(thr, rud, target_speed=cmd.speed_ref)
        return obs

    def _extra_forces(self, wrench: np.ndarray):
        if getattr(self, "boarding", None) is not None:
            self.boarding.add_forces(wrench, self._add_point_force)

    def update_visuals(self):
        super().update_visuals()
        if getattr(self, "boarding", None) is not None:
            self.boarding.apply_visuals()

    # ---------------------------------------------------------------- 관측·판정
    def _observe(self) -> dict:
        obs = super()._observe()
        return self._observe_mission(obs) if hasattr(self, "brain") and hasattr(self, "gps") else obs

    def _observe_mission(self, obs: dict) -> dict:
        base = np.asarray(self.cfg.mission.base, float)
        tr = self.brain.tracker
        obs.update({
            "phase": self.brain.phase if self.mode == "mission" else "TOW_ONLY",
            "gps_x": float(self.gps.measure(self.data.xipos[self.robot_id])[0]),
            "gps_y": float(self.gps.measure(self.data.xipos[self.robot_id])[1]),
            "gps_error": float(np.linalg.norm(self.gps.error)),
            "goal": None if self.brain.goal is None else self.brain.goal.tolist(),
            "track": None if (tr.est is None or not tr.confirmed) else tr.est.tolist(),
            "human_home_dist": float(np.linalg.norm(self._torso_xy() - base)),
            "attached": bool(self.tether.attached),
            "tube": self.handoff.state,
            "tube_reach": self.handoff.reach_dist,     # 가까운 어깨 ~ 튜브 [m] (건네는 중일 때)
            "boarding": self.boarding.state if getattr(self, "boarding", None) else "CALM",
            "hull_gap": self.boarding.gap if getattr(self, "boarding", None) else float("inf"),  # 어깨 ~ 선체 [m]
        })
        return obs

    def _check_done(self, obs: dict):
        if self.mode != "mission":
            return super()._check_done(obs)
        mc = self.cfg.mission
        if obs["tilt"] > CAPSIZE_ANGLE:
            self.outcome = Outcome(True, "capsized")
        elif getattr(self, "climbed", False):
            self.outcome = Outcome(True, "victim_climbed")
        elif self.tether.attached and obs["tether_broken"]:
            self.outcome = Outcome(True, "tether_broken")
        elif self.tether.attached and obs["human_home_dist"] < mc.home_radius:
            self.outcome = Outcome(True, "success", True)
        elif self.brain.phase == DONE and self.brain.fail_reason:
            self.outcome = Outcome(True, self.brain.fail_reason)
        elif obs["t"] >= mc.max_time:
            self.outcome = Outcome(True, "timeout")
        if self.outcome.done:
            self.brain.finish()

    def metrics(self) -> MissionMetrics:
        log = self.log.array()
        col = {k: i for i, k in enumerate(LOG_FIELDS)}
        t0 = self.goal_received_t
        tow_air, max_t = 0.0, 0.0
        if len(log):
            max_t = float(log[:, col["tension"]].max())
            if self.attach_t is not None:
                m = log[:, col["t"]] >= self.attach_t
                if m.any():
                    tow_air = float(log[m, col["airway_ok"]].mean())
        first = self.brain.tracker.first_confirm_t
        return MissionMetrics(
            reason=self.outcome.reason, success=self.outcome.success, total_time=self.time,
            detect_time=None if (first is None or t0 is None) else first - t0,
            attach_time=None if (self.attach_t is None or t0 is None) else self.attach_t - t0,
            goal_error=self.goal_error, gps_error_at_arrival=self.gps_err_arrival,
            tow_airway_ratio=tow_air, max_tension=max_t, false_alarms=self.brain.tracker.false_alarms)


def run_mission_episode(cfg: TowConfig, seed: int | None = None, goal=None) -> tuple[MissionSim, MissionMetrics]:
    """한 판 끝까지. goal=None이면 목격 지점(sighting)을 목표로 보냅니다."""
    sim = MissionSim(cfg)
    sim.reset(seed)
    if sim.brain.goal is None:
        sim.set_goal(sim.sighting if goal is None else goal)
    while not sim.done:
        sim.step_auto()
    return sim, sim.metrics()
