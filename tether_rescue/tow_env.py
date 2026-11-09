"""AirwayTow 시뮬레이션: 로봇이 줄로 사람을 끌고 오는 가짜 바다.

박자
- 물리: physics_hz (기본 250Hz) — 부력·저항·줄·추진력을 매 물리 스텝마다 계산
- 조종·기록: control_hz (기본 50Hz) — step() 한 번 = 물리 스텝 여러 번

사용 예
    sim = TowSim(load_config())
    obs = sim.reset()
    while not sim.done:
        obs = sim.step(throttle=0.5, rudder=0.0)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .config import TowConfig
from .handoff import TubeHandoff
from .ocean.hydro import HydroSpec, HydroSystem, box_blocks, sphere_blocks
from .ocean.waves import WaveField
from .scene import (WATER_CENTER_X, WATER_ELEV, WATER_HALF_X, WATER_HALF_Y, WATER_NCOL,
                    WATER_NROW, build_mjcf)
from .tow.tether import Tether

CAPSIZE_ANGLE = math.radians(60.0)

# 기록 한 줄에 들어가는 항목 (순서 고정). 앞 6개는 ⑤ TetherSense의 입력 후보.
LOG_FIELDS = [
    "t", "tension", "acc_x", "acc_y", "acc_z", "throttle", "rudder",
    "gyro_x", "gyro_y", "gyro_z",
    "robot_speed", "human_speed", "target_speed",
    "roll", "pitch", "yaw",
    "robot_x", "robot_y", "human_x", "human_y", "progress",
    "mouth_z", "water_z", "bow_rise", "mouth_clearance", "airway_ok",
    "tether_length", "tether_taut", "tether_broken",
]


@dataclass
class Outcome:
    done: bool = False
    reason: str = ""          # success / tether_broken / capsized / timeout (구조 임무: + not_found / victim_climbed)
    success: bool = False


@dataclass
class StepLog:
    rows: list[list[float]] = field(default_factory=list)

    def append(self, row: dict):
        self.rows.append([float(row[k]) for k in LOG_FIELDS])

    def array(self) -> np.ndarray:
        return np.asarray(self.rows, dtype=float).reshape(-1, len(LOG_FIELDS))


def quat_to_euler(q) -> tuple[float, float, float]:
    """MuJoCo 쿼터니언(w,x,y,z) → (roll, pitch, yaw) [rad]."""
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class TowSim:
    def __init__(self, cfg: TowConfig):
        self.cfg = cfg
        if cfg.sim.physics_hz % cfg.sim.control_hz != 0:
            raise ValueError("physics_hz는 control_hz의 정수배여야 합니다")
        self.n_sub = cfg.sim.physics_hz // cfg.sim.control_hz
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(cfg))
        self.data = mujoco.MjData(self.model)
        m = self.model
        oc = cfg.ocean

        def bid(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)

        def sid(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)

        self.robot_id, self.torso_id, self.head_id = bid("robot"), bid("torso"), bid("head")
        self.tow_site, self.harness_site = sid("tow_point"), sid("harness")
        self.mouth_site, self.thruster_site = sid("mouth"), sid("thruster")
        self.hfield_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_HFIELD, "water")
        # MuJoCo 구동기: 추진 모터, 조향 서보
        self.thrust_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust")
        self.steer_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "steer")
        self._thrust_actadr = m.actuator_actadr[self.thrust_act]
        self.steer_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "steer")]
        self._acc_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_acc")]
        self._gyro_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")]

        r, h = cfg.robot, cfg.human
        specs = [
            HydroSpec("robot", *box_blocks([r.length, r.width, r.height], r.blocks),
                      r.drag_lin, r.drag_quad, r.ang_drag),
            HydroSpec("torso", *box_blocks(h.torso_size, h.blocks), h.drag_lin, h.drag_quad, h.ang_drag),
            HydroSpec("head", *sphere_blocks(h.head_radius), [0.0] * 3, [h.head_drag_quad] * 3, [0.5] * 3),
        ]
        self.hydro = HydroSystem(m, specs, oc.rho, oc.g)

        self.tether = Tether(cfg.tether)
        self.waves = WaveField(oc, cfg.sim.seed)
        self._v6 = np.zeros(6)
        self.handoff = TubeHandoff(self)
        self.reset()

    # ------------------------------------------------------------------ 준비
    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.cfg.sim.seed = seed
        self.waves = WaveField(self.cfg.ocean, self.cfg.sim.seed)
        self._wave_target = None
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.tether.reset()
        self._before_settle()
        self.handoff.reset(worn=self.tether.attached)   # 끌고 오기만: 처음부터 튜브 착용
        mujoco.mj_forward(self.model, self.data)
        self.throttle = 0.0
        self.thrust_level = 0.0   # 실제 엔진 세기 (throttle을 늦게 따라감)
        self.rudder = 0.0
        self.outcome = Outcome()
        self.log = StepLog()
        self.last_tension = 0.0

        # 물 위에 자리 잡기 (추진력 0, 기록 안 함)
        n_settle = int(round(self.cfg.sim.settle_time * self.cfg.sim.physics_hz))
        for _ in range(n_settle):
            self._physics_step()
        self.t0 = self.data.time
        self.start_x = float(self.data.xipos[self.torso_id][0])
        self.update_water_surface()
        self.update_visuals()
        self._last_obs = self._observe()
        return self._last_obs

    def update_visuals(self):
        """구명튜브·팔 모양 표시 (갑판 위 / 건네는 중 / 사람 가슴). 물리에는 영향 없음."""
        self.handoff.apply_visuals()

    def _before_settle(self):
        """자리 잡기 전에 몸 위치를 바꾸고 싶을 때 하위 클래스에서 덮어씁니다 (구조 임무 등)."""

    @property
    def time(self) -> float:
        """에피소드 시간 [s] (자리 잡기 시간 제외)"""
        return self.data.time - getattr(self, "t0", 0.0)

    @property
    def done(self) -> bool:
        return self.outcome.done

    def set_wave_height(self, hs: float, ramp_time: float = 1.0):
        """실행 중에 파도 높이를 바꿉니다. 갑자기 바꾸면 충격이 생기므로 ramp_time 초에 걸쳐 바꿉니다.
        물결 모양(주기·방향·위상)은 그대로라 화면이 튀지 않습니다."""
        hs = max(float(hs), 0.0)
        self.cfg.ocean.wave_height = hs
        self._wave_target = hs
        start = self.waves.wave_height
        self._wave_rate = abs(hs - start) / ramp_time if ramp_time > 0 else float("inf")

    def _update_wave_ramp(self, dt: float):
        target = getattr(self, "_wave_target", None)
        if target is None:
            return
        cur = self.waves.wave_height
        step = self._wave_rate * dt
        if abs(target - cur) <= step:
            self.waves.set_wave_height(target)
            self._wave_target = None
        else:
            self.waves.set_wave_height(cur + step * np.sign(target - cur))

    # ------------------------------------------------------------------ 물리
    def _site_vel(self, site_id: int) -> np.ndarray:
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, site_id, self._v6, 0)
        return self._v6[3:].copy()

    def _add_point_force(self, wrench: np.ndarray, body: int, force: np.ndarray, point: np.ndarray):
        wrench[body, :3] += force
        r = point - self.data.xipos[body]
        wrench[body, 3] += r[1] * force[2] - r[2] * force[1]
        wrench[body, 4] += r[2] * force[0] - r[0] * force[2]
        wrench[body, 5] += r[0] * force[1] - r[1] * force[0]

    def _extra_forces(self, wrench: np.ndarray):
        """하위 클래스가 힘을 더 넣고 싶을 때 덮어씀 (구조 임무: 사람이 배에 매달리는 힘)."""

    def _physics_step(self):
        m, d, cfg = self.model, self.data, self.cfg
        t = d.time
        wrench = np.zeros((m.nbody, 6))
        self._update_wave_ramp(m.opt.timestep)

        # 부력 + 물 저항
        res = self.hydro.compute(m, d, self.waves, t)
        wrench[self.hydro.body_ids, :3] += res.force
        wrench[self.hydro.body_ids, 3:] += res.torque

        # 추진·조향: MuJoCo 구동기에 명령만 넣음 (힘 계산은 MuJoCo가 함)
        # 추진 모터: 목표 추진력 [N]. 엔진 반응 지연은 구동기의 filterexact가 처리 (패널에서 바꿔도 반영)
        m.actuator_dynprm[self.thrust_act, 0] = max(cfg.robot.thrust_tau, 1e-6)
        d.ctrl[self.thrust_act] = self.throttle * cfg.robot.max_thrust
        # 조향 서보: rudder > 0 = 왼쪽(반시계)으로 돌기. 선외기를 시계 방향(음수 각)으로 돌려야
        # 선미가 오른쪽으로 밀리고 뱃머리가 왼쪽으로 돕니다.
        d.ctrl[self.steer_act] = -self.rudder * cfg.robot.max_rudder

        # 줄
        p_r, p_h = d.site_xpos[self.tow_site].copy(), d.site_xpos[self.harness_site].copy()
        ts = self.tether.compute(p_r, p_h, self._site_vel(self.tow_site), self._site_vel(self.harness_site),
                                 m.opt.timestep)
        if ts.tension > 0:
            f = ts.tension * ts.direction
            self._add_point_force(wrench, self.robot_id, f, p_r)
            self._add_point_force(wrench, self.torso_id, -f, p_h)
        self.last_tension = ts.tension

        self._extra_forces(wrench)
        d.xfrc_applied[:] = wrench
        mujoco.mj_step(m, d)
        # 실제 엔진 세기 = 모터의 현재 추진력 / 최대 추진력
        mt = cfg.robot.max_thrust
        self.thrust_level = float(d.act[self._thrust_actadr]) / mt if mt > 0 else 0.0

    # ------------------------------------------------------------------ 조종
    def step(self, throttle: float, rudder: float, target_speed: float = float("nan")) -> dict:
        """조종 한 번(1/control_hz 초). throttle: -max_reverse~1, rudder: -1~1"""
        if self.outcome.done:
            return self._last_obs
        self.throttle = float(np.clip(throttle, -self.cfg.robot.max_reverse, 1.0))
        self.rudder = float(np.clip(rudder, -1.0, 1.0))
        peak = 0.0
        for _ in range(self.n_sub):
            self._physics_step()
            peak = max(peak, self.last_tension)
        obs = self._observe()
        obs["tension_peak"] = peak
        obs["target_speed"] = target_speed
        self.update_visuals()
        self.log.append(obs)
        self._check_done(obs)
        self._last_obs = obs
        return obs

    # ------------------------------------------------------------------ 관측
    def _observe(self) -> dict:
        m, d, cfg = self.model, self.data, self.cfg
        roll, pitch, yaw = quat_to_euler(d.xquat[self.robot_id])
        # 주의: mj_objectVelocity의 local 옵션은 관성 주축 기준이라 축 순서가 바뀔 수 있음.
        # 월드 속도를 구해 몸체 x축(앞쪽)에 직접 투영합니다.
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.robot_id, self._v6, 0)
        forward = d.xmat[self.robot_id].reshape(3, 3)[:, 0]
        robot_speed = float(self._v6[3:] @ forward)   # 앞쪽 속도 [m/s]
        yaw_rate = float(d.sensordata[self._gyro_adr + 2])

        # 사람 속도: 물에 대한 수평 상대 속도 (선수파 계산용)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.torso_id, self._v6, 0)
        torso_pos = d.xipos[self.torso_id]
        v_rel = self._v6[3:] - self.waves.velocity(torso_pos, d.time)
        human_speed = float(np.hypot(v_rel[0], v_rel[1]))

        mouth = d.site_xpos[self.mouth_site]
        water_z = float(self.waves.height(mouth[0], mouth[1], d.time))
        bow_rise = cfg.human.bow_wave_coef * human_speed**2 / (2 * cfg.ocean.g)
        clearance = float(mouth[2] - (water_z + bow_rise))
        ts = self.tether.state
        acc = d.sensordata[self._acc_adr:self._acc_adr + 3]
        gyro = d.sensordata[self._gyro_adr:self._gyro_adr + 3]

        return {
            "t": self.time,
            "tension": self.last_tension,
            "acc_x": float(acc[0]), "acc_y": float(acc[1]), "acc_z": float(acc[2]),
            "throttle": self.throttle, "rudder": self.rudder,
            "thrust_level": self.thrust_level,   # 실제 엔진 세기 (최대 추진력 대비 비율)
            "steer_angle": float(d.qpos[self.steer_qadr]),   # 실제 선외기 조향각 [rad] (+ = 반시계)
            "gyro_x": float(gyro[0]), "gyro_y": float(gyro[1]), "gyro_z": float(gyro[2]),
            "yaw_rate": yaw_rate,
            "robot_speed": robot_speed, "human_speed": human_speed, "target_speed": float("nan"),
            "roll": roll, "pitch": pitch, "yaw": yaw,
            "robot_x": float(d.xipos[self.robot_id][0]), "robot_y": float(d.xipos[self.robot_id][1]),
            "human_x": float(torso_pos[0]), "human_y": float(torso_pos[1]),
            "progress": float(torso_pos[0] - getattr(self, "start_x", torso_pos[0])),
            "mouth_z": float(mouth[2]), "water_z": water_z, "bow_rise": bow_rise,
            "mouth_clearance": clearance, "airway_ok": float(clearance > 0.0),
            "tether_length": ts.length, "tether_taut": float(ts.taut), "tether_broken": float(ts.broken),
            "tilt": self.robot_tilt(),
        }

    def robot_tilt(self) -> float:
        """로봇이 수직에서 기운 각 [rad]"""
        zz = self.data.xmat[self.robot_id][8]
        return math.acos(max(-1.0, min(1.0, zz)))

    def _check_done(self, obs: dict):
        if obs["progress"] >= self.cfg.sim.goal_distance:
            self.outcome = Outcome(True, "success", True)
        elif obs["tether_broken"]:
            self.outcome = Outcome(True, "tether_broken")
        elif obs["tilt"] > CAPSIZE_ANGLE:
            self.outcome = Outcome(True, "capsized")
        elif obs["t"] >= self.cfg.sim.max_time:
            self.outcome = Outcome(True, "timeout")

    # ------------------------------------------------------------------ 화면용
    def update_water_surface(self):
        """물 표면 그림(hfield)을 현재 파도 모양으로 갱신. 화면에 보여 줄 때만 필요."""
        xs = WATER_CENTER_X + np.linspace(-WATER_HALF_X, WATER_HALF_X, WATER_NCOL)
        ys = np.linspace(-WATER_HALF_Y, WATER_HALF_Y, WATER_NROW)
        gx, gy = np.meshgrid(xs, ys)          # (nrow, ncol)
        eta = self.waves.height(gx, gy, self.data.time)
        norm = np.clip(eta / WATER_ELEV + 0.5, 0.0, 1.0)
        adr = self.model.hfield_adr[self.hfield_id]
        self.model.hfield_data[adr:adr + norm.size] = norm.ravel()
