"""물리 검사: 파도·부력·줄·PID가 맞게 움직이는지 확인합니다.

실행:  python -m pytest -q
"""
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tether_rescue.config import OceanConfig, PIDConfig, TetherConfig, load_config  # noqa: E402
from tether_rescue.metrics import run_pid_episode  # noqa: E402
from tether_rescue.ocean.waves import WaveField  # noqa: E402
from tether_rescue.tow.pid import PID, TowPIDController  # noqa: E402
from tether_rescue.tow.tether import Tether  # noqa: E402
from tether_rescue.tow_env import TowSim  # noqa: E402


def flat_cfg(**sections):
    ov = {"ocean": {"wave_height": 0.0}}
    for k, v in sections.items():
        ov.setdefault(k, {}).update(v)
    return load_config(overrides=ov)


# ---------------------------------------------------------------- 파도
def test_waves_same_seed_same_result():
    a, b = WaveField(OceanConfig(), seed=3), WaveField(OceanConfig(), seed=3)
    x, y = np.linspace(-10, 10, 50), np.zeros(50)
    assert np.allclose(a.height(x, y, 7.3), b.height(x, y, 7.3))


def test_waves_different_seed_differs():
    a, b = WaveField(OceanConfig(), seed=1), WaveField(OceanConfig(), seed=2)
    x = np.linspace(-10, 10, 50)
    assert not np.allclose(a.height(x, 0 * x, 1.0), b.height(x, 0 * x, 1.0))


def test_wave_height_statistics():
    """긴 시간 동안 물 높이 표준편차 ≈ Hs/4"""
    w = WaveField(OceanConfig(wave_height=1.0, n_components=8), seed=0)
    t = np.arange(0, 2000, 0.25)
    eta = np.array([w.height(0.0, 0.0, ti) for ti in t])
    assert abs(4 * eta.std() - 1.0) < 0.1


def test_sample_matches_slow_functions():
    w = WaveField(OceanConfig(), seed=0)
    pts = np.random.default_rng(0).uniform(-5, 5, (20, 3))
    pts[:, 2] = np.minimum(pts[:, 2], 0.3)
    eta, vel = w.sample(pts, 2.5)
    assert np.allclose(eta, w.height(pts[:, 0], pts[:, 1], 2.5))
    assert np.allclose(vel, w.velocity(pts, 2.5))


# ---------------------------------------------------------------- 부력
def test_robot_floats_at_expected_draft():
    cfg = flat_cfg()
    sim = TowSim(cfg)
    r = cfg.robot
    expected_center = -r.mass / (cfg.ocean.rho * r.length * r.width) + r.height / 2
    for _ in range(100):
        sim.step(0.0, 0.0)
    z = sim.data.xipos[sim.robot_id][2]
    assert abs(z - expected_center) < 0.01


def test_human_mouth_above_water_at_rest():
    sim = TowSim(flat_cfg())
    for _ in range(100):
        obs = sim.step(0.0, 0.0)
    assert 0.0 < obs["mouth_clearance"] < 0.2
    assert obs["airway_ok"] == 1.0


def test_robot_rights_itself_after_tilt():
    """30도 기울여 놓으면 오뚝이처럼 다시 선다"""
    sim = TowSim(flat_cfg())
    q = sim.data.qpos
    half = math.radians(30) / 2
    q[3:7] = [math.cos(half), math.sin(half), 0, 0]   # roll 30도
    mujoco.mj_forward(sim.model, sim.data)
    assert sim.robot_tilt() > math.radians(25)
    for _ in range(250):
        sim.step(0.0, 0.0)
    assert sim.robot_tilt() < math.radians(3)


# ---------------------------------------------------------------- 줄
def test_tether_slack_gives_zero_force():
    t = Tether(TetherConfig(rest_length=3.0))
    s = t.compute([0, 0, 0], [-2.9, 0, 0], np.zeros(3), np.zeros(3), 0.004)
    assert s.tension == 0.0 and not s.taut


def test_tether_taut_is_spring():
    cfg = TetherConfig(rest_length=3.0, stiffness=600.0, damping=30.0)
    t = Tether(cfg)
    s = t.compute([0, 0, 0], [-3.1, 0, 0], np.zeros(3), np.zeros(3), 0.004)
    assert s.taut and s.tension == pytest.approx(60.0)


def test_tether_never_pushes():
    """줄이 줄어드는 중(빠르게 가까워짐)이어도 힘이 음수가 되면 안 됨"""
    t = Tether(TetherConfig(rest_length=3.0, stiffness=600.0, damping=30.0))
    s = t.compute([0, 0, 0], [-3.01, 0, 0], np.zeros(3), np.array([5.0, 0, 0]), 0.004)
    assert s.tension >= 0.0


def test_tether_breaks_and_stays_broken():
    cfg = TetherConfig(rest_length=3.0, stiffness=600.0, damping=0.0, break_force=294.0, break_hold=0.02)
    t = Tether(cfg)
    for _ in range(10):   # 0.04 s 동안 360 N
        s = t.compute([0, 0, 0], [-3.6, 0, 0], np.zeros(3), np.zeros(3), 0.004)
    assert s.broken and s.tension == 0.0
    s = t.compute([0, 0, 0], [-3.1, 0, 0], np.zeros(3), np.zeros(3), 0.004)
    assert s.broken and s.tension == 0.0


def test_tether_brief_spike_does_not_break():
    cfg = TetherConfig(rest_length=3.0, stiffness=600.0, damping=0.0, break_force=294.0, break_hold=0.05)
    t = Tether(cfg)
    t.compute([0, 0, 0], [-3.6, 0, 0], np.zeros(3), np.zeros(3), 0.004)   # 한 번만 튐
    s = t.compute([0, 0, 0], [-3.1, 0, 0], np.zeros(3), np.zeros(3), 0.004)
    assert not s.broken


# ---------------------------------------------------------------- PID
def test_pid_output_is_clamped_and_no_windup():
    pid = PID(1.0, 1.0, 0.0, -1.0, 1.0)
    for _ in range(1000):
        out = pid.update(10.0, 0.02)
    assert out == 1.0
    assert pid.integral < 1.0      # 포화 중엔 적분이 쌓이지 않음


def test_speed_tracking_flat_water():
    cfg = flat_cfg(pid={"target_speed": 1.0})
    sim = TowSim(cfg)
    obs = sim.reset()
    ctrl = TowPIDController(cfg.pid, cfg.robot.max_reverse)
    for _ in range(int(25 * cfg.sim.control_hz)):
        thr, rud = ctrl.act(obs["t"], 0.02, obs["robot_speed"], obs["yaw"], obs["yaw_rate"])
        obs = sim.step(thr, rud)
    assert abs(obs["robot_speed"] - 1.0) < 0.1
    assert abs(obs["yaw"]) < math.radians(5)


def test_rudder_positive_turns_left():
    sim = TowSim(flat_cfg())
    for _ in range(150):
        obs = sim.step(0.4, 1.0)
    assert obs["yaw"] > math.radians(10)


# ---------------------------------------------------------------- 전체
def test_full_episode_is_deterministic():
    cfg = load_config(overrides={"pid": {"target_speed": 1.2}, "sim": {"max_time": 15.0}})
    sim_a, m_a = run_pid_episode(cfg, seed=1)
    sim_b, m_b = run_pid_episode(cfg, seed=1)
    assert np.array_equal(sim_a.log.array(), sim_b.log.array())
    assert m_a == m_b


def test_too_fast_submerges_face_more_than_slow():
    """기획서 핵심 가설: 너무 빨리 끌면 얼굴이 더 많이 잠긴다"""
    slow = run_pid_episode(flat_cfg(pid={"target_speed": 1.0}, sim={"goal_distance": 25.0}), 0)[1]
    fast = run_pid_episode(flat_cfg(pid={"target_speed": 1.9}, sim={"goal_distance": 25.0}), 0)[1]
    assert slow.airway_ratio > fast.airway_ratio
    assert slow.rescue_time > fast.rescue_time


# ---------------------------------------------------------------- 겉모양 (배·구명튜브)
def test_visual_parts_do_not_affect_physics():
    """보이는 부품(배 모양·튜브 등)은 질량 0, 충돌 없음. 질량·관성은 물리용 상자와 같아야 함."""
    sim = TowSim(flat_cfg())
    m = sim.model
    for i in range(m.ngeom):
        name = m.geom(i).name
        if name in ("hull", "torso", "head") or name.startswith("water"):
            continue
        assert m.geom_contype[i] == 0 and m.geom_conaffinity[i] == 0, name
    r = sim.cfg.robot
    # 로봇 전체 질량 = 선체 상자 + 선외기
    assert m.body_subtreemass[sim.robot_id] == pytest.approx(r.mass)
    hull = r.mass - r.outboard_mass
    assert m.body_mass[sim.robot_id] == pytest.approx(hull)
    assert m.body_inertia[sim.robot_id][0] == pytest.approx(hull * (r.width ** 2 + r.height ** 2) / 12)


def test_thrust_motor_follows_command_with_lag():
    """추진 모터(MuJoCo 구동기)는 명령을 thrust_tau 지연으로 따라감 (1 tau 뒤 약 63%)"""
    cfg = flat_cfg()
    sim = TowSim(cfg)
    n = int(round(cfg.robot.thrust_tau * cfg.sim.control_hz))
    for _ in range(n):
        sim.step(1.0, 0.0)
    assert sim.thrust_level == pytest.approx(1 - math.exp(-1), abs=0.03)
    f = sim.data.actuator_force[sim.thrust_act]
    # 힘은 스텝 시작 시점 값, thrust_level은 스텝 끝 값이라 한 스텝(4ms)만큼 차이 남
    assert f == pytest.approx(sim.thrust_level * cfg.robot.max_thrust, rel=0.02)


def test_outboard_steers_to_commanded_angle():
    """조향 서보: rudder=1 이면 선외기가 -max_rudder 까지 돌고, 0이면 다시 가운데로"""
    cfg = flat_cfg()
    sim = TowSim(cfg)
    for _ in range(50):
        obs = sim.step(0.5, 1.0)
    assert obs["steer_angle"] == pytest.approx(-cfg.robot.max_rudder, abs=0.02)
    for _ in range(50):
        obs = sim.step(0.5, 0.0)
    assert obs["steer_angle"] == pytest.approx(0.0, abs=0.02)


def test_rescue_tube_moves_from_deck_to_water_to_victim():
    from tether_rescue.handoff import WORN
    from tether_rescue.mission.mission_sim import MissionSim

    def shown(sim, prefix):
        a = [sim.model.geom_rgba[i, 3] for i in range(sim.model.ngeom) if sim.model.geom(i).name.startswith(prefix)]
        assert a, prefix
        return all(x == 1 for x in a) if a[0] == 1 else (False if any(x == 1 for x in a) else None)

    sim = MissionSim(flat_cfg(), mode="mission")
    sim.reset(0)
    assert shown(sim, "tube_robot_") and shown(sim, "tube_human_") is None and shown(sim, "tube_free_") is None
    sim.handoff.offer()                                   # 뱃머리 앞 물 위로 내밂
    sim.update_visuals()
    assert shown(sim, "tube_free_") and shown(sim, "tube_robot_") is None and shown(sim, "tube_human_") is None
    sim.handoff._go(WORN)                                 # 사람이 끼움
    sim.update_visuals()
    assert shown(sim, "tube_human_") and shown(sim, "tube_robot_") is None and shown(sim, "tube_free_") is None


def test_victim_grabs_offered_tube_only_within_arm_reach(monkeypatch):
    """튜브가 손 닿는 거리 안에 있어야 잡고, 잡은 뒤 끼우면 'worn'으로 줄 연결 신호."""
    from tether_rescue.handoff import DONNING, OFFERED, TubeHandoff
    from tether_rescue.mission.mission_sim import MissionSim

    sim = MissionSim(flat_cfg(), mode="mission")
    sim.reset(0)
    h = sim.handoff
    dt = 1.0 / sim.cfg.sim.control_hz
    torso = sim.data.xpos[sim.torso_id].copy()
    shoulder = torso + sim.data.xmat[sim.torso_id].reshape(3, 3) @ h.puppet.shape.shoulder[+1]
    q = np.array([1.0, 0.0, 0.0, 0.0])

    far = (shoulder + np.array([3.0, 0.0, 0.0]), q)
    monkeypatch.setattr(TubeHandoff, "offered_pose", lambda self: far)   # 멀리 내밂 → 안 잡음
    h.offer()
    for _ in range(100):
        assert h.update(dt, 0.0) is None
    assert h.state == OFFERED

    near = (shoulder + np.array([0.5, 0.0, 0.0]), q)       # 고리 가장자리가 어깨에서 약 0.2 m
    monkeypatch.setattr(TubeHandoff, "offered_pose", lambda self: near)
    events = []
    for _ in range(int((sim.cfg.human.grab_reaction + sim.cfg.human.grab_time + 0.2) / dt)):
        events.append(h.update(dt, 0.0))
        sim.data.time += dt
    assert "grabbed" in events and h.state == DONNING
    for _ in range(int((sim.cfg.human.don_time + 0.2) / dt)):
        events.append(h.update(dt, 0.0))
        sim.data.time += dt
    assert "worn" in events


def test_tether_attached_beyond_rest_length_reels_in_without_snapping():
    from tether_rescue.tow.tether import Tether

    cfg = flat_cfg().tether
    t = Tether(cfg)
    t.reset(attached=False)
    t.attach(cfg.rest_length + 0.6)                         # 0.6 m 더 멀 때 연결 (그냥이면 360 N → 끊어짐)
    z = np.zeros(3)
    p = np.array([cfg.rest_length + 0.6, 0.0, 0.0])
    first = t.compute(z, p, z, z, 0.004)
    assert first.tension == 0 and not first.broken
    near = np.array([cfg.rest_length + 0.2, 0.0, 0.0])     # 로봇이 다가와서 거리가 줄어듦
    for _ in range(int(0.8 / cfg.reel_speed / 0.004)):
        s = t.compute(z, near, z, z, 0.004)
    assert t.rest == pytest.approx(cfg.rest_length)        # 다 감으면 원래 자연 길이
    assert s.tension == pytest.approx(cfg.stiffness * 0.2) and not s.broken


def _robot_facing_shoulder(sim, bow_gap: float):
    """로봇을 사람 어깨를 바라보게, 뱃머리 ~ 어깨가 bow_gap 이 되도록 옮김."""
    d = sim.data
    sh = d.xpos[sim.torso_id] + d.xmat[sim.torso_id].reshape(3, 3) @ sim.handoff.puppet.shape.shoulder[+1]
    away = np.array([1.0, 0.0])
    pos = sh[:2] + away * (sim.cfg.robot.length / 2 + bow_gap)
    yaw = math.atan2(-away[1], -away[0])
    adr = sim.model.jnt_qposadr[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "robot_free")]
    sim.data.qpos[adr:adr + 2] = pos
    sim.data.qpos[adr + 3:adr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)


def test_victim_without_tube_clings_to_close_robot():
    """튜브 없는 사람에게 배가 위험 거리(1.5 m) 안에 있으면 달려들어 매달림 → 실패(매달림 또는 뒤집힘)"""
    from tether_rescue.mission.boarding import CALM, CLINGING
    from tether_rescue.mission.mission_sim import MissionSim

    sim = MissionSim(flat_cfg(), mode="mission")
    sim.reset(0)
    _robot_facing_shoulder(sim, 1.0)
    b = sim.boarding
    assert b.measure()[0] < sim.cfg.human.grab_zone
    dt = 1.0 / sim.cfg.sim.control_hz
    seen, failed = set(), False
    for _ in range(int(12.0 / dt)):
        failed = failed or b.update(dt, has_tube=False) == "failed"
        seen.add(b.state)
        sim.step(0.0, 0.0)
        if failed or sim.done:
            break
    assert CLINGING in seen
    assert failed or sim.outcome.reason == "capsized"
    # 멀리 있으면 아무 일도 없음
    sim.reset(0)
    _robot_facing_shoulder(sim, 2.2)
    for _ in range(int(3.0 / dt)):
        sim.boarding.update(dt, has_tube=False)
        sim.step(0.0, 0.0)
    assert sim.boarding.state == CALM


def test_victim_holding_tube_does_not_lunge_at_robot():
    from tether_rescue.mission.boarding import CALM
    from tether_rescue.mission.mission_sim import MissionSim

    sim = MissionSim(flat_cfg(), mode="mission")
    sim.reset(0)
    _robot_facing_shoulder(sim, 1.0)
    dt = 1.0 / sim.cfg.sim.control_hz
    for _ in range(int(3.0 / dt)):
        sim.boarding.update(dt, has_tube=True)
        sim.step(0.0, 0.0)
    assert sim.boarding.state == CALM
