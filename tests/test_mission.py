"""구조 임무 검사: GPS·센서·추적·임무 전체·ROS2 메시지/UDP 연결."""
import json
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tether_rescue.bridge.messages import build_state, goal_message, marker_specs, parse_goal  # noqa: E402
from tether_rescue.bridge.udp_link import SimLink  # noqa: E402
from tether_rescue.config import (BridgeConfig, MissionConfig, SonarConfig, ThermalConfig,  # noqa: E402
                                  load_config)
from tether_rescue.metrics import run_pid_episode  # noqa: E402
from tether_rescue.mission.gps import GpsModel  # noqa: E402
from tether_rescue.mission.guidance import TargetTracker, square_spiral  # noqa: E402
from tether_rescue.mission.mission_sim import MissionSim, run_mission_episode  # noqa: E402
from tether_rescue.mission.sensors import Detection, Sonar, ThermalCamera  # noqa: E402


def flat(**sections):
    ov = {"ocean": {"wave_height": 0.0}}
    for k, v in sections.items():
        ov.setdefault(k, {}).update(v)
    return load_config(overrides=ov)


# ---------------------------------------------------------------- GPS
def test_gps_error_statistics():
    cfg = MissionConfig(gps_error=2.5, gps_tau=30.0)
    gps = GpsModel(cfg, np.random.default_rng(0))
    errs = []
    for _ in range(200_000):                 # 4000초 (dt 0.02)
        gps.update(0.02)
        errs.append(gps.error.copy())
    errs = np.array(errs)
    assert abs(errs.std() - 2.5) < 0.4       # 오래 보면 표준편차 ≈ σ
    step = np.abs(np.diff(errs[:, 0])).mean()
    assert step < 0.04 * 2.5                  # 한 걸음 변화는 오차 크기의 수 % — 천천히 흘러감


# ---------------------------------------------------------------- 센서
def test_thermal_sees_only_in_fov_and_when_head_visible():
    cam = ThermalCamera(ThermalConfig(range=20, fov=60, rate=1000, bearing_noise=0.0), np.random.default_rng(0))
    hits = sum(len(cam.sense(i * 0.01, (0, 0), 0.0, (5, 1), True, 0.0)) for i in range(100))
    assert hits > 90                          # 가깝고 앞쪽: 거의 항상 보임
    assert all(not cam.sense(1 + i * 0.01, (0, 0), 0.0, (-5, 0), True, 0.0) for i in range(50))   # 뒤쪽
    assert all(not cam.sense(2 + i * 0.01, (0, 0), 0.0, (5, 1), False, 0.0) for i in range(50))   # 머리 잠김
    d = cam.sense(3.0, (0, 0), 0.0, (5, 5 * math.tan(math.radians(20))), True, 0.0)
    assert d and d[0].range is None and d[0].bearing == pytest.approx(math.radians(20), abs=1e-6)


def test_sonar_range_and_false_alarms_scale_with_waves():
    cfg = SonarConfig(range=25, fov=120, rate=1000, range_noise=0.0, bearing_noise=0.0, false_rate=1.0)
    son = Sonar(cfg, np.random.default_rng(1))
    real = [d for i in range(200) for d in son.sense(i * 0.01, (0, 0), 0.0, (10, 0), 0.0)]
    assert real and all(d.real and d.range == pytest.approx(10.0) for d in real)
    calm = [d for i in range(200) for d in son.sense(10 + i * 0.01, (0, 0), 0.0, (100, 0), 0.0)]
    rough = [d for i in range(200) for d in son.sense(20 + i * 0.01, (0, 0), 0.0, (100, 0), 1.5)]
    assert not calm and len(rough) > 200      # 파도 없으면 가짜 탐지 없음, 파도 있으면 많음


# ---------------------------------------------------------------- 추적
def _sonar(t, x, y, real=True):
    r = math.hypot(x, y)
    return Detection("sonar", t, math.atan2(y, x), r, real)


def test_tracker_needs_repeated_sonar_hits():
    tr = TargetTracker()
    tr.update(0.0, [_sonar(0.0, 10, 2)], (0, 0), 0.0)
    tr.update(0.5, [_sonar(0.5, 10.3, 2.2)], (0, 0), 0.0)
    assert not tr.confirmed
    tr.update(1.0, [_sonar(1.0, 9.8, 1.9)], (0, 0), 0.0)
    assert tr.confirmed and np.linalg.norm(tr.est - [10, 2]) < 0.6


def test_tracker_ignores_scattered_false_returns():
    tr = TargetTracker()
    rng = np.random.default_rng(3)
    for i in range(40):
        x, y = rng.uniform(3, 25), rng.uniform(-20, 20)
        tr.update(i * 0.5, [_sonar(i * 0.5, x, y, real=False)], (0, 0), 0.0)
    assert not tr.confirmed


def test_tracker_thermal_and_sonar_agree_confirms_fast():
    tr = TargetTracker()
    tr.update(0.0, [_sonar(0.0, 12, 0)], (0, 0), 0.0)
    tr.update(0.1, [Detection("thermal", 0.1, math.radians(1.0), None, True)], (0, 0), 0.0)
    assert tr.confirmed


def test_tracker_corrects_wrong_range_when_sonar_agrees_with_thermal():
    """가짜 반사(20 m)로 확인된 뒤, 열화상과 같은 방향의 진짜 소나(6 m)가 오면 그 거리로 바로잡음"""
    tr = TargetTracker()
    tr.update(0.0, [_sonar(0.0, 20, 0, real=False)], (0, 0), 0.0)
    tr.update(0.1, [Detection("thermal", 0.1, 0.0, None, True)], (0, 0), 0.0)
    assert tr.confirmed and tr.est[0] == pytest.approx(20.0)
    tr.update(0.2, [Detection("thermal", 0.2, 0.0, None, True), _sonar(0.2, 6, 0.2)], (0, 0), 0.0)
    assert np.linalg.norm(tr.est - [6, 0.2]) < 0.5


def test_robot_stays_put_without_goal_in_waves():
    """좌표를 안 주면 엔진 0 → 파도 속에서도 거의 제자리 (부가질량 힘이 쌓여 떠밀리던 버그 방지)"""
    sim = MissionSim(load_config(overrides={"ocean": {"wave_height": 1.0}}))
    sim.reset(1)
    p0 = sim.data.xipos[sim.robot_id][:2].copy()
    for _ in range(int(30 * sim.cfg.sim.control_hz)):
        sim.step_auto()
    assert sim.brain.phase == "WAIT_GOAL"
    assert np.linalg.norm(sim.data.xipos[sim.robot_id][:2] - p0) < 3.0


def test_square_spiral_expands():
    pts = square_spiral((0, 0), 5.0, 8)
    assert np.allclose(pts[0], [5, 0]) and np.allclose(pts[1], [5, 5]) and np.allclose(pts[2], [-5, 5])
    assert max(np.abs(p).max() for p in pts) == pytest.approx(10.0)   # 8변이면 바깥 끝이 10 m


# ---------------------------------------------------------------- 임무 전체
def test_mission_flat_water_succeeds_through_all_phases():
    sim = MissionSim(flat())
    sim.reset(0)
    assert sim.brain.phase == "WAIT_GOAL" and not sim.tether.attached
    for _ in range(100):                      # 좌표가 없으면 제자리
        sim.step_auto()
    assert np.linalg.norm(sim.data.xipos[sim.robot_id][:2]) < 1.0
    sim.set_goal(sim.sighting)
    seen = []
    while not sim.done:
        o = sim.step_auto()
        if not seen or seen[-1] != o["phase"]:
            seen.append(o["phase"])
    m = sim.metrics()
    assert m.reason == "success"
    assert seen[:1] == ["TRANSIT"] and "APPROACH" in seen and "TOW" in seen
    assert m.detect_time is not None and m.attach_time > m.detect_time


def test_mission_is_deterministic():
    _, a = run_mission_episode(load_config(overrides={"mission": {"max_time": 40.0}}), seed=2)
    _, b = run_mission_episode(load_config(overrides={"mission": {"max_time": 40.0}}), seed=2)
    assert a == b


def test_far_wrong_goal_ends_not_found():
    cfg = flat(mission={"search_timeout": 20.0})
    sim, m = run_mission_episode(cfg, seed=0, goal=(10.0, 25.0))   # 익수자와 먼 곳
    assert m.reason == "not_found"


def test_tow_mode_matches_original_towsim():
    cfg = flat(pid={"target_speed": 1.2}, sim={"max_time": 20.0})
    ref, _ = run_pid_episode(cfg, seed=0)
    sim = MissionSim(flat(pid={"target_speed": 1.2}, sim={"max_time": 20.0}), mode="tow")
    sim.reset(0)
    while not sim.done:
        sim.step_auto()
    assert np.allclose(ref.log.array(), sim.log.array())
    assert sim.set_goal((10, 0)) is False     # 끌고 오기만 모드에서는 목표 좌표 무시


# ---------------------------------------------------------------- ROS2 메시지 / UDP
def test_state_message_and_markers():
    sim = MissionSim(flat())
    sim.reset(0)
    sim.set_goal(sim.sighting)
    for _ in range(300):
        sim.step_auto()
    st = build_state(sim, "mission")
    json.dumps(st)                            # JSON으로 보낼 수 있어야 함
    specs = marker_specs(st)
    ns = {s["ns"] for s in specs}
    assert {"robot", "human", "base", "goal", "thermal_fov", "sonar_fov", "robot_gps"} <= ns
    assert all(len(s["color"]) == 4 and len(s["pose"]) == 4 for s in specs)


def test_parse_goal_rejects_bad_messages():
    assert parse_goal(goal_message(3.0, -2.0, "rviz2")) == (3.0, -2.0, "rviz2")
    assert parse_goal({"type": "state"}) is None
    assert parse_goal({"type": "goal", "x": "nan", "y": 1}) is None
    assert parse_goal({"type": "goal", "x": 1}) is None


def test_udp_link_roundtrip_with_fake_bridge():
    bridge = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    bridge.bind(("127.0.0.1", 0))
    bridge.settimeout(3.0)
    port = bridge.getsockname()[1]
    link = SimLink(BridgeConfig(host="127.0.0.1", port=port, rate=100.0))
    for _ in range(50):
        if link.host:
            break
        time.sleep(0.02)
    assert link.maybe_send({"type": "state", "t": 0.0})
    data, addr = bridge.recvfrom(65535)
    assert json.loads(data)["type"] == "state"
    g = json.dumps(goal_message(12.5, -3.0, "rviz2 2D Goal Pose")).encode()
    bridge.sendto(g, addr)
    bridge.sendto(g, addr)                    # 브리지는 두 번 보냄 → 한 번만 처리돼야 함
    time.sleep(0.1)
    goals = link.poll_goals()
    assert goals == [(12.5, -3.0, "rviz2 2D Goal Pose")]
    assert link.bridge_alive
    link.close()
    bridge.close()
