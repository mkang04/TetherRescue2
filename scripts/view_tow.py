"""3D 화면 + 실험 설정 패널 (MuJoCo 뷰어).

예)  python scripts/view_tow.py                   # 전체 구조 임무 모드 + 설정 패널
     python scripts/view_tow.py --mode tow        # 끌고 오기만
     python scripts/view_tow.py --ros             # ROS2(rviz2) 연결을 켠 채로 시작
     python scripts/view_tow.py --no-panel        # 뷰어만

모드
- 전체 구조 임무: 좌표를 받으면(지도 클릭 / rviz2 2D Goal Pose / 자동 신고) GPS로 이동 →
  가까이서 열화상·소나로 직접 감지해 접근 → 줄 연결 → 기지로 견인
- 끌고 오기만: 줄이 연결된 상태에서 PID로 끌고 오기 (기존)

창 두 개
- 실험 설정 패널 (왼쪽): 탭 [실험 값] [구조 임무 · 지도] [실험 결과]
- MuJoCo 뷰어 (오른쪽): 왼쪽 위 상태, 오른쪽 실시간 그래프, 왼쪽 아래 단축키 안내
  줄 색: 노랑=느슨, 주황→빨강=팽팽, 회색=끊어짐 / 빨간 기둥=목표 좌표, 초록 공=감지로 찾은 사람,
  주황 선=열화상 시야, 파랑 선=소나 시야, 노란 공=로봇이 생각하는 자기 위치(GPS)

단축키 (뷰어 창에서. MuJoCo 뷰어 기본 단축키와 겹치지 않게 골랐습니다)
- Space      일시정지 / 계속
- Backspace  처음부터 다시
- Up / Down  목표(견인) 속도 +0.1 / -0.1 m/s
- = / -      빠르게 / 느리게 (슬로모션)
- 6 / 7 / 8  카메라: 사람 따라가기 / 로봇 따라가기 / 위에서 보기
- 9          그래프 켜기 / 끄기
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

from tether_rescue.bridge.messages import build_state  # noqa: E402
from tether_rescue.bridge.udp_link import SimLink  # noqa: E402
from tether_rescue.config import load_config  # noqa: E402
from tether_rescue.handoff import STATE_EN  # noqa: E402
from tether_rescue.metrics import compute_metrics  # noqa: E402
from tether_rescue.mission.mission_sim import MissionSim  # noqa: E402
from tether_rescue.ui.control_panel import ControlPanel, set_cfg  # noqa: E402

# GLFW 키 코드
KEY_SPACE, KEY_BACKSPACE, KEY_UP, KEY_DOWN = 32, 259, 265, 264
KEY_MINUS, KEY_EQUAL = 45, 61
KEY_6, KEY_7, KEY_8, KEY_9 = 54, 55, 56, 57

VIEWER_TITLE = "MuJoCo : airway_tow"   # MuJoCo 뷰어 창 제목 = "MuJoCo : " + 모델 이름(scene.py)
RESTART_DELAY = 3.0
PLOT_SECONDS = 15.0
SLOWMO_STEPS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]   # 1 = 실시간, 2 = 두 배 느리게
SPEED_MIN, SPEED_MAX = 0.2, 3.0

# 뷰어 글꼴은 한글을 못 그리므로 화면 글자는 영어로 씁니다.
REASON_TEXT = {"success": "RESCUED", "tether_broken": "TETHER BROKEN", "capsized": "ROBOT CAPSIZED",
               "timeout": "TIMEOUT", "not_found": "VICTIM NOT FOUND"}
PHASE_TEXT = {"WAIT_GOAL": "waiting for coordinate", "TRANSIT": "GPS transit to coordinate",
              "SEARCH": "searching nearby", "APPROACH": "direct sensing approach",
              "HANDOFF": "handing tube to victim", "TOW": "towing to base",
              "DONE": "done", "TOW_ONLY": "towing"}
CAMERA_NAMES = {6: "follow human", 7: "follow robot", 8: "top view"}

HELP_KEYS = "\n".join(["Space", "Backspace", "Up / Down", "= / -", "6 / 7 / 8", "9", "Mouse"])
HELP_DESC = "\n".join(["pause / resume", "restart", "tow speed +/- 0.1", "faster / slower (slow motion)",
                       "camera: human / robot / top", "graphs on / off", "L-drag rotate, R-drag move, wheel zoom"])


class Plot:
    """mjvFigure 하나 (최근 PLOT_SECONDS 초 롤링 그래프)."""

    def __init__(self, title: str, lines: list[tuple[str, tuple[float, float, float]]], y_range, n_max):
        f = mujoco.MjvFigure()
        f.title = title
        f.flg_extend = 0
        f.flg_legend = 1
        f.flg_ticklabel[:] = [1, 1]
        f.gridsize[:] = [4, 4]
        f.range[1] = y_range
        f.figurergba[:] = [0.08, 0.08, 0.10, 0.75]
        f.panergba[:] = [0.15, 0.15, 0.18, 0.75]
        f.gridrgb[:] = [0.35, 0.35, 0.4]
        f.textrgb[:] = [1, 1, 1]
        f.linewidth = 2.0
        f.yformat = "%.1f" if y_range[1] - y_range[0] < 20 else "%.0f"
        for i, (name, rgb) in enumerate(lines):
            f.linename[i] = name
            f.linergb[i] = rgb
        self.fig = f
        self.n_lines = len(lines)
        self.buf = [deque(maxlen=n_max) for _ in range(self.n_lines + 1)]   # [t, y0, y1, ...]

    def clear(self):
        for b in self.buf:
            b.clear()

    def push(self, t: float, *ys: float):
        self.buf[0].append(t)
        for b, y in zip(self.buf[1:], ys):
            b.append(y)

    def refresh(self):
        t = np.fromiter(self.buf[0], float)
        n = len(t)
        t_end = t[-1] if n else 0.0
        self.fig.range[0] = [t_end - PLOT_SECONDS, t_end]
        for i in range(self.n_lines):
            self.fig.linepnt[i] = n
            if n:
                self.fig.linedata[i, 0:2 * n:2] = t
                self.fig.linedata[i, 1:2 * n:2] = np.fromiter(self.buf[i + 1], float)


# --------------------------------------------------------------------------- 3D 장면에 임무 표시 그리기
_EYE = np.eye(3).flatten()


def _add(scn, gtype, size, pos, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], gtype, np.asarray(size, float), np.asarray(pos, float), _EYE,
                        np.asarray(rgba, np.float32))
    scn.ngeom += 1


def _line(scn, p0, p1, width, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.zeros(9),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(p0, float), np.asarray(p1, float))
    scn.ngeom += 1


def draw_mission(scn, st: dict, show_truth: bool):
    scn.ngeom = 0
    if st.get("mode") != "mission":
        return
    z = 0.7
    rx, ry, ryaw = st["robot"]
    s = st["sensors"]
    for rng, fov, col in ((s["thermal_range"], s["thermal_fov"], (1.0, 0.5, 0.1, 0.9)),
                          (s["sonar_range"], s["sonar_fov"], (0.3, 0.6, 1.0, 0.7))):
        half = math.radians(fov) / 2
        pts = [(rx + rng * math.cos(ryaw + a), ry + rng * math.sin(ryaw + a), z)
               for a in np.linspace(-half, half, 9)]
        _line(scn, (rx, ry, z), pts[0], 2, col)
        _line(scn, (rx, ry, z), pts[-1], 2, col)
        for p0, p1 in zip(pts[:-1], pts[1:]):
            _line(scn, p0, p1, 2, col)
    if st.get("goal"):
        gx, gy = st["goal"]
        _add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.15, 1.5, 0], [gx, gy, 1.5], (1.0, 0.1, 0.1, 0.9))
        _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.35, 0, 0], [gx, gy, 3.1], (1.0, 0.1, 0.1, 0.9))
    if st.get("track"):
        _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.45, 0, 0], [st["track"][0], st["track"][1], 1.0],
             (0.1, 1.0, 0.3, 0.6))
    gx, gy = st["robot_gps"]
    _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.25, 0, 0], [gx, gy, 1.2], (1.0, 0.9, 0.2, 0.9))
    _line(scn, (rx, ry, 1.2), (gx, gy, 1.2), 2, (1.0, 0.9, 0.2, 0.7))
    path = st.get("search_path") or []
    for p0, p1 in zip(path[:-1], path[1:]):
        _line(scn, (p0[0], p0[1], 0.4), (p1[0], p1[1], 0.4), 1, (0.8, 0.8, 1.0, 0.6))
    for d in st["detections"][-25:]:
        col = (1.0, 0.5, 0.1, 1.0) if d["sensor"] == "thermal" else (0.3, 0.7, 1.0, 1.0)
        _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.12, 0, 0], [d["x"], d["y"], 0.8], col)
    if show_truth:
        hx, hy = st["human"]
        _add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.05, 1.2, 0], [hx, hy, 1.2], (1.0, 0.75, 0.6, 0.8))


def place_viewer_window(title: str, left: int) -> bool:
    """(Windows 전용) 뷰어 창을 패널 오른쪽 남은 공간에 맞춰 배치. 창을 찾았으면 True."""
    if sys.platform != "win32":
        return True
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, title)
    if not hwnd:
        return False
    area = wintypes.RECT()
    user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(area), 0)   # SPI_GETWORKAREA: 작업 표시줄 제외
    width = max(640, area.right - left)
    user32.ShowWindow(hwnd, 9)                                         # SW_RESTORE (최대화 해제)
    user32.SetWindowPos(hwnd, 0, left, area.top, width, area.bottom - area.top, 0x0004)  # SWP_NOZORDER
    return True


def result_row(cfg, sim, mode: str, modified: bool) -> dict:
    """결과 표 / CSV 한 줄 (끝난 시점의 설정값)."""
    row = {
        "mode": mode, "seed": cfg.sim.seed, "speed": cfg.pid.target_speed, "wave": cfg.ocean.wave_height,
        "changed_mid_run": int(modified),
        "wave_period": cfg.ocean.peak_period, "wave_direction_deg": round(math.degrees(cfg.ocean.direction)),
        "max_thrust": cfg.robot.max_thrust, "break_force": cfg.tether.break_force,
        "tether_stiffness": cfg.tether.stiffness, "tether_length": cfg.tether.rest_length,
        "bow_wave_coef": cfg.human.bow_wave_coef,
        "speed_kp": cfg.pid.speed_kp, "speed_ki": cfg.pid.speed_ki,
        "heading_kp": cfg.pid.heading_kp, "heading_kd": cfg.pid.heading_kd, "ramp_time": cfg.pid.ramp_time,
    }
    if mode == "mission":
        mm = sim.metrics()
        f = (lambda v: "-" if v is None else f"{v:.0f}")
        row.update({
            "reason": mm.reason, "time": mm.total_time, "airway": mm.tow_airway_ratio * 100,
            "tension": mm.max_tension, "score": float("nan"),
            "note": f"감지 {f(mm.detect_time)}s · 연결 {f(mm.attach_time)}s · 목표오차 {f(mm.goal_error)}m",
            "detect_time": mm.detect_time, "attach_time": mm.attach_time, "goal_error": mm.goal_error,
            "gps_error_at_arrival": mm.gps_error_at_arrival, "false_alarms": mm.false_alarms,
            "gps_error_sigma": cfg.mission.gps_error, "report_error": cfg.mission.report_error,
            "thermal_range": cfg.thermal.range, "thermal_fov": cfg.thermal.fov, "sonar_range": cfg.sonar.range,
            "sonar_false_rate": cfg.sonar.false_rate, "transit_speed": cfg.mission.transit_speed,
            "victim_distance": cfg.mission.victim_distance,
        })
    else:
        mt = compute_metrics(sim)
        row.update({"reason": mt.reason, "time": mt.rescue_time, "airway": mt.airway_ratio * 100,
                    "tension": mt.max_tension, "score": mt.score, "note": f"점수 {mt.score:.0f}",
                    "goal_distance": cfg.sim.goal_distance})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["mission", "tow"], default="mission")
    ap.add_argument("--speed", type=float, default=None, help="목표(견인) 속도 [m/s]")
    ap.add_argument("--wave", type=float, default=None, help="유의파고 [m]")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--slowmo", type=float, default=1.0, help="1=실시간, 2=두 배 느리게")
    ap.add_argument("--no-loop", action="store_true", help="한 판 끝나면 다시 시작하지 않음")
    ap.add_argument("--no-panel", action="store_true", help="설정 패널 창 없이 뷰어만")
    ap.add_argument("--ros", action="store_true", help="ROS2(rviz2) 연결을 켠 채로 시작")
    ap.add_argument("--auto-report", action="store_true", help="클릭 없이 목격 지점으로 자동 출발")
    args = ap.parse_args()

    ov: dict = {}
    if args.speed is not None:
        ov.setdefault("pid", {})["target_speed"] = args.speed
    if args.wave is not None:
        ov.setdefault("ocean", {})["wave_height"] = args.wave
    if args.seed is not None:
        ov.setdefault("sim", {})["seed"] = args.seed
    if args.auto_report:
        ov.setdefault("mission", {})["auto_report"] = True
    cfg = load_config(args.config, ov)
    sim = MissionSim(cfg, mode=args.mode)
    ctrl = sim.ctrl
    dt = 1.0 / cfg.sim.control_hz
    m = sim.model
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, "tether_vis")

    panel = None
    if not args.no_panel:
        panel = ControlPanel(cfg, load_config(args.config), ROOT / "results", mode=args.mode)
        panel.set_playback(args.slowmo)
        panel.auto_restart.set(not args.no_loop)
        panel.ros_enabled.set(args.ros)
        if args.mode == "mission":
            panel.notebook.select(1)              # 구조 임무 · 지도 탭을 먼저 보여 줌
            panel.camera.set(7)

    n_pts = int(PLOT_SECONDS * cfg.sim.control_hz) + 1
    plots = [
        Plot("Speed [m/s]", [("target", (0.7, 0.7, 0.7)), ("robot", (1.0, 0.55, 0.1)),
                             ("human", (0.3, 0.8, 1.0))], [0.0, 2.6], n_pts),
        Plot("Tether tension [N]", [("tension", (1.0, 0.3, 0.3)), ("break", (1.0, 1.0, 1.0))],
             [0.0, cfg.tether.break_force * 1.1], n_pts),
        Plot("Mouth above water [cm]", [("mouth", (0.3, 1.0, 0.5)), ("water level", (0.3, 0.5, 1.0))],
             [-15.0, 15.0], n_pts),
    ]

    # 단축키는 뷰어 스레드에서 들어오므로 여기에는 '요청'만 적고, 처리는 메인 루프에서 합니다.
    keys = {"pause": False, "restart": False, "speed_delta": 0.0, "slowmo_mul": 1.0, "camera": None,
            "graphs": False}

    def on_key(key):
        if key == KEY_SPACE:
            keys["pause"] = True
        elif key == KEY_BACKSPACE:
            keys["restart"] = True
        elif key == KEY_UP:
            keys["speed_delta"] += 0.1
        elif key == KEY_DOWN:
            keys["speed_delta"] -= 0.1
        elif key == KEY_EQUAL:
            keys["slowmo_mul"] /= 2
        elif key == KEY_MINUS:
            keys["slowmo_mul"] *= 2
        elif key in (KEY_6, KEY_7, KEY_8):
            keys["camera"] = key - KEY_6 + 6
        elif key == KEY_9:
            keys["graphs"] = True

    ui = {"paused": False, "slowmo": args.slowmo, "camera": 7 if args.mode == "mission" else 6, "graphs": True,
          "auto": not args.no_loop, "truth": False}
    run = {"modified": False}       # 한 판 도중에 값을 바꿨는지 (결과 표에 표시)
    link: SimLink | None = SimLink(cfg.ros_bridge) if (args.ros and panel is None) else None

    def start():
        o = sim.reset(cfg.sim.seed)
        for p in plots:
            p.clear()
        run["modified"] = False
        if sim.mode == "mission":
            print(f"\n▶ 구조 임무 시작: 시드 {cfg.sim.seed}, 파고 {cfg.ocean.wave_height:.1f} m, "
                  f"목격 지점 ({sim.sighting[0]:.1f}, {sim.sighting[1]:.1f}) — 지도에서 좌표를 찍으세요")
        else:
            print(f"\n▶ 끌고 오기 시작: 목표 속도 {cfg.pid.target_speed:.1f} m/s, 파고 {cfg.ocean.wave_height:.1f} m, "
                  f"시드 {cfg.sim.seed}")
        return o

    def apply_restart_values(auto: bool):
        prev_seed = cfg.sim.seed
        if panel:
            for k, v in panel.restart_values().items():
                set_cfg(cfg, k, v)
            sim.mode = panel.mode_applied
        if auto and cfg.sim.seed == prev_seed:          # 사용자가 시드를 안 바꿨으면 다음 시드로
            cfg.sim.seed = (cfg.sim.seed + 1) % 1000
            if panel:
                panel.set_value("sim.seed", cfg.sim.seed, mark_applied=True)

    def apply_camera(viewer):
        cam = viewer.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        if ui["camera"] == 7:
            cam.trackbodyid, cam.distance, cam.elevation, cam.azimuth = sim.robot_id, 9.0, -25, 200
        elif ui["camera"] == 8:
            cam.trackbodyid, cam.distance, cam.elevation, cam.azimuth = sim.robot_id, 30.0, -70, 90
        else:
            cam.trackbodyid, cam.distance, cam.elevation, cam.azimuth = sim.torso_id, 9.0, -22, 135

    def give_goal(x: float, y: float, source: str):
        if sim.mode != "mission":
            print(f"  ({source}) 끌고 오기만 모드에서는 목표 좌표를 쓰지 않습니다")
            return
        ok = sim.set_goal((x, y))
        print(f"  목표 좌표 ({source}): ({x:.1f}, {y:.1f}) → {'접수' if ok else '무시 (견인 중/끝남)'}")

    obs = start()
    result_line = ""
    end_time = None
    last_sim_time = sim.data.time
    last_restart = time.perf_counter()
    camera_applied = None

    print("설정 패널에서 값을 바꾸면 바로 반영됩니다. 둘 중 한 창을 닫으면 종료.")
    with mujoco.viewer.launch_passive(m, sim.data, key_callback=on_key,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        frame = 0
        placed = panel is None
        while viewer.is_running():
            tick = time.perf_counter()
            want_restart = False

            # 처음 한 번: 패널은 왼쪽, 뷰어는 오른쪽 남은 공간 (창이 뜰 때까지 몇 프레임 재시도)
            if not placed and frame % 5 == 0:
                placed = place_viewer_window(VIEWER_TITLE, panel.right_edge()) or frame > 250

            # ---- 설정 패널
            if panel:
                if not panel.poll():
                    break                                  # 패널을 닫으면 종료
                for act in panel.actions:
                    if act == "pause":
                        ui["paused"] = not ui["paused"]
                    elif act == "restart":
                        want_restart = True
                panel.actions.clear()
                for k, v in panel.take_live_changes().items():
                    if k == "ocean.wave_height":
                        sim.set_wave_height(v)             # 1초에 걸쳐 부드럽게
                    else:
                        set_cfg(cfg, k, v)
                    if k.startswith("pid."):
                        ctrl.update_gains()
                    if not sim.done:
                        run["modified"] = True
                for gx, gy in panel.take_goal_requests():
                    give_goal(gx, gy, "지도 클릭")
                ui["camera"] = panel.camera.get()
                ui["slowmo"] = panel.playback_slowmo()
                ui["graphs"] = panel.show_graphs.get()
                ui["auto"] = panel.auto_restart.get()
                ui["truth"] = panel.show_truth.get()
                cfg.mission.auto_report = panel.auto_report.get()
                # ROS2 연결 켜기/끄기
                if panel.ros_enabled.get() and link is None:
                    link = SimLink(cfg.ros_bridge)
                elif not panel.ros_enabled.get() and link is not None:
                    link.close()
                    link = None
                    panel.set_ros_status("꺼짐  —  WSL에서 ros_bridge_wsl.bat 실행 후 체크")

            # ---- ROS2 브리지에서 온 목표 (rviz2 2D Goal Pose / Publish Point)
            if link is not None:
                for gx, gy, src in link.poll_goals():
                    give_goal(gx, gy, src)
                if panel:
                    panel.set_ros_status(link.status)

            # 자동 신고: 좌표를 기다리는 중이면 목격 지점으로 출발
            if sim.mode == "mission" and cfg.mission.auto_report and sim.brain.phase == "WAIT_GOAL" \
                    and not sim.done:
                give_goal(*sim.sighting, "자동 신고")

            # ---- 단축키 (패널 표시도 같이 맞춤)
            if keys["pause"]:
                ui["paused"] = not ui["paused"]
                keys["pause"] = False
            if keys["restart"]:
                want_restart = True
                keys["restart"] = False
            if keys["speed_delta"]:
                cfg.pid.target_speed = round(min(SPEED_MAX, max(SPEED_MIN,
                                                                cfg.pid.target_speed + keys["speed_delta"])), 2)
                keys["speed_delta"] = 0.0
                if panel:
                    panel.set_value("pid.target_speed", cfg.pid.target_speed)
                if not sim.done:
                    run["modified"] = True
            if keys["slowmo_mul"] != 1.0:
                ui["slowmo"] = min(SLOWMO_STEPS[-1], max(SLOWMO_STEPS[0], ui["slowmo"] * keys["slowmo_mul"]))
                keys["slowmo_mul"] = 1.0
                if panel:
                    panel.set_playback(ui["slowmo"])
            if keys["camera"] is not None:
                ui["camera"] = keys["camera"]
                keys["camera"] = None
                if panel:
                    panel.camera.set(ui["camera"])
            if keys["graphs"]:
                ui["graphs"] = not ui["graphs"]
                keys["graphs"] = False
                if panel:
                    panel.show_graphs.set(ui["graphs"])

            # ---- 다시 시작 (버튼·Backspace, 뷰어가 직접 리셋한 경우, 끝난 뒤 자동)
            # Backspace는 뷰어도 자체적으로 데이터를 리셋하므로, 방금 재시작했다면 한 번 더 하지 않음
            viewer_reset = (sim.data.time < last_sim_time - 1e-9
                            and time.perf_counter() - last_restart > 0.5)
            auto = (ui["auto"] and end_time is not None and not ui["paused"]
                    and time.perf_counter() - end_time > RESTART_DELAY)
            if want_restart or viewer_reset or auto:
                apply_restart_values(auto)
                with viewer.lock():
                    obs = start()
                result_line, end_time = "", None
                last_restart = time.perf_counter()
                camera_applied = None

            if ui["camera"] != camera_applied:
                with viewer.lock():
                    apply_camera(viewer)
                camera_applied = ui["camera"]

            # ---- 시뮬레이션 한 걸음
            brk = cfg.tether.break_force
            if not sim.done and not ui["paused"]:
                with viewer.lock():
                    obs = sim.step_auto()
                tgt = sim.command.speed_ref if sim.command is not None else ctrl.target_speed(obs["t"])
                plots[0].push(obs["t"], tgt, obs["robot_speed"], obs["human_speed"])
                plots[1].push(obs["t"], obs["tension"], brk)
                plots[2].push(obs["t"], obs["mouth_clearance"] * 100, 0.0)
                if sim.done:
                    row = result_row(cfg, sim, sim.mode, run["modified"])
                    result_line = (f"{REASON_TEXT.get(row['reason'], row['reason'])}   time {row['time']:.1f} s   "
                                   f"airway {row['airway']:.0f} %")
                    print(f"■ 끝: {row['reason']}  시간 {row['time']:.1f}s  {row['note']}")
                    end_time = time.perf_counter()
                    if panel:
                        panel.add_result(row)
            last_sim_time = sim.data.time

            # ---- 상태 dict (지도·rviz2·3D 표시 공통)
            need_state = (panel is not None and frame % 5 == 0) or link is not None or sim.mode == "mission"
            st = build_state(sim, sim.mode) if need_state else None
            if link is not None and st is not None:
                link.maybe_send(st)
            if panel is not None and st is not None and frame % 5 == 0:
                panel.update_map(st)
            with viewer.lock():
                if st is not None:
                    draw_mission(viewer.user_scn, st, ui["truth"])
                else:
                    viewer.user_scn.ngeom = 0

            # 줄 색 (연결 전에는 안 보이게)
            ratio = min(1.0, obs["tension"] / brk)
            if not sim.tether.attached:
                m.tendon_rgba[tid] = [0, 0, 0, 0]
            elif obs["tether_broken"]:
                m.tendon_rgba[tid] = [0.5, 0.5, 0.5, 0.6]
            elif obs["tension"] > 0:
                m.tendon_rgba[tid] = [1.0, 0.7 * (1 - ratio), 0.1, 1.0]
            else:
                m.tendon_rgba[tid] = [0.95, 0.95, 0.2, 1.0]

            # 물 표면 (2프레임마다 = 25Hz)
            if frame % 2 == 0:
                sim.update_water_surface()
                viewer.update_hfield(sim.hfield_id)
            frame += 1

            # ---- 화면 글자
            if ui["paused"]:
                run_state, run_ko = "PAUSED", "일시정지"
            elif sim.done:
                run_state, run_ko = "FINISHED", "끝"
            else:
                run_state, run_ko = "RUNNING", "진행 중"
            sm = ui["slowmo"]
            play = "real time" if sm == 1 else (f"{1 / sm:.0f}x faster" if sm < 1 else f"{sm:.0f}x slower")
            phase = obs.get("phase", "TOW_ONLY")
            if panel:
                prog = (f"사람-기지 {obs['human_home_dist']:.0f} m" if sim.mode == "mission"
                        else f"{obs['progress']:.1f} / {cfg.sim.goal_distance:.0f} m")
                panel.set_status(f"{run_ko}  ·  {obs['t']:.1f} s  ·  {prog}", ui["paused"])
            rows = [("STATE", f"{run_state}  ({play})"), ("time", f"{obs['t']:.1f} s")]
            if sim.mode == "mission":
                tube = STATE_EN.get(obs.get("tube"), "")
                if obs.get("tube") in ("OFFERED", "GRABBING"):
                    tube += f"  (hand {obs['tube_reach']:.2f} / reach {cfg.human.arm_reach:.2f} m)"
                rows += [("phase", PHASE_TEXT.get(phase, phase)),
                         ("rescue tube", tube),
                         ("GPS error", f"{obs['gps_error']:.1f} m"),
                         ("victim to base", f"{obs['human_home_dist']:.1f} m")]
            else:
                rows += [("distance", f"{obs['progress']:.1f} / {cfg.sim.goal_distance:.0f} m")]
            rows += [("", ""), ("tow speed set", f"{cfg.pid.target_speed:.1f} m/s"),
                     ("robot speed", f"{obs['robot_speed']:.2f} m/s"),
                     ("human speed", f"{obs['human_speed']:.2f} m/s"),
                     ("thrust", f"{obs['thrust_level'] * 100:+.0f}% of max  "
                                f"({obs['thrust_level'] * cfg.robot.max_thrust:+.0f} N)"), ("", ""),
                     ("tether", ("not attached" if not sim.tether.attached else
                                 "BROKEN" if obs["tether_broken"] else
                                 f"{obs['tension']:.0f} N  ({ratio * 100:.0f}% of break)")),
                     ("mouth", f"{obs['mouth_clearance'] * 100:+.1f} cm"),
                     ("airway", "OK" if obs["airway_ok"] else "SUBMERGED !"),
                     ("robot tilt", f"{math.degrees(obs['tilt']):.1f} deg"), ("", ""),
                     ("waves / seed", f"Hs {sim.waves.wave_height:.1f} m / {cfg.sim.seed}"),
                     ("camera", CAMERA_NAMES[ui["camera"]])]
            if link is not None:
                rows.append(("ROS2", "bridge connected" if link.bridge_alive else "sending (no bridge reply)"))
            texts = [
                (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                 "\n".join(r[0] for r in rows), "\n".join(r[1] for r in rows)),
                (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, HELP_KEYS, HELP_DESC),
            ]
            # 가운데 위 알림 (한 칸에 몰아 써야 글자가 겹치지 않음)
            banner = None
            if result_line:
                banner = result_line + ("\nnext run in 3 s  (Backspace = same seed again)" if ui["auto"]
                                        else "\nBackspace = run again")
            elif ui["paused"]:
                banner = "PAUSED  -  press Space to resume"
            elif sim.mode == "mission" and phase == "WAIT_GOAL":
                banner = "WAITING FOR VICTIM COORDINATE\nclick the map in the panel  (or rviz2 '2D Goal Pose')"
            if banner:
                texts.append((mujoco.mjtFontScale.mjFONTSCALE_200, mujoco.mjtGridPos.mjGRID_TOP, banner, None))
            viewer.set_texts(texts)

            # ---- 그래프 (오른쪽, 창 크기에 맞춰 배치). 값이 바뀌면 축 범위도 따라감
            plots[0].fig.range[1] = [0.0, max(2.6, cfg.pid.target_speed + 0.6)]
            plots[1].fig.range[1] = [0.0, brk * 1.1]
            vp = viewer.viewport
            if ui["graphs"] and vp is not None and vp.width > 0:
                w = max(320, int(vp.width * 0.30))
                h = max(150, int(vp.height * 0.26))
                figs = []
                for i, p in enumerate(plots):
                    p.refresh()
                    rect = mujoco.MjrRect(vp.width - w, vp.height - (i + 1) * h, w, h)
                    figs.append((rect, p.fig))
                viewer.set_figures(figs)
            else:
                viewer.clear_figures()

            viewer.sync()
            wait = dt * ui["slowmo"] - (time.perf_counter() - tick)
            if wait > 0:
                time.sleep(wait)

    if link is not None:
        link.close()
    if panel:
        panel.destroy()


if __name__ == "__main__":
    main()
