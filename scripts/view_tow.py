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
- 실험 설정 패널 (왼쪽): 탭 [실험 설정 · 지도] [실험 결과]
- MuJoCo 뷰어 (오른쪽): 왼쪽 위 상태, 오른쪽 실시간 그래프, 왼쪽 아래 단축키 안내 (모두 한글)
  판이 끝날 때마다 results/실험기록.csv 에 회차별로 자동 기록
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
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

from tether_rescue.bridge.messages import build_state  # noqa: E402
from tether_rescue.bridge.udp_link import SimLink  # noqa: E402
from tether_rescue.config import load_config  # noqa: E402
from tether_rescue.handoff import STATE_KO  # noqa: E402
from tether_rescue.mission.boarding import STATE_KO as BOARD_KO  # noqa: E402
from tether_rescue.metrics import compute_metrics  # noqa: E402
from tether_rescue.mission.guidance import PHASE_KO  # noqa: E402
from tether_rescue.mission.mission_sim import MissionSim  # noqa: E402
from tether_rescue.ui import overlay  # noqa: E402
from tether_rescue.ui.control_panel import ControlPanel, set_cfg  # noqa: E402
from tether_rescue.ui.run_log import REASON_KO  # noqa: E402

# GLFW 키 코드
KEY_SPACE, KEY_BACKSPACE, KEY_UP, KEY_DOWN = 32, 259, 265, 264
KEY_MINUS, KEY_EQUAL = 45, 61
KEY_6, KEY_7, KEY_8, KEY_9 = 54, 55, 56, 57

VIEWER_TITLE = "MuJoCo : airway_tow"   # MuJoCo 뷰어 창 제목 = "MuJoCo : " + 모델 이름(scene.py)
RESTART_DELAY = 3.0
PLOT_SECONDS = 15.0
SLOWMO_STEPS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]   # 1 = 실시간, 2 = 두 배 느리게
SPEED_MIN, SPEED_MAX = 0.2, 3.0

# 화면 글자는 한글 글꼴로 그림을 만들어 얹습니다 (tether_rescue/ui/overlay.py).
OVERLAY_INTERVAL = 0.08          # 글자·그래프 그림을 다시 만드는 간격 [s] (초당 약 12번)
HELP_ROWS = [("Space", "일시정지 / 계속"), ("Backspace", "처음부터 다시"), ("↑ / ↓", "견인 속도 +0.1 / -0.1"),
             ("= / -", "빠르게 / 느리게"), ("6 / 7 / 8", "카메라: 사람 / 로봇 / 위"), ("9", "그래프 켜기 / 끄기"),
             ("마우스", "왼쪽 끌기 회전 · 오른쪽 끌기 이동 · 휠 확대")]


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


class HFieldUploader:
    """물 표면(hfield)을 뷰어로 올리는 일을 따로 돌리는 스레드.

    viewer.update_hfield()는 뷰어 그리기 스레드가 올려 줄 때까지 기다립니다.
    뷰어 창을 닫은 직후에 부르면 그 스레드가 이미 멈춰 있어서 영원히 기다리고,
    메인 루프가 멈춰 프로그램이 안 꺼졌습니다. 그래서 따로 돌리고, 메인 루프는 기다리지 않습니다.
    (daemon 스레드라 거기서 멈춰 있어도 프로그램 종료를 막지 않음)"""

    def __init__(self, viewer, hfield_id: int):
        self.viewer, self.hfield_id = viewer, hfield_id
        self.idle = threading.Event()
        self.idle.set()
        self.request = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            self.request.wait()
            self.request.clear()
            try:
                if self.viewer.is_running():
                    self.viewer.update_hfield(self.hfield_id)
            except Exception:
                pass
            self.idle.set()

    def try_upload(self, update_data) -> None:
        """이전 업로드가 끝났을 때만 물 표면을 새로 계산해서 올려 달라고 요청 (아니면 이번 프레임은 건너뜀)."""
        if not self.idle.is_set():
            return
        self.idle.clear()
        update_data()
        self.request.set()


def make_overlay(vp, sim, cfg, obs, ui, plots, link, run_ko, play, phase, ratio, result_line) -> list:
    """뷰어 위에 얹을 한글 그림들: 왼쪽 위 상태, 왼쪽 아래 단축키, 오른쪽 그래프, 가운데 위 알림.
    반환: [(MjrRect, RGB 배열), ...]  (MjrRect는 왼쪽 아래가 원점)"""
    W, H = vp.width, vp.height
    size = int(min(max(H / 78, 11), 15))          # 글자 (3D 화면을 덜 가리게 작게)
    size_g = int(min(max(H / 50, 13), 24))        # 그래프 (크기 그대로)
    margin = 6
    G, B, Y, I = overlay.GOOD, overlay.BAD, overlay.WARN, overlay.INFO

    rows = [("상태", f"{run_ko}  ({play})", Y if ui["paused"] else None), ("시간", f"{obs['t']:.1f} 초")]
    if sim.mode == "mission":
        tube = STATE_KO.get(obs.get("tube"), "")
        if obs.get("tube") in ("OFFERED", "GRABBING"):
            tube += f"  (손까지 {obs['tube_reach']:.2f} / 팔 {cfg.human.arm_reach:.2f} m)"
        rows += [("단계", PHASE_KO.get(phase, phase), I), ("구명튜브", tube)]
        if not obs.get("attached") and math.isfinite(obs.get("hull_gap", math.inf)) and phase in ("APPROACH", "HANDOFF"):
            gap, zone = obs["hull_gap"], cfg.human.grab_zone
            board = obs.get("boarding", "CALM")
            txt = f"{gap:.1f} m  (위험 {zone:.1f} m 안)" if board == "CALM" else f"{gap:.1f} m  ← {BOARD_KO[board]}!"
            rows.append(("사람 ~ 배", txt, B if board != "CALM" or gap < zone else (Y if gap < zone + 0.3 else G)))
        rows += [("GPS 오차", f"{obs['gps_error']:.1f} m"), ("사람 ~ 기지", f"{obs['human_home_dist']:.1f} m")]
    else:
        rows += [("단계", "끌고 오기", I), ("이동 거리", f"{obs['progress']:.1f} / {cfg.sim.goal_distance:.0f} m")]
    if not sim.tether.attached:
        tether, tcol = "연결 안 됨", None
    elif obs["tether_broken"]:
        tether, tcol = "끊어짐!", B
    else:
        tether, tcol = f"{obs['tension']:.0f} N  (끊어지는 힘의 {ratio * 100:.0f}%)", (Y if ratio > 0.8 else None)
    rows += [("", ""),
             ("견인 목표 속도", f"{cfg.pid.target_speed:.1f} m/s"),
             ("로봇 속도", f"{obs['robot_speed']:.2f} m/s"),
             ("사람 속도", f"{obs['human_speed']:.2f} m/s"),
             ("추진력", f"엔진 {obs['thrust_level'] * 100:+.0f}%  → 실제 {obs['thrust_force']:+.0f} N"
                      f"  (효율 {obs['prop_factor'] * 100:.0f}%)", Y if obs['prop_factor'] < 0.5 else None),
             ("", ""),
             ("줄 힘", tether, tcol),
             ("입 높이", f"{obs['mouth_clearance'] * 100:+.1f} cm (수면 기준)"),
             ("얼굴", "물 밖 (숨 쉼)" if obs["airway_ok"] else "물에 잠김!", G if obs["airway_ok"] else B),
             ("로봇 기울기", f"{math.degrees(obs['tilt']):.1f}도"),
             ("파고 / 시드", f"{sim.waves.wave_height:.1f} m / {cfg.sim.seed}")]
    if link is not None:
        rows.append(("ROS2", "브리지 연결됨" if link.bridge_alive else "보내는 중 (브리지 응답 없음)",
                     G if link.bridge_alive else Y))

    out = []
    img = overlay.info_box(rows, size, int(W * 0.30), int(H * 0.50))
    out.append((mujoco.MjrRect(0, H - img.shape[0], img.shape[1], img.shape[0]), img))   # 모서리에 붙임
    img = overlay.info_box(HELP_ROWS, max(10, size - 2), int(W * 0.30), int(H * 0.20))
    out.append((mujoco.MjrRect(margin, margin, img.shape[1], img.shape[0]), img))

    if ui["graphs"]:
        gw = min(max(340, int(W * 0.30)), W // 2)
        gh = max(150, int(H * 0.26))
        # 눈금 4칸이 딱 떨어지는 범위 (값이 바뀌면 따라감)
        v_hi = 3.0 if cfg.pid.target_speed + 0.6 <= 3.0 else 5.0
        f_hi = math.ceil(cfg.tether.break_force * 1.1 / 100) * 100
        ranges = [(-1.0, v_hi), (0.0, f_hi), (-16.0, 16.0)]
        for i, (p, yr) in enumerate(zip(plots, ranges)):
            if (i + 1) * gh > H:
                break
            img = p.render(gw, gh, yr, PLOT_SECONDS, max(11, int(size_g * 0.85)))
            out.append((mujoco.MjrRect(W - gw, H - (i + 1) * gh, gw, gh), img))

    # 가운데 위 알림
    if result_line:
        title, color, detail = result_line
        sub = detail + ("\n3초 뒤 다음 판 (Backspace = 같은 시드로 다시)" if ui["auto"] else "\nBackspace = 다시 하기")
        img = overlay.banner(title, sub, color, size, int(W * 0.4), int(H * 0.15))
    elif ui["paused"]:
        img = overlay.banner("일시정지", "Space = 계속", Y, size, int(W * 0.4), int(H * 0.15))
    elif sim.mode == "mission" and phase == "WAIT_GOAL":
        img = overlay.banner("익수자 좌표 기다리는 중", "패널 지도를 클릭하세요", I, size, int(W * 0.4), int(H * 0.15))
    else:
        img = None
    if img is not None:
        out.append((mujoco.MjrRect((W - img.shape[1]) // 2, H - img.shape[0] - margin, img.shape[1], img.shape[0]),
                    img))
    return out


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
        t0 = sim.goal_received_t
        row.update({
            "reason": mm.reason, "time": mm.total_time, "airway": mm.tow_airway_ratio * 100,
            "tension": mm.max_tension, "score": float("nan"),
            "note": f"감지 {f(mm.detect_time)}s · 연결 {f(mm.attach_time)}s · 목표오차 {f(mm.goal_error)}m",
            "detect_time": mm.detect_time, "attach_time": mm.attach_time, "goal_error": mm.goal_error,
            "grab_time": None if (sim.grab_t is None or t0 is None) else sim.grab_t - t0,
            "handoff_standoff": cfg.mission.handoff_standoff, "arm_reach": cfg.human.arm_reach,
            "grab_zone": cfg.human.grab_zone,
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
            panel.camera.set(7)

    n_pts = int(PLOT_SECONDS * cfg.sim.control_hz) + 1
    plots = [
        overlay.Chart("속도 (m/s)", [("목표", (180, 180, 180)), ("로봇", (255, 140, 25)),
                                    ("사람", (75, 205, 255))], n_pts),
        overlay.Chart("줄 힘 (N)", [("줄 힘", (255, 80, 80)), ("끊어지는 힘", (240, 240, 240))], n_pts),
        overlay.Chart("입 높이 (cm, 수면 기준)", [("입", (80, 255, 130)), ("수면", (80, 130, 255))], n_pts),
    ]
    hud = {"t": -1.0, "images": []}     # 마지막으로 만든 화면 그림 (OVERLAY_INTERVAL마다 새로 만듦)

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
        water = HFieldUploader(viewer, sim.hfield_id)
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
                    no = ""
                    if panel:
                        no = f"{panel.add_result(row)['회차']}회차  "      # 기록 파일에 자동 저장
                    reason_ko = REASON_KO.get(row["reason"], row["reason"])
                    result_line = (f"{no}{reason_ko}",
                                   overlay.GOOD if row["reason"] == "success" else overlay.BAD,
                                   f"걸린 시간 {row['time']:.1f}초  ·  얼굴 물 밖 {row['airway']:.0f}%  ·  "
                                   f"최대 줄 힘 {row['tension']:.0f} N")
                    print(f"■ {no}끝: {reason_ko}  시간 {row['time']:.1f}s  {row['note']}")
                    end_time = time.perf_counter()
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
                water.try_upload(sim.update_water_surface)
            frame += 1

            # ---- 화면 글자 (한글)
            run_ko = "일시정지" if ui["paused"] else ("끝" if sim.done else "진행 중")
            sm = ui["slowmo"]
            play = "실시간" if sm == 1 else (f"{1 / sm:.0f}배 빠르게" if sm < 1 else f"{sm:.0f}배 느리게")
            phase = obs.get("phase", "TOW_ONLY")
            if panel:
                prog = (f"사람-기지 {obs['human_home_dist']:.0f} m" if sim.mode == "mission"
                        else f"{obs['progress']:.1f} / {cfg.sim.goal_distance:.0f} m")
                panel.set_status(f"{run_ko}  ·  {obs['t']:.1f} s  ·  {prog}", ui["paused"])

            now = time.perf_counter()
            vp = viewer.viewport
            if vp is not None and vp.width > 0 and now - hud["t"] >= OVERLAY_INTERVAL:
                hud["t"] = now
                hud["images"] = make_overlay(vp, sim, cfg, obs, ui, plots, link, run_ko, play, phase, ratio,
                                             result_line)
            if hud["images"]:
                # 하늘(skybox) 텍스처를 그린 뒤 남는 OpenGL 상태 때문에 그림이 안 보이는 문제가 있음.
                # 그림보다 먼저 그려지는 글자를 하나(빈 칸) 넣으면 그 상태가 정리되어 그림이 보임.
                # (빈 칸 글자는 왼쪽 위 상태판에 가려져 안 보임)
                viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT, " ", " ")])
                viewer.set_images(hud["images"])

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
