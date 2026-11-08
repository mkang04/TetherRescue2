"""실험 설정 패널 (tkinter 창). MuJoCo 뷰어 옆에 띄워서 값을 바꾸며 실험합니다.

탭
- 실험 설정 · 지도: 모드 선택, 지도(클릭 = 목표 좌표, rviz2 "2D Goal Pose"와 같은 역할), ROS2 연결,
  그 아래 모든 값 (구조 임무·센서·끌고 오기·바다·줄·PID). 화면보다 길면 마우스 휠로 스크롤
- 실험 결과: 판이 끝날 때마다 회차별로 자동 기록 (results/실험기록.csv), 성공 초록 · 실패 빨강

"바로 적용" 값은 바꾸는 즉시 반영, "다시 시작할 때 적용" 값은 [적용하고 다시 시작] 을 눌러야 반영.
이 패널은 메인 스레드에서 poll()을 자주 불러 주는 방식으로 동작합니다 (자체 mainloop 없음).
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

from ..config import TowConfig
from .run_log import RunLog

FONT = ("Malgun Gothic", 10)
FONT_S = ("Malgun Gothic", 9)
FONT_B = ("Malgun Gothic", 10, "bold")
FONT_TITLE = ("Malgun Gothic", 12, "bold")


@dataclass(frozen=True)
class Param:
    key: str            # "section.attr"
    label: str
    lo: float
    hi: float
    step: float
    live: bool          # True=바로 적용, False=다시 시작할 때 적용
    group: str
    to_cfg: float = 1.0  # 패널 값 * to_cfg = cfg 값 (각도 deg→rad 등)
    tip: str = ""


G_TOW, G_SEA, G_PID, G_RESTART = "끌고 오기", "바다 · 줄 · 사람", "PID 계수", "다시 시작할 때 적용"
G_MISSION, G_SENSOR, G_MRESTART = "구조 임무", "근거리 센서", "임무 · 다시 시작할 때 적용"

PARAMS: list[Param] = [
    # ---- 실험 값 탭
    Param("pid.target_speed", "목표 속도 (m/s)", 0.2, 3.0, 0.1, True, G_TOW,
          tip="견인 속도. 빠를수록 얼굴이 잠기고 줄 힘이 커짐"),
    Param("pid.ramp_time", "출발 가속 시간 (s)", 0.0, 15.0, 0.5, True, G_TOW,
          tip="0에서 목표 속도까지 천천히 올리는 시간"),
    Param("robot.max_thrust", "최대 추진력 (N)", 100, 500, 10, True, G_TOW),
    Param("sim.goal_distance", "구조 거리 (m)", 10, 80, 5, True, G_TOW,
          tip="[끌고 오기만] 사람이 이만큼 이동하면 구조 성공"),
    Param("ocean.wave_height", "파도 높이 Hs (m)", 0.0, 2.0, 0.1, True, G_SEA, tip="1초에 걸쳐 부드럽게 바뀜"),
    Param("tether.break_force", "줄 끊어지는 힘 (N)", 100, 600, 5, True, G_SEA, tip="294N ≈ 30kg 무게"),
    Param("tether.stiffness", "줄 탄성 (N/m)", 200, 3000, 50, True, G_SEA, tip="클수록 덜 늘어나고 충격이 큼"),
    Param("tether.rest_length", "줄 길이 (m)", 1.5, 6.0, 0.1, True, G_SEA),
    Param("human.bow_wave_coef", "선수파 계수", 0.0, 2.0, 0.05, True, G_SEA,
          tip="빨리 끌 때 얼굴 앞에 물이 쌓이는 정도 (임시값)"),
    Param("pid.speed_kp", "속도 Kp", 0.0, 3.0, 0.05, True, G_PID),
    Param("pid.speed_ki", "속도 Ki", 0.0, 2.0, 0.05, True, G_PID),
    Param("pid.heading_kp", "방향 Kp", 0.0, 5.0, 0.1, True, G_PID),
    Param("pid.heading_kd", "방향 Kd", 0.0, 2.0, 0.05, True, G_PID),
    Param("ocean.peak_period", "파도 주기 (s)", 2.0, 10.0, 0.5, False, G_RESTART),
    Param("ocean.direction", "파도 방향 (도)", -180, 180, 15, False, G_RESTART, to_cfg=math.pi / 180,
          tip="0 = +x(동쪽)으로 파도가 진행"),
    Param("sim.seed", "시드", 0, 999, 1, False, G_RESTART, tip="같은 시드 = 같은 파도, 같은 익수자 위치"),
    # ---- 구조 임무 탭
    Param("mission.transit_speed", "이동 속도 (m/s)", 0.5, 2.5, 0.1, True, G_MISSION,
          tip="좌표(GPS)로 갈 때 속도"),
    Param("mission.gps_error", "GPS 오차 (m)", 0.0, 8.0, 0.5, True, G_MISSION,
          tip="로봇이 아는 자기 위치의 오차. 지도에서 노란 원 = 로봇이 생각하는 자기 위치"),
    Param("mission.handoff_standoff", "튜브 건넬 거리 (m)", 1.2, 2.5, 0.05, True, G_MISSION,
          tip="튜브를 건넬 때 로봇 중심 ~ 사람 거리. 튜브는 뱃머리 앞에 내밂"),
    Param("human.arm_reach", "팔 닿는 거리 (m)", 0.4, 1.0, 0.05, True, G_MISSION,
          tip="어깨에서 손끝까지. 튜브가 이 안에 오면 사람이 잡음"),
    Param("thermal.range", "열화상 거리 (m)", 5, 40, 1, True, G_SENSOR, tip="머리가 물 밖에 있어야 보임. 방향만 앎"),
    Param("thermal.fov", "열화상 시야각 (도)", 20, 120, 5, True, G_SENSOR),
    Param("sonar.range", "소나 거리 (m)", 5, 40, 1, True, G_SENSOR, tip="물속 몸통. 거리+방향. 가짜 탐지 있음"),
    Param("sonar.false_rate", "소나 가짜 탐지", 0.0, 2.0, 0.1, True, G_SENSOR,
          tip="핑 1번당 가짜 탐지 평균 개수 (파고 1m 기준, 파고에 비례)"),
    Param("mission.victim_distance", "익수자 거리 (m)", 10, 45, 1, False, G_MRESTART,
          tip="기지에서 익수자까지 평균 거리 (시드마다 ±20%)"),
    Param("mission.report_error", "목격 오차 (m)", 0.0, 15.0, 0.5, False, G_MRESTART,
          tip="지도의 빨간 점선 원 = 목격 지점과 그 오차"),
]

PLAYBACK = [("4배 빠르게", 0.25), ("2배 빠르게", 0.5), ("실시간", 1.0),
            ("2배 느리게", 2.0), ("4배 느리게", 4.0), ("8배 느리게", 8.0)]
CAMERAS = [("사람", 6), ("로봇", 7), ("위에서", 8)]
MODES = [("끌고 오기만", "tow"), ("전체 구조 임무 (좌표 → 탐지 → 견인)", "mission")]
# 결과 표: (기록 열 이름, 표 제목, 너비). 기록 파일(run_log)의 한글 열을 그대로 보여 줌
RESULT_COLS = [("회차", "회차", 40), ("시각", "시각", 64), ("모드", "모드", 66), ("결과", "결과", 78),
               ("시드", "시드", 40), ("견인 목표 속도(m/s)", "견인속도", 58), ("파고(m)", "파고", 40),
               ("걸린 시간(s)", "시간 s", 54), ("얼굴 물 밖 비율(%)", "얼굴밖 %", 62),
               ("최대 줄 힘(N)", "최대줄힘", 62), ("비고", "비고", 190)]
TABLE_MAX_ROWS = 300      # 표에는 최근 기록만 (파일에는 전부)
PHASE_KO = {"WAIT_GOAL": "좌표 기다림", "TRANSIT": "좌표로 이동(GPS)", "SEARCH": "주변 탐색", "APPROACH": "직접 감지 접근",
            "HANDOFF": "튜브 건네기", "TOW": "견인", "DONE": "끝", "TOW_ONLY": "끌고 오기"}

# 지도 범위 (월드 m) — 물 표면 그림과 같은 범위
MAP_X = (-20.0, 70.0)
MAP_Y = (-30.0, 30.0)
MAP_W, MAP_H = 600, 400


def get_cfg(cfg: TowConfig, key: str):
    sec, attr = key.split(".")
    return getattr(getattr(cfg, sec), attr)


def set_cfg(cfg: TowConfig, key: str, value):
    sec, attr = key.split(".")
    cur = getattr(getattr(cfg, sec), attr)
    setattr(getattr(cfg, sec), attr, int(round(value)) if isinstance(cur, int) else float(value))


class ControlPanel:
    def __init__(self, cfg: TowConfig, defaults: TowConfig, results_dir: Path, mode: str = "tow"):
        self.cfg, self.defaults, self.results_dir = cfg, defaults, Path(results_dir)
        self.closed = False
        self.actions: list[str] = []        # "restart", "pause" — 메인 루프가 꺼내 감
        self.changed: set[str] = set()      # 바뀐 '바로 적용' 값
        self.goal_requests: list[tuple[float, float]] = []   # 지도 클릭
        self._suspend = False
        self.log = RunLog(self.results_dir)
        self.mode_applied = mode

        root = tk.Tk()
        root.title("AirwayTow 실험 설정")
        root.geometry("+0+0")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.option_add("*Font", FONT)
        style = ttk.Style(root)
        style.configure("Treeview", font=FONT, rowheight=22)
        style.configure("Treeview.Heading", font=FONT_B)
        style.configure("TButton", font=FONT_B, padding=4)
        style.configure("TNotebook.Tab", font=FONT_B, padding=(10, 4))
        self.root = root

        outer = ttk.Frame(root, padding=8)
        outer.pack(fill="both", expand=True)

        # ---- 상단: 상태 + 버튼 + 보기
        top = ttk.Frame(outer)
        top.pack(fill="x")
        ttk.Label(top, text="AirwayTow 실험 설정", font=FONT_TITLE).pack(side="left")
        self.status = tk.StringVar(value="준비")
        ttk.Label(top, textvariable=self.status, foreground="#1a5fb4", font=FONT_B).pack(side="right")
        btns = ttk.Frame(outer)
        btns.pack(fill="x", pady=(6, 4))
        self.pause_btn = ttk.Button(btns, text="⏸ 일시정지", command=lambda: self.actions.append("pause"))
        self.pause_btn.pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(btns, text="↺ 적용하고 다시 시작", command=lambda: self.actions.append("restart")).pack(
            side="left", expand=True, fill="x", padx=2)
        ttk.Button(btns, text="기본값 복원", command=self._reset_defaults).pack(side="left", expand=True, fill="x", padx=2)

        view = ttk.Frame(outer)
        view.pack(fill="x", pady=(2, 4))
        ttk.Label(view, text="재생").pack(side="left")
        self.playback = tk.StringVar(value="실시간")
        ttk.Combobox(view, textvariable=self.playback, values=[p[0] for p in PLAYBACK], state="readonly",
                     width=9).pack(side="left", padx=(4, 10))
        ttk.Label(view, text="카메라").pack(side="left")
        self.camera = tk.IntVar(value=6)
        for name, val in CAMERAS:
            ttk.Radiobutton(view, text=name, value=val, variable=self.camera).pack(side="left", padx=1)
        self.show_graphs = tk.BooleanVar(value=True)
        ttk.Checkbutton(view, text="그래프", variable=self.show_graphs).pack(side="left", padx=(10, 0))
        self.auto_restart = tk.BooleanVar(value=True)
        ttk.Checkbutton(view, text="끝나면 자동 재시작", variable=self.auto_restart).pack(side="left", padx=(6, 0))

        self.pending = tk.StringVar(value="")
        ttk.Label(outer, textvariable=self.pending, foreground="#c01c28", font=FONT_B).pack(anchor="w")

        # ---- 탭
        nb = ttk.Notebook(outer)
        nb.pack(fill="both", expand=True, pady=(2, 0))
        self.notebook = nb
        # 탭 두 개: [실험 설정 · 지도] (모드·지도·모든 값, 스크롤) / [실험 결과]
        tab_set, tab_res = ttk.Frame(nb), ttk.Frame(nb, padding=6)
        nb.add(tab_set, text="실험 설정 · 지도")
        nb.add(tab_res, text="실험 결과")
        tab_mission = _ScrollFrame(tab_set).inner

        self.vars: dict[str, tk.Variable] = {}

        mtop = ttk.Frame(tab_mission)
        mtop.pack(fill="x")
        ttk.Label(mtop, text="모드", font=FONT_B).pack(side="left")
        self.mode = tk.StringVar(value=mode)
        for name, val in MODES:
            ttk.Radiobutton(mtop, text=name, value=val, variable=self.mode).pack(side="left", padx=4)
        self.mode.trace_add("write", lambda *_a: self._update_pending() if hasattr(self, "applied_restart") else None)
        self.map = tk.Canvas(tab_mission, width=MAP_W, height=MAP_H, background="#1d4f7a", highlightthickness=1,
                             highlightbackground="#888", cursor="crosshair")
        self.map.pack(pady=(4, 2))
        self.map.bind("<Button-1>", self._on_map_click)
        self.map_hint = tk.StringVar(value="")
        ttk.Label(tab_mission, textvariable=self.map_hint, foreground="#1a5fb4", font=FONT_S).pack(anchor="w")
        opts = ttk.Frame(tab_mission)
        opts.pack(fill="x", pady=(2, 2))
        self.show_truth = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="실제 익수자 위치 보기 (정답)", variable=self.show_truth).pack(side="left")
        self.auto_report = tk.BooleanVar(value=bool(cfg.mission.auto_report))
        ttk.Checkbutton(opts, text="자동 신고 (클릭 없이 목격 지점으로 출발)", variable=self.auto_report).pack(
            side="left", padx=(10, 0))
        ros = ttk.Frame(tab_mission)
        ros.pack(fill="x", pady=(0, 4))
        self.ros_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(ros, text="ROS2 연결 (rviz2)", variable=self.ros_enabled).pack(side="left")
        self.ros_status = tk.StringVar(value="꺼짐  —  WSL에서 ros_bridge_wsl.bat 실행 후 체크")
        ttk.Label(ros, textvariable=self.ros_status, foreground="#555", font=FONT_S).pack(side="left", padx=6)
        # 값: 두 열. 왼쪽 = 구조 임무 → 끌고 오기 → 바다, 오른쪽 = 센서 → PID → 다시 시작할 때 적용
        cols = ttk.Frame(tab_mission)
        cols.pack(fill="x")
        left, right = ttk.Frame(cols), ttk.Frame(cols)
        left.pack(side="left", fill="both", expand=True, padx=(0, 4), anchor="n")
        right.pack(side="left", fill="both", expand=True, padx=(4, 0), anchor="n")
        layout = [(left, [G_MISSION, G_MRESTART, G_TOW, G_SEA]), (right, [G_SENSOR, G_PID, G_RESTART])]
        live = {p.group: p.live for p in PARAMS}
        frames: dict[str, ttk.LabelFrame] = {}
        for col, groups in layout:
            for g in groups:
                frames[g] = ttk.LabelFrame(col, text=g + ("  (바로 적용)" if live[g] else ""), padding=4)
                frames[g].pack(fill="x", pady=3)
                frames[g].columnconfigure(1, weight=1)
        for p in PARAMS:
            self._add_slider(frames[p.group], p)
        ttk.Label(tab_mission, text="단축키(뷰어 창): Space 일시정지 · Backspace 다시 시작 · ↑/↓ 목표 속도 · "
                                    "=/- 재생 속도 · 6/7/8 카메라 · 9 그래프",
                  foreground="#555", font=FONT_S).pack(anchor="w", pady=(6, 0))

        # 결과 탭: 위 요약, 가운데 표, 아래 버튼
        self.summary = tk.StringVar(value="")
        ttk.Label(tab_res, textvariable=self.summary, font=FONT_B).pack(side="top", anchor="w", pady=(0, 4))
        rb = ttk.Frame(tab_res)
        rb.pack(side="bottom", fill="x", pady=(4, 0))
        self.tree = ttk.Treeview(tab_res, columns=[c[0] for c in RESULT_COLS], show="headings", height=16)
        for key, name, w in RESULT_COLS:
            self.tree.heading(key, text=name)
            self.tree.column(key, width=w, anchor="w" if key == "비고" else "center", stretch=key == "비고")
        self.tree.tag_configure("ok", background="#dff5e3")
        self.tree.tag_configure("fail", background="#fbe0de")
        sb = ttk.Scrollbar(tab_res, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        ttk.Button(rb, text="기록 파일 열기 (엑셀)", command=lambda: self._open(self.log.path)).pack(side="left")
        ttk.Button(rb, text="기록 폴더 열기", command=lambda: self._open(self.log.path.parent)).pack(
            side="left", padx=4)
        self.save_msg = tk.StringVar(value="")
        ttk.Label(rb, textvariable=self.save_msg, foreground="#555", font=FONT_S).pack(side="left", padx=6)
        for r in self.log.rows[-TABLE_MAX_ROWS:]:       # 지난 기록도 보여 줌
            self._insert_row(r)
        self._update_summary()

        self.applied_restart = {p.key: self._panel_value_from_cfg(p) for p in PARAMS if not p.live}
        self._update_pending()
        self.update_map(None)

    # ------------------------------------------------------------------ 위젯
    def _add_slider(self, parent, p: Param):
        row = len(parent.grid_slaves()) // 3
        if self._is_int(p):     # 정수 값은 "300.0" 대신 "300"으로 보이게
            var = tk.IntVar(value=int(round(self._panel_value_from_cfg(p))))
        else:
            var = tk.DoubleVar(value=self._panel_value_from_cfg(p))
        self.vars[p.key] = var
        lab = ttk.Label(parent, text=p.label, width=16)
        lab.grid(row=row, column=0, sticky="w")
        digits = 0 if p.step >= 1 else (1 if p.step >= 0.1 else 2)
        scale = tk.Scale(parent, from_=p.lo, to=p.hi, resolution=p.step, orient="horizontal", variable=var,
                         showvalue=False, length=110, highlightthickness=0, sliderlength=16)
        scale.grid(row=row, column=1, sticky="ew", padx=4)
        spin = ttk.Spinbox(parent, from_=p.lo, to=p.hi, increment=p.step, textvariable=var, width=6,
                           format=f"%.{digits}f")
        spin.grid(row=row, column=2, sticky="e")
        if p.tip:
            for w in (lab, scale):
                _Tooltip(w, p.tip)
        var.trace_add("write", lambda *_a, k=p.key: self._on_var(k))

    @staticmethod
    def _is_int(p: Param) -> bool:
        return float(p.step).is_integer() and float(p.lo).is_integer()

    def _set_var(self, p: Param, v: float):
        self.vars[p.key].set(int(round(v)) if self._is_int(p) else v)

    def _panel_value_from_cfg(self, p: Param) -> float:
        return round(get_cfg(self.cfg, p.key) / p.to_cfg, 6)

    def _read(self, key: str) -> float | None:
        try:
            return float(self.vars[key].get())
        except (tk.TclError, ValueError):
            return None         # 칸에 숫자를 입력하는 중

    def _on_var(self, key: str):
        if self._suspend:
            return
        p = next(q for q in PARAMS if q.key == key)
        if p.live:
            self.changed.add(key)
        else:
            self._update_pending()

    def _update_pending(self):
        diff = [p.label for p in PARAMS if not p.live
                and self._read(p.key) is not None and abs(self._read(p.key) - self.applied_restart[p.key]) > 1e-9]
        if self.mode.get() != self.mode_applied:
            diff.insert(0, "모드")
        self.pending.set(("변경됨: " + ", ".join(diff) + "  →  [적용하고 다시 시작] 을 누르세요") if diff else "")

    # ------------------------------------------------------------------ 지도
    def _w2c(self, x: float, y: float) -> tuple[float, float]:
        cx = (x - MAP_X[0]) / (MAP_X[1] - MAP_X[0]) * MAP_W
        cy = (MAP_Y[1] - y) / (MAP_Y[1] - MAP_Y[0]) * MAP_H
        return cx, cy

    def _c2w(self, cx: float, cy: float) -> tuple[float, float]:
        x = MAP_X[0] + cx / MAP_W * (MAP_X[1] - MAP_X[0])
        y = MAP_Y[1] - cy / MAP_H * (MAP_Y[1] - MAP_Y[0])
        return x, y

    def _px(self, meters: float) -> float:
        return meters / (MAP_X[1] - MAP_X[0]) * MAP_W

    def _on_map_click(self, e):
        x, y = self._c2w(e.x, e.y)
        self.goal_requests.append((x, y))

    def update_map(self, st: dict | None):
        """상태 dict(bridge.messages.build_state 와 같은 모양)로 지도를 다시 그림."""
        c = self.map
        c.delete("all")
        for gx in range(int(MAP_X[0]), int(MAP_X[1]) + 1, 10):
            x0, _ = self._w2c(gx, 0)
            c.create_line(x0, 0, x0, MAP_H, fill="#2f6690")
            c.create_text(x0 + 2, MAP_H - 8, text=f"{gx}", anchor="w", fill="#9cc3e6", font=FONT_S)
        for gy in range(int(MAP_Y[0]), int(MAP_Y[1]) + 1, 10):
            _, y0 = self._w2c(0, gy)
            c.create_line(0, y0, MAP_W, y0, fill="#2f6690")
            c.create_text(4, y0 - 7, text=f"{gy}", anchor="w", fill="#9cc3e6", font=FONT_S)
        c.create_text(MAP_W - 6, 10, text="10 m 격자 · 위 = 북(+y)", anchor="e", fill="#9cc3e6", font=FONT_S)
        if st is None:
            return

        def circle(x, y, r_px, **kw):
            cx, cy = self._w2c(x, y)
            return c.create_oval(cx - r_px, cy - r_px, cx + r_px, cy + r_px, **kw)

        bx, by = st["base"]
        circle(bx, by, self._px(st["home_radius"]), outline="#4cd964", width=2, dash=(4, 2))
        cbx, cby = self._w2c(bx, by)
        c.create_text(cbx, cby, text="기지", fill="#4cd964", font=FONT_B)
        rx, ry, ryaw = st["robot"]
        mission = st.get("mode") == "mission"

        if mission:
            s = st["sensors"]
            for rng, fov, col, dash in ((s["sonar_range"], s["sonar_fov"], "#5aa9ff", (2, 3)),
                                        (s["thermal_range"], s["thermal_fov"], "#ff8a3d", ())):
                cx, cy = self._w2c(rx, ry)
                r = self._px(rng)
                c.create_arc(cx - r, cy - r, cx + r, cy + r, start=math.degrees(ryaw) - fov / 2, extent=fov,
                             style="pieslice", outline=col, dash=dash)
            sx, sy = st["sighting"]
            circle(sx, sy, self._px(st["sighting_radius"]), outline="#ff6b6b", dash=(3, 3))
            csx, csy = self._w2c(sx, sy)
            c.create_text(csx, csy - self._px(st["sighting_radius"]) - 8, text="목격 지점", fill="#ff6b6b",
                          font=FONT_S)
            if st["search_path"]:
                pts = [v for p in st["search_path"] for v in self._w2c(*p)]
                if len(pts) >= 4:
                    c.create_line(*pts, fill="#c8c8ff", dash=(2, 2))
            for cand in st["candidates"]:
                circle(cand[0], cand[1], 4, outline="#cccccc")
            for d in st["detections"]:
                col = "#ff8a3d" if d["sensor"] == "thermal" else "#5aa9ff"
                circle(d["x"], d["y"], 2.5, fill=col, outline="")
            if st["goal"]:
                gx, gy = self._w2c(*st["goal"])
                c.create_line(gx - 7, gy - 7, gx + 7, gy + 7, fill="#ff3030", width=3)
                c.create_line(gx - 7, gy + 7, gx + 7, gy - 7, fill="#ff3030", width=3)
                c.create_text(gx + 12, gy - 12, text="목표", fill="#ff3030", anchor="sw", font=FONT_B)
            if st["track"]:
                circle(st["track"][0], st["track"][1], 8, outline="#3cff6e", width=3)
            gxp, gyp = self._w2c(*st["robot_gps"])
            rxp, ryp = self._w2c(rx, ry)
            c.create_line(rxp, ryp, gxp, gyp, fill="#ffe14d", dash=(1, 2))
            c.create_oval(gxp - 6, gyp - 6, gxp + 6, gyp + 6, outline="#ffe14d", width=2)

        if self.show_truth.get() or not mission:
            hx, hy = st["human"]
            circle(hx, hy, 5, fill="#ffb3a7", outline="white")
            chx, chy = self._w2c(hx, hy)
            c.create_text(chx, chy + 14, text="실제 익수자", fill="#ffb3a7", anchor="n", font=FONT_S)
        if st["tether"]["attached"] and not st["tether"]["broken"]:
            c.create_line(*self._w2c(rx, ry), *self._w2c(*st["human"]), fill="#ffd60a", width=2)

        # 로봇 (실제 위치, 방향 삼각형)
        size = 9
        pts = []
        for ang, rad in ((0, size * 1.4), (2.5, size), (-2.5, size)):
            a = ryaw + ang
            px, py = self._w2c(rx, ry)
            pts += [px + rad * math.cos(a), py - rad * math.sin(a)]
        c.create_polygon(*pts, fill="#ff8c1a", outline="white")

        legend = ("▲로봇  ○노랑=GPS 위치  ✕목표  ◯초록=감지한 사람  점: 주황 열화상·파랑 소나"
                  if mission else "끌고 오기만 모드 (지도 클릭은 구조 임무 모드에서 사용)")
        c.create_rectangle(0, 0, MAP_W, 18, fill="#0e2b45", outline="")
        c.create_text(4, 9, text=legend, anchor="w", fill="white", font=FONT_S)
        if mission:
            ph = PHASE_KO.get(st["phase"], st["phase"])
            hint = f"단계: {ph}"
            if st["phase"] == "WAIT_GOAL":
                hint += "   ← 지도에서 익수자가 있을 것 같은 곳을 클릭하세요 (rviz2 2D Goal Pose 도 가능)"
            self.map_hint.set(hint)
        else:
            self.map_hint.set("")

    # ------------------------------------------------------------------ 메인 루프에서 부름
    def poll(self) -> bool:
        """창 이벤트 처리. 창이 닫혔으면 False."""
        if self.closed:
            return False
        try:
            self.root.update()
        except tk.TclError:
            self.closed = True
        return not self.closed

    def take_live_changes(self) -> dict[str, float]:
        """바뀐 '바로 적용' 값을 cfg 단위로 돌려줍니다."""
        out = {}
        for key in list(self.changed):
            v = self._read(key)
            if v is None:
                continue
            p = next(q for q in PARAMS if q.key == key)
            v = min(max(v, p.lo), p.hi)
            out[key] = v * p.to_cfg
            self.changed.discard(key)
        return out

    def take_goal_requests(self) -> list[tuple[float, float]]:
        out, self.goal_requests = self.goal_requests, []
        return out

    def restart_values(self) -> dict[str, float]:
        """'다시 시작할 때 적용' 값을 cfg 단위로 돌려주고, 적용된 것으로 표시합니다."""
        out = {}
        for p in PARAMS:
            if p.live:
                continue
            v = self._read(p.key)
            if v is None:
                v = self.applied_restart[p.key]
            v = min(max(v, p.lo), p.hi)
            self.applied_restart[p.key] = v
            out[p.key] = v * p.to_cfg
        self.mode_applied = self.mode.get()
        self._update_pending()
        return out

    def set_value(self, key: str, cfg_value: float, mark_applied: bool = False):
        """프로그램 쪽에서 값을 바꿨을 때 (단축키, 자동 시드 증가 등) 패널 표시만 맞춥니다."""
        p = next(q for q in PARAMS if q.key == key)
        v = round(cfg_value / p.to_cfg, 6)
        self._suspend = True
        self._set_var(p, v)
        self._suspend = False
        if mark_applied and not p.live:
            self.applied_restart[key] = v
        self._update_pending()

    def playback_slowmo(self) -> float:
        return dict(PLAYBACK).get(self.playback.get(), 1.0)

    def set_playback(self, slowmo: float):
        name = min(PLAYBACK, key=lambda x: abs(math.log(x[1] / slowmo)))[0]
        self.playback.set(name)

    def set_status(self, text: str, paused: bool):
        self.status.set(text)
        self.pause_btn.configure(text="▶ 계속" if paused else "⏸ 일시정지")

    def set_ros_status(self, text: str):
        self.ros_status.set(text)

    def add_result(self, res: dict) -> dict[str, str]:
        """한 판 결과(영어 키 dict)를 기록 파일에 붙이고 표에 보여 줌. 붙인 기록 줄을 돌려줌."""
        row = self.log.add(res)
        self._insert_row(row)
        self.tree.yview_moveto(1.0)
        self._update_summary()
        return row

    @staticmethod
    def _note(r: dict) -> str:
        if r.get("모드") == "구조 임무":
            parts = [f"{name} {r[key]}s" for name, key in (("감지", "감지까지(s)"), ("튜브 잡기", "튜브 잡기까지(s)"),
                                                          ("연결", "줄 연결까지(s)")) if r.get(key)]
            note = " · ".join(parts)
        else:
            note = f"점수 {r['점수(끌고 오기)']}" if r.get("점수(끌고 오기)") else ""
        return note + ("  (도중에 값 바꿈)" if r.get("도중에 값 바꿈") else "")

    def _insert_row(self, r: dict):
        values = [self._note(r) if key == "비고" else r.get(key, "") for key, _n, _w in RESULT_COLS]
        self.tree.insert("", "end", values=values, tags=("ok" if r.get("성공") == "O" else "fail",))
        kids = self.tree.get_children()
        if len(kids) > TABLE_MAX_ROWS:
            self.tree.delete(kids[0])

    def _update_summary(self):
        rows = self.log.rows
        n = len(rows)
        ok = sum(r.get("성공") == "O" for r in rows)
        if n:
            last = rows[-1]
            self.summary.set(f"전체 {n}회  ·  성공 {ok}  ·  실패 {n - ok}  ·  성공률 {ok / n * 100:.0f}%"
                             f"      마지막: {last.get('회차')}회차 {last.get('결과')}")
        else:
            self.summary.set("아직 기록이 없습니다. 한 판이 끝날 때마다 여기에 자동으로 쌓입니다.")
        self.save_msg.set(self.log.error or f"자동 저장: {self.log.path}")

    def _open(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            self.save_msg.set("아직 기록 파일이 없습니다 (한 판이 끝나면 생김)")
            return
        if sys.platform == "win32":
            os.startfile(path)        # noqa: S606  (엑셀/탐색기로 열기)
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def right_edge(self) -> int:
        """패널 창의 오른쪽 끝 x 좌표 (테두리 포함)."""
        self.root.update_idletasks()
        return self.root.winfo_rootx() + self.root.winfo_width() + 8

    # ------------------------------------------------------------------ 내부
    def _reset_defaults(self):
        for p in PARAMS:
            self._set_var(p, round(get_cfg(self.defaults, p.key) / p.to_cfg, 6))

    def _on_close(self):
        self.closed = True
        self.root.destroy()

    def destroy(self):
        if not self.closed:
            self.closed = True
            try:
                self.root.destroy()
            except tk.TclError:
                pass


class _ScrollFrame:
    """세로 스크롤 영역. 내용은 self.inner 에 넣습니다. 높이는 화면에 맞춰 자동으로 정해짐 (마우스 휠로 스크롤)."""

    RESERVED_H = 260      # 화면 높이에서 패널 위쪽(제목·버튼·탭 머리)과 작업 표시줄 몫으로 남겨 둘 높이

    def __init__(self, parent):
        self.canvas = tk.Canvas(parent, highlightthickness=0, borderwidth=0)
        self.bar = ttk.Scrollbar(parent, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.inner = ttk.Frame(self.canvas, padding=6)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.bar.pack(side="left", fill="y")
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        self.canvas.bind("<Enter>", lambda _e: self.canvas.bind_all("<MouseWheel>", self._on_wheel))
        self.canvas.bind("<Leave>", lambda _e: self.canvas.unbind_all("<MouseWheel>"))

    def _on_inner(self, _e=None):
        w, h = self.inner.winfo_reqwidth(), self.inner.winfo_reqheight()
        max_h = max(300, self.canvas.winfo_screenheight() - self.RESERVED_H)
        self.canvas.configure(scrollregion=(0, 0, w, h), width=w, height=min(h, max_h))

    def _on_wheel(self, e):
        if self.canvas.yview() != (0.0, 1.0):          # 다 보이면 스크롤 안 함
            self.canvas.yview_scroll(int(-e.delta / 120) * 3, "units")


class _Tooltip:
    """마우스를 올리면 설명이 뜨는 작은 창."""

    def __init__(self, widget, text: str):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _e):
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, background="#ffffe0", relief="solid", borderwidth=1,
                 font=FONT, padx=4, pady=2).pack()

    def _hide(self, _e):
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None
