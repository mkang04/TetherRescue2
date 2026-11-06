"""시뮬레이션 상태 → 주고받는 메시지(JSON) / rviz2 마커 모양.

이 파일은 ROS2도 MuJoCo도 import하지 않습니다 (순수 파이썬).
- Windows 시뮬레이터: build_state() 로 상태 dict를 만들어 UDP로 보냄
- WSL ROS2 브리지: marker_specs() 로 마커 모양을 만들어 visualization_msgs/Marker 로 바꿈
- 설정 패널 지도도 같은 상태 dict를 그립니다

좌표: ROS 관례대로 "map" 좌표계, x=동(앞), y=북(왼쪽), 단위 m. 시뮬레이터 월드 좌표와 같음.
"""
from __future__ import annotations

import math
from typing import Any

PROTOCOL_VERSION = 1


def build_state(sim, mode: str) -> dict[str, Any]:
    """시뮬레이터(MissionSim) → 상태 dict (JSON으로 보낼 수 있는 값만)."""
    import numpy as np   # 시뮬레이터 쪽에서만 불림

    obs = sim._last_obs
    cfg = sim.cfg
    rp = sim.data.xipos[sim.robot_id]
    hp = sim.data.xipos[sim.torso_id]
    brain = sim.brain
    st: dict[str, Any] = {
        "type": "state", "v": PROTOCOL_VERSION, "t": round(float(obs["t"]), 3), "mode": mode,
        "phase": obs.get("phase", "TOW_ONLY"),
        "robot": [float(rp[0]), float(rp[1]), float(obs["yaw"])],
        "human": [float(hp[0]), float(hp[1])],
        "base": [float(v) for v in cfg.mission.base],
        "home_radius": cfg.mission.home_radius,
        "tether": {"attached": bool(sim.tether.attached), "broken": bool(obs["tether_broken"]),
                   "tension": round(float(obs["tension"]), 1), "break_force": cfg.tether.break_force},
        "airway_ok": bool(obs["airway_ok"]),
        "speed": round(float(obs["robot_speed"]), 3),
        "outcome": sim.outcome.reason if sim.outcome.done else "",
    }
    if mode == "mission":
        tr = brain.tracker
        st.update({
            "robot_gps": [float(obs["gps_x"]), float(obs["gps_y"])],
            "goal": obs["goal"],
            "sighting": [float(v) for v in sim.sighting],
            "sighting_radius": cfg.mission.report_error,
            "track": obs["track"],
            "candidates": [[float(c.pos[0]), float(c.pos[1])] for c in tr.candidates],
            "search_path": [[float(p[0]), float(p[1])] for p in brain.search_path],
            "detections": _detections_world(sim, np),
            "sensors": {"thermal_range": cfg.thermal.range, "thermal_fov": cfg.thermal.fov,
                        "sonar_range": cfg.sonar.range, "sonar_fov": cfg.sonar.fov},
        })
    return st


def _detections_world(sim, np) -> list[dict]:
    """최근 2초 탐지를 지도 위 점으로 (로봇이 생각하는 위치 = GPS 기준)."""
    out = []
    gx, gy = sim._last_obs["gps_x"], sim._last_obs["gps_y"]
    yaw = sim._last_obs["yaw"]
    for d in sim.recent_dets[-40:]:
        bw = yaw + d.bearing
        r = d.range if d.range is not None else min(sim.cfg.thermal.range, 12.0)
        out.append({"sensor": d.sensor, "x": float(gx + r * math.cos(bw)), "y": float(gy + r * math.sin(bw)),
                    "range_known": d.range is not None})
    return out


def goal_message(x: float, y: float, source: str) -> dict:
    return {"type": "goal", "v": PROTOCOL_VERSION, "x": float(x), "y": float(y), "source": source}


def parse_goal(msg: dict) -> tuple[float, float, str] | None:
    """받은 메시지가 목표 좌표면 (x, y, 출처), 아니면 None."""
    if msg.get("type") != "goal":
        return None
    try:
        x, y = float(msg["x"]), float(msg["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y, str(msg.get("source", "?"))


# --------------------------------------------------------------------------- rviz2 마커
def _m(ns, mid, mtype, x=0.0, y=0.0, z=0.0, yaw=0.0, scale=(1, 1, 1), color=(1, 1, 1, 1), points=None, text=""):
    return {"ns": ns, "id": mid, "type": mtype, "pose": [x, y, z, yaw], "scale": list(scale),
            "color": list(color), "points": points or [], "text": text}


def _arc(cx, cy, yaw, r, fov_deg, n=16, z=0.05):
    half = math.radians(fov_deg) / 2
    pts = [[cx, cy, z]]
    for i in range(n + 1):
        a = yaw - half + 2 * half * i / n
        pts.append([cx + r * math.cos(a), cy + r * math.sin(a), z])
    pts.append([cx, cy, z])
    return pts


def marker_specs(st: dict) -> list[dict]:
    """상태 dict → rviz2 마커 모양 목록 (ROS 메시지로 바꾸기 전 단계)."""
    ms = []
    rx, ry, ryaw = st["robot"]
    bx, by = st["base"]
    ms.append(_m("base", 0, "CYLINDER", bx, by, 0.0, scale=(2 * st["home_radius"],) * 2 + (0.05,),
                 color=(0.2, 0.9, 0.3, 0.25)))
    ms.append(_m("base", 1, "TEXT_VIEW_FACING", bx, by, 1.5, scale=(0, 0, 2.0), color=(1, 1, 1, 1), text="BASE"))
    ms.append(_m("robot", 0, "ARROW", rx, ry, 0.3, ryaw, scale=(3.0, 1.2, 1.2), color=(1.0, 0.5, 0.05, 1)))
    ms.append(_m("human", 0, "SPHERE", st["human"][0], st["human"][1], 0.1, scale=(1.2, 1.2, 1.2),
                 color=(0.95, 0.75, 0.6, 1)))
    if st["tether"]["attached"] and not st["tether"]["broken"]:
        ratio = min(1.0, st["tether"]["tension"] / max(st["tether"]["break_force"], 1e-6))
        ms.append(_m("tether", 0, "LINE_STRIP", scale=(0.08, 0, 0), color=(1.0, 1.0 - ratio, 0.1, 1),
                     points=[[rx, ry, 0.3], [st["human"][0], st["human"][1], 0.1]]))
    label = st["phase"] + (f"  [{st['outcome']}]" if st.get("outcome") else "")
    ms.append(_m("label", 0, "TEXT_VIEW_FACING", rx, ry + 3.0, 2.2, scale=(0, 0, 1.8), color=(1, 1, 1, 1), text=label))

    if st.get("mode") != "mission":
        return ms
    s = st["sensors"]
    ms.append(_m("thermal_fov", 0, "LINE_STRIP", scale=(0.05, 0, 0), color=(1.0, 0.4, 0.1, 0.8),
                 points=_arc(rx, ry, ryaw, s["thermal_range"], s["thermal_fov"])))
    ms.append(_m("sonar_fov", 0, "LINE_STRIP", scale=(0.04, 0, 0), color=(0.2, 0.6, 1.0, 0.6),
                 points=_arc(rx, ry, ryaw, s["sonar_range"], s["sonar_fov"])))
    gx, gy = st["robot_gps"]
    ms.append(_m("robot_gps", 0, "SPHERE", gx, gy, 0.3, scale=(1.0, 1.0, 1.0), color=(1.0, 0.9, 0.2, 0.8)))
    sx, sy = st["sighting"]
    ms.append(_m("sighting", 0, "CYLINDER", sx, sy, 0.0, scale=(2 * st["sighting_radius"],) * 2 + (0.03,),
                 color=(1.0, 0.3, 0.3, 0.15)))
    if st.get("goal"):
        ms.append(_m("goal", 0, "CYLINDER", st["goal"][0], st["goal"][1], 1.0, scale=(0.8, 0.8, 2.0),
                     color=(1.0, 0.1, 0.1, 0.9)))
    if st.get("track"):
        ms.append(_m("track", 0, "SPHERE", st["track"][0], st["track"][1], 0.4, scale=(1.8, 1.8, 1.8),
                     color=(0.1, 1.0, 0.3, 0.7)))
    if st.get("search_path"):
        ms.append(_m("search", 0, "LINE_STRIP", scale=(0.06, 0, 0), color=(0.8, 0.8, 1.0, 0.6),
                     points=[[p[0], p[1], 0.05] for p in st["search_path"]]))
    pts_t = [[d["x"], d["y"], 0.2] for d in st["detections"] if d["sensor"] == "thermal"]
    pts_s = [[d["x"], d["y"], 0.2] for d in st["detections"] if d["sensor"] == "sonar"]
    ms.append(_m("det_thermal", 0, "POINTS", scale=(0.4, 0.4, 0), color=(1.0, 0.4, 0.1, 1), points=pts_t))
    ms.append(_m("det_sonar", 0, "POINTS", scale=(0.3, 0.3, 0), color=(0.3, 0.7, 1.0, 1), points=pts_s))
    return ms
