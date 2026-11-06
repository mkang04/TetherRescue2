"""끌고 오기 성적 매기기 + PID로 한 판 돌리기."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import TowConfig
from .tow.pid import TowPIDController
from .tow_env import LOG_FIELDS, TowSim

REFERENCE_SPEED = 2.0   # 점수 계산용 기준 속도 [m/s] (임시값, 02 명세서와 맞출 것)


@dataclass
class TowMetrics:
    reason: str
    success: bool
    rescue_time: float          # 끝날 때까지 걸린 시간 [s]
    progress: float             # 사람이 이동한 거리 [m]
    mean_speed: float           # 평균 속도 = 이동 거리 / 시간 [m/s]
    airway_ratio: float         # 입이 물 밖에 있던 시간 비율 0~1
    max_tension: float          # 가장 세게 당겨진 힘 [N]
    max_tilt_deg: float         # 로봇이 가장 많이 기운 각 [deg]
    score: float                # 종합 점수 0~100 (임시 공식)


def compute_metrics(sim: TowSim) -> TowMetrics:
    log = sim.log.array()
    col = {k: i for i, k in enumerate(LOG_FIELDS)}
    if len(log) == 0:
        return TowMetrics(sim.outcome.reason, False, 0, 0, 0, 0, 0, 0, 0)
    t_end = float(log[-1, col["t"]])
    progress = float(log[-1, col["progress"]])
    airway = float(log[:, col["airway_ok"]].mean())
    max_tension = float(log[:, col["tension"]].max())
    tilt = np.degrees(np.arccos(np.clip(np.cos(log[:, col["roll"]]) * np.cos(log[:, col["pitch"]]), -1, 1)))
    mean_speed = progress / t_end if t_end > 0 else 0.0
    goal = sim.cfg.sim.goal_distance

    # 임시 종합 점수: 얼굴 보호 70점 + 빠르기 30점
    #   - 줄 끊김·뒤집힘은 0점
    #   - 시간 초과는 간 거리 비율만큼만 인정
    if sim.outcome.reason in ("tether_broken", "capsized"):
        score = 0.0
    else:
        speed_part = min(1.0, mean_speed / REFERENCE_SPEED)
        done_part = min(1.0, max(0.0, progress / goal))
        speed_part = max(0.0, speed_part)
        score = done_part * (70.0 * airway + 30.0 * speed_part)

    return TowMetrics(sim.outcome.reason, sim.outcome.success, t_end, progress, mean_speed,
                      airway, max_tension, float(tilt.max()), score)


def run_pid_episode(cfg: TowConfig, seed: int | None = None) -> tuple[TowSim, TowMetrics]:
    """PID로 끝날 때까지 한 판 돌립니다."""
    sim = TowSim(cfg)
    obs = sim.reset(seed)
    ctrl = TowPIDController(cfg.pid, cfg.robot.max_reverse, goal_heading=0.0)
    dt = 1.0 / cfg.sim.control_hz
    while not sim.done:
        thr, rud = ctrl.act(obs["t"], dt, obs["robot_speed"], obs["yaw"], obs["yaw_rate"])
        obs = sim.step(thr, rud, target_speed=ctrl.target_speed(obs["t"]))
    return sim, compute_metrics(sim)


def save_log_csv(sim: TowSim, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(LOG_FIELDS)
        w.writerows(sim.log.rows)


def metrics_dict(m: TowMetrics) -> dict:
    return asdict(m)
