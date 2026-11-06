"""PID로 한 판 끌고 오기 → 결과 출력 + CSV 기록 + 그래프 저장.

예)  python scripts/run_tow.py --speed 1.2 --wave 1.0 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tether_rescue.config import load_config  # noqa: E402
from tether_rescue.metrics import metrics_dict, run_pid_episode, save_log_csv  # noqa: E402
from tether_rescue.tow_env import LOG_FIELDS  # noqa: E402


def plot_log(sim, path: Path, title: str):
    log = sim.log.array()
    c = {k: i for i, k in enumerate(LOG_FIELDS)}
    t = log[:, c["t"]]
    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(t, log[:, c["target_speed"]], "k--", lw=1, label="target")
    ax[0].plot(t, log[:, c["robot_speed"]], label="robot")
    ax[0].plot(t, log[:, c["human_speed"]], label="human (rel. water)")
    ax[0].set_ylabel("speed [m/s]")
    ax[0].legend(loc="lower right")
    ax[1].plot(t, log[:, c["tension"]], color="tab:red")
    ax[1].axhline(sim.cfg.tether.break_force, color="k", ls=":", lw=1, label="break force")
    ax[1].set_ylabel("tension [N]")
    ax[1].legend(loc="upper right")
    ax[2].plot(t, log[:, c["mouth_clearance"]] * 100, color="tab:blue")
    ax[2].axhline(0, color="k", lw=1)
    bad = log[:, c["airway_ok"]] < 0.5
    ax[2].fill_between(t, -20, 20, where=bad, color="tab:red", alpha=0.15, label="face submerged")
    ax[2].set_ylim(-20, 20)
    ax[2].set_ylabel("mouth clearance [cm]")
    ax[2].legend(loc="upper right")
    ax[3].plot(t, log[:, c["throttle"]], label="throttle")
    ax[3].plot(t, log[:, c["rudder"]], label="rudder")
    ax[3].plot(t, np.degrees(log[:, c["roll"]]) / 30, label="roll / 30deg")
    ax[3].set_ylabel("command")
    ax[3].set_xlabel("time [s]")
    ax[3].legend(loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="AirwayTow PID 한 판 실행")
    ap.add_argument("--config", default=None, help="설정 YAML (기본: config/tow_default.yaml)")
    ap.add_argument("--speed", type=float, default=None, help="목표 속도 [m/s]")
    ap.add_argument("--wave", type=float, default=None, help="유의파고 [m]")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "run_tow"))
    args = ap.parse_args()

    ov: dict = {}
    if args.speed is not None:
        ov.setdefault("pid", {})["target_speed"] = args.speed
    if args.wave is not None:
        ov.setdefault("ocean", {})["wave_height"] = args.wave
    cfg = load_config(args.config, ov)
    seed = cfg.sim.seed if args.seed is None else args.seed

    t0 = time.perf_counter()
    sim, m = run_pid_episode(cfg, seed)
    wall = time.perf_counter() - t0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"v{cfg.pid.target_speed:.2f}_hs{cfg.ocean.wave_height:.1f}_s{seed}"
    save_log_csv(sim, out / f"log_{tag}.csv")
    plot_log(sim, out / f"plot_{tag}.png",
             f"AirwayTow PID  target={cfg.pid.target_speed} m/s  Hs={cfg.ocean.wave_height} m  seed={seed}  -> {m.reason}")
    md = metrics_dict(m)
    md["wall_time_s"] = wall
    md["realtime_factor"] = m.rescue_time / wall if wall > 0 else 0
    (out / f"metrics_{tag}.json").write_text(json.dumps(md, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"결과: {m.reason}  (성공={m.success})")
    print(f"  걸린 시간      {m.rescue_time:6.1f} s")
    print(f"  평균 속도      {m.mean_speed:6.2f} m/s")
    print(f"  얼굴 물 밖 비율 {m.airway_ratio * 100:6.1f} %")
    print(f"  최대 줄 힘     {m.max_tension:6.1f} N  (끊어짐 {cfg.tether.break_force:.0f} N)")
    print(f"  최대 기울기    {m.max_tilt_deg:6.1f} deg")
    print(f"  종합 점수      {m.score:6.1f} / 100 (임시 공식)")
    print(f"  시뮬 속도      실시간의 {md['realtime_factor']:.1f}배")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
