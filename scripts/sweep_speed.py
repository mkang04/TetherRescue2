"""목표 속도를 바꿔 가며 '빠르게 vs 안전하게' 비교 (M2 그래프의 첫 버전).

시드 3개로 세 번씩 돌려 평균을 냅니다.
예)  python scripts/sweep_speed.py --speeds 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 --wave 1.0
"""
from __future__ import annotations

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tether_rescue.config import load_config  # noqa: E402
from tether_rescue.metrics import run_pid_episode  # noqa: E402


def _one(job):
    cfg_path, speed, wave, seed = job
    cfg = load_config(cfg_path, {"pid": {"target_speed": speed}, "ocean": {"wave_height": wave}})
    _, m = run_pid_episode(cfg, seed)
    return speed, seed, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--speeds", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ap.add_argument("--wave", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results" / "sweep_speed"))
    args = ap.parse_args()

    jobs = [(args.config, v, args.wave, s) for v in args.speeds for s in args.seeds]
    print(f"{len(jobs)}판 실행 중 (프로세스 {args.workers}개)...")
    with Pool(args.workers) as pool:
        results = pool.map(_one, jobs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for speed, seed, m in results:
        rows.append({"target_speed": speed, "seed": seed, "reason": m.reason, "success": int(m.success),
                     "rescue_time": m.rescue_time, "mean_speed": m.mean_speed, "airway_ratio": m.airway_ratio,
                     "max_tension": m.max_tension, "max_tilt_deg": m.max_tilt_deg, "score": m.score})
    with open(out / f"sweep_hs{args.wave:.1f}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def agg(key):
        return np.array([np.mean([r[key] for r in rows if r["target_speed"] == v]) for v in args.speeds])

    v = np.array(args.speeds)
    air, tmax, succ = agg("airway_ratio"), agg("max_tension"), agg("success")
    rtime, score = agg("rescue_time"), agg("score")

    print(f"\n{'목표속도':>8} {'성공률':>6} {'얼굴밖%':>7} {'시간s':>7} {'최대줄힘N':>9} {'점수':>6}")
    for i, sp in enumerate(v):
        print(f"{sp:8.2f} {succ[i]:6.2f} {air[i] * 100:7.1f} {rtime[i]:7.1f} {tmax[i]:9.1f} {score[i]:6.1f}")
    best = v[int(np.argmax(score))]
    print(f"\n점수가 가장 높은 목표 속도: {best:.2f} m/s (임시 점수 공식 기준)")

    cfg = load_config(args.config)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(v, air * 100, "o-", label="airway clear [%]")
    ax[0].plot(v, succ * 100, "s--", label="success [%]")
    ax[0].set_xlabel("target speed [m/s]")
    ax[0].set_ylabel("%")
    ax[0].set_ylim(-5, 105)
    ax[0].legend()
    ax[0].set_title("Face protection vs speed")
    ax[1].plot(v, tmax, "o-", color="tab:red")
    ax[1].axhline(cfg.tether.break_force, color="k", ls=":", label="break force")
    ax[1].set_xlabel("target speed [m/s]")
    ax[1].set_ylabel("max tension [N]")
    ax[1].legend()
    ax[1].set_title("Peak tether load")
    sc = ax[2].scatter(rtime, air * 100, c=v, cmap="viridis", s=60)
    for i, sp in enumerate(v):
        ax[2].annotate(f"{sp:.1f}", (rtime[i], air[i] * 100), textcoords="offset points", xytext=(4, 4))
    ax[2].set_xlabel("rescue time [s]")
    ax[2].set_ylabel("airway clear [%]")
    ax[2].set_title("Fast vs safe (trade-off)")
    fig.colorbar(sc, ax=ax[2], label="target speed")
    fig.suptitle(f"AirwayTow PID speed sweep  Hs={args.wave} m  seeds={args.seeds}")
    fig.tight_layout()
    fig.savefig(out / f"sweep_hs{args.wave:.1f}.png", dpi=110)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
