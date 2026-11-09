"""실험 기록: 한 판이 끝날 때마다 results/실험기록.csv 에 한 줄씩 자동으로 쌓습니다.

- 회차 번호는 파일에 이어서 매깁니다 (프로그램을 껐다 켜도 계속).
- 열 이름은 한글, 엑셀에서 바로 열리게 UTF-8(BOM)으로 저장합니다.
- 끌고 오기·구조 임무 두 모드가 같은 열을 씁니다 (해당 없는 칸은 비움).
- 열 구성이 바뀌면 예전 파일은 '실험기록_이전_날짜.csv'로 이름을 바꿔 두고 새로 시작합니다.
"""
from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

LOG_NAME = "실험기록.csv"

REASON_KO = {"success": "구조 성공", "tether_broken": "줄 끊어짐", "capsized": "로봇 뒤집힘", "timeout": "시간 초과",
             "not_found": "못 찾음", "victim_climbed": "사람이 로봇에 매달림"}
MODE_KO = {"mission": "구조 임무", "tow": "끌고 오기"}

# (열 이름, 결과 dict 키, 소수 자리). 키가 None이면 따로 채움.
COLUMNS: list[tuple[str, str | None, int]] = [
    ("회차", None, 0), ("날짜", None, 0), ("시각", None, 0), ("모드", None, 0), ("성공", None, 0), ("결과", None, 0),
    ("시드", "seed", 0), ("파고(m)", "wave", 1), ("견인 목표 속도(m/s)", "speed", 1),
    ("걸린 시간(s)", "time", 1), ("얼굴 물 밖 비율(%)", "airway", 0), ("최대 줄 힘(N)", "tension", 0),
    ("점수(끌고 오기)", "score", 0),
    ("감지까지(s)", "detect_time", 1), ("튜브 잡기까지(s)", "grab_time", 1), ("줄 연결까지(s)", "attach_time", 1),
    ("목표 좌표 오차(m)", "goal_error", 1), ("도착 때 GPS 오차(m)", "gps_error_at_arrival", 1),
    ("가짜 탐지 수", "false_alarms", 0), ("도중에 값 바꿈", None, 0),
    # 설정값
    ("파도 주기(s)", "wave_period", 1), ("파도 방향(도)", "wave_direction_deg", 0),
    ("최대 추진력(N)", "max_thrust", 0), ("줄 끊어지는 힘(N)", "break_force", 0),
    ("줄 탄성(N/m)", "tether_stiffness", 0), ("줄 길이(m)", "tether_length", 1),
    ("선수파 계수", "bow_wave_coef", 2), ("속도 Kp", "speed_kp", 2), ("속도 Ki", "speed_ki", 2),
    ("방향 Kp", "heading_kp", 2), ("방향 Kd", "heading_kd", 2), ("출발 가속 시간(s)", "ramp_time", 1),
    ("구조 거리(m)", "goal_distance", 0),
    ("GPS 오차(m)", "gps_error_sigma", 1), ("목격 오차(m)", "report_error", 1),
    ("열화상 거리(m)", "thermal_range", 0), ("열화상 시야각(도)", "thermal_fov", 0),
    ("소나 거리(m)", "sonar_range", 0), ("소나 가짜 탐지", "sonar_false_rate", 1),
    ("이동 속도(m/s)", "transit_speed", 1), ("익수자 거리(m)", "victim_distance", 0),
    ("튜브 건넬 거리(m)", "handoff_standoff", 2), ("팔 닿는 거리(m)", "arm_reach", 2),
    ("배 붙잡는 거리(m)", "grab_zone", 1),
]
HEADER = [c[0] for c in COLUMNS]


def _fmt(v, digits: int) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, (int, float)):
        return f"{v:.{digits}f}"
    return str(v)


class RunLog:
    def __init__(self, results_dir: Path):
        self.path = Path(results_dir) / LOG_NAME
        self.rows: list[dict[str, str]] = []      # 파일에 있는 기록 전체 (한글 열 이름)
        self.unsaved: list[dict[str, str]] = []   # 아직 파일에 못 쓴 기록
        self.error = ""
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            with open(self.path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != HEADER:          # 열 구성이 바뀜 → 예전 파일은 보관
                    f.close()
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.path.rename(self.path.with_name(f"실험기록_이전_{stamp}.csv"))
                    return
                self.rows = list(reader)
        except OSError as e:
            self.error = f"기록 파일을 읽지 못했습니다: {e}"

    @property
    def next_no(self) -> int:
        nos = [int(r["회차"]) for r in self.rows if r.get("회차", "").isdigit()]
        return (max(nos) + 1) if nos else 1

    def to_row(self, res: dict) -> dict[str, str]:
        """결과 dict(영어 키) → 기록 한 줄(한글 열)."""
        now = datetime.now()
        row = {}
        for name, key, digits in COLUMNS:
            row[name] = _fmt(res.get(key), digits) if key else ""
        row.update({
            "회차": str(self.next_no), "날짜": now.strftime("%Y-%m-%d"), "시각": now.strftime("%H:%M:%S"),
            "모드": MODE_KO.get(res.get("mode"), res.get("mode", "")),
            "성공": "O" if res.get("reason") == "success" else "X",
            "결과": REASON_KO.get(res.get("reason"), res.get("reason", "")),
            "도중에 값 바꿈": "예" if res.get("changed_mid_run") else "",
        })
        return row

    def add(self, res: dict) -> dict[str, str]:
        """한 판 결과를 파일 끝에 붙이고, 붙인 줄(한글 열)을 돌려줍니다.
        파일이 엑셀에서 열려 있어 못 쓰면 기억해 뒀다가 다음 판 때 같이 씁니다."""
        row = self.to_row(res)
        self.rows.append(row)
        self.unsaved.append(row)
        self.flush()
        return row

    def flush(self):
        if not self.unsaved:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.path.exists()
            with open(self.path, "a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
                w = csv.DictWriter(f, fieldnames=HEADER)
                if new:
                    w.writeheader()
                w.writerows(self.unsaved)
            self.unsaved.clear()
            self.error = ""
        except OSError as e:
            self.error = (f"기록 파일에 쓰지 못했습니다 (엑셀에서 열려 있나요?) — {len(self.unsaved)}개 대기 중, "
                          f"다음 판 때 다시 씁니다: {e.strerror}")
