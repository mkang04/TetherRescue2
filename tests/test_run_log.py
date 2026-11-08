"""실험 기록 (results/실험기록.csv) 검사."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tether_rescue.ui.run_log import HEADER, LOG_NAME, RunLog  # noqa: E402

MISSION = {"mode": "mission", "seed": 3, "wave": 1.0, "speed": 1.0, "reason": "success", "time": 71.12,
           "airway": 99.6, "tension": 113.4, "score": float("nan"), "detect_time": 8.0, "grab_time": None,
           "attach_time": 32.7, "changed_mid_run": 0}
TOW = {"mode": "tow", "seed": 0, "wave": 1.5, "speed": 2.0, "reason": "tether_broken", "time": 12.0,
       "airway": 80.0, "tension": 300.0, "score": 0.0, "changed_mid_run": 1}


def read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_each_run_is_appended_with_running_number(tmp_path):
    log = RunLog(tmp_path)
    log.add(MISSION)
    log.add(TOW)
    rows = read(tmp_path / LOG_NAME)
    assert [r["회차"] for r in rows] == ["1", "2"]
    assert rows[0]["모드"] == "구조 임무" and rows[0]["성공"] == "O" and rows[0]["결과"] == "구조 성공"
    assert rows[0]["걸린 시간(s)"] == "71.1" and rows[0]["튜브 잡기까지(s)"] == "" and rows[0]["점수(끌고 오기)"] == ""
    assert rows[1]["결과"] == "줄 끊어짐" and rows[1]["성공"] == "X" and rows[1]["도중에 값 바꿈"] == "예"

    again = RunLog(tmp_path)                 # 프로그램을 다시 켜도 번호가 이어짐
    assert len(again.rows) == 2
    assert again.add(MISSION)["회차"] == "3"
    assert [r["회차"] for r in read(tmp_path / LOG_NAME)] == ["1", "2", "3"]


def test_old_format_file_is_kept_aside(tmp_path):
    (tmp_path / LOG_NAME).write_text("no,reason\n1,success\n", encoding="utf-8-sig")
    log = RunLog(tmp_path)
    assert log.rows == []
    assert list(tmp_path.glob("실험기록_이전_*.csv"))
    log.add(MISSION)
    with open(tmp_path / LOG_NAME, newline="", encoding="utf-8-sig") as f:
        assert next(csv.reader(f)) == HEADER


def test_unwritable_file_is_retried_next_time(tmp_path, monkeypatch):
    log = RunLog(tmp_path)
    import builtins
    real_open = builtins.open

    def locked(*a, **k):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(builtins, "open", locked)      # 엑셀이 파일을 잡고 있는 상황
    log.add(MISSION)
    assert log.error and len(log.unsaved) == 1
    monkeypatch.setattr(builtins, "open", real_open)
    log.add(TOW)
    assert not log.error and not log.unsaved
    assert [r["회차"] for r in read(tmp_path / LOG_NAME)] == ["1", "2"]
