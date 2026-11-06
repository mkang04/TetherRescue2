"""실험 설정 패널: 값 변경이 cfg로 제대로 전달되는지, 실행 중 파도 변경이 부드러운지."""
import gc
import math
import sys
import tkinter as tk
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tether_rescue.config import load_config  # noqa: E402
from tether_rescue.tow_env import TowSim  # noqa: E402


@pytest.fixture
def panel(tmp_path, capfd):
    from tether_rescue.ui.control_panel import ControlPanel
    # pytest가 출력(fd)을 가로채는 중에 Tk를 만들면 Windows에서 Tcl 초기화 파일 읽기가 가끔 실패함.
    # 창을 만드는 동안만 가로채기를 끕니다. (실제 프로그램에는 해당 없음)
    try:
        with capfd.disabled():
            p = ControlPanel(load_config(), load_config(), tmp_path)
    except tk.TclError as e:
        pytest.skip(f"tkinter 창을 만들 수 없음 (화면 없음?): {e}")
    yield p
    p.destroy()
    del p
    gc.collect()


def test_live_change_is_reported_in_cfg_units(panel):
    panel.vars["pid.target_speed"].set(1.7)
    panel.vars["ocean.direction"].set(90)          # 다시 시작할 때 적용 → 바로 적용 목록에 없어야 함
    changes = panel.take_live_changes()
    assert changes == {"pid.target_speed": pytest.approx(1.7)}
    assert panel.take_live_changes() == {}          # 한 번 꺼내면 비워짐


def test_restart_values_and_pending_message(panel):
    panel.vars["ocean.direction"].set(90)
    panel.vars["sim.seed"].set(5)
    panel.poll()
    assert "시드" in panel.pending.get() and "파도 방향" in panel.pending.get()
    vals = panel.restart_values()
    assert vals["ocean.direction"] == pytest.approx(math.pi / 2)   # 도 → 라디안
    assert vals["sim.seed"] == 5
    assert panel.pending.get() == ""


def test_program_side_update_does_not_count_as_user_change(panel):
    panel.set_value("pid.target_speed", 2.0)        # 단축키로 바꾼 경우
    assert panel.take_live_changes() == {}
    assert float(panel.vars["pid.target_speed"].get()) == pytest.approx(2.0)


def test_reset_defaults(panel):
    panel.vars["tether.break_force"].set(150)
    panel.take_live_changes()
    panel._reset_defaults()
    assert panel.take_live_changes()["tether.break_force"] == pytest.approx(load_config().tether.break_force)


def test_live_wave_height_ramps_smoothly():
    sim = TowSim(load_config(overrides={"ocean": {"wave_height": 0.0}}))
    sim.set_wave_height(1.5, ramp_time=1.0)
    heights = []
    for _ in range(60):                             # 1.2초
        sim.step(0.0, 0.0)
        heights.append(sim.waves.wave_height)
    assert heights[0] < 0.1
    assert all(b >= a for a, b in zip(heights, heights[1:]))   # 계속 커지기만
    assert heights[-1] == pytest.approx(1.5)
    assert sim.cfg.ocean.wave_height == 1.5
