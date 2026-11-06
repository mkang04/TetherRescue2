"""설정값 묶음. YAML 파일을 읽어 dataclass로 바꿉니다."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, asdict
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "tow_default.yaml"


@dataclass
class SimConfig:
    physics_hz: int = 250
    control_hz: int = 50
    max_time: float = 120.0
    goal_distance: float = 50.0
    seed: int = 0
    settle_time: float = 2.0


@dataclass
class OceanConfig:
    rho: float = 1025.0
    g: float = 9.81
    wave_height: float = 1.0
    peak_period: float = 5.0
    n_components: int = 8
    direction: float = 0.0
    spread: float = 0.5
    current: list[float] = field(default_factory=lambda: [0.0, 0.0])


@dataclass
class RobotConfig:
    length: float = 1.4
    width: float = 0.6
    height: float = 0.3
    mass: float = 28.0
    max_thrust: float = 300.0
    max_reverse: float = 0.3
    thrust_tau: float = 0.4
    max_rudder: float = 0.6
    drag_lin: list[float] = field(default_factory=lambda: [5.0, 60.0, 400.0])
    drag_quad: list[float] = field(default_factory=lambda: [12.0, 150.0, 200.0])
    ang_drag: list[float] = field(default_factory=lambda: [8.0, 8.0, 6.0])
    blocks: list[int] = field(default_factory=lambda: [6, 3, 3])
    tow_point: list[float] = field(default_factory=lambda: [-0.7, 0.0, 0.1])


@dataclass
class HumanConfig:
    torso_size: list[float] = field(default_factory=lambda: [1.5, 0.38, 0.14])
    torso_mass: float = 65.0
    head_radius: float = 0.11
    head_mass: float = 5.0
    neck_stiffness: float = 60.0
    neck_damping: float = 6.0
    neck_range: float = 0.6
    drag_lin: list[float] = field(default_factory=lambda: [20.0, 40.0, 120.0])
    drag_quad: list[float] = field(default_factory=lambda: [60.0, 200.0, 300.0])
    head_drag_quad: float = 15.0
    ang_drag: list[float] = field(default_factory=lambda: [15.0, 25.0, 15.0])
    blocks: list[int] = field(default_factory=lambda: [8, 3, 3])
    harness_point: list[float] = field(default_factory=lambda: [0.6, 0.0, 0.07])
    mouth_offset: list[float] = field(default_factory=lambda: [0.02, 0.0, 0.10])
    bow_wave_coef: float = 0.6
    start_gap: float = 0.3


@dataclass
class TetherConfig:
    rest_length: float = 3.0
    stiffness: float = 600.0
    damping: float = 30.0
    break_force: float = 294.0
    break_hold: float = 0.05


@dataclass
class PIDConfig:
    target_speed: float = 1.0
    ramp_time: float = 6.0
    speed_kp: float = 0.8
    speed_ki: float = 0.3
    speed_kd: float = 0.0
    heading_kp: float = 1.5
    heading_ki: float = 0.0
    heading_kd: float = 0.4


@dataclass
class MissionConfig:
    """전체 구조 임무: 좌표 받기 → GPS로 이동 → 가까이서 직접 감지 → 줄 연결 → 기지로 견인."""
    base: list[float] = field(default_factory=lambda: [0.0, 0.0])
    victim_distance: float = 32.0
    victim_bearing: float = 35.0
    auto_report: bool = False
    report_error: float = 5.0
    gps_error: float = 2.5
    gps_tau: float = 30.0
    transit_speed: float = 1.5
    arrive_radius: float = 4.0
    search_leg: float = 5.0
    search_speed: float = 0.9
    search_timeout: float = 90.0
    approach_speed_max: float = 1.0
    capture_distance: float = 2.0
    capture_speed: float = 0.6
    track_lost_time: float = 8.0
    home_radius: float = 5.0
    max_time: float = 300.0


@dataclass
class ThermalConfig:
    range: float = 20.0
    fov: float = 60.0
    rate: float = 10.0
    bearing_noise: float = 1.5


@dataclass
class SonarConfig:
    range: float = 25.0
    fov: float = 120.0
    rate: float = 2.0
    range_noise: float = 0.3
    bearing_noise: float = 3.0
    false_rate: float = 0.3


@dataclass
class BridgeConfig:
    host: str = "auto"
    port: int = 47800
    rate: float = 10.0
    wsl_distro: str = "Ubuntu-24.04"


@dataclass
class TowConfig:
    sim: SimConfig = field(default_factory=SimConfig)
    ocean: OceanConfig = field(default_factory=OceanConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    human: HumanConfig = field(default_factory=HumanConfig)
    tether: TetherConfig = field(default_factory=TetherConfig)
    pid: PIDConfig = field(default_factory=PIDConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    sonar: SonarConfig = field(default_factory=SonarConfig)
    ros_bridge: BridgeConfig = field(default_factory=BridgeConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fill(cls, values: dict[str, Any] | None):
    """dict 값을 dataclass에 채웁니다. 모르는 키가 있으면 오타일 수 있으니 에러를 냅니다."""
    obj = cls()
    if not values:
        return obj
    names = {f.name: f for f in fields(cls)}
    for key, val in values.items():
        if key not in names:
            raise KeyError(f"{cls.__name__}에 없는 설정 키: '{key}'")
        cur = getattr(obj, key)
        if is_dataclass(cur):
            val = _fill(type(cur), val)
        setattr(obj, key, val)
    return obj


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> TowConfig:
    """YAML 설정을 읽습니다. overrides는 {"pid": {"target_speed": 1.5}} 같은 모양입니다."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for section, vals in (overrides or {}).items():
        data.setdefault(section, {}).update(vals)
    return _fill(TowConfig, data)
