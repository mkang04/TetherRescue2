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
    added_mass_scale: float = 1.0   # 부가질량 배율 (0 = 끔, 1 = 평판 공식 그대로)


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
    outboard_mass: float = 3.0   # 선외기(모터 머리·축·프로펠러) 질량 [kg], 전체 mass 안에 포함
    steer_kp: float = 40.0       # 조향 서보 강성 [N·m/rad]
    steer_tau: float = 0.1       # 조향 서보 반응 지연 (1차 지연 시정수) [s]
    prop_radius: float = 0.07    # 프로펠러 반지름 [m] (물 밖으로 나온 비율만큼 추력 감소)
    thrust_zero_speed: float = 5.0  # 프로펠러 전진 속도가 이 값이면 추력 0 [m/s] (속도에 따른 추력 감소)
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
    arm_reach: float = 0.7        # 어깨에서 손끝까지 [m] (튜브를 잡을 수 있는 거리)
    grab_reaction: float = 0.8    # 튜브가 손 닿는 곳에 이만큼 있어야 잡으려고 함 [s]
    grab_time: float = 1.2        # 팔을 뻗어 잡는 데 걸리는 시간 [s]
    don_time: float = 3.0         # 잡은 튜브를 가슴(겨드랑이 아래)에 끼우는 시간 [s]
    # 로봇에 매달리기 (튜브가 없는 사람이 배를 붙잡으려 함)
    grab_zone: float = 1.5        # 어깨 ~ 선체 거리가 이 안이면 배를 향해 달려듦 [m]
    climb_reaction: float = 1.0   # 위험 거리 안에 이만큼 있으면 달려들기 시작 [s]
    lunge_force: float = 60.0     # 배를 향해 헤엄치는 힘 [N]
    climb_force_ratio: float = 0.4  # 매달려 올라타려 할 때 뱃전을 누르는 힘 = 체중의 이 비율
    climb_fail_time: float = 3.0  # 이만큼 매달려 있으면 구조 실패 [s]


@dataclass
class TetherConfig:
    rest_length: float = 3.0
    stiffness: float = 600.0
    damping: float = 30.0
    break_force: float = 294.0
    break_hold: float = 0.05
    reel_speed: float = 0.2       # 연결 때 더 풀려 있던 줄을 감아 들이는 빠르기 [m/s]


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
    """전체 구조 임무: 좌표 받기 → GPS로 이동 → 가까이서 직접 감지 → 튜브 건네기 → 기지로 견인."""
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
    handoff_standoff: float = 3.0      # 튜브 건넬 때 로봇 중심 ~ 사람(추정) 거리 [m] (이 거리를 유지)
    tube_line_start: float = 0.9       # 처음 내밀 때 뱃머리 ~ 튜브 중심 거리 [m]
    tube_line_max: float = 3.6         # 줄을 이만큼까지 풀어 튜브를 사람 쪽으로 보냄 [m]
    tube_payout_speed: float = 0.2     # 줄 푸는 빠르기 [m/s]
    tube_payout_delay: float = 2.0     # 내민 뒤 이만큼 기다렸다가 줄을 풀기 시작 [s]
    handoff_timeout: float = 20.0     # 이 시간 동안 못 잡으면 추정이 틀린 것으로 보고 다시 탐색 [s]
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
class ImuConfig:
    """IMU 잡음 (MuJoCo는 센서 noise 속성을 적용하지 않으므로 파이썬에서 더함). 기본값: MPU-6050 데이터시트"""
    acc_noise_density: float = 400e-6   # 가속도 잡음 밀도 [g/√Hz]
    gyro_noise_density: float = 0.005   # 자이로 잡음 밀도 [deg/s/√Hz]
    bandwidth: float = 25.0             # 측정 대역폭 [Hz] (조종 박자 50 Hz의 절반)
    gyro_bias: float = 0.1              # 켤 때마다 정해지는 자이로 고정 오차 표준편차 [deg/s]


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
    imu: ImuConfig = field(default_factory=ImuConfig)
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
