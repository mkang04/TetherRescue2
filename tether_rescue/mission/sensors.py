"""가상 근거리 센서: 열화상 카메라와 소나.

둘 다 '로봇 기준 상대값'을 줍니다. 그래서 GPS 오차와 상관없이 가까이서는 정확히 다가갈 수 있습니다.
(두 단계 유도의 핵심: 멀리서는 GPS 좌표, 가까이서는 직접 감지)

열화상 (뱃머리 카메라)
- 시야각 안, 최대 거리 안에 있는 '물 밖으로 나온 머리'의 방향(bearing)만 줌. 거리는 모름.
- 머리가 물에 잠기면 안 보임. 멀수록, 파도가 높을수록 놓칠 확률이 커짐 (파도에 가려짐).

소나 (물속)
- 물속 몸통의 거리 + 방향을 줌. 머리가 잠겨도 보임. 대신 방향이 덜 정확하고, 파도 거품 때문에 가짜 탐지가 섞임.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import SonarConfig, ThermalConfig
from ..tow.pid import wrap_angle


@dataclass
class Detection:
    sensor: str                 # "thermal" / "sonar"
    t: float
    bearing: float              # 로봇 앞쪽 기준 상대 방향 [rad] (왼쪽 +)
    range: float | None         # [m], 열화상은 None
    real: bool                  # 실제 사람인지 (분석용. 로봇 두뇌는 이 값을 보면 안 됨)


def _relative(robot_xy, robot_yaw, target_xy):
    d = np.asarray(target_xy, float)[:2] - np.asarray(robot_xy, float)[:2]
    rng = float(np.hypot(d[0], d[1]))
    bearing = wrap_angle(math.atan2(d[1], d[0]) - robot_yaw)
    return rng, bearing


class ThermalCamera:
    def __init__(self, cfg: ThermalConfig, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        self.next_t = 0.0

    def sense(self, t, robot_xy, robot_yaw, head_xy, head_visible: bool, wave_height: float,
              line_of_sight: bool | None = None) -> list[Detection]:
        """line_of_sight: 카메라 ~ 머리 사이를 파도가 가리는지 실제로 계산한 결과 (구조 임무가 넘겨 줌).
        None이면 예전처럼 파고에 따른 확률로 가려짐을 흉내 냄."""
        if t < self.next_t:
            return []
        self.next_t = t + 1.0 / self.cfg.rate
        r, b = _relative(robot_xy, robot_yaw, head_xy)
        half = math.radians(self.cfg.fov) / 2
        if not head_visible or r > self.cfg.range or abs(b) > half or line_of_sight is False:
            return []
        # 멀수록 머리가 화면에서 작아져 놓치기 쉬움. 파도 가림은 line_of_sight가 있으면 그걸로 이미 판정
        p = max(0.0, 1.0 - (r / self.cfg.range) ** 2)
        if line_of_sight is None:
            p *= math.exp(-0.6 * wave_height * r / 10.0)
        if self.rng.random() > p:
            return []
        b_meas = b + self.rng.normal(0.0, math.radians(self.cfg.bearing_noise))
        return [Detection("thermal", t, wrap_angle(b_meas), None, True)]


class Sonar:
    def __init__(self, cfg: SonarConfig, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        self.next_t = 0.0

    def sense(self, t, robot_xy, robot_yaw, body_xy, wave_height: float) -> list[Detection]:
        if t < self.next_t:
            return []
        self.next_t = t + 1.0 / self.cfg.rate
        out = []
        half = math.radians(self.cfg.fov) / 2
        r, b = _relative(robot_xy, robot_yaw, body_xy)
        if r <= self.cfg.range and abs(b) <= half:
            p = 0.9 * max(0.0, 1.0 - (r / self.cfg.range) ** 2)
            if self.rng.random() < p:
                out.append(Detection(
                    "sonar", t,
                    wrap_angle(b + self.rng.normal(0.0, math.radians(self.cfg.bearing_noise))),
                    max(0.1, r + self.rng.normal(0.0, self.cfg.range_noise)), True))
        # 가짜 탐지 (파도 거품·쓰레기). 파고에 비례.
        n_false = self.rng.poisson(self.cfg.false_rate * max(wave_height, 0.0))
        for _ in range(n_false):
            out.append(Detection("sonar", t, float(self.rng.uniform(-half, half)),
                                 float(self.rng.uniform(2.0, self.cfg.range)), False))
        return out

    def sense_rays(self, t, robot_yaw, cast, wave_height: float, elevations=(-3.0, 0.0, 3.0),
                   beam_step: float = 2.0) -> list[Detection]:
        """MuJoCo 광선(mj_ray)으로 실제 빔을 쏘는 소나.
        cast(azimuth, elevation) → (거리 또는 None, 사람인지): 로봇 앞쪽 기준 각도 [rad]로 빔 하나를 쏨.
        시야각 안에 beam_step 간격으로 빔을 쏴서, 사람(몸통)에 맞은 빔들의 가장 가까운 거리와 가운데 방향을 씀.
        → 다른 물체에 가려지거나, 몸이 빔 사이로 빠지면 안 보임."""
        if t < self.next_t:
            return []
        self.next_t = t + 1.0 / self.cfg.rate
        out = []
        half = math.radians(self.cfg.fov) / 2
        n = max(1, int(round(self.cfg.fov / beam_step)))
        hits = []
        for i in range(n + 1):
            az = -half + 2 * half * i / n
            for el in elevations:
                dist, is_target = cast(az, math.radians(el))
                if dist is not None and is_target and dist <= self.cfg.range:
                    hits.append((dist, az))
        if hits:
            r = min(h[0] for h in hits)
            b = float(np.mean([h[1] for h in hits]))
            p = 0.9 * max(0.0, 1.0 - (r / self.cfg.range) ** 2)     # 멀수록 반사 신호가 약해짐
            if self.rng.random() < p:
                out.append(Detection(
                    "sonar", t,
                    wrap_angle(b + self.rng.normal(0.0, math.radians(self.cfg.bearing_noise))),
                    max(0.1, r + self.rng.normal(0.0, self.cfg.range_noise)), True))
        n_false = self.rng.poisson(self.cfg.false_rate * max(wave_height, 0.0))
        for _ in range(n_false):
            out.append(Detection("sonar", t, float(self.rng.uniform(-half, half)),
                                 float(self.rng.uniform(2.0, self.cfg.range)), False))
        return out
