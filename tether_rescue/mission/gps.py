"""GPS 모델: 실제 위치 + 천천히 바뀌는 오차 (수 m).

실제 GPS 오차는 매 순간 무작위로 튀지 않고 수십 초에 걸쳐 천천히 흘러갑니다.
그래서 '1차 가우스-마르코프' 과정으로 흉내 냅니다:
    e(t+dt) = a·e(t) + sqrt(1-a²)·σ·N(0,1),   a = exp(-dt/τ)
→ 오래 보면 표준편차 σ, 변하는 시간 τ.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import MissionConfig


class GpsModel:
    def __init__(self, cfg: MissionConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.error = rng.normal(0.0, cfg.gps_error, 2)

    def update(self, dt: float):
        sigma, tau = self.cfg.gps_error, max(self.cfg.gps_tau, 1e-6)
        a = math.exp(-dt / tau)
        self.error = a * self.error + math.sqrt(1 - a * a) * sigma * self.rng.normal(0.0, 1.0, 2)

    def measure(self, true_xy) -> np.ndarray:
        return np.asarray(true_xy, float)[:2] + self.error
