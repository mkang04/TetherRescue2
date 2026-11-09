"""파도: 크기·빠르기·방향이 다른 물결(사인파) 여러 개를 겹칩니다.

같은 시드를 넣으면 언제나 같은 파도가 나옵니다.
깊은 바다 가정: 각진동수 w^2 = g * k (k: 파수).
"""
from __future__ import annotations

import numpy as np

from ..config import OceanConfig


class WaveField:
    def __init__(self, cfg: OceanConfig, seed: int = 0):
        self.cfg = cfg
        self.g = cfg.g
        self.current = np.array([cfg.current[0], cfg.current[1], 0.0], dtype=float)
        n = int(cfg.n_components)
        rng = np.random.default_rng(seed)

        if n <= 0:
            self.unit_amp = self.amp = np.zeros(0)
            self.k = self.omega = self.phase = self.dir_x = self.dir_y = np.zeros(0)
            return

        # 유의파고 Hs = 4 * sqrt(m0),  m0 = sum(a_i^2 / 2)
        # Hs = 1m 기준 진폭을 저장해 두고, 실제 진폭 = 기준 진폭 * Hs.
        # 파고가 0이어도 물결 성분은 만들어 두어서 실행 중에 파고를 키울 수 있게 합니다.
        self.unit_amp = np.full(n, np.sqrt(2.0 * (1.0 / 4.0) ** 2 / n))
        self.amp = self.unit_amp * max(cfg.wave_height, 0.0)
        periods = cfg.peak_period * rng.uniform(0.7, 1.5, size=n)
        self.omega = 2.0 * np.pi / periods
        self.k = self.omega**2 / self.g
        theta = cfg.direction + rng.uniform(-cfg.spread, cfg.spread, size=n)
        self.dir_x = np.cos(theta)
        self.dir_y = np.sin(theta)
        self.phase = rng.uniform(0.0, 2.0 * np.pi, size=n)

    @property
    def wave_height(self) -> float:
        """현재 유의파고 [m]"""
        return float(self.amp[0] / self.unit_amp[0]) if self.unit_amp.size else 0.0

    def set_wave_height(self, hs: float):
        """물결 모양(주기·방향·위상)은 그대로 두고 높이만 바꿉니다."""
        self.amp = self.unit_amp * max(float(hs), 0.0)

    def _arg(self, x, y, t):
        x = np.asarray(x, dtype=float)[..., None]
        y = np.asarray(y, dtype=float)[..., None]
        return self.k * (x * self.dir_x + y * self.dir_y) - self.omega * t + self.phase

    def height(self, x, y, t: float):
        """(x, y) 위치, t 시각의 물 높이 [m]. 배열도 넣을 수 있습니다."""
        if self.amp.size == 0:
            return np.zeros(np.shape(x))
        return np.sum(self.amp * np.cos(self._arg(x, y, t)), axis=-1)

    def sample(self, pts: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """점 여러 개 (n, 3)의 물 높이 (n,)와 물 입자 속도 (n, 3)를 한 번에 계산 (빠른 버전)."""
        n = len(pts)
        vel = np.empty((n, 3))
        vel[:] = self.current
        if self.amp.size == 0:
            return np.zeros(n), vel
        arg = np.outer(pts[:, 0], self.k * self.dir_x) + np.outer(pts[:, 1], self.k * self.dir_y)
        arg += self.phase - self.omega * t
        c, s = np.cos(arg), np.sin(arg)
        eta = c @ self.amp
        aw = np.exp(np.minimum(pts[:, 2:3], 0.0) * self.k) * (self.amp * self.omega)
        awc = aw * c
        vel[:, 0] += awc @ self.dir_x
        vel[:, 1] += awc @ self.dir_y
        vel[:, 2] += (aw * s).sum(axis=1)
        return eta, vel

    def velocity(self, pos, t: float):
        """물 입자 속도 [m/s] (파도 궤도 속도 + 해류). pos: (..., 3)"""
        pos = np.asarray(pos, dtype=float)
        out = np.broadcast_to(self.current, pos.shape).copy()
        if self.amp.size == 0:
            return out
        arg = self._arg(pos[..., 0], pos[..., 1], t)
        # 깊어질수록 exp(k z)로 약해짐. 수면 위 점은 수면 값 사용.
        z = np.minimum(pos[..., 2], 0.0)[..., None]
        decay = np.exp(self.k * z)
        aw = self.amp * self.omega * decay
        c = aw * np.cos(arg)
        out[..., 0] += np.sum(c * self.dir_x, axis=-1)
        out[..., 1] += np.sum(c * self.dir_y, axis=-1)
        out[..., 2] += np.sum(aw * np.sin(arg), axis=-1)
        return out

    def acceleration(self, pos, t: float):
        """물 입자 가속도 [m/s^2] (파도 궤도 운동, 해류는 일정해서 0). pos: (..., 3)"""
        pos = np.asarray(pos, dtype=float)
        out = np.zeros(pos.shape)
        if self.amp.size == 0:
            return out
        arg = self._arg(pos[..., 0], pos[..., 1], t)
        z = np.minimum(pos[..., 2], 0.0)[..., None]
        aw2 = self.amp * self.omega ** 2 * np.exp(self.k * z)     # velocity()를 시간으로 미분
        s = aw2 * np.sin(arg)
        out[..., 0] = np.sum(s * self.dir_x, axis=-1)
        out[..., 1] = np.sum(s * self.dir_y, axis=-1)
        out[..., 2] = -np.sum(aw2 * np.cos(arg), axis=-1)
        return out
