"""줄(테더) 모델.

- 느슨하면(거리 <= 자연 길이) 힘 0
- 팽팽하면 고무줄처럼 당김:  T = k * (늘어난 길이) + c * (늘어나는 속도, 늘어날 때만)
- 줄은 밀 수 없으므로 T >= 0
- T가 끊어지는 힘을 break_hold 시간 동안 계속 넘으면 끊어짐 (이후 영원히 0)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import TetherConfig


@dataclass
class TetherState:
    tension: float = 0.0      # 줄 당기는 힘 [N]
    length: float = 0.0       # 두 묶음점 사이 거리 [m]
    stretch: float = 0.0      # 늘어난 길이 [m] (느슨하면 0)
    taut: bool = False
    broken: bool = False
    direction: np.ndarray | None = None   # 로봇 묶음점 → 사람 묶음점 단위벡터


class Tether:
    def __init__(self, cfg: TetherConfig):
        self.cfg = cfg
        self.reset()

    def reset(self, attached: bool = True):
        """attached=False면 아직 사람에게 연결 안 된 상태 (구조 임무 시작 시)."""
        self.broken = False
        self.attached = attached
        self._over_time = 0.0
        self.state = TetherState()

    def attach(self):
        self.attached = True
        self.broken = False
        self._over_time = 0.0

    def compute(self, p_robot, p_human, v_robot, v_human, dt: float) -> TetherState:
        """묶음점 위치·속도로 줄 힘을 계산합니다.

        반환된 tension은 로봇은 사람 쪽으로(+direction), 사람은 로봇 쪽으로(-direction) 당깁니다.
        """
        d = np.asarray(p_human, float) - np.asarray(p_robot, float)
        length = float(np.linalg.norm(d))
        u = d / length if length > 1e-9 else np.zeros(3)

        if not self.attached:
            self.state = TetherState(0.0, length, 0.0, False, False, u)
            return self.state
        if self.broken:
            self.state = TetherState(0.0, length, 0.0, False, True, u)
            return self.state

        stretch = length - self.cfg.rest_length
        if stretch <= 0.0:
            self._over_time = 0.0
            self.state = TetherState(0.0, length, 0.0, False, False, u)
            return self.state

        rate = float(np.dot(np.asarray(v_human, float) - np.asarray(v_robot, float), u))
        tension = self.cfg.stiffness * stretch + self.cfg.damping * max(rate, 0.0)
        tension = max(tension, 0.0)

        if tension > self.cfg.break_force:
            self._over_time += dt
            if self._over_time >= self.cfg.break_hold:
                self.broken = True
                self.state = TetherState(0.0, length, stretch, False, True, u)
                return self.state
        else:
            self._over_time = 0.0

        self.state = TetherState(tension, length, stretch, True, False, u)
        return self.state
