"""물에 뜨는 힘(부력)과 물 저항.

MuJoCo는 물을 계산하지 않으므로, 몸을 작은 블록 조각으로 나눠
각 조각이 물에 잠긴 만큼 위로 밀어 주고(부력), 물에 대한 상대 속도만큼
반대로 밀어 줍니다(저항). 조각마다 힘을 주기 때문에 기울어지면
자연스럽게 다시 서려는 힘(복원 모멘트)이 생깁니다.

속도를 위해 여러 몸(로봇·몸통·머리)의 블록을 한 배열로 묶어 한 번에 계산합니다.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .waves import WaveField


def box_blocks(size, counts):
    """박스(전체 크기 size)를 counts 개로 나눈 블록 중심점, 블록 부피, 블록 높이."""
    size = np.asarray(size, dtype=float)
    counts = np.asarray(counts, dtype=int)
    cell = size / counts
    axes = [(-size[i] / 2 + cell[i] * (np.arange(counts[i]) + 0.5)) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    vol = float(np.prod(cell))
    return pts, np.full(len(pts), vol), float(cell[2])


def sphere_blocks(radius, n=5):
    """구를 격자로 나눈 뒤 구 안의 점만 남깁니다. 전체 부피는 정확히 맞춥니다."""
    ax = -radius + (2 * radius / n) * (np.arange(n) + 0.5)
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    pts = pts[np.linalg.norm(pts, axis=1) <= radius]
    total = 4.0 / 3.0 * np.pi * radius**3
    return pts, np.full(len(pts), total / len(pts)), 2 * radius / n


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """np.cross보다 빠른 (n,3) x (n,3) 외적."""
    out = np.empty(np.broadcast_shapes(a.shape, b.shape))
    out[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    out[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    out[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return out


@dataclass
class HydroSpec:
    """몸 하나의 물 관련 설정."""
    body: str
    blocks: np.ndarray         # (n, 3) body 좌표계 블록 중심 (body 원점 기준)
    block_vol: np.ndarray      # (n,)
    block_h: float             # 블록 높이 (잠긴 비율 계산용)
    drag_lin: list[float]      # 몸체 좌표 x, y, z
    drag_quad: list[float]
    ang_drag: list[float]


@dataclass
class HydroResult:
    force: np.ndarray        # (nb, 3) 월드 좌표 힘 [N] (질량중심에 작용)
    torque: np.ndarray       # (nb, 3) 질량중심 기준 토크 [N·m]
    submerged: np.ndarray    # (nb,) 잠긴 부피 비율 0~1
    buoyancy: np.ndarray     # (nb,) 부력 크기 [N]


class HydroSystem:
    def __init__(self, model: mujoco.MjModel, specs: list[HydroSpec], rho=1025.0, g=9.81):
        self.rho, self.g = rho, g
        self.names = [s.body for s in specs]
        self.body_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s.body) for s in specs])
        if (self.body_ids < 0).any():
            raise ValueError(f"없는 body가 있음: {self.names}")
        nb = len(specs)
        self.owner = np.concatenate([np.full(len(s.blocks), i) for i, s in enumerate(specs)])
        self.local = np.concatenate([np.asarray(s.blocks, float) for s in specs])
        self.vol = np.concatenate([np.asarray(s.block_vol, float) for s in specs])
        self.h = np.concatenate([np.full(len(s.blocks), float(s.block_h)) for s in specs])
        total = np.array([np.sum(s.block_vol) for s in specs])
        self.total_vol = total
        self.weight = self.vol / total[self.owner]                     # 몸 안에서 블록 비중
        self.drag_lin = np.array([np.asarray(s.drag_lin, float) * np.ones(3) for s in specs])[self.owner]
        self.drag_quad = np.array([np.asarray(s.drag_quad, float) * np.ones(3) for s in specs])[self.owner]
        self.ang_drag = np.array([np.asarray(s.ang_drag, float) * np.ones(3) for s in specs])
        # 블록 → 몸 합산 행렬 (nb, n)
        self.sum_mat = np.zeros((nb, len(self.owner)))
        self.sum_mat[self.owner, np.arange(len(self.owner))] = 1.0
        self._v6 = np.zeros(6)
        self._w = np.zeros((nb, 3))
        self._v = np.zeros((nb, 3))

    def compute(self, model, data, waves: WaveField, t: float) -> HydroResult:
        ids, own = self.body_ids, self.owner
        R = data.xmat[ids].reshape(-1, 3, 3)                 # (nb, 3, 3)
        com = data.xipos[ids]
        Rb = R[own]                                          # (n, 3, 3)
        pts = data.xpos[ids][own] + np.einsum("nij,nj->ni", Rb, self.local)

        for i, b in enumerate(ids):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, self._v6, 0)
            self._w[i] = self._v6[:3]
            self._v[i] = self._v6[3:]

        eta, wvel = waves.sample(pts, t)

        # 1) 부력: 블록 아래면부터 물 높이까지 잠긴 비율
        frac = np.clip((eta - (pts[:, 2] - self.h / 2)) / self.h, 0.0, 1.0)
        buoy = self.rho * self.g * self.vol * frac

        # 2) 저항: 블록마다 (물에 대한 상대 속도)로 계산, 잠긴 만큼만
        r = pts - com[own]
        v_rel = self._v[own] + _cross(self._w[own], r) - wvel
        v_loc = np.einsum("nji,nj->ni", Rb, v_rel)           # 몸체 좌표계로 (R^T v)
        f_loc = -(frac * self.weight)[:, None] * (self.drag_lin * v_loc + self.drag_quad * np.abs(v_loc) * v_loc)
        f = np.einsum("nij,nj->ni", Rb, f_loc)
        f[:, 2] += buoy

        force = self.sum_mat @ f
        torque = self.sum_mat @ _cross(r, f)

        # 3) 회전 감쇠 (물속에서 빙글빙글 도는 걸 막음)
        sub = (self.sum_mat @ (frac * self.vol)) / self.total_vol
        w_loc = np.einsum("bji,bj->bi", R, self._w)
        torque += np.einsum("bij,bj->bi", R, -sub[:, None] * self.ang_drag * w_loc)

        return HydroResult(force, torque, sub, self.sum_mat @ buoy)
