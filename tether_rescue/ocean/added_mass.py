"""부가질량 (added mass): 물속 물체가 가속하려면 주변 물도 같이 밀어야 해서 더 무겁게 움직입니다.

계수는 고전 유체역학의 평판 부가질량 공식을 길이 방향으로 더하는 방식(strip theory)으로 선체 크기에서 구합니다.
(물에 뜬 상자를 '높이 = 흘수(T)인 세운 판'과 '폭 B인 바닥 판'으로 봄, 수면 = 거울면)
  좌우(sway)  a22 = ρπT²L/2          위아래(heave) a33 = ρπB²L/8
  앞뒤(surge) a11 = 질량의 5% (가정값, 길쭉한 몸은 보통 수 % 수준)
  롤 a44 = ρπB⁴L/256   피치 a55 = a33·L²/12   요 a66 = a22·L²/12
물에 잠긴 정도가 평소보다 줄면(파도에 들리면) 그만큼 줄입니다.

MuJoCo에 넣는 방법
- 대각 성분: 자유 관절 자유도의 armature(관성 추가)로 넣음 → 암시적으로 계산되어 안정적
  (이동 자유도는 월드 좌표라, 몸체 좌표 계수를 월드로 돌린 행렬의 대각을 넣음)
- 비대각 성분(비스듬히 움직일 때)은 생략: 이전 스텝 가속도로 넣으면 힘이 한쪽으로 쌓여 배가 저절로 떠밀림
- 물이 파도로 가속할 때 몸을 같이 끄는 힘: F = M_A · (물 입자 가속도)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class AddedMassSpec:
    body: str
    joint: str                 # 그 몸의 자유 관절
    size: list[float]          # 상자 크기 [L, B, H]
    mass: float                # 물에 뜨는 데 쓰이는 전체 질량 [kg]
    surge_ratio: float = 0.05


def box_coefficients(size, mass: float, rho: float, surge_ratio: float = 0.05) -> tuple[np.ndarray, np.ndarray, float]:
    """(이동 계수 [a11, a22, a33], 회전 계수 [a44, a55, a66], 평소 잠김 비율)"""
    L, B, H = (float(v) for v in size)
    sub = min(1.0, mass / (rho * L * B * H))
    T = sub * H
    a22 = rho * math.pi * T ** 2 * L / 2
    a33 = rho * math.pi * B ** 2 * L / 8
    lin = np.array([surge_ratio * mass, a22, a33])
    rot = np.array([rho * math.pi * B ** 4 * L / 256, a33 * L ** 2 / 12, a22 * L ** 2 / 12])
    return lin, rot, sub


class AddedMass:
    def __init__(self, model: mujoco.MjModel, specs: list[AddedMassSpec], rho: float, scale: float = 1.0):
        self.items = []
        for s in specs:
            lin, rot, sub = box_coefficients(s.size, s.mass, rho, s.surge_ratio)
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s.body)
            dof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, s.joint)]
            self.items.append((bid, dof, lin, rot, sub))
        self.base_armature = model.dof_armature.copy()
        self.scale = scale

    def coefficients(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        _, _, lin, rot, _ = self.items[i]
        return lin * self.scale, rot * self.scale

    def apply(self, model, data, sub_frac, waves, t: float, wrench: np.ndarray):
        """sub_frac: 몸별 지금 잠긴 비율 (items 순서). armature를 바꾸고 비대각·물 가속 힘을 wrench에 더함."""
        for i, (bid, dof, lin, rot, sub0) in enumerate(self.items):
            f = self.scale * min(1.0, float(sub_frac[i]) / max(sub0, 1e-6))
            R = data.xmat[bid].reshape(3, 3)
            Mw = R @ np.diag(lin * f) @ R.T                       # 몸체 → 월드
            model.dof_armature[dof:dof + 3] = self.base_armature[dof:dof + 3] + np.diag(Mw)
            model.dof_armature[dof + 3:dof + 6] = self.base_armature[dof + 3:dof + 6] + rot * f
            # 비대각 성분은 넣지 않음: 이전 스텝 가속도로 계산하면 한 박자 늦은 힘이 한쪽으로 쌓여
            # 엔진이 꺼져 있어도 파도 속에서 배가 떠밀려 감 (파고 1 m에서 30초에 약 22 m)
            a_water = waves.acceleration(data.xipos[bid], t)
            wrench[bid, :3] += Mw @ a_water
