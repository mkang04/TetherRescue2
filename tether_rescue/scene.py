"""MuJoCo 장면(MJCF) 만들기: 로봇(구조정) + 사람(머리+몸통) + 줄(그림용) + 물 표면(그림용).

물리적인 물 효과(부력·저항)와 줄 힘은 MuJoCo가 아니라 파이썬에서 계산해 넣습니다.
여기 있는 줄(tendon)과 물 표면(hfield)은 화면에 보여 주기 위한 것이며 힘을 내지 않습니다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TowConfig

# 물 표면 그림(hfield) 범위
WATER_CENTER_X = 25.0
WATER_HALF_X = 45.0
WATER_HALF_Y = 30.0
WATER_NCOL = 121      # x 방향 칸 수 (0.75 m 간격)
WATER_NROW = 81       # y 방향 칸 수
WATER_ELEV = 4.0      # 표현 가능한 높이 범위 (-2 ~ +2 m)


@dataclass
class SceneLayout:
    robot_pos: np.ndarray
    torso_pos: np.ndarray
    head_offset: np.ndarray


def _fmt(v) -> str:
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


def compute_layout(cfg: TowConfig) -> SceneLayout:
    r, h, t = cfg.robot, cfg.human, cfg.tether
    rho = cfg.ocean.rho

    # 로봇: 물에 잠기는 깊이(흘수) = 질량 / (밀도 * 바닥 면적)
    draft = r.mass / (rho * r.length * r.width)
    robot_pos = np.array([0.0, 0.0, -draft + r.height / 2])

    # 사람: 몸통이 머리 무게 대부분을 떠받친다고 보고 대략적인 평형 위치에서 시작
    L, W, H = h.torso_size
    head_vol = 4 / 3 * np.pi * h.head_radius**3
    net = h.torso_mass + h.head_mass - rho * head_vol * 0.5
    t_draft = min(net / (rho * L * W), H)
    torso_z = -t_draft + H / 2

    tow_world_x = robot_pos[0] + r.tow_point[0]
    harness_world_x = tow_world_x - (t.rest_length - h.start_gap)
    torso_x = harness_world_x - h.harness_point[0]
    torso_pos = np.array([torso_x, 0.0, torso_z])
    head_offset = np.array([L / 2 + 0.8 * h.head_radius, 0.0, H / 2 - 0.02])
    return SceneLayout(robot_pos, torso_pos, head_offset)


# --------------------------------------------------------------------------- 겉모양 (힘·충돌 없음)
# 물리 계산(부력·저항·충돌)은 지금까지처럼 상자 모양 그대로이고, 아래는 '보이는 모양'만 정합니다.
# 보이는 부품은 질량 0, 충돌 없음(contype=0) → 시뮬레이션 결과에 영향 없음.
TUBE_MAJOR = 0.28      # 구명튜브 고리 반지름 (튜브 중심선) [m]  → 바깥 지름 약 0.68 m
TUBE_MINOR = 0.06      # 튜브 굵기 반지름 [m]
TUBE_SEGMENTS = 16     # 고리를 이루는 캡슐 개수 (4개마다 흰 줄무늬)
TUBE_CHEST_X = 0.55    # 몸통 기준 튜브 위치 (가슴, 겨드랑이 아래)
_VIS = 'contype="0" conaffinity="0" mass="0"'


def _boat_hull_vertices(L: float, W: float, H: float) -> np.ndarray:
    """작은 구조정 선체 꼭짓점 (뾰족한 뱃머리, V자 바닥). MuJoCo가 볼록 껍질로 면을 만듭니다."""
    base = np.array([
        # 갑판 (위)                     # 바닥 (아래)
        [-0.70, 0.30, 0.15], [-0.70, -0.30, 0.15],
        [0.25, 0.30, 0.15], [0.25, -0.30, 0.15],
        [0.52, 0.19, 0.16], [0.52, -0.19, 0.16],
        [0.70, 0.00, 0.19],                                   # 뱃머리 끝 (살짝 들림)
        [-0.70, 0.22, -0.12], [-0.70, -0.22, -0.12],
        [0.15, 0.22, -0.12], [0.15, -0.22, -0.12],
        [-0.70, 0.00, -0.17], [0.20, 0.00, -0.17],            # 용골 (V자 바닥)
        [0.52, 0.00, -0.04],                                  # 비스듬한 뱃머리 아래
    ])
    return base * np.array([L / 1.4, W / 0.6, H / 0.3])


def _ring(prefix: str, center, axis: str, rgba_a: str, rgba_b: str, group: int = 0) -> str:
    """캡슐을 둥글게 이어 붙인 구명튜브 고리. axis='x'면 고리 면이 y-z, 'z'면 x-y."""
    c = np.asarray(center, float)
    out = []
    th = np.linspace(0, 2 * np.pi, TUBE_SEGMENTS + 1)
    for i in range(TUBE_SEGMENTS):
        if axis == "x":
            p0 = c + TUBE_MAJOR * np.array([0.0, np.cos(th[i]), np.sin(th[i])])
            p1 = c + TUBE_MAJOR * np.array([0.0, np.cos(th[i + 1]), np.sin(th[i + 1])])
        else:
            p0 = c + TUBE_MAJOR * np.array([np.cos(th[i]), np.sin(th[i]), 0.0])
            p1 = c + TUBE_MAJOR * np.array([np.cos(th[i + 1]), np.sin(th[i + 1]), 0.0])
        rgba = rgba_b if i % 4 == 0 else rgba_a
        out.append(f'<geom name="{prefix}{i}" type="capsule" size="{TUBE_MINOR}" fromto="{_fmt(p0)} {_fmt(p1)}" '
                   f'rgba="{rgba}" group="{group}" {_VIS}/>')
    return "\n      ".join(out)


def set_tube_visibility(model, attached: bool):
    """줄 연결 전: 갑판 위 튜브만 보이고 줄은 안 보임 / 연결 후: 사람 가슴의 튜브와 줄이 보임."""
    tid = model.tendon("tether_vis").id
    model.tendon_rgba[tid, 3] = 1.0 if attached else 0.0
    for i in range(model.ngeom):
        name = model.geom(i).name
        if name.startswith("tube_robot_"):
            model.geom_rgba[i, 3] = 0.0 if attached else 1.0
        elif name.startswith("tube_human_"):
            model.geom_rgba[i, 3] = 1.0 if attached else 0.0


def build_mjcf(cfg: TowConfig) -> str:
    r, h = cfg.robot, cfg.human
    lay = compute_layout(cfg)
    dt = 1.0 / cfg.sim.physics_hz
    hs = np.array(h.torso_size) / 2
    rs = np.array([r.length, r.width, r.height]) / 2
    hull_v = " ".join(f"{v:.4f}" for v in _boat_hull_vertices(r.length, r.width, r.height).ravel())
    top = rs[2]                                    # 갑판 높이 (몸체 좌표)
    sx = r.length / 1.4                            # 부품 위치를 선체 크기에 맞춰 늘림
    tube_robot = _ring("tube_robot_", [0.12 * sx, 0.0, top + TUBE_MINOR], "z", "1 0.35 0.05 1", "0.97 0.97 0.97 1")
    tube_human = _ring("tube_human_", [TUBE_CHEST_X, 0.0, 0.0], "x", "1 0.35 0.05 1", "0.97 0.97 0.97 1")
    post = np.array([-0.62 * sx, 0.0, top + 0.10])     # 견인 기둥 꼭대기 (줄이 시작하는 곳으로 보임)

    return f"""
<mujoco model="airway_tow">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{dt:.6g}" integrator="implicitfast" gravity="0 0 -{cfg.ocean.g}"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
    <map znear="0.01"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.75 0.95" rgb2="0.95 0.97 1" width="256" height="256"/>
    <hfield name="water" nrow="{WATER_NROW}" ncol="{WATER_NCOL}" size="{WATER_HALF_X} {WATER_HALF_Y} {WATER_ELEV} 0.05"/>
    <material name="water" rgba="0.1 0.4 0.7 0.55" specular="0.6" shininess="0.8"/>
    <material name="hull" rgba="1 0.45 0.05 1" specular="0.4" shininess="0.6"/>
    <mesh name="boat_hull" vertex="{hull_v}"/>
    <material name="skin" rgba="0.95 0.75 0.6 1"/>
    <material name="suit" rgba="0.2 0.25 0.35 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 30" dir="0 0 -1" directional="true"/>
    <geom name="water_surface" type="hfield" hfield="water" material="water"
          pos="{WATER_CENTER_X} 0 {-WATER_ELEV / 2}" contype="0" conaffinity="0" group="1"/>
    <site name="goal" pos="{lay.torso_pos[0] + cfg.sim.goal_distance} 0 0" type="cylinder" size="0.3 1.5" rgba="0 0.9 0.2 0.6"/>

    <body name="robot" pos="{_fmt(lay.robot_pos)}">
      <freejoint name="robot_free"/>
      <!-- 물리용 선체: 상자 (질량·충돌). 그룹 3이라 화면에는 안 보임 -->
      <geom name="hull" type="box" size="{_fmt(rs)}" mass="{r.mass}" rgba="1 0.45 0.05 0.3" group="3"/>
      <!-- 보이는 모양: 작은 구조정 -->
      <geom name="boat_hull" type="mesh" mesh="boat_hull" material="hull" {_VIS}/>
      <geom name="rub_rail_l" type="box" size="{0.475 * sx:.4g} 0.014 0.022" pos="{-0.225 * sx:.4g} {rs[1]:.4g} {top - 0.025:.4g}"
            rgba="0.15 0.15 0.18 1" {_VIS}/>
      <geom name="rub_rail_r" type="box" size="{0.475 * sx:.4g} 0.014 0.022" pos="{-0.225 * sx:.4g} {-rs[1]:.4g} {top - 0.025:.4g}"
            rgba="0.15 0.15 0.18 1" {_VIS}/>
      <geom name="deck_pad" type="box" size="{0.33 * sx:.4g} {rs[1] * 0.8:.4g} 0.004" pos="{-0.30 * sx:.4g} 0 {top + 0.004:.4g}"
            rgba="0.92 0.92 0.92 1" {_VIS}/>
      <geom name="console" type="box" size="0.10 0.14 0.07" pos="{-0.38 * sx:.4g} 0 {top + 0.07:.4g}"
            rgba="0.95 0.95 0.97 1" {_VIS}/>
      <geom name="console_screen" type="box" size="0.005 0.10 0.04" pos="{-0.38 * sx + 0.10:.4g} 0 {top + 0.09:.4g}"
            rgba="0.1 0.2 0.3 1" {_VIS}/>
      <geom name="tow_post" type="cylinder" size="0.025" fromto="{post[0]:.4g} 0 {top:.4g} {_fmt(post)}"
            rgba="1 0.85 0.1 1" {_VIS}/>
      <geom name="motor_cowl" type="box" size="0.07 0.06 0.09" pos="{-rs[0] - 0.06:.4g} 0 {top + 0.06:.4g}"
            rgba="0.2 0.2 0.22 1" {_VIS}/>
      <geom name="motor_shaft" type="cylinder" size="0.022" fromto="{-rs[0] - 0.06:.4g} 0 {top:.4g} {-rs[0] - 0.06:.4g} 0 {-rs[2] - 0.05:.4g}"
            rgba="0.25 0.25 0.27 1" {_VIS}/>
      <geom name="propeller" type="cylinder" size="0.07 0.012" pos="{-rs[0] - 0.09:.4g} 0 {-rs[2] - 0.05:.4g}" euler="0 1.5708 0"
            rgba="0.75 0.75 0.78 1" {_VIS}/>
      <geom name="cam_mast" type="cylinder" size="0.015" fromto="{0.48 * sx:.4g} 0 {top:.4g} {0.48 * sx:.4g} 0 {top + 0.14:.4g}"
            rgba="0.2 0.2 0.2 1" {_VIS}/>
      <geom name="thermal_cam" type="box" size="0.04 0.05 0.035" pos="{0.48 * sx:.4g} 0 {top + 0.17:.4g}"
            rgba="0.08 0.08 0.1 1" {_VIS}/>
      <geom name="thermal_lens" type="cylinder" size="0.025 0.01" pos="{0.48 * sx + 0.045:.4g} 0 {top + 0.17:.4g}" euler="0 1.5708 0"
            rgba="0.3 0.1 0.5 1" {_VIS}/>
      <!-- 갑판에 실린 구명튜브 (줄 연결 전에만 보임) -->
      {tube_robot}
      <site name="imu" pos="0 0 0" size="0.03"/>
      <site name="tow_point" pos="{_fmt(r.tow_point)}" size="0.04" rgba="1 1 0 1" group="3"/>
      <site name="tow_post_top" pos="{_fmt(post)}" size="0.03" rgba="1 0.85 0.1 1" group="3"/>
      <site name="thruster" pos="{-rs[0]:.4g} 0 {-rs[2]:.4g}" size="0.04" rgba="0.2 0.2 0.2 1" group="3"/>
    </body>

    <body name="torso" pos="{_fmt(lay.torso_pos)}">
      <freejoint name="torso_free"/>
      <geom name="torso" type="box" size="{_fmt(hs)}" mass="{h.torso_mass}" material="suit"/>
      <site name="harness" pos="{_fmt(h.harness_point)}" size="0.04" rgba="1 1 0 1" group="3"/>
      <!-- 가슴에 걸린 구명튜브 (줄 끝) -->
      {tube_human}
      <site name="tube_attach" pos="{TUBE_CHEST_X} 0 {TUBE_MAJOR + TUBE_MINOR:.4g}" size="0.03" rgba="1 1 1 1" group="3"/>
      <body name="head" pos="{_fmt(lay.head_offset)}">
        <joint name="neck" type="hinge" axis="0 1 0" stiffness="{h.neck_stiffness}" damping="{h.neck_damping}"
               range="{-h.neck_range} {h.neck_range}"/>
        <geom name="head" type="sphere" size="{h.head_radius}" mass="{h.head_mass}" material="skin"/>
        <site name="mouth" pos="{_fmt(h.mouth_offset)}" size="0.025" rgba="1 0 0 1"/>
      </body>
    </body>
  </worldbody>

  <contact>
    <exclude body1="torso" body2="head"/>
  </contact>

  <tendon>
    <!-- 보이는 줄: 견인 기둥 꼭대기 → 구명튜브 (힘은 파이썬에서 tow_point ↔ harness 로 계산) -->
    <spatial name="tether_vis" width="0.012" rgba="0.95 0.95 0.2 1">
      <site site="tow_post_top"/>
      <site site="tube_attach"/>
    </spatial>
  </tendon>

  <sensor>
    <accelerometer name="imu_acc" site="imu"/>
    <gyro name="imu_gyro" site="imu"/>
  </sensor>
</mujoco>
"""
