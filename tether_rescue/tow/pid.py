"""PID 조종기와 끌고 오기용 속도·방향 조종 (AirwayTow 기준선)."""
from __future__ import annotations

import math

from ..config import PIDConfig


class PID:
    """출력 제한 + 적분 누적 방지(anti-windup)가 있는 PID."""

    def __init__(self, kp: float, ki: float, kd: float, out_min: float, out_max: float):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_err: float | None = None

    def update(self, err: float, dt: float, d_meas: float | None = None) -> float:
        """d_meas를 주면 미분 항에 오차 미분 대신 그 값을 씁니다 (각속도 등)."""
        if d_meas is not None:
            deriv = d_meas
        elif self.prev_err is None:
            deriv = 0.0
        else:
            deriv = (err - self.prev_err) / dt
        self.prev_err = err

        p_d = self.kp * err + self.kd * deriv
        new_int = self.integral + err * dt
        out = p_d + self.ki * new_int
        # 포화 상태에서 같은 방향으로 더 쌓이면 적분 중지
        if (out > self.out_max and err > 0) or (out < self.out_min and err < 0):
            out = p_d + self.ki * self.integral
        else:
            self.integral = new_int
        return min(max(out, self.out_min), self.out_max)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class TowPIDController:
    """목표 속도로 육지(+x) 쪽을 향해 끌고 오는 조종기.

    출력: throttle (-max_reverse ~ 1), rudder (-1 ~ 1, 최대 조향각 대비 비율)
    """

    def __init__(self, cfg: PIDConfig, max_reverse: float = 0.3, goal_heading: float = 0.0):
        self.cfg = cfg
        self.goal_heading = goal_heading
        self.speed_pid = PID(cfg.speed_kp, cfg.speed_ki, cfg.speed_kd, -max_reverse, 1.0)
        self.heading_pid = PID(cfg.heading_kp, cfg.heading_ki, cfg.heading_kd, -1.0, 1.0)

    def reset(self):
        self.speed_pid.reset()
        self.heading_pid.reset()

    def update_gains(self):
        """cfg의 PID 계수를 다시 읽습니다 (실행 중 조정용). 적분값은 유지합니다."""
        c = self.cfg
        self.speed_pid.kp, self.speed_pid.ki, self.speed_pid.kd = c.speed_kp, c.speed_ki, c.speed_kd
        self.heading_pid.kp, self.heading_pid.ki, self.heading_pid.kd = c.heading_kp, c.heading_ki, c.heading_kd

    def target_speed(self, t: float) -> float:
        if self.cfg.ramp_time <= 0:
            return self.cfg.target_speed
        return self.cfg.target_speed * min(1.0, t / self.cfg.ramp_time)

    def act(self, t: float, dt: float, speed: float, yaw: float, yaw_rate: float) -> tuple[float, float]:
        return self.track(dt, self.target_speed(t), self.goal_heading, speed, yaw, yaw_rate)

    def track(self, dt: float, speed_ref: float, heading_ref: float, speed: float, yaw: float,
              yaw_rate: float) -> tuple[float, float]:
        """원하는 속도·방향을 따라가는 저수준 조종 (구조 임무에서도 재사용)."""
        throttle = self.speed_pid.update(speed_ref - speed, dt)
        err = wrap_angle(heading_ref - yaw)
        # 목표 방향이 천천히 바뀐다고 보면 오차의 미분 ≈ -각속도. 측정값으로 미분해서 튐을 줄임.
        rudder = self.heading_pid.update(err, dt, d_meas=-yaw_rate)
        return throttle, rudder
