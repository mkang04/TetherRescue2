"""시뮬레이터 쪽 UDP 연결 (ROS2 없이 동작).

- 시뮬레이터가 상태를 ROS2 브리지로 '먼저' 보냅니다 (기본 1초 10번).
- 브리지는 받은 주소로 목표 좌표를 '답장'합니다.
  → Windows 방화벽은 먼저 보낸 상대의 답장을 통과시키므로, 방화벽 설정 없이 WSL과 주고받을 수 있습니다.
- 브리지가 꺼져 있어도 시뮬레이터는 아무 영향 없이 계속 돕니다 (보내기만 하고 기다리지 않음).

host="auto": Windows면 `wsl hostname -I` 로 WSL 주소를 찾고, 그 외(리눅스/WSL 안)면 127.0.0.1.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time

from ..config import BridgeConfig
from .messages import parse_goal


def find_wsl_ip(distro: str, timeout: float = 15.0) -> str | None:
    try:
        out = subprocess.run(["wsl", "-d", distro, "--", "hostname", "-I"], capture_output=True, timeout=timeout,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        ips = out.stdout.decode(errors="ignore").replace("\x00", "").split()
        return ips[0] if ips else None
    except (OSError, subprocess.TimeoutExpired):
        return None


class SimLink:
    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        self.host: str | None = None
        self.status = "주소 찾는 중..."
        self.last_rx = 0.0
        self._last_tx = 0.0
        self._last_goal, self._last_goal_t = None, 0.0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", 0))
        self.sock.setblocking(False)
        self.closed = False
        threading.Thread(target=self._resolve, daemon=True).start()

    def _resolve(self):
        host = self.cfg.host
        if host == "auto":
            host = find_wsl_ip(self.cfg.wsl_distro) if sys.platform == "win32" else "127.0.0.1"
            if host is None:
                self.status = f"WSL({self.cfg.wsl_distro}) 주소를 못 찾음 → 설정의 ros_bridge.host 확인"
                return
        self.host = host
        self.status = f"보내는 중 → {host}:{self.cfg.port} (브리지 응답 없음)"

    @property
    def bridge_alive(self) -> bool:
        return time.monotonic() - self.last_rx < 3.0

    def maybe_send(self, state: dict) -> bool:
        """rate에 맞춰 상태를 보냄. 보냈으면 True."""
        if self.closed or self.host is None:
            return False
        now = time.monotonic()
        if now - self._last_tx < 1.0 / self.cfg.rate:
            return False
        self._last_tx = now
        try:
            self.sock.sendto(json.dumps(state, separators=(",", ":")).encode(), (self.host, self.cfg.port))
        except OSError as e:
            self.status = f"보내기 실패: {e}"
            return False
        return True

    def poll_goals(self) -> list[tuple[float, float, str]]:
        """브리지가 보낸 목표 좌표들 (rviz2 2D Goal Pose / Publish Point)."""
        goals = []
        while not self.closed:
            try:
                data, _addr = self.sock.recvfrom(65535)
            except (BlockingIOError, ConnectionResetError, OSError):
                break
            self.last_rx = time.monotonic()
            try:
                msg = json.loads(data.decode())
            except (ValueError, UnicodeDecodeError):
                continue
            g = parse_goal(msg)
            if g:
                # 브리지는 유실 대비로 같은 목표를 두 번 보냄 → 1초 안의 같은 좌표는 한 번만
                key = (round(g[0], 3), round(g[1], 3))
                if key == self._last_goal and time.monotonic() - self._last_goal_t < 1.0:
                    continue
                self._last_goal, self._last_goal_t = key, time.monotonic()
                goals.append(g)
        if self.host:
            self.status = (f"브리지 연결됨 ({self.host}:{self.cfg.port})" if self.bridge_alive
                           else f"보내는 중 → {self.host}:{self.cfg.port} (브리지 응답 없음)")
        return goals

    def close(self):
        self.closed = True
        try:
            self.sock.close()
        except OSError:
            pass
