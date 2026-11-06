#!/usr/bin/env python3
"""TetherRescue ROS2 브리지 (WSL/우분투, ROS2 Jazzy 에서 실행).

시뮬레이터(Windows) ──UDP 상태──▶ 이 노드 ──▶ /tether_rescue/markers (rviz2 지도 표시)
                     ◀──UDP 목표──       ◀── /goal_pose     (rviz2 "2D Goal Pose" 도구)
                                         ◀── /clicked_point (rviz2 "Publish Point" 도구)

발행 토픽
  /tether_rescue/markers     visualization_msgs/MarkerArray   로봇·익수자·목표·센서 시야·탐지·탐색 경로
  /tether_rescue/robot_pose  geometry_msgs/PoseStamped        로봇 위치 (map 좌표)
  /tether_rescue/state       std_msgs/String                  상태 JSON (녹화·디버그용)
  TF map → base_link                                          로봇 위치

구독 토픽
  /goal_pose      geometry_msgs/PoseStamped    → 시뮬레이터에 목표 좌표로 전달
  /clicked_point  geometry_msgs/PointStamped   → 〃

실행:  bash scripts/run_ros_bridge.sh        (rviz2까지 같이 켬)
       python3 ros2/tether_rescue_bridge.py --port 47800
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rclpy  # noqa: E402
from geometry_msgs.msg import Point, PointStamped, PoseStamped, TransformStamped  # noqa: E402
from rclpy.duration import Duration  # noqa: E402
from rclpy.node import Node  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from tf2_ros import TransformBroadcaster  # noqa: E402
from visualization_msgs.msg import Marker, MarkerArray  # noqa: E402

from tether_rescue.bridge.messages import goal_message, marker_specs  # noqa: E402

MARKER_TYPES = {"ARROW": Marker.ARROW, "CUBE": Marker.CUBE, "SPHERE": Marker.SPHERE, "CYLINDER": Marker.CYLINDER,
                "LINE_STRIP": Marker.LINE_STRIP, "LINE_LIST": Marker.LINE_LIST, "POINTS": Marker.POINTS,
                "TEXT_VIEW_FACING": Marker.TEXT_VIEW_FACING}


def yaw_quat(yaw: float):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


class TetherRescueBridge(Node):
    def __init__(self, port: int, frame: str):
        super().__init__("tether_rescue_bridge")
        self.frame = frame
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.setblocking(False)
        self.sim_addr = None
        self.last_state_time = 0.0
        self.n_states = 0

        self.pub_markers = self.create_publisher(MarkerArray, "/tether_rescue/markers", 10)
        self.pub_pose = self.create_publisher(PoseStamped, "/tether_rescue/robot_pose", 10)
        self.pub_state = self.create_publisher(String, "/tether_rescue/state", 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(PoseStamped, "/goal_pose", self.on_goal_pose, 10)
        self.create_subscription(PointStamped, "/clicked_point", self.on_clicked_point, 10)
        self.create_timer(0.02, self.poll_udp)
        self.create_timer(1.0, self.heartbeat)
        self.get_logger().info(f"UDP {port} 에서 시뮬레이터 상태를 기다립니다. rviz2의 '2D Goal Pose'로 목표를 찍으세요.")

    # ------------------------------------------------------------ 시뮬레이터 → ROS2
    def poll_udp(self):
        latest = None
        while True:
            try:
                data, addr = self.sock.recvfrom(65535)
            except (BlockingIOError, OSError):
                break
            try:
                msg = json.loads(data.decode())
            except (ValueError, UnicodeDecodeError):
                continue
            if msg.get("type") == "state":
                if self.sim_addr != addr:
                    self.get_logger().info(f"시뮬레이터 연결: {addr[0]}:{addr[1]}")
                self.sim_addr, latest = addr, msg
        if latest is not None:
            self.last_state_time = time.monotonic()
            self.n_states += 1
            self.publish_state(latest)

    def publish_state(self, st: dict):
        now = self.get_clock().now().to_msg()
        x, y, yaw = st["robot"]
        qx, qy, qz, qw = yaw_quat(yaw)

        pose = PoseStamped()
        pose.header.frame_id, pose.header.stamp = self.frame, now
        pose.pose.position.x, pose.pose.position.y = float(x), float(y)
        pose.pose.orientation.x, pose.pose.orientation.y = qx, qy
        pose.pose.orientation.z, pose.pose.orientation.w = qz, qw
        self.pub_pose.publish(pose)

        tf = TransformStamped()
        tf.header.frame_id, tf.header.stamp, tf.child_frame_id = self.frame, now, "base_link"
        tf.transform.translation.x, tf.transform.translation.y = float(x), float(y)
        tf.transform.rotation.x, tf.transform.rotation.y = qx, qy
        tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
        self.tf.sendTransform(tf)

        arr = MarkerArray()
        for spec in marker_specs(st):
            arr.markers.append(self.to_marker(spec, now))
        self.pub_markers.publish(arr)
        self.pub_state.publish(String(data=json.dumps(st)))

    def to_marker(self, s: dict, stamp) -> Marker:
        m = Marker()
        m.header.frame_id, m.header.stamp = self.frame, stamp
        m.ns, m.id, m.type, m.action = s["ns"], int(s["id"]), MARKER_TYPES[s["type"]], Marker.ADD
        x, y, z, yaw = s["pose"]
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), float(z)
        qx, qy, qz, qw = yaw_quat(yaw)
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = qx, qy, qz, qw
        m.scale.x, m.scale.y, m.scale.z = (float(v) for v in s["scale"])
        m.color.r, m.color.g, m.color.b, m.color.a = (float(v) for v in s["color"])
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in s["points"]]
        m.text = s["text"]
        m.lifetime = Duration(seconds=0.6).to_msg()      # 상태가 끊기면 사라지게
        return m

    # ------------------------------------------------------------ ROS2 → 시뮬레이터
    def send_goal(self, x: float, y: float, source: str):
        if self.sim_addr is None:
            self.get_logger().warn("아직 시뮬레이터 상태를 못 받아서 목표를 보낼 곳을 모릅니다. 시뮬레이터에서 'ROS2 연결'을 켜세요.")
            return
        data = json.dumps(goal_message(x, y, source)).encode()
        for _ in range(2):                                # UDP 유실 대비 두 번
            self.sock.sendto(data, self.sim_addr)
        self.get_logger().info(f"목표 좌표 전송 ({source}): x={x:.1f} y={y:.1f}")

    def on_goal_pose(self, msg: PoseStamped):
        self.send_goal(msg.pose.position.x, msg.pose.position.y, "rviz2 2D Goal Pose")

    def on_clicked_point(self, msg: PointStamped):
        self.send_goal(msg.point.x, msg.point.y, "rviz2 Publish Point")

    def heartbeat(self):
        if self.sim_addr is not None:
            self.sock.sendto(b'{"type":"ack"}', self.sim_addr)   # 시뮬레이터 패널에 '브리지 연결됨' 표시용


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=47800)
    ap.add_argument("--frame", default="map")
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = TetherRescueBridge(args.port, args.frame)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
