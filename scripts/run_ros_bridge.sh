#!/bin/bash
# WSL(우분투) / 리눅스에서 실행: ROS2 브리지 + rviz2
#   bash scripts/run_ros_bridge.sh            # 브리지 + rviz2
#   NO_RVIZ=1 bash scripts/run_ros_bridge.sh  # 브리지만
cd "$(dirname "$0")/.." || exit 1
ROS_DISTRO_NAME="${ROS_DISTRO_NAME:-jazzy}"
if [ ! -f "/opt/ros/$ROS_DISTRO_NAME/setup.bash" ]; then
    echo "[오류] /opt/ros/$ROS_DISTRO_NAME 이 없습니다. ROS2 $ROS_DISTRO_NAME 를 설치하세요."
    exit 1
fi
# shellcheck disable=SC1090
source "/opt/ros/$ROS_DISTRO_NAME/setup.bash"

python3 ros2/tether_rescue_bridge.py "$@" &
BRIDGE_PID=$!
trap 'kill $BRIDGE_PID 2>/dev/null' EXIT INT TERM

if [ "${NO_RVIZ:-0}" = "1" ]; then
    wait $BRIDGE_PID
else
    rviz2 -d config/rviz/tether_rescue.rviz
fi
