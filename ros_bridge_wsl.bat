@echo off
chcp 65001 > nul
cd /d "%~dp0"
set DISTRO=Ubuntu-24.04

echo ============================================
echo   TetherRescue ROS2 브리지 + rviz2 (WSL %DISTRO%)
echo   1) 이 창은 켜 둔 채로
echo   2) view_simulation.bat 패널에서 [ROS2 연결] 체크
echo   3) rviz2 위쪽 "2D Goal Pose" 를 누르고 지도를 클릭
echo ============================================

wsl -d %DISTRO% --cd "%~dp0" -- bash scripts/run_ros_bridge.sh
if errorlevel 1 (
    echo.
    echo [오류] WSL 실행 실패. "wsl -l -v" 로 배포판 이름을 확인하고 이 파일의 DISTRO 를 고치세요.
    pause
)
