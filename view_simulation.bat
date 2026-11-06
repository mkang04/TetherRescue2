@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [설치] 처음 실행이라 가상환경을 만듭니다. 몇 분 걸릴 수 있어요...
    py -3.12 -m venv .venv || goto :fail
    ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt || goto :fail
)

echo ============================================
echo   AirwayTow 시뮬레이션
echo   - 실험 설정 패널: 값을 바꾸면 바로 반영
echo   - 3D 뷰어: 시뮬레이션 화면
echo   둘 중 한 창을 닫으면 종료됩니다.
echo ============================================

".venv\Scripts\python.exe" -u scripts\view_tow.py %*
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
echo [오류] 실행에 실패했습니다. 위 메시지를 확인하세요.
echo mujoco.dll 에러라면: winget install Microsoft.VCRedist.2015+.x64
pause
exit /b 1
