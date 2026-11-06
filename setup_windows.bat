@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo ============================================
echo   TetherRescue 처음 설정 (새 PC에서 한 번만)
echo ============================================

rem 1) Python 3.12 확인
py -3.12 --version > nul 2>&1
if errorlevel 1 (
    echo [필요] Python 3.12 이 없습니다. 아래 명령으로 설치한 뒤 이 파일을 다시 실행하세요.
    echo        winget install Python.Python.3.12
    pause
    exit /b 1
)

rem 2) 가상환경 + 패키지
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] 가상환경 만드는 중...
    py -3.12 -m venv .venv || goto :fail
)
echo [2/3] 패키지 설치 중... (처음에는 몇 분 걸려요)
".venv\Scripts\python.exe" -m pip install -q --upgrade pip
".venv\Scripts\python.exe" -m pip install -q -r requirements.txt || goto :fail

rem 3) MuJoCo 동작 확인 (Visual C++ 재배포 패키지가 없으면 여기서 실패)
echo [3/3] MuJoCo 확인 중...
".venv\Scripts\python.exe" -c "import mujoco; print('MuJoCo', mujoco.__version__, 'OK')"
if errorlevel 1 (
    echo.
    echo [필요] Microsoft Visual C++ 재배포 패키지를 설치합니다. 관리자 권한 창이 뜨면 허용하세요.
    winget install --id Microsoft.VCRedist.2015+.x64 -e --accept-source-agreements --accept-package-agreements
    ".venv\Scripts\python.exe" -c "import mujoco; print('MuJoCo', mujoco.__version__, 'OK')" || goto :fail
)

echo.
echo 설정 완료! 이제 view_simulation.bat 을 실행하세요.
pause
exit /b 0

:fail
echo.
echo [오류] 설정에 실패했습니다. 위 메시지를 확인하세요.
pause
exit /b 1
