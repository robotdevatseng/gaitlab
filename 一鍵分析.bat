@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
set PY=
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
if not defined PY (
  py -3.11 -c "import mediapipe" >nul 2>nul && set PY=py -3.11
)
if not defined PY (
  echo 還沒安裝：請先雙擊「安裝.bat」
  pause
  exit /b 1
)
%PY% -m gaitlab auto %1
pause
