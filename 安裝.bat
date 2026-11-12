@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo GaitLab 安裝中（需要網路，約 5-10 分鐘）
set BASEPY=
py -3.11 --version >nul 2>nul && set BASEPY=py -3.11
if not defined BASEPY py -3.12 --version >nul 2>nul && set BASEPY=py -3.12
if not defined BASEPY py -3.10 --version >nul 2>nul && set BASEPY=py -3.10
if not defined BASEPY (
  echo 找不到 Python 3.10 - 3.12。請到 https://www.python.org/downloads/ 下載 Python 3.11（64 位元），
  echo 安裝時勾選 "Add python.exe to PATH"，裝好後再雙擊這個檔案。
  pause
  exit /b 1
)
if not exist .venv\Scripts\python.exe %BASEPY% -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip -q
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python tests\check_install.py
pause
