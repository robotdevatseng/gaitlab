@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo GaitLab 安裝中（需要網路，約 5-10 分鐘）
if not exist .venv\Scripts\python.exe python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip -q
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python tests\check_install.py
pause
