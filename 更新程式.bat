@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .git (
  echo 這個資料夾不是用 git 下載的，無法自動更新。
  echo 請到 https://github.com/robotdevatseng/gaitlab 按 Code - Download ZIP，
  echo 解壓縮後取代舊的 gaitlab 資料夾；.venv 資料夾可以留著，不用重新安裝。
  pause
  exit /b 1
)
where git >nul 2>nul
if errorlevel 1 (
  echo 電腦上沒有 git。請安裝 Git for Windows（https://git-scm.com/download/win），
  echo 或改用 GitHub Desktop 按 Fetch origin / Pull。
  pause
  exit /b 1
)
for /f "delims=" %%i in ('git log -1 --format^=%%h') do echo 目前版本：%%i
git pull --ff-only
if errorlevel 1 (
  echo 更新失敗。常見原因：沒有網路，或程式資料夾裡的檔案被改過。請把這個畫面拍下來回報。
  pause
  exit /b 1
)
for /f "delims=" %%i in ('git log -1 --format^=%%h') do echo 更新後版本：%%i
echo 完成。不必重跑舊資料，除非研究者要求。
pause
