@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .git (
  echo 這個資料夾不是從 GitHub 下載的，無法自動更新。
  echo 請到 https://github.com/robotdevatseng/gaitlab 按 Code - Download ZIP，解壓縮後整個取代舊資料夾。
  pause
  exit /b 1
)
where git >nul 2>nul || (
  echo 電腦上沒有 git，請改用 GitHub Desktop 按 Fetch origin / Pull。
  pause
  exit /b 1
)
git pull --ff-only
echo 完成。之前分析過的受試者，再雙擊「一鍵分析」重跑即可。
pause
