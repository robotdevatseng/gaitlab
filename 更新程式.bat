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
echo 正在抓取標籤，並切換到鎖版 v0.4.1（不再更新到 main）。

set "SETTINGS_BAK=%TEMP%\gaitlab-settings-backup.txt"
set "HAD_SETTINGS=0"
if not exist "設定.txt" goto :do_update
copy /Y "設定.txt" "%SETTINGS_BAK%" >nul
if errorlevel 1 goto :backup_fail
set "HAD_SETTINGS=1"
git ls-files --error-unmatch -- "設定.txt" >nul 2>nul
if errorlevel 1 (
  del /f /q "設定.txt"
  goto :do_update
)
git checkout -- "設定.txt"
if errorlevel 1 goto :restore_and_fail

:do_update
git fetch origin --tags
if errorlevel 1 goto :restore_and_fail
git -c advice.detachedHead=false checkout v0.4.1
if errorlevel 1 goto :restore_and_fail
if not "%HAD_SETTINGS%"=="1" goto :done
copy /Y "%SETTINGS_BAK%" "設定.txt" >nul
if errorlevel 1 goto :restore_failed
del /f /q "%SETTINGS_BAK%" >nul 2>nul
echo 已保留你本機的 設定.txt，沒有用鎖版裡的設定蓋過。

:done
for /f "delims=" %%i in ('git log -1 --format^=%%h') do echo 更新後版本：%%i（v0.4.1）
echo 完成。不必重跑舊資料，除非研究者要求。
pause
exit /b 0

:backup_fail
echo 無法備份 設定.txt，已停止，沒有更新。
pause
exit /b 1

:restore_failed
echo 已切換到 v0.4.1，但還原 設定.txt 失敗。備份在：%SETTINGS_BAK%
pause
exit /b 1

:restore_and_fail
echo 更新失敗。常見原因：沒有網路，或程式資料夾裡的檔案被改過。請把這個畫面拍下來回報。
if not "%HAD_SETTINGS%"=="1" goto :fail_pause
copy /Y "%SETTINGS_BAK%" "設定.txt" >nul
echo 你的 設定.txt 已還原。
:fail_pause
pause
exit /b 1
