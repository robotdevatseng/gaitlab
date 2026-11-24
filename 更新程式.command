#!/bin/bash
# 雙擊執行：切換到鎖版 v0.4.2（只更新程式碼，不動你的受試者資料夾與設定.txt）
cd "$(dirname "$0")"
if [ ! -d .git ]; then
  echo "這個資料夾不是從 GitHub 下載的，無法自動更新。"
  echo "請到 https://github.com/robotdevatseng/gaitlab 按 Code → Download ZIP，解壓縮後整個取代舊資料夾（受試者資料夾另外放，不受影響）。"
  read -n 1 -s -r -p "按任意鍵關閉這個視窗…"; exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "電腦上沒有 git。Mac 請在終端機輸入 xcode-select --install，或改用 GitHub Desktop 按 Fetch origin / Pull。"
  read -n 1 -s -r -p "按任意鍵關閉這個視窗…"; exit 1
fi
echo "目前版本：$(git log -1 --format='%h %ci')"
echo "正在抓取標籤，並切換到鎖版 v0.4.2（不再更新到 main）。"

bak=""
cleared_deletion=0
restore_settings() {
  if [ -n "$bak" ] && [ -f "$bak" ]; then
    if cp -p "$bak" "設定.txt"; then
      rm -f "$bak"
      bak=""
      echo "你的 設定.txt 已還原。"
    else
      echo "還原 設定.txt 失敗。請用這個備份手動還原：$bak"
    fi
  fi
}
pause_exit() {
  echo
  read -n 1 -s -r -p "按任意鍵關閉這個視窗…"
  exit "$1"
}

if [ -f "設定.txt" ]; then
  tmp="$(mktemp "${TMPDIR:-/tmp}/gaitlab-settings.XXXXXX")" || {
    echo "無法備份 設定.txt，已停止，沒有更新。"
    pause_exit 1
  }
  if ! cp -p "設定.txt" "$tmp"; then
    rm -f "$tmp"
    echo "無法備份 設定.txt，已停止，沒有更新。"
    pause_exit 1
  fi
  bak="$tmp"
  if git ls-files --error-unmatch -- "設定.txt" >/dev/null 2>&1; then
    if ! git checkout -- "設定.txt"; then
      echo "更新失敗。常見原因：沒有網路，或程式資料夾裡的檔案被改過。請把這個畫面拍下來回報。"
      restore_settings
      pause_exit 1
    fi
  else
    rm -f "設定.txt"
  fi
elif git ls-files --error-unmatch -- "設定.txt" >/dev/null 2>&1; then
  # 本機已刪掉、舊版仍追蹤：先清掉刪除狀態，避免 checkout 把它留在工作樹。
  if ! git checkout -- "設定.txt"; then
    echo "更新失敗。常見原因：沒有網路，或程式資料夾裡的檔案被改過。請把這個畫面拍下來回報。"
    pause_exit 1
  fi
  cleared_deletion=1
fi

fail_update() {
  echo "更新失敗。常見原因：沒有網路，或程式資料夾裡的檔案被改過。請把這個畫面拍下來回報。"
  if [ -n "$bak" ]; then
    if cp -p "$bak" "設定.txt"; then
      rm -f "$bak"
      bak=""
      echo "你的 設定.txt 已還原。"
    else
      echo "還原 設定.txt 失敗。請用這個備份手動還原：$bak"
    fi
  elif [ "$cleared_deletion" = 1 ]; then
    rm -f "設定.txt"
  fi
  pause_exit 1
}

if ! git fetch origin --tags; then
  fail_update
fi
if ! git -c advice.detachedHead=false checkout v0.4.2; then
  fail_update
fi

if [ -n "$bak" ]; then
  if ! cp -p "$bak" "設定.txt"; then
    echo "已切換到 v0.4.2，但還原 設定.txt 失敗。備份在：$bak"
    pause_exit 1
  fi
  rm -f "$bak"
  bak=""
  echo "已保留你本機的 設定.txt，沒有用鎖版裡的設定蓋過。"
fi
if [ ! -f "設定.txt" ] && git ls-files --error-unmatch -- "設定.txt" >/dev/null 2>&1; then
  if ! git checkout HEAD -- "設定.txt"; then
    fail_update
  fi
fi
echo "更新後版本：$(git log -1 --format='%h %ci')（v0.4.2）"
echo "完成。之前分析過的受試者，再雙擊「一鍵分析」重跑即可（姿態檔已有快取，只需幾十秒）。"
pause_exit 0
