#!/bin/bash
# 雙擊執行：從 GitHub 抓最新版程式（只更新程式碼，不動你的受試者資料夾）
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
git pull --ff-only && echo "更新後版本：$(git log -1 --format='%h %ci')" && echo "完成。之前分析過的受試者，再雙擊「一鍵分析」重跑即可（姿態檔已有快取，只需幾十秒）。"
echo
read -n 1 -s -r -p "按任意鍵關閉這個視窗…"
