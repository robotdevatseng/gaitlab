#!/bin/bash
# 雙擊執行：選受試者資料夾 → 自動分析正面、側面 → 產生「寄給研究者_受試者.zip」
cd "$(dirname "$0")"
PY=""
if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  # 以前照說明書用 python3.11 安裝過的，直接沿用
  for p in python3.11 python3.12 python3.10 python3; do
    if command -v "$p" >/dev/null 2>&1 && "$p" -c "import mediapipe, scipy, matplotlib" >/dev/null 2>&1; then
      PY="$p"; break
    fi
  done
fi
if [ -z "$PY" ]; then
  echo "還沒安裝：請先雙擊「安裝.command」。"
  read -n 1 -s -r -p "按任意鍵關閉這個視窗…"; exit 1
fi
"$PY" -m gaitlab auto "$@"
echo
read -n 1 -s -r -p "按任意鍵關閉這個視窗…"
