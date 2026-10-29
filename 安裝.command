#!/bin/bash
# 第一次使用（或拿到新版本）時雙擊一次。需要網路，約 5–10 分鐘。
cd "$(dirname "$0")"
echo "GaitLab 安裝中…"
PY=""
for p in python3.11 python3.12 python3.10 python3; do
  if command -v "$p" >/dev/null 2>&1; then
    v=$("$p" -c 'import sys;print("%d.%d"%sys.version_info[:2])')
    case "$v" in 3.10|3.11|3.12) PY="$p"; break;; esac
  fi
done
if [ -z "$PY" ]; then
  echo "找不到 Python 3.10–3.12。請先到 python.org 下載安裝 Python 3.11，再雙擊一次這個檔案。"
  read -n 1 -s -r -p "按任意鍵關閉這個視窗…"; exit 1
fi
echo "使用 $PY（$v）"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip -q
.venv/bin/python -m pip install -r requirements.txt
echo
.venv/bin/python tests/check_install.py
echo
echo "看到「檢查通過 ✔」就完成了。之後每次分析，雙擊「一鍵分析.command」即可。"
read -n 1 -s -r -p "按任意鍵關閉這個視窗…"
