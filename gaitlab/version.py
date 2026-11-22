"""程式版本：把目前的 git commit 寫進輸出，讓每一份結果都能對回是哪一版程式算的。

取不到 git 資訊時（例如用 Download ZIP 取得的程式資料夾），一律寫明「未取得」，不留空白、不虛構 hash。
"""
from __future__ import annotations

import os
import re
import subprocess

PROGRAM_VERSION = "0.4.0"      # 人工維護的版本號；每次鎖定版本時更新
UNKNOWN = "未取得"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _git(root: str, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _read_head_without_git(root: str) -> str | None:
    """電腦上沒有 git 指令、但資料夾是 git 工作複本時（例如只裝了 GitHub Desktop），直接讀 .git 裡的紀錄。"""
    gd = os.path.join(root, ".git")
    try:
        if os.path.isfile(gd):                      # worktree：.git 是一個指向實際位置的文字檔
            txt = open(gd, encoding="utf-8").read().strip()
            if txt.startswith("gitdir:"):
                gd = os.path.normpath(os.path.join(root, txt.split(":", 1)[1].strip()))
        head = open(os.path.join(gd, "HEAD"), encoding="utf-8").read().strip()
        if _HEX40.match(head):
            return head
        if not head.startswith("ref:"):
            return None
        ref = head.split(":", 1)[1].strip()
        common = gd
        cfile = os.path.join(gd, "commondir")
        if os.path.exists(cfile):
            common = os.path.normpath(os.path.join(gd, open(cfile, encoding="utf-8").read().strip()))
        for base in (gd, common):
            p = os.path.join(base, *ref.split("/"))
            if os.path.exists(p):
                h = open(p, encoding="utf-8").read().strip()
                return h if _HEX40.match(h) else None
        packed = os.path.join(common, "packed-refs")
        if os.path.exists(packed):
            for line in open(packed, encoding="utf-8"):
                parts = line.strip().split(" ")
                if len(parts) == 2 and parts[1] == ref and _HEX40.match(parts[0]):
                    return parts[0]
    except Exception:
        return None
    return None


def get_version(root: str | None = None) -> dict:
    """回傳 {"commit": 完整 hash 或「未取得」, "commit_short": 前 7 碼或「未取得」,
    "program_version": 版本號, "worktree_modified": True／False／None, "source": 取得方式}。

    只認「程式資料夾本身就是 git 工作複本的最上層」：避免程式資料夾被放在別的 git 專案底下時，誤抓到別人的 commit。
    """
    root = os.path.abspath(root or ROOT)
    commit, modified, source = None, None, UNKNOWN
    if os.path.exists(os.path.join(root, ".git")):
        top = _git(root, "rev-parse", "--show-toplevel")
        if top is not None and os.path.realpath(top) == os.path.realpath(root):
            h = _git(root, "rev-parse", "HEAD")
            if h and _HEX40.match(h):
                commit, source = h, "git 指令"
                st = _git(root, "status", "--porcelain", "--untracked-files=no")
                modified = bool(st) if st is not None else None
        if commit is None:
            h = _read_head_without_git(root)
            if h:
                commit, source = h, "讀取 .git 目錄"
    if commit is None:
        return {"commit": UNKNOWN, "commit_short": UNKNOWN, "program_version": PROGRAM_VERSION,
                "worktree_modified": None, "source": UNKNOWN}
    return {"commit": commit, "commit_short": commit[:7], "program_version": PROGRAM_VERSION,
            "worktree_modified": modified, "source": source}


def version_meta(root: str | None = None) -> dict:
    """要併入 summary.json／side_summary.json 的 meta 的欄位。"""
    v = get_version(root)
    return {"commit": v["commit"], "program_version": v["program_version"],
            "commit_worktree_modified": v["worktree_modified"]}
