"""鎖版安裝：缺 設定.txt 時從範例複製，且 設定.txt 不進 git。

執行：python tests/test_settings_lock.py
不需要影片。會在暫存資料夾裡用本機 git 演練一次 Mac 更新腳本。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaitlab import auto  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(cond, msg):
    print(("  ✔ " if cond else "  ✘ ") + msg)
    if not cond:
        raise SystemExit(f"失敗：{msg}")


def _ignore_names():
    path = os.path.join(ROOT, ".gitignore")
    names = []
    for line in open(path, encoding="utf-8"):
        names.append(line.split("#", 1)[0].strip())
    return [n for n in names if n]


def test_gitignore_lists_settings():
    print(".gitignore 含 設定.txt，不含範例")
    names = _ignore_names()
    check("設定.txt" in names, ".gitignore 有 設定.txt")
    check("設定.範例.txt" not in names, ".gitignore 沒有排除 設定.範例.txt")
    listed = subprocess.run(["git", "check-ignore", "--no-index", "-q", "設定.txt"], cwd=ROOT)
    check(listed.returncode == 0, "git check-ignore 認定 設定.txt 被忽略")
    example = subprocess.run(["git", "check-ignore", "--no-index", "-q", "設定.範例.txt"], cwd=ROOT)
    check(example.returncode != 0, "設定.範例.txt 仍會被追蹤")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", "設定.txt"], cwd=ROOT,
                             capture_output=True, text=True)
    check(tracked.returncode != 0, "設定.txt 已移出版控")


def test_missing_settings_copied_from_example():
    print("沒有 設定.txt 時從範例複製，已有的檔不覆蓋")
    d = tempfile.mkdtemp()
    old_settings, old_example = auto.SETTINGS_FILE, auto.EXAMPLE_FILE
    try:
        example = os.path.join(d, "設定.範例.txt")
        settings = os.path.join(d, "設定.txt")
        open(example, "w", encoding="utf-8").write("# 範例\n量測區寬度 = 1.0\n赤腳 = 否\n")
        auto.EXAMPLE_FILE = example
        auto.SETTINGS_FILE = settings
        vals = auto.load_settings()
        check(os.path.exists(settings), "已產生 設定.txt")
        check(open(settings, encoding="utf-8").read() == open(example, encoding="utf-8").read(),
              "內容與範例相同")
        check(vals["量測區寬度"] == 1.0 and vals["赤腳"] == "否", "讀到範例裡的值")
        open(settings, "w", encoding="utf-8").write("量測區寬度 = 9.9\n赤腳 = 否\n")
        again = auto.load_settings()
        check(again["量測區寬度"] == 9.9, "已有 設定.txt 時不覆蓋")
        check("9.9" in open(settings, encoding="utf-8").read(), "檔案內容維持使用者改過的值")
    finally:
        auto.SETTINGS_FILE, auto.EXAMPLE_FILE = old_settings, old_example
        shutil.rmtree(d)


def _git(args, cwd, env):
    subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True)


def _git_env():
    env = os.environ.copy()
    env.update({
        "GIT_AUTHOR_NAME": "lock-test",
        "GIT_AUTHOR_EMAIL": "lock-test@example.com",
        "GIT_COMMITTER_NAME": "lock-test",
        "GIT_COMMITTER_EMAIL": "lock-test@example.com",
    })
    return env


def _repo_with_tag(root):
    """做一個只有本機 origin 的倉庫，tag v0.4.2 裡有被追蹤的 設定.txt。"""
    env = _git_env()
    bare = os.path.join(root, "bare.git")
    work = os.path.join(root, "work")
    _git(["init", "--bare", "-b", "main", bare], root, env)
    _git(["clone", bare, work], root, env)
    script = os.path.join(work, "更新程式.command")
    shutil.copy(os.path.join(ROOT, "更新程式.command"), script)
    os.chmod(script, 0o755)
    open(os.path.join(work, "設定.txt"), "w", encoding="utf-8").write("TAG-SETTINGS\n")
    open(os.path.join(work, "README"), "w", encoding="utf-8").write("x\n")
    _git(["add", "設定.txt", "README", "更新程式.command"], work, env)
    _git(["commit", "-m", "base"], work, env)
    _git(["tag", "-a", "v0.4.2", "-m", "gaitlab 0.4.1 鎖版"], work, env)
    _git(["push", "origin", "HEAD", "v0.4.2"], work, env)
    return work, env


def test_update_command_keeps_modified_settings():
    print("Mac 更新腳本：設定.txt 有修改時仍切到 v0.4.2，且不覆蓋")
    d = tempfile.mkdtemp()
    try:
        work, env = _repo_with_tag(d)
        open(os.path.join(work, "設定.txt"), "w", encoding="utf-8").write("USER-SETTINGS\n")
        proc = subprocess.run(["bash", os.path.join(work, "更新程式.command")],
                               input="x", text=True, capture_output=True, cwd=work, env=env)
        check(proc.returncode == 0, f"更新腳本結束碼 0（{proc.returncode}）\n{proc.stdout}\n{proc.stderr}")
        got = open(os.path.join(work, "設定.txt"), encoding="utf-8").read()
        check(got == "USER-SETTINGS\n", f"本機設定還在（實際 {got!r}）")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=work, text=True).strip()
        tag = subprocess.check_output(["git", "rev-parse", "v0.4.2^{}"], cwd=work, text=True).strip()
        check(head == tag, "工作目錄停在 v0.4.2")
        command = open(os.path.join(ROOT, "更新程式.command"), encoding="utf-8").read()
        check("git pull" not in command and "git fetch origin --tags" in command and "checkout v0.4.2" in command,
              "Mac 腳本改為 fetch tags 後 checkout v0.4.2")
        bat = open(os.path.join(ROOT, "更新程式.bat"), encoding="utf-8").read()
        check("git pull" not in bat and "git fetch origin --tags" in bat and "checkout v0.4.2" in bat,
              "Windows 腳本改為 fetch tags 後 checkout v0.4.2")
        check("gaitlab-settings-backup.txt" not in bat and "%RANDOM%" in bat,
              "Windows 備份檔名含 %RANDOM%，不是固定檔名")
        check("if errorlevel 1 goto :restore_copy_fail" in bat and "請用這個備份手動還原" in bat,
              "Windows 還原失敗才印備份路徑，成功才說已還原")
        check('git checkout HEAD -- "設定.txt"' in bat and 'git checkout HEAD -- "設定.txt"' in command,
              "兩邊都會清掉殘留的刪除狀態")
    finally:
        shutil.rmtree(d)


def test_update_command_keeps_untracked_settings():
    print("Mac 更新腳本：未被追蹤的 設定.txt 不會讓 checkout 失敗")
    d = tempfile.mkdtemp()
    try:
        work, env = _repo_with_tag(d)
        _git(["rm", "設定.txt"], work, env)
        _git(["commit", "-m", "stop tracking settings"], work, env)
        _git(["push", "origin", "HEAD"], work, env)
        open(os.path.join(work, "設定.txt"), "w", encoding="utf-8").write("USER-UNTRACKED\n")
        proc = subprocess.run(["bash", os.path.join(work, "更新程式.command")],
                               input="x", text=True, capture_output=True, cwd=work, env=env)
        check(proc.returncode == 0, f"更新腳本結束碼 0（{proc.returncode}）\n{proc.stdout}\n{proc.stderr}")
        got = open(os.path.join(work, "設定.txt"), encoding="utf-8").read()
        check(got == "USER-UNTRACKED\n", f"未追蹤的本機設定還在（實際 {got!r}）")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=work, text=True).strip()
        tag = subprocess.check_output(["git", "rev-parse", "v0.4.2^{}"], cwd=work, text=True).strip()
        check(head == tag, "工作目錄停在 v0.4.2")
    finally:
        shutil.rmtree(d)


def test_deleted_tracked_settings_leaves_clean_tree():
    print("本機刪掉仍被追蹤的 設定.txt 後，更新完工作樹不再留著刪除狀態")
    d = tempfile.mkdtemp()
    try:
        work, env = _repo_with_tag(d)
        os.remove(os.path.join(work, "設定.txt"))
        proc = subprocess.run(["bash", os.path.join(work, "更新程式.command")],
                               input="x", text=True, capture_output=True, cwd=work, env=env)
        check(proc.returncode == 0, f"更新腳本結束碼 0（{proc.returncode}）\n{proc.stdout}\n{proc.stderr}")
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=work, text=True)
        check("設定.txt" not in status, f"工作樹沒有留下刪除狀態（實際 {status!r}）")
        got = open(os.path.join(work, "設定.txt"), encoding="utf-8").read()
        check(got == "TAG-SETTINGS\n", f"刪掉的追蹤檔回到鎖版內容（實際 {got!r}）")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=work, text=True).strip()
        tag = subprocess.check_output(["git", "rev-parse", "v0.4.2^{}"], cwd=work, text=True).strip()
        check(head == tag, "工作目錄停在 v0.4.2")
    finally:
        shutil.rmtree(d)


def main():
    test_gitignore_lists_settings()
    test_missing_settings_copied_from_example()
    test_update_command_keeps_modified_settings()
    test_update_command_keeps_untracked_settings()
    test_deleted_tracked_settings_leaves_clean_tree()
    print("鎖版設定檢查通過 ✔")


if __name__ == "__main__":
    main()
