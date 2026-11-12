"""逐格姿態估計（完全離線）。

使用 MediaPipe BlazePose（33 個關鍵點，含腳跟 29/30 與腳尖 31/32）。
mediapipe 0.10.21 的安裝檔內已包含 pose_landmark_full 模型，不需要連網下載。
"""
from __future__ import annotations

import numpy as np
import cv2

# BlazePose 關鍵點索引
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32
N_LANDMARKS = 33


_ASCII_COPIES: dict = {}


def _windows_fallback_paths(path: str):
    """Windows 上 OpenCV 有時打不開含中文的路徑（例如 S02\\正面\\IMG_1511.MOV）。
    依序嘗試：8.3 短路徑 → 複製一份到暫存資料夾（純英數檔名）。只在直接開啟失敗時才會用到。"""
    import os
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(1024)
        n = ctypes.windll.kernel32.GetShortPathNameW(os.path.abspath(path), buf, 1024)
        if 0 < n < 1024 and buf.value.isascii():
            yield buf.value
    except Exception:
        pass
    try:
        import hashlib
        import shutil
        import tempfile
        key = os.path.abspath(path)
        if key not in _ASCII_COPIES:
            ext = os.path.splitext(path)[1]
            ext = ext if ext.isascii() else ".mov"
            dst = os.path.join(tempfile.gettempdir(), "gaitlab_" + hashlib.md5(key.encode("utf-8")).hexdigest()[:12] + ext)
            if dst.isascii():
                if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(path)):
                    print("  （Windows：路徑含中文，先複製一份到暫存資料夾再讀取）")
                    shutil.copyfile(path, dst)
                _ASCII_COPIES[key] = dst
        if key in _ASCII_COPIES:
            yield _ASCII_COPIES[key]
    except Exception:
        pass


def open_video(path: str) -> cv2.VideoCapture:
    """開啟影片，並要求依手機的旋轉標記自動轉正（iPad 直拍的影片常帶 90° 標記）。"""
    import os
    cap = cv2.VideoCapture(path)
    if not cap.isOpened() and os.name == "nt" and os.path.exists(path) and not path.isascii():
        for alt in _windows_fallback_paths(path):
            cap = cv2.VideoCapture(alt)
            if cap.isOpened():
                break
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片：{path}（檔名、路徑是否正確？iPad 格式是否設為「最相容」？）")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return cap


def video_info(path: str) -> dict:
    cap = open_video(path)
    ok, frame = cap.read()
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(frame.shape[1]) if ok else 0,   # 以實際影格尺寸為準（已轉正）
        "height": int(frame.shape[0]) if ok else 0,
    }
    cap.release()
    return info


def read_frame(path: str, index: int = 0) -> np.ndarray:
    cap = open_video(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("無法讀取影格")
    return frame


def extract_landmarks(path: str, model_complexity: int = 1, progress: bool = True) -> dict:
    """回傳 dict：xy (T,33,2) 像素座標、vis (T,33) 可見度、t (T,) 秒、fps、size。

    偵測不到人的影格以 NaN 填入。
    """
    import mediapipe as mp  # 延遲載入，讓不需要姿態模型的步驟也能執行

    info = video_info(path)
    fps = info["fps"]
    cap = open_video(path)
    xy, vis, ts = [], [], []
    w = h = None
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        h, w = frame.shape[:2]
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            xy.append([[p.x * w, p.y * h] for p in lm])
            vis.append([p.visibility for p in lm])
        else:
            xy.append(np.full((N_LANDMARKS, 2), np.nan))
            vis.append(np.zeros(N_LANDMARKS))
        i += 1
        if progress and i % 60 == 0:
            print(f"  姿態估計 {i}/{info['n_frames']} 格", flush=True)
    cap.release()
    pose.close()
    xy = np.asarray(xy, dtype=float).reshape(-1, N_LANDMARKS, 2)
    vis = np.asarray(vis, dtype=float).reshape(-1, N_LANDMARKS)
    if len(xy) == 0:
        raise RuntimeError(f"讀不到任何影格：{path}")
    xy, vis, t, fps, vfr = _uniform_time(xy, vis, np.asarray(ts), fps)
    if vfr:
        print(f"  注意：此影片為變動影格率，已重新取樣為 {fps:.1f} fps")
    return {"xy": xy, "vis": vis, "t": t, "fps": fps, "size": (w, h), "vfr": vfr}


def _uniform_time(xy, vis, ts, fps):
    """iPad 在光線不足時會錄成「變動影格率」。用影片內的真實時間戳記重新取樣成等間隔，
    避免把每格當成 1/fps 秒而算錯時間。"""
    n = len(xy)
    t_nom = np.arange(n) / fps
    if n < 3 or len(ts) != n:
        return xy, vis, t_nom, fps, False
    dt = np.diff(ts)
    if not np.all(np.isfinite(ts)) or np.any(dt <= 0):
        return xy, vis, t_nom, fps, False   # 時間戳記不可用，退回名目影格率
    med = float(np.median(dt))
    if np.std(dt) / med < 0.05:
        return xy, vis, ts - ts[0], fps, False
    fps_u = 1.0 / med
    t_src = ts - ts[0]
    t_u = np.arange(0, t_src[-1] + 1e-9, med)
    xy_u = np.full((len(t_u), N_LANDMARKS, 2), np.nan)
    vis_u = np.zeros((len(t_u), N_LANDMARKS))
    for k in range(N_LANDMARKS):
        vis_u[:, k] = np.interp(t_u, t_src, vis[:, k])
        for d in range(2):
            col = xy[:, k, d]
            ok = ~np.isnan(col)
            if ok.sum() >= 2:
                xy_u[:, k, d] = np.interp(t_u, t_src[ok], col[ok], left=np.nan, right=np.nan)
    # 掉格處（附近 1.5 個影格週期內沒有原始影格）設為缺值，不做直線內插
    j = np.clip(np.searchsorted(t_src, t_u), 1, len(t_src) - 1)
    near = np.minimum(np.abs(t_src[j] - t_u), np.abs(t_src[j - 1] - t_u))
    gap = near > 1.5 * med
    xy_u[gap] = np.nan
    vis_u[gap] = 0.0
    return xy_u, vis_u, t_u, fps_u, True


def save_landmarks(lm: dict, path: str) -> None:
    np.savez_compressed(path, xy=lm["xy"], vis=lm["vis"], t=lm["t"],
                        fps=lm["fps"], size=np.array(lm["size"]))


def load_landmarks(path: str) -> dict:
    d = np.load(path)
    return {"xy": d["xy"], "vis": d["vis"], "t": d["t"],
            "fps": float(d["fps"]), "size": tuple(int(v) for v in d["size"])}


NATIVE_SIZES = {(720, 1280), (1080, 1920), (2160, 3840)}


def is_native_size(size) -> bool:
    """影片解析度是否為相機原始規格（720p／1080p／4K，直拍或橫拍）。
    在相簿調色、套濾鏡、裁切或經通訊軟體傳送後，解析度常變成 2116×3762 這類數字，
    畫面被裁切過，地板校正與尺度都不能信任。"""
    try:
        a, b = sorted(int(v) for v in size)
    except Exception:
        return False
    return (a, b) in NATIVE_SIZES


def size_problem(size) -> str | None:
    if is_native_size(size):
        return None
    return (f"影片解析度 {int(size[0])}×{int(size[1])} 不是相機原始規格（720p／1080p／4K），"
            "表示影片被編輯（調色、濾鏡、裁切）或轉檔過，畫面位置已改變，無法可靠校正。"
            "請在 iPhone 相簿對該影片按「編輯 → 回復 → 回復原狀」，再用 AirDrop 或接線傳原始檔。")
