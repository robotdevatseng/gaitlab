"""地板校正：把影像像素轉成地板上的公尺座標。

在走道上貼一個已知尺寸的長方形（例如 寬 1.0 m x 長 6.0 m），四個角點順序固定：
  0 = 近左、1 = 近右、2 = 遠右、3 = 遠左（「左右」以 iPad 畫面看到的為準，「近」= 靠近鏡頭）
地板座標定義：X = 往畫面右邊（公尺），Y = 往遠離鏡頭的方向（公尺），近左角為原點。
"""
from __future__ import annotations

import json
import numpy as np
import cv2

ARUCO_DICT = cv2.aruco.DICT_4X4_50


def floor_corners(width_m: float, length_m: float) -> np.ndarray:
    return np.float32([[0, 0], [width_m, 0], [width_m, length_m], [0, length_m]])


def homography(img_pts, width_m: float, length_m: float) -> np.ndarray:
    img_pts = np.float32(img_pts).reshape(4, 2)
    H, _ = cv2.findHomography(img_pts, floor_corners(width_m, length_m))
    if H is None:
        raise ValueError("校正失敗：四個點可能共線或順序錯誤")
    return H


def to_floor(H: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    """(…,2) 像素 → (…,2) 公尺；NaN 會保留為 NaN。"""
    pts = np.asarray(pts_px, dtype=float)
    shape = pts.shape
    flat = pts.reshape(-1, 2)
    out = np.full_like(flat, np.nan)
    ok = ~np.isnan(flat).any(axis=1)
    if ok.any():
        out[ok] = cv2.perspectiveTransform(flat[ok].reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    return out.reshape(shape)


def to_image(H: np.ndarray, pts_m: np.ndarray) -> np.ndarray:
    Hinv = np.linalg.inv(H)
    pts = np.asarray(pts_m, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2)


def _project_cam(params, Pw, size):
    f, pitch, roll, yaw, xc, yc, h = params
    w, hh = size
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy_, sy = np.cos(yaw), np.sin(yaw)
    fwd = np.array([sy * cp, cy_ * cp, -sp])
    right0 = np.array([cy_, -sy, 0.0])
    down0 = np.cross(fwd, right0)
    cr, sr = np.cos(roll), np.sin(roll)
    R = np.stack([cr * right0 + sr * down0, -sr * right0 + cr * down0, fwd])
    pc = (Pw - np.array([xc, yc, h])) @ R.T
    return np.stack([f * pc[:, 0] / pc[:, 2] + w / 2, f * pc[:, 1] / pc[:, 2] + hh / 2], -1)


def check_geometry(img_pts, width_m: float, length_m: float, size) -> dict:
    """從四個地板點反推相機（焦距、俯角、位置、高度），用來檢查佈置是否量錯。

    四個點與長方形尺寸本身一定解得出 H；但若再假設「主點在畫面中心、像素為正方形」，
    反推出的鏡頭高度與近端距離就可以拿來跟實際佈置比對。
    例：左右標記間距其實是 1.18 m 卻輸入 1.0 m，鏡頭 1.2 m 會被反推成約 1.0 m。
    回傳 cam_height_m、near_dist_m（鏡頭到量測區近端）、focal_px、pitch_deg、resid_px。
    """
    from scipy.optimize import least_squares
    W, L = float(width_m), float(length_m)
    Pw = np.array([[0, 0, 0], [W, 0, 0], [W, L, 0], [0, L, 0]], float)
    img = np.asarray(img_pts, float).reshape(4, 2)
    best = None
    for h0 in (0.8, 1.2, 1.6):
        for pit in (0.05, 0.2, 0.4):
            p0 = [float(max(size)), pit, 0.0, 0.0, W / 2, -2.0, h0]
            try:
                r = least_squares(lambda p: (_project_cam(p, Pw, size) - img).ravel(), p0, method="lm", max_nfev=20000)
            except Exception:  # pragma: no cover
                continue
            if r.x[6] > 0 and r.x[0] > 0 and (best is None or r.cost < best.cost):
                best = r
    if best is None:
        return {}
    f, pitch, roll, yaw, xc, yc, h = best.x
    return {"cam_height_m": float(h), "near_dist_m": float(-yc), "focal_px": float(f),
            "pitch_deg": float(np.degrees(pitch)), "roll_deg": float(np.degrees(roll)),
            "resid_px": float(np.sqrt(np.mean(best.fun ** 2)))}


def parse_points(text: str) -> np.ndarray:
    """"x0,y0;x1,y1;x2,y2;x3,y3" → (4,2)"""
    pts = [[float(v) for v in p.split(",")] for p in text.strip().split(";") if p.strip()]
    if len(pts) != 4:
        raise ValueError("需要 4 個點，格式：x0,y0;x1,y1;x2,y2;x3,y3（近左、近右、遠右、遠左）")
    return np.float32(pts)


def detect_aruco_ids(frame: np.ndarray, ids=(0, 1, 2, 3)) -> dict:
    """偵測指定 ID 的 ArUco 標記（預設 0–3），回傳 {ID: 中心點}（可能只找到部分）。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
                                       cv2.aruco.DetectorParameters())
    corners, ids_found, _ = detector.detectMarkers(gray)
    if ids_found is None:
        return {}
    want = set(ids)
    return {int(i): c.reshape(4, 2).mean(axis=0) for i, c in zip(ids_found.flatten(), corners) if int(i) in want}


def detect_aruco(frame: np.ndarray) -> np.ndarray | None:
    """4 個都找到才回傳（依 ID 排序），否則 None。"""
    found = detect_aruco_ids(frame)
    if not all(k in found for k in range(4)):
        return None
    return np.float32([found[k] for k in range(4)])


def snap_to_marker(frame: np.ndarray, x: float, y: float, r: int = 60) -> tuple[float, float]:
    """把點選位置吸附到附近黑色標記的中心（遠端標記在畫面上很扁，手點容易偏）。找不到就維持原點。"""
    h, w = frame.shape[:2]
    x0, x1 = max(0, int(x) - 2 * r), min(w, int(x) + 2 * r)
    y0, y1 = max(0, int(y) - r), min(h, int(y) + r)
    roi = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    if roi.size == 0:
        return x, y
    _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(bw)
    best, bd = None, 1e9
    for i in range(1, n):
        bx, by, bw_, bh, area = st[i]
        if area < 30 or bw_ > 3 * r:
            continue
        cx, cy = bx + bw_ / 2 + x0, by + bh / 2 + y0
        dist = np.hypot(cx - x, cy - y)
        if dist < bd and dist < r:
            best, bd = (cx, cy), dist
    return best if best else (x, y)


def click_points(frame: np.ndarray, known: dict | None = None,
                 labels=("0 near-left", "1 near-right", "2 far-right", "3 far-left")) -> np.ndarray:
    """點選地板四角。已自動偵測到的標記（known）會先填好，只需點缺的角。
    點選後自動吸附到附近黑色標記的中心。按 r 全部重點、Enter 確認、Esc 取消。"""
    n_pts = len(labels)
    known = dict(known or {})
    pts: dict[int, tuple[float, float]] = {k: tuple(v) for k, v in known.items()}
    scale = min(1.0, 1400 / frame.shape[1], 900 / frame.shape[0])
    disp0 = cv2.resize(frame, None, fx=scale, fy=scale)

    def missing():
        return [k for k in range(n_pts) if k not in pts]

    def draw():
        d = disp0.copy()
        for k, (x, y) in pts.items():
            col = (0, 200, 0) if k in known else (0, 0, 255)
            cv2.circle(d, (int(x * scale), int(y * scale)), 6, col, -1)
            cv2.putText(d, labels[k], (int(x * scale) + 8, int(y * scale) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        if len(pts) == n_pts and n_pts > 2:
            cv2.polylines(d, [np.int32([[pts[k][0] * scale, pts[k][1] * scale] for k in range(n_pts)])],
                          True, (0, 255, 255), 2)
        m = missing()
        msg = f"Click: {labels[m[0]]}" if m else "Enter = OK, r = redo"
        cv2.putText(d, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("floor calibration", d)

    def on_mouse(event, x, y, flags, param):
        m = missing()
        if event == cv2.EVENT_LBUTTONDOWN and m:
            pts[m[0]] = snap_to_marker(frame, x / scale, y / scale)
            draw()

    cv2.namedWindow("floor calibration")
    cv2.setMouseCallback("floor calibration", on_mouse)
    draw()
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10) and not missing():
            break
        if key == ord("r"):          # 全部重點（包含自動偵測到的）
            pts.clear()
            known.clear()
            draw()
        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("已取消校正")
    cv2.destroyAllWindows()
    return np.float32([pts[k] for k in range(n_pts)])


def save_calib(path: str, img_pts, width_m: float, length_m: float, method: str,
               image_size=None, video: str | None = None) -> dict:
    H = homography(img_pts, width_m, length_m)
    calib = {
        "method": method,
        "image_size_px": [int(v) for v in image_size] if image_size is not None else None,
        "video": video,
        "image_points_px": np.asarray(img_pts).tolist(),
        "floor_width_m": width_m,
        "floor_length_m": length_m,
        "H": H.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)
    return calib


def load_calib(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        c = json.load(f)
    c["H"] = np.asarray(c["H"], dtype=float)
    return c


def make_markers(out_pdf: str, marker_cm: float = 18.0, ids=(0, 1, 2, 3), names=None) -> str:
    """產生 ID 0–3 的 ArUco 標記，A4 每頁一個、實際尺寸 marker_cm。列印時選「實際大小／100%」。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    dic = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    if names is None:
        names = ["0  近左 near-left", "1  近右 near-right", "2  遠右 far-right", "3  遠左 far-left"]
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "PingFang TC", "Noto Sans CJK TC", "Noto Sans CJK JP", "DejaVu Sans"]
    a4w, a4h = 21.0 / 2.54, 29.7 / 2.54
    with PdfPages(out_pdf) as pdf:
        for j, k in enumerate(ids):
            img = cv2.aruco.generateImageMarker(dic, k, 600)
            fig = plt.figure(figsize=(a4w, a4h))
            s = marker_cm / 2.54
            ax = fig.add_axes([(a4w - s) / 2 / a4w, (a4h - s) / 2 / a4h, s / a4w, s / a4h])
            ax.imshow(img, cmap="gray", interpolation="nearest", extent=(0, 1, 0, 1))
            ax.plot([0.5], [0.5], "+", color="red", ms=18, mew=1.5)
            ax.axis("off")
            fig.text(0.5, 0.88, f"ArUco ID {names[j]}", ha="center", fontsize=22)
            fig.text(0.5, 0.12, f"黑框邊長應為 {marker_cm:g} cm（請以 100% 實際大小列印後用尺確認）\n"
                     "標記中心（紅十字）對準地板長方形的角點", ha="center", fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)
    return out_pdf
