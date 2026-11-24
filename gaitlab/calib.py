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
# 地板四角只用這四個 ID。放寬偵測參數時會冒出 17、37 這類雜訊，一律丟掉。
FLOOR_MARKER_IDS = (0, 1, 2, 3)
FAR_MARKER_IDS = (2, 3)
ROI_UPSCALE_SCALES = (2, 3, 4)
# 遠端放大的固定取樣上限：前 240 格、每隔一格（最多 120 張）。
# 不再在每個 ID 湊滿 15 次時停。4K 直拍上，搜尋帶放大 2／3／4 倍一格大約 0.1–0.4 秒，
# 120 格大約十幾秒到一分鐘，所以用這個上限，不把整支影片掃完。
FAR_SAMPLE_END = 240
FAR_SAMPLE_STEP = 2
# 校正檔裡每個角點實際的來源。重算、沿用前一支時只接受這些，不寫成 aruco+click。
POINT_METHODS = ("aruco", "aruco_roi_upscale", "click", "points", "reused_previous")
# 遠端搜尋帶：從近端標記那一列，沿「遠離鏡頭」的方向量畫面高度的比例。
# 正面直拍、近端靠近畫面下緣時，5 m 量測區的遠端大約落在 0.25–0.40
# （實測 IMG_1554 約 0.27；合成的強透視約 0.37）。0.08–0.48 兩者都包得住，又不會蓋到近端標記。
FAR_ROI_Y_ABOVE = (0.08, 0.48)
FAR_ROI_X_MARGIN_FRAC = 0.08


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


def _as_points(centers) -> np.ndarray:
    if isinstance(centers, dict):
        if not centers:
            return np.zeros((0, 2))
        return np.asarray(list(centers.values()), float).reshape(-1, 2)
    return np.asarray(centers, float).reshape(-1, 2)


def far_marker_roi(frame_shape, near_centers=None, *,
                   y_above: tuple[float, float] = FAR_ROI_Y_ABOVE,
                   x_margin_frac: float = FAR_ROI_X_MARGIN_FRAC) -> tuple[int, int, int, int]:
    """遠端標記（ID 2、3）的搜尋範圍，像素座標 (x0, y0, x1, y1)，右下不含。

    不把範圍寫死。有近端標記時：遠端在畫面上位於近端的「遠離鏡頭」那一側
    （近端靠近下緣就往上找，靠近上緣就往下找），左右不超出近端兩點再加一點邊。
    沒有近端標記時，用畫面中央偏上的一塊預設帶。y_above、x_margin_frac 可改。
    """
    h, w = int(frame_shape[0]), int(frame_shape[1])

    def fallback():
        return (int(0.15 * w), int(0.45 * h), int(0.85 * w), int(0.72 * h))

    pts = _as_points(near_centers) if near_centers is not None else np.zeros((0, 2))
    if len(pts) == 0:
        return fallback()
    y_near = float(np.median(pts[:, 1]))
    x_lo, x_hi = float(pts[:, 0].min()), float(pts[:, 0].max())
    if len(pts) < 2 or (x_hi - x_lo) < 0.05 * w:
        x_lo, x_hi, margin = 0.15 * w, 0.85 * w, 0.0
    else:
        margin = float(x_margin_frac) * w
    # 近端靠近哪一側的邊緣，遠端就往另一側找
    if (h - 1 - y_near) <= y_near:
        y_a = y_near - y_above[1] * h
        y_b = y_near - y_above[0] * h
    else:
        y_a = y_near + y_above[0] * h
        y_b = y_near + y_above[1] * h
    x0 = max(0, int(np.floor(x_lo - margin)))
    x1 = min(w, int(np.ceil(x_hi + margin)))
    y0 = max(0, int(np.floor(min(y_a, y_b))))
    y1 = min(h, int(np.ceil(max(y_a, y_b))))
    if x1 <= x0 or y1 <= y0:
        return fallback()
    return (x0, y0, x1, y1)


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _aruco_detector():
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
                                   cv2.aruco.DetectorParameters())


def detect_roi_upscale(frame: np.ndarray, roi, *,
                       ids=FLOOR_MARKER_IDS,
                       scales=ROI_UPSCALE_SCALES) -> dict[int, list]:
    """把 roi 用 INTER_CUBIC 放大 scales 倍後，以預設參數偵測。

    中心是四個角的平均，再換算回整張畫面的像素。只留下 ID 0–3（而且要在 ids 裡）；
    17、37 這類假標記直接丟棄。回傳 {id: [中心, ...]}，三種倍率的結果都留著。
    """
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return {}
    allowed = {int(i) for i in ids if int(i) in FLOOR_MARKER_IDS}
    if not allowed:
        return {}
    gray = _gray(frame)[y0:y1, x0:x1]
    detector = _aruco_detector()
    found: dict[int, list] = {}
    origin = np.array([x0, y0], float)
    for scale in scales:
        up = cv2.resize(gray, None, fx=float(scale), fy=float(scale), interpolation=cv2.INTER_CUBIC)
        corners, ids_found, _ = detector.detectMarkers(up)
        if ids_found is None:
            continue
        for cc, i in zip(corners, ids_found.flatten()):
            i = int(i)
            if i not in allowed:
                continue
            center = cc.reshape(4, 2).mean(axis=0) / float(scale) + origin
            found.setdefault(i, []).append(center)
    return found


def search_far_markers(frames, near_centers=None, *,
                       wanted=FAR_MARKER_IDS,
                       scales=ROI_UPSCALE_SCALES,
                       frame_shape=None,
                       roi=None) -> dict[int, dict]:
    """遠端標記後備搜尋。frames 是 BGR 影格（可為產生器）。

    範圍用 far_marker_roi 從近端位置推出來（roi 有給就用給的）。
    傳進來的影格全部掃完，不在某個 ID 湊滿幾次時提早停。格數上限由呼叫端決定
    （正面校正是前 FAR_SAMPLE_END 格、每隔 FAR_SAMPLE_STEP 格）。
    中心取所有偵測（每格、每種倍率各算一次）的中位數。
    """
    wanted = tuple(int(i) for i in wanted if int(i) in FLOOR_MARKER_IDS)
    buckets: dict[int, list] = {i: [] for i in wanted}
    used_roi = roi
    for frame in frames:
        if used_roi is None:
            used_roi = far_marker_roi(frame_shape if frame_shape is not None else frame.shape, near_centers)
        for i, centers in detect_roi_upscale(frame, used_roi, ids=wanted, scales=scales).items():
            buckets[i].extend(centers)
    out = {}
    for i, centers in buckets.items():
        if not centers:
            continue
        stacked = np.stack(centers, 0)
        out[i] = {
            "method": "aruco_roi_upscale",
            "center": np.median(stacked, 0),
            "n": int(stacked.shape[0]),
            "sd": stacked.std(0),
            "roi_px": None if used_roi is None else [int(v) for v in used_roi],
            "scales": [int(s) for s in scales],
        }
    return out


def records_from_saved_detection(calib: dict) -> dict | None:
    """從校正檔的 marker_detection 讀回每個點的 method、次數與標準差。

    重算或沿用前一支影片時用這份紀錄，保留 aruco／aruco_roi_upscale／click，
    不要把 method 改寫成 aruco+click。四個點不齊，或某個點的 method 不是已知來源時，回傳 None。
    """
    det = calib.get("marker_detection")
    if not isinstance(det, dict):
        return None
    records = {}
    points = calib.get("image_points_px")
    for k in range(4):
        src = det.get(str(k))
        if not isinstance(src, dict):
            return None
        method = src.get("method")
        if method not in POINT_METHODS:
            return None
        center = src.get("center_px")
        if center is None:
            if not points or len(points) != 4:
                return None
            center = points[k]
        sd = src.get("sd_px")
        records[k] = {
            "method": method,
            "center": np.asarray(center, float),
            "n": None if src.get("n") is None else int(src["n"]),
            "sd": None if sd is None else np.asarray(sd, float),
        }
    return records


def format_calib_method(records: dict, note: str | None = None) -> str:
    """校正檔的 method 字串。四點來源相同且是一般全畫面偵測時維持 "aruco"。

    來源不相同（例如有的點是放大後找到的、有的是點選的）就逐點寫出來，
    並附上偵測次數與 x/y 標準差，不再把整組標成 aruco+click。
    """
    methods = [records[k]["method"] for k in range(4)]
    if len(set(methods)) == 1 and methods[0] == "aruco":
        text = "aruco"
    elif len(set(methods)) == 1 and methods[0] in ("click", "points"):
        text = methods[0]
    else:
        parts = []
        for k in range(4):
            r = records[k]
            n = r.get("n")
            sd = r.get("sd")
            if n is None:
                parts.append(f"{k}:{r['method']}")
            elif sd is None:
                parts.append(f"{k}:{r['method']}(n={int(n)})")
            else:
                parts.append(f"{k}:{r['method']}(n={int(n)},sd={float(sd[0]):.2f}/{float(sd[1]):.2f})")
        text = ";".join(parts)
    if note:
        text = f"{text}｜{note}"
    return text


def marker_detection_json(records: dict) -> dict:
    """每個角點的來源、次數、標準差。JSON 物件的鍵是 "0"–"3"。"""
    out = {}
    for k in range(4):
        r = records[k]
        sd = r.get("sd")
        item = {
            "method": r["method"],
            "n": None if r.get("n") is None else int(r["n"]),
            "sd_px": None if sd is None else [float(sd[0]), float(sd[1])],
            "center_px": [float(r["center"][0]), float(r["center"][1])],
        }
        if r.get("roi_px") is not None:
            item["roi_px"] = [int(v) for v in r["roi_px"]]
        if r.get("scales") is not None:
            item["scales"] = [int(v) for v in r["scales"]]
        out[str(k)] = item
    return out


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
    manual: set[int] = set()
    click_points.last_manual = manual
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
            manual.add(m[0])
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
            manual.clear()
            draw()
        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("已取消校正")
    cv2.destroyAllWindows()
    return np.float32([pts[k] for k in range(n_pts)])


def save_calib(path: str, img_pts, width_m: float, length_m: float, method: str,
               image_size=None, video: str | None = None, marker_detection: dict | None = None) -> dict:
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
    if marker_detection is not None:
        calib["marker_detection"] = marker_detection
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
