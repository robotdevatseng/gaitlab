"""遠端標記 ROI 放大的自我檢查。執行：python tests/test_far_marker_roi.py

不需要實拍影片。把 18 cm 的 ArUco（DICT_4X4_50）貼在 1.0×5.0 m 的地板上，
用針孔相機投到 2160×3840，讓遠端兩點大約 85×20 像素。
全畫面的預設偵測器會漏掉 ID 2、3；同一張圖走校正用的 ROI 放大後，中心要回到 1 像素以內。
四點都找得到時不該去跑放大。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from gaitlab import calib as C  # noqa: E402
from gaitlab import cli as CLI  # noqa: E402
from gaitlab import pose as P  # noqa: E402


def check(cond, msg):
    print(("  ✔ " if cond else "  ✘ ") + msg)
    if not cond:
        raise SystemExit(f"失敗：{msg}")


class _Camera:
    """與 tests/synth.py 同一個針孔模型：世界 X 右、Y 遠離鏡頭、Z 上。"""

    def __init__(self, h=1.6, f=3600.0, pitch=6.0, size=(2160, 3840)):
        self.pos = np.array((0.5, -2.5, h), float)
        self.f = float(f)
        self.w, self.h = size
        p = np.radians(pitch)
        fwd = np.array([0, np.cos(p), -np.sin(p)])
        right = np.array([1.0, 0, 0])
        down = np.cross(fwd, right)
        self.R = np.stack([right, down, fwd])

    def project(self, Pw):
        pc = (np.asarray(Pw, float) - self.pos) @ self.R.T
        return np.stack([self.f * pc[..., 0] / pc[..., 2] + self.w / 2,
                         self.f * pc[..., 1] / pc[..., 2] + self.h / 2], -1)


def render_floor(blur: float):
    """四個 18 cm 標記的中心在長方形四角。回傳 BGR 圖、四角平均的真值中心、軸對齊尺寸。

    blur 是整張圖的高斯標準差（像素）。遠端標記只有約 20 像素高，
    輕微失焦就會讓全畫面預設偵測器放棄它們；近端大很多，仍然找得到。
    OpenCV 4.11 對完全清晰的 85×20 標記仍能在全畫面找到，所以合成圖要帶這一層失焦，
    才會重現「全畫面只有 0、1，放大後才找回 2、3」。
    """
    cam = _Camera()
    half = 0.09
    base = [(-half, half), (half, half), (half, -half), (-half, -half)]
    order = base[3:] + base[:3]
    centers = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 5.0), 3: (0.0, 5.0)}
    img = np.full((cam.h, cam.w), 236, np.uint8)
    gt, sizes = {}, {}
    side = 900
    dic = cv2.aruco.getPredefinedDictionary(C.ARUCO_DICT)
    for i, (cx, cy) in centers.items():
        world = np.array([[cx + lx, cy + ly, 0.0] for lx, ly in order], float)
        quad = cam.project(world).astype(np.float32)
        mk = cv2.aruco.generateImageMarker(dic, i, side)
        src = np.float32([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]])
        H = cv2.getPerspectiveTransform(src, quad)
        warped = cv2.warpPerspective(mk, H, (cam.w, cam.h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full((side, side), 255, np.uint8), H, (cam.w, cam.h))
        img[mask > 0] = warped[mask > 0]
        gt[i] = quad.mean(axis=0)
        sizes[i] = quad.max(axis=0) - quad.min(axis=0)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), gt, sizes


def _inside(roi, xy) -> bool:
    x, y = float(xy[0]), float(xy[1])
    return roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]


def test_roi_band_covers_known_layouts(gt):
    print("範圍：由近端位置推出來，實測幾何與合成幾何都落在裡面")
    # IMG_1554（2160×3840）公布的四個中心。沒有畫面，只核對預設範圍蓋得住遠端。
    near = {0: (409.0, 3438.5), 1: (1700.25, 3469.75)}
    roi = C.far_marker_roi((3840, 2160), near)
    for xy, name in (((1283.625, 2443.375), "遠右"), ((858.75, 2430.9375), "遠左")):
        check(_inside(roi, xy), f"IMG_1554 {name} {xy} 在預設範圍 {roi} 內")
    roi_default = C.far_marker_roi((3840, 2160), None)
    for xy, name in (((1283.625, 2443.375), "遠右"), ((858.75, 2430.9375), "遠左")):
        check(_inside(roi_default, xy), f"沒有近端標記時，預設帶仍蓋住 IMG_1554 {name}")
    roi_s = C.far_marker_roi((3840, 2160), {0: gt[0], 1: gt[1]})
    for i, name in ((2, "遠右"), (3, "遠左")):
        check(_inside(roi_s, gt[i]), f"合成圖 {name} 在由近端推出的範圍內")


def test_reject_ids_outside_0_3():
    print("只接受 ID 0–3")
    dic = cv2.aruco.getPredefinedDictionary(C.ARUCO_DICT)
    gray = np.full((480, 480), 255, np.uint8)
    gray[40:200, 40:200] = cv2.aruco.generateImageMarker(dic, 17, 160)
    gray[260:420, 260:420] = cv2.aruco.generateImageMarker(dic, 37, 160)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    raw = C._aruco_detector().detectMarkers(gray)
    raw_ids = [] if raw[1] is None else sorted(int(i) for i in raw[1].flatten())
    check(17 in raw_ids and 37 in raw_ids, f"未過濾時這張圖找得到 17 與 37（實際 {raw_ids}）")
    kept = C.detect_roi_upscale(bgr, (0, 0, 480, 480), ids=(0, 1, 2, 3, 17, 37))
    check(kept == {}, f"即使呼叫端把 17、37 放進 ids，結果仍是空的（實際 {sorted(kept)}）")


def test_method_string_names_each_point():
    print("method 逐點記錄，不再整組寫成 aruco+click")
    pure = {k: {"method": "aruco", "n": 1, "sd": [0.0, 0.0], "center": [0, 0]} for k in range(4)}
    check(C.format_calib_method(pure) == "aruco", "四點都是全畫面偵測時，method 維持 aruco")
    mixed = {
        0: {"method": "aruco", "n": 2, "sd": [0.10, 0.20], "center": [0, 0]},
        1: {"method": "aruco", "n": 2, "sd": [0.00, 0.10], "center": [1, 1]},
        2: {"method": "aruco_roi_upscale", "n": 4, "sd": [0.35, 0.12], "center": [2, 2]},
        3: {"method": "click", "n": 1, "sd": None, "center": [3, 3]},
    }
    text = C.format_calib_method(mixed)
    check("aruco+click" not in text, "不再出現 aruco+click")
    check("0:aruco" in text and "1:aruco" in text, "近端兩點標成 aruco")
    check("2:aruco_roi_upscale" in text and "n=4" in text and "sd=0.35/0.12" in text, "遠右標成放大偵測，並寫次數與 SD")
    check("3:click" in text, "點選的那一點標成 click")
    check(C.format_calib_method({k: {"method": "click", "n": 1, "sd": None, "center": [0, 0]} for k in range(4)}) == "click",
          "四點都是點選時，method 是 click")


class _Cap:
    def __init__(self, frame):
        self.frame = frame
        self.left = 1

    def set(self, *a, **k):
        return True

    def read(self):
        if self.left <= 0:
            return False, None
        self.left -= 1
        return True, self.frame

    def release(self):
        pass

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 1.0
        return 0.0


def _run_calibrate(frame, search_counter):
    orig = (P.open_video, P.read_frame, P.video_info, C.click_points, C.search_far_markers)

    def open_video(path):
        return _Cap(frame)

    def read_frame(path, index=0):
        return frame

    def video_info(path):
        return {"fps": 30.0, "n_frames": 1, "width": int(frame.shape[1]), "height": int(frame.shape[0])}

    def no_click(*a, **k):
        raise AssertionError("這張合成圖不該打開點選視窗")

    def wrapped_search(*a, **k):
        search_counter["n"] += 1
        return orig[4](*a, **k)

    P.open_video, P.read_frame, P.video_info = open_video, read_frame, video_info
    C.click_points = no_click
    C.search_far_markers = wrapped_search
    fd, path = tempfile.mkstemp(suffix="_calib.json")
    os.close(fd)
    try:
        cal = CLI.do_calibrate("synthetic.mov", (1.0, 5.0), "auto", None, path)
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return cal, raw
    finally:
        P.open_video, P.read_frame, P.video_info, C.click_points, C.search_far_markers = orig
        if os.path.exists(path):
            os.remove(path)


def test_perspective_fallback(flat, gt, sizes):
    print("強透視：全畫面漏掉遠端，放大後中心回到 1 像素以內")
    fw, fh = float(sizes[2][0]), float(sizes[2][1])
    print(f"  遠端標記軸對齊尺寸 {fw:.1f}×{fh:.1f} 像素（近端 {sizes[0][0]:.1f}×{sizes[0][1]:.1f}）")
    check(70 <= fw <= 110 and 14 <= fh <= 26, f"遠端大約 85×20 像素（實際 {fw:.1f}×{fh:.1f}）")
    full = C.detect_aruco_ids(flat)
    check(0 in full and 1 in full, f"全畫面找得到近端 {sorted(full)}")
    check(2 not in full and 3 not in full, f"全畫面找不到遠端 2、3（實際 {sorted(full)}）")
    found = C.search_far_markers([flat], {0: full[0], 1: full[1]}, wanted=(2, 3), frame_shape=flat.shape)
    check(set(found) == {2, 3}, f"ROI 放大找回 2 與 3（實際 {sorted(found)}）")
    for i, name in ((2, "遠右"), (3, "遠左")):
        err = float(np.linalg.norm(found[i]["center"] - gt[i]))
        print(f"  ID {i} {name}：{found[i]['n']} 次偵測，誤差 {err:.3f} 像素，SD {found[i]['sd'][0]:.3f}/{found[i]['sd'][1]:.3f}")
        check(err <= 1.0, f"ID {i} 中心誤差 {err:.3f} 像素 ≤ 1")
        check(found[i]["method"] == "aruco_roi_upscale", "來源標成 aruco_roi_upscale")
        check(all(int(s) in (2, 3, 4) for s in found[i]["scales"]), "只用 2、3、4 倍")

    calls = {"n": 0}
    _cal, raw = _run_calibrate(flat, calls)
    check(calls["n"] == 1, "校正流程在缺遠端時有跑放大搜尋")
    check("aruco+click" not in raw["method"], f"校正檔 method 不是 aruco+click（{raw['method']}）")
    det = raw["marker_detection"]
    check(det["0"]["method"] == "aruco" and det["1"]["method"] == "aruco", "近端兩點的 method 是 aruco")
    check(det["2"]["method"] == "aruco_roi_upscale" and det["3"]["method"] == "aruco_roi_upscale",
          "遠端兩點的 method 是 aruco_roi_upscale")
    for key in "0123":
        check(isinstance(det[key]["n"], int) and det[key]["n"] >= 1, f"點 {key} 有偵測次數")
        check(det[key]["sd_px"] is not None and len(det[key]["sd_px"]) == 2, f"點 {key} 有 x/y 標準差")
    for i in (2, 3):
        got = np.asarray(raw["image_points_px"][i], float)
        err = float(np.linalg.norm(got - gt[i]))
        check(err <= 1.0, f"寫進校正檔的 ID {i} 誤差 {err:.3f} 像素 ≤ 1")
    print(f"  校正檔 method = {raw['method']}")


def test_all_four_skips_upscale(sharp):
    print("四點都找到時，不跑放大，method 維持 aruco")
    full = C.detect_aruco_ids(sharp)
    check(set(full) == {0, 1, 2, 3}, f"清晰合成圖全畫面四點都在（實際 {sorted(full)}）")
    calls = {"n": 0}
    _cal, raw = _run_calibrate(sharp, calls)
    check(calls["n"] == 0, "沒有呼叫遠端放大")
    check(raw["method"] == "aruco", f"method 是 aruco（實際 {raw['method']}）")
    for i in range(4):
        got = np.asarray(raw["image_points_px"][i], float)
        err = float(np.linalg.norm(got - full[i]))
        check(err < 1e-6, f"ID {i} 用的是全畫面第一次看到的位置")
        check(raw["marker_detection"][str(i)]["method"] == "aruco", f"ID {i} 的 marker_detection 是 aruco")


def main():
    print(f"OpenCV {cv2.__version__}")
    flat, gt, sizes = render_floor(1.5)
    sharp, _gt2, _sz2 = render_floor(0.0)
    test_roi_band_covers_known_layouts(gt)
    test_reject_ids_outside_0_3()
    test_method_string_names_each_point()
    test_perspective_fallback(flat, gt, sizes)
    test_all_four_skips_upscale(sharp)
    print("遠端標記檢查通過 ✔")


if __name__ == "__main__":
    main()
