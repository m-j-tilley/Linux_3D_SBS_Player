"""
tracker.py — real-time 3D eye position from the Odyssey 3D's built-in stereo camera.

Camera: UVC "3D Stereo WebCam" (VID 04E8/PID 20D7), MJPG 1280x480 = two 640x480 monochrome/IR
views side-by-side (L | R). MVP: use the LEFT view with MediaPipe **FaceLandmarker (Tasks API)**
(478 landmarks incl. iris) and estimate metric depth from the iris diameter (~11.7 mm).

Coordinate output:
  * camera frame (cm): origin at the left camera, +x right, +y down, +z forward (OpenCV convention)
  * display frame: camera frame shifted by Tracker2DisplayTransform.ini offsets (rotation TODO)

Refinement roadmap: stereo-triangulate both views (calib/intrinsics+extrinsics) for true z,
one-euro/Kalman smoothing, full camera->display rotation. For now this proves the camera path and
gives a usable eye signal for the weaver.

Needs models/face_landmarker.task (downloaded once). Cross-platform: Windows=DirectShow, Linux=V4L2.
"""
from __future__ import annotations
import os
import math
import platform
import configparser
import subprocess

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = os.path.dirname(__file__)
CALIB_DIR = os.path.join(HERE, "..", "calib")
MODEL_PATH = os.path.join(HERE, "..", "models", "face_landmarker.task")
REAL_IRIS_DIAMETER_MM = 11.7  # population mean (MediaPipe Iris single-camera depth, <10% err)

# FaceLandmarker refined-iris landmark indices (same as legacy FaceMesh)
IRIS_L = dict(center=468, right=469, top=470, left=471, bottom=472)
IRIS_R = dict(center=473, right=474, top=475, left=476, bottom=477)


# ---------- calibration ----------

def load_intrinsics(calib_dir: str = CALIB_DIR):
    """(fx, fy, cx, cy) for camera 1 (left) from intrinsics.yml (OpenCV FileStorage)."""
    fs = cv2.FileStorage(os.path.join(calib_dir, "intrinsics.yml"), cv2.FILE_STORAGE_READ)
    M1 = fs.getNode("M1").mat()
    fs.release()
    return float(M1[0, 0]), float(M1[1, 1]), float(M1[0, 2]), float(M1[1, 2])


def load_transform(calib_dir: str = CALIB_DIR) -> dict:
    cp = configparser.ConfigParser()
    cp.read(os.path.join(calib_dir, "Tracker2DisplayTransform.ini"))
    return {k: float(v) for k, v in cp["Transform"].items()}


# ---------- camera ----------

def open_stereo_cam(index: int | None = None, width: int = 1280, height: int = 480, fps: int = 60,
                    exposure: int = 500, gain: int = 100, raw_mjpg: bool = False):
    """Open the SBS stereo camera; if index is None, probe 0..5 for the 1280x480 mode.
    exposure/gain: SHORT manual exposure (the IR illuminator gives steady light) so the camera delivers the full
    60fps with minimal motion-blur/integration-lag. Auto-exposure picks ~1100 -> only ~18-35fps. exposure<=500=60fps
    (env GPU_CAPTURE_EXPOSURE / _GAIN override)."""
    exposure = int(os.environ.get("WEAVE_CAM_EXPOSURE", exposure)); gain = int(os.environ.get("WEAVE_CAM_GAIN", gain))
    backend = cv2.CAP_MSMF if platform.system() == "Windows" else cv2.CAP_V4L2  # MSMF honors MJPEG@60 (DShow stuck at YUYV@15)
    for i in ([index] if index is not None else range(6)):
        cap = cv2.VideoCapture(i, backend)
        if not cap.isOpened():
            cap.release(); continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        ok, frame = cap.read()
        if ok and frame is not None and frame.shape[1] >= width and frame.shape[0] == height:
            print(f"[tracker] camera index {i}: {frame.shape[1]}x{frame.shape[0]}")
            if platform.system() != "Windows":
                # SHORT MANUAL exposure -> full 60fps + low motion-blur (auto-exposure picked ~1100 -> ~18-35fps).
                try:
                    subprocess.run(["v4l2-ctl", "-d", f"/dev/video{i}", "--set-ctrl",
                                    f"auto_exposure=1,exposure_time_absolute={exposure},gain={gain}"],
                                   capture_output=True, timeout=3)
                    print(f"[tracker] manual exposure={exposure} gain={gain} (-> 60fps)")
                except Exception as e:
                    print(f"[tracker] exposure set skipped: {e}")
                try:
                    from ir_control import ir_on          # CONFIRMED: extension unit 3 selector 2 = 1
                    if ir_on(f"/dev/video{i}"):
                        print("[tracker] IR illuminator ON")
                except Exception as e:
                    print(f"[tracker] IR illuminator enable skipped: {e}")
            if raw_mjpg:                                   # subsequent cap.read() returns RAW MJPG bytes; the caller
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)     # imdecodes GRAY (~1.4ms) — OpenCV's internal MJPG->BGR is ~35ms
                print("[tracker] raw-MJPG mode (CONVERT_RGB=0) -> fast gray decode")
            return cap
        cap.release()
    raise RuntimeError("stereo camera (1280x480) not found; pass --cam <index>")


def split_lr(frame: np.ndarray):
    w = frame.shape[1] // 2
    return frame[:, :w], frame[:, w:]


# ---------- tracking ----------

class EyeTracker:
    def __init__(self, calib_dir: str = CALIB_DIR, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"missing {model_path} (download face_landmarker.task)")
        self.fx, self.fy, self.cx, self.cy = load_intrinsics(calib_dir)
        self.transform = load_transform(calib_dir)
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self._ts = 0
        self._diam = {}   # per-eye smoothed iris diameter (depth is slow -> stabilises lateral x)

    def _eye_xyz(self, lms, idx: dict, key: str, w: int, h: int, diam_alpha: float = 0.1):
        """3D position (cm, camera frame) of one eye from its iris landmarks.
        Depth (from iris diameter) is heavily smoothed: it changes slowly, but its per-frame noise
        otherwise scales straight into the lateral x = (u-cx)*z/fx and dominates the jitter."""
        u, v = lms[idx["center"]].x * w, lms[idx["center"]].y * h
        rx, ry = lms[idx["right"]].x * w, lms[idx["right"]].y * h
        lx, ly = lms[idx["left"]].x * w,  lms[idx["left"]].y * h
        tx, ty = lms[idx["top"]].x * w,   lms[idx["top"]].y * h
        bx, by = lms[idx["bottom"]].x * w, lms[idx["bottom"]].y * h
        diam_px = max(math.hypot(rx - lx, ry - ly), math.hypot(tx - bx, ty - by), 1e-3)
        d = self._diam.get(key)
        d = diam_px if d is None else d + diam_alpha * (diam_px - d)   # slow EMA on depth
        self._diam[key] = d
        z_mm = self.fx * REAL_IRIS_DIAMETER_MM / d
        x_mm = (u - self.cx) * z_mm / self.fx   # u raw (fast lateral) x stable z scale
        y_mm = (v - self.cy) * z_mm / self.fy
        return np.array([x_mm, y_mm, z_mm]) / 10.0  # -> cm

    def to_display(self, xyz_cm: np.ndarray) -> np.ndarray:
        t = self.transform  # MVP: translation only (rotation TODO)
        return xyz_cm + np.array([t["xoff_cm"], t["yoff_cm"], t["zoff_cm"]])

    def process(self, gray_left: np.ndarray):
        """Return dict (eyeL/eyeR/mid cm + display-space mid + IPD) or None if no face."""
        h, w = gray_left.shape[:2]
        rgb = cv2.cvtColor(gray_left, cv2.COLOR_GRAY2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        self._ts += 16
        res = self.landmarker.detect_for_video(mp_img, self._ts)
        if not res.face_landmarks:
            return None
        lms = res.face_landmarks[0]
        eL = self._eye_xyz(lms, IRIS_L, "L", w, h)
        eR = self._eye_xyz(lms, IRIS_R, "R", w, h)
        mid = (eL + eR) / 2.0
        return {"eyeL_cm": eL, "eyeR_cm": eR, "mid_cm": mid,
                "mid_display_cm": self.to_display(mid),
                "ipd_mm": float(np.linalg.norm(eL - eR) * 10.0)}


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Odyssey 3D eye tracker (MVP: FaceLandmarker + iris depth)")
    ap.add_argument("--cam", type=int, default=None, help="camera index (default: auto-probe)")
    ap.add_argument("--frames", type=int, default=150, help="frames to process then exit")
    a = ap.parse_args()
    cap = open_stereo_cam(a.cam)
    et = EyeTracker()
    print("[tracker] running — please look at the monitor's camera...")
    seen = 0
    for n in range(a.frames):
        ok, frame = cap.read()
        if not ok:
            continue
        left, _ = split_lr(frame)
        gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
        r = et.process(gray)
        if r:
            seen += 1
            if seen % 10 == 1:
                m = r["mid_display_cm"]
                print(f"  frame {n:4d}: eyes@display x={m[0]:+6.1f} y={m[1]:+6.1f} z={m[2]:+6.1f} cm  IPD={r['ipd_mm']:.0f}mm")
    cap.release()
    print(f"[tracker] done — face detected in {seen}/{a.frames} frames")


if __name__ == "__main__":
    _main()
