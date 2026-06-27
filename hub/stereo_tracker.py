"""
stereo_tracker.py — steady 3D eye position by TRIANGULATING the iris centres seen in BOTH IR views
(640x480 each) using the per-unit stereo calibration (intrinsics.yml M1/D1/M2/D2 + extrinsics.yml R/T,
baseline ~70 mm). This removes the depth-from-iris-diameter noise that dominated the mono tracker's jitter,
giving a low-noise x AND z. Result is in the LEFT-camera frame, cm.

Detect iris centres (landmarks 468=left-eye, 473=right-eye) with MediaPipe FaceLandmarker in each view,
undistort, cv2.triangulatePoints -> 3D per eye -> midpoint. Needs models/face_landmarker.task.
"""
from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = os.path.dirname(__file__)
CALIB_DIR = os.path.join(HERE, "..", "calib")
MODEL_PATH = os.path.join(HERE, "..", "models", "face_landmarker.task")
IRIS_LE, IRIS_RE = 468, 473   # left-eye, right-eye iris centres


def _opts(model_path, gpu=False):
    # GPU delegate runs FaceMesh on the (idle) GPU: ~1-3ms/view vs ~8-16ms CPU -> the single biggest tracking
    # latency cut, keeping full iris precision. Falls back to CPU if the build/driver can't.
    base = mp_python.BaseOptions(model_asset_path=model_path,
                                 delegate=(mp_python.BaseOptions.Delegate.GPU if gpu
                                           else mp_python.BaseOptions.Delegate.CPU))
    return vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        min_face_detection_confidence=0.4, min_tracking_confidence=0.4)


class StereoEyeTracker:
    def __init__(self, calib_dir: str = CALIB_DIR, model_path: str = MODEL_PATH, gpu: bool = False):
        fs = cv2.FileStorage(os.path.join(calib_dir, "intrinsics.yml"), cv2.FILE_STORAGE_READ)
        self.M1, self.D1 = fs.getNode("M1").mat(), fs.getNode("D1").mat()
        self.M2, self.D2 = fs.getNode("M2").mat(), fs.getNode("D2").mat()
        fs.release()
        fs = cv2.FileStorage(os.path.join(calib_dir, "extrinsics.yml"), cv2.FILE_STORAGE_READ)
        R, T = fs.getNode("R").mat(), fs.getNode("T").mat().reshape(3, 1)
        fs.release()
        self.P1 = self.M1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        self.P2 = self.M2 @ np.hstack([R, T])          # T in mm -> triangulated points in mm
        self._gpu = gpu                                # GPU delegate -> run views SEQUENTIALLY (concurrent GPU contexts contend)
        try:
            self.lmL = vision.FaceLandmarker.create_from_options(_opts(model_path, gpu))
            self.lmR = vision.FaceLandmarker.create_from_options(_opts(model_path, gpu))
        except Exception as e:
            if gpu:
                print(f"[tracker] GPU delegate init failed ({e}); falling back to CPU", flush=True)
                self._gpu = False
                self.lmL = vision.FaceLandmarker.create_from_options(_opts(model_path, False))
                self.lmR = vision.FaceLandmarker.create_from_options(_opts(model_path, False))
            else:
                raise
        self._ts = 0
        self._pool = ThreadPoolExecutor(max_workers=2)   # run both views' detection concurrently
        self._last_z = None; self._Tx = float(T[0, 0])    # for mono fallback (one view + last depth; T.x = baseline mm)

    @staticmethod
    def _iris_px(landmarker, gray, ts, w, h):
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                    data=np.ascontiguousarray(rgb)), ts)
        if not res.face_landmarks:
            return None
        lm = res.face_landmarks[0]
        return np.array([[lm[IRIS_LE].x * w, lm[IRIS_LE].y * h],
                         [lm[IRIS_RE].x * w, lm[IRIS_RE].y * h]], dtype=np.float64)  # (2,2): [LE, RE]

    def detect(self, gray_left, gray_right):
        h, w = gray_left.shape[:2]
        self._ts += 16
        if self._gpu:                                                                # GPU: sequential (avoid concurrent-context tensor contention); each ~1-3ms
            pL = self._iris_px(self.lmL, gray_left, self._ts, w, h)
            pR = self._iris_px(self.lmR, gray_right, self._ts, w, h)
        else:
            fL = self._pool.submit(self._iris_px, self.lmL, gray_left, self._ts, w, h)   # both views in parallel
            fR = self._pool.submit(self._iris_px, self.lmR, gray_right, self._ts, w, h)
            pL = fL.result(); pR = fR.result()
        if pL is not None and pR is not None:
            uL = cv2.undistortPoints(pL.reshape(-1, 1, 2), self.M1, self.D1, P=self.M1).reshape(-1, 2)
            uR = cv2.undistortPoints(pR.reshape(-1, 1, 2), self.M2, self.D2, P=self.M2).reshape(-1, 2)
            X4 = cv2.triangulatePoints(self.P1, self.P2, uL.T, uR.T)
            X3 = (X4[:3] / X4[3]).T                                # (2,3) mm, left-cam frame
            mid_mm = X3.mean(axis=0); self._last_z = float(mid_mm[2])
            return {"mid_cm": mid_mm / 10.0, "eyeL_cm": X3[0] / 10.0, "eyeR_cm": X3[1] / 10.0,
                    "ipd_mm": float(np.linalg.norm(X3[0] - X3[1])), "mode": "stereo"}
        # MONO fallback: one camera lost the eye (far off-centre) -> estimate from the other view + last depth
        if self._last_z is None or (pL is None and pR is None):
            return None                                      # no detection at all (both views) -> let caller hold/wait
        z = self._last_z
        if pL is not None: M, pts, bx = self.M1, pL, 0.0
        else: M, pts, bx = self.M2, pR, self._Tx            # right-cam frame -> left-cam: x += baseline
        fx, fy, cx, cy = M[0, 0], M[1, 1], M[0, 2], M[1, 2]
        e = np.array([[(u - cx) * z / fx + bx, (v - cy) * z / fy, z] for (u, v) in pts])
        mid_mm = e.mean(axis=0)
        return {"mid_cm": mid_mm / 10.0, "eyeL_cm": e[0] / 10.0, "eyeR_cm": e[1] / 10.0,
                "ipd_mm": float(np.linalg.norm(e[0] - e[1])), "mode": "mono"}


BLAZE_PATH = os.path.join(HERE, "..", "models", "blaze_face_short_range.tflite")


class StereoBlazeTracker:
    """Fast SR-style tracker: BlazeFace FaceDetector (like the SR MTCNN) on BOTH IR views -> the two eye
    keypoints -> triangulate to 3D. Much lighter than the 478-pt mesh, so ~60 fps (low latency)."""
    def __init__(self, calib_dir: str = CALIB_DIR, det_path: str = BLAZE_PATH):
        fs = cv2.FileStorage(os.path.join(calib_dir, "intrinsics.yml"), cv2.FILE_STORAGE_READ)
        self.M1, self.D1 = fs.getNode("M1").mat(), fs.getNode("D1").mat()
        self.M2, self.D2 = fs.getNode("M2").mat(), fs.getNode("D2").mat()
        fs.release()
        fs = cv2.FileStorage(os.path.join(calib_dir, "extrinsics.yml"), cv2.FILE_STORAGE_READ)
        R, T = fs.getNode("R").mat(), fs.getNode("T").mat().reshape(3, 1); fs.release()
        self.P1 = self.M1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        self.P2 = self.M2 @ np.hstack([R, T])
        opts = lambda: vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=det_path),
            running_mode=vision.RunningMode.VIDEO, min_detection_confidence=0.4)
        self.dL = vision.FaceDetector.create_from_options(opts())
        self.dR = vision.FaceDetector.create_from_options(opts())
        self._ts = 0

    @staticmethod
    def _eyes_px(det, gray, ts, w, h):
        res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                            data=np.ascontiguousarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))), ts)
        if not res.detections:
            return None
        kp = res.detections[0].keypoints   # 0=right eye, 1=left eye (normalized)
        return np.array([[kp[0].x * w, kp[0].y * h], [kp[1].x * w, kp[1].y * h]], dtype=np.float64)

    def detect(self, gray_left, gray_right):
        h, w = gray_left.shape[:2]; self._ts += 16
        pL = self._eyes_px(self.dL, gray_left, self._ts, w, h)
        pR = self._eyes_px(self.dR, gray_right, self._ts, w, h)
        if pL is None or pR is None:
            return None
        uL = cv2.undistortPoints(pL.reshape(-1, 1, 2), self.M1, self.D1, P=self.M1).reshape(-1, 2)
        uR = cv2.undistortPoints(pR.reshape(-1, 1, 2), self.M2, self.D2, P=self.M2).reshape(-1, 2)
        X4 = cv2.triangulatePoints(self.P1, self.P2, uL.T, uR.T)
        X3 = (X4[:3] / X4[3]).T
        return {"mid_cm": X3.mean(axis=0) / 10.0, "ipd_mm": float(np.linalg.norm(X3[0] - X3[1]))}


def _main():
    import argparse, time
    from tracker import open_stereo_cam, split_lr
    ap = argparse.ArgumentParser(); ap.add_argument("--cam", type=int, default=None); ap.add_argument("--frames", type=int, default=200)
    a = ap.parse_args()
    cap = open_stereo_cam(a.cam); et = StereoEyeTracker()
    rev = os.environ.get("SBS3D_DEBUG_DIR", "/tmp")   # where the diagnostic L/R snapshots are written
    print("[stereo] hold STILL ~3s, then move. running...")
    xs, zs = [], []; t0 = time.time(); seen = lseen = rseen = reads = 0
    for n in range(a.frames):
        ok, frame = cap.read()
        if not ok or frame is None: continue
        reads += 1
        L, Rv = split_lr(frame)
        gL = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY) if L.ndim == 3 else L
        gR = cv2.cvtColor(Rv, cv2.COLOR_BGR2GRAY) if Rv.ndim == 3 else Rv
        if reads == 3:
            cv2.imwrite(os.path.join(rev, "stereo_L.png"), gL); cv2.imwrite(os.path.join(rev, "stereo_R.png"), gR)
            print(f"  frame {gL.shape[1]}x{gL.shape[0]} grayL mean={gL.mean():.0f} grayR mean={gR.mean():.0f} (saved stereo_L/R.png)")
        et._ts += 16
        pL = et._iris_px(et.lmL, gL, et._ts, gL.shape[1], gL.shape[0])
        pR = et._iris_px(et.lmR, gR, et._ts, gR.shape[1], gR.shape[0])
        if pL is not None: lseen += 1
        if pR is not None: rseen += 1
        if pL is not None and pR is not None:
            uL = cv2.undistortPoints(pL.reshape(-1, 1, 2), et.M1, et.D1, P=et.M1).reshape(-1, 2)
            uR = cv2.undistortPoints(pR.reshape(-1, 1, 2), et.M2, et.D2, P=et.M2).reshape(-1, 2)
            X4 = cv2.triangulatePoints(et.P1, et.P2, uL.T, uR.T); m = (X4[:3] / X4[3]).T.mean(0) / 10.0
            seen += 1; xs.append(m[0]); zs.append(m[2])
            if seen % 15 == 1:
                print(f"  f{n:3d} mid x={m[0]:+6.1f} y={m[1]:+6.1f} z={m[2]:+6.1f}cm")
    cap.release()
    fps = reads / (time.time() - t0)
    print(f"[stereo] reads={reads} leftFaces={lseen} rightFaces={rseen} bothTriangulated={seen} ~{fps:.0f}fps")
    if len(xs) > 30:
        print(f"[stereo] first-3s jitter std: x={np.std(xs[:90]):.3f}cm z={np.std(zs[:90]):.3f}cm")
    print("[stereo] done")


if __name__ == "__main__":
    _main()
