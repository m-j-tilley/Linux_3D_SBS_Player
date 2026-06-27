"""
sbs.py — side-by-side stereo input: load/split content into Left/Right views, and a calibration
test pattern. Returns RGB uint8 arrays (H, W, 3) sized to the panel.

Layouts:
  full : frame is [ Left | Right ], each half already correct aspect.
  half : Half-SBS, each half squished horizontally -> un-squish (stretch) back to full width.
"""
from __future__ import annotations
import numpy as np
import cv2

PANEL_W, PANEL_H = 3840, 2160


def _to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_sbs(path: str, layout: str = "full", panel=(PANEL_W, PANEL_H)):
    """Load an SBS image file -> (left_rgb, right_rgb) sized to the panel."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    half = w // 2
    left, right = img[:, :half], img[:, half:2 * half]
    pw, ph = panel
    left = cv2.resize(left, (pw, ph), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (pw, ph), interpolation=cv2.INTER_AREA)
    return _to_rgb(left), _to_rgb(right)


def make_test_pair(panel=(PANEL_W, PANEL_H), disparity=60):
    """A calibration test pair: clear per-eye labels + a grid offset by `disparity` between eyes.

    When weaving + lens are correct: closing the left eye shows only 'RIGHT', closing the right eye
    shows only 'LEFT' (no ghosting), and with both eyes the grid shows depth.
    """
    pw, ph = panel
    left = np.full((ph, pw, 3), 40, np.uint8)
    right = np.full((ph, pw, 3), 40, np.uint8)
    for x in range(0, pw, 160):
        cv2.line(left, (x, 0), (x, ph), (170, 170, 170), 2)
        cv2.line(right, (x + disparity, 0), (x + disparity, ph), (170, 170, 170), 2)
    for y in range(0, ph, 160):
        cv2.line(left, (0, y), (pw, y), (90, 90, 90), 1)
        cv2.line(right, (0, y), (pw, y), (90, 90, 90), 1)
    f, sc, th = cv2.FONT_HERSHEY_SIMPLEX, 8.0, 18
    cv2.putText(left, "LEFT", (pw // 2 - 420, ph // 2), f, sc, (60, 120, 255), th)
    cv2.putText(right, "RIGHT", (pw // 2 - 520, ph // 2), f, sc, (255, 120, 60), th)
    # central depth target (a filled box with disparity -> should float in front)
    cv2.rectangle(left, (pw // 2 - 150, ph // 2 + 200), (pw // 2 + 150, ph // 2 + 500), (255, 255, 255), -1)
    cv2.rectangle(right, (pw // 2 - 150 + disparity, ph // 2 + 200), (pw // 2 + 150 + disparity, ph // 2 + 500), (255, 255, 255), -1)
    return _to_rgb(left), _to_rgb(right)
