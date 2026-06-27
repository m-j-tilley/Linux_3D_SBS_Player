# Calibration

Glasses-free 3D on the Odyssey 3D (G90XF) is **fixed panel-design geometry that is identical across every
unit**. So this repo **ships the standard G90XF calibration** — the lenticular weave field
(`hub/_fields.pkl`), the correction textures (`calib/3DStackCorrection_A/B.png`), and the tracker/eye
calibration — and you get **clean, ghost-free 3D out of the box, with no per-monitor capture**.

Everything below is **optional**: small nudges for your seating position, or regenerating the calibration
if you're bringing up different hardware. Most people never need it.

## 1. Optional: live sweet-spot tuning (a few minutes, no extra tools)

Run the player on a clear SBS clip and tune it. The alignment nudge persists to `calib/` automatically.

- **Centre the sweet-spot** — use `Ctrl+Alt+←/→/↑/↓` to nudge the alignment until the 3D "locks" with
  the least ghosting from your normal seating position. Saved live to `calib/eye_offset.json`
  (`Ctrl+Alt+0` resets it).
- **Depth amount** — edit `bscale` in `calib/baseline_scale.json` and relaunch (≈0.9–1.1; **>1**
  exaggerates depth, **<1** flattens it). The player reads it at startup; there is no live key for it.
- **Swap / flip** — if depth is inverted use `Ctrl+Alt+F` (swap L/R); if the image is upside down use
  `Ctrl+Alt+V` (or launch with `GPU_CAPTURE_MPV_FLIPY=0`).

The full live keybindings are printed to the console at startup and shown in the on-screen HUD; see
`hub/hotkeys.py`. This alone gets most setups to "good".

## 2. Optional: regenerate the calibration for your own hardware

The shipped standard G90XF calibration already gives a crisp weave. You only need this section if you're
adapting the player to different hardware or want to re-measure from scratch.

### a) Stereo webcam calibration → `calib/intrinsics.yml`, `calib/extrinsics.yml`
Standard OpenCV stereo calibration of the built-in webcam (it presents as one 1280×480 device = left|right
640×480). Print a checkerboard, capture pairs, run `cv2.stereoCalibrate`, and write `M1/D1/M2/D2` and
`R/T` in OpenCV `FileStorage` YAML (match the format of the shipped files). This fixes eye-tracking depth.

### b) Camera→display transform → `calib/Tracker2DisplayTransform.ini`
The offset/rotation between the webcam and the panel centre (cm / degrees). Tune `Yoff_cm` first (the
camera's height relative to screen centre); adjust until the tracked sweet-spot matches where you
actually sit.

### c) Weave correction + field → `calib/3DStackCorrection_A.png`, `calib/3DStackCorrection_B.png`, `hub/_fields.pkl`
These encode the panel's exact lenticular weave and **are shipped here as the standard G90XF calibration**
(the lens geometry is identical across units), so the weave is pixel-exact out of the box. If you delete
them or set `SBS3D_NOMINAL=1`, the player falls back to a nominal analytic weave synthesized from the lens
geometry constants (`PX_CM`, `SLANT`, `PX_LENS_PX`, `OPTPOS`, …) in `hub/linux_oracle.py` — runnable, but
not pixel-exact. To regenerate them for different hardware, capture the weave field and correction textures
from the vendor weaving pipeline.

## Tips

- Calibrate at your **real viewing distance** (~60 cm) and lighting.
- Good, **bright** room light (or the webcam's IR illuminator) makes eye-tracking far more stable.
- Re-run the quick self-calibration any time the image starts to ghost — it's fast.
