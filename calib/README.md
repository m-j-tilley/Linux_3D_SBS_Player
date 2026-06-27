# calib/ — standard G90XF calibration (shipped)

This folder holds the calibration the weave and head-tracker read at runtime. Glasses-free 3D on the
Odyssey 3D is fixed panel-design geometry — **the same for every G90XF unit** — so this repo ships the
**standard G90XF calibration** and the player gives clean, ghost-free 3D **out of the box**. No
per-monitor capture is required. Optional fine-tuning knobs are in [`../CALIBRATION.md`](../CALIBRATION.md).

| File | What it is | Shipped |
|------|------------|---------|
| `3DStackCorrection_A.png`, `3DStackCorrection_B.png` | lenticular weave correction textures (feed the shader) | standard G90XF |
| `../hub/_fields.pkl` | captured lenticular weave field (the pixel-exact weave) | standard G90XF |
| `intrinsics.yml`, `extrinsics.yml` | stereo webcam intrinsics + R/T (≈70 mm baseline) | measured |
| `Tracker2DisplayTransform.ini` | camera→display offset/rotation | measured |
| `tracker_calibration.yml` | stereo webcam identifiers + image size | standard |
| `eye_offset.json` | sweet-spot nudge (set live with `Ctrl+Alt+arrows`) | standard |
| `baseline_scale.json` | depth / camera-separation scale (edit the file; no live key) | standard |
| `weave_params*.json`, `screen.ini`, `player.ini`, `calib_defaults.json`, `tune.json` | weave/panel reference params + RESET defaults | standard |

These are the **same physical-design constants for every G90XF**, not per-unit secrets — they contain no
serial/identity data. If you ever want to regenerate them for your own hardware, see `CALIBRATION.md`.
(The encrypted Samsung `3dstack.eini` blob is intentionally **not** shipped — it is never read at runtime.)
