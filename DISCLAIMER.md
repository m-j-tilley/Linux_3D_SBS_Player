# Disclaimer & interoperability notice

**Use at your own risk.** This software is provided "as is", without warranty of any kind (see the
[MIT License](LICENSE)). You are solely responsible for how you use it.

## Not affiliated with Samsung

This is an **independent** project, built by me (with the help of Claude Code). It is **not** produced, endorsed, or supported by
Samsung, Leia, Dimenco, or any display vendor. "Samsung", "Odyssey", and "G90XF" are used only to state
factual hardware compatibility (nominative use); no affiliation or endorsement is implied. No vendor
logos or trademarks are included.

## What it does to your hardware

The player talks to the monitor's lens controller over a USB CDC-ACM serial interface and switches the
panel into its 3D mode. This was achieved by **reverse-engineering for interoperability** — making
independently-written software work with hardware **you own** on an operating system the manufacturer
does not officially support. It does not modify firmware and makes no changes that persist across a
power cycle.

### Lens authentication

The lens controller gates 3D mode behind a challenge/response handshake. `hub/lens.py` includes a
re-implementation of that handshake so the lens can be enabled natively on Linux. It is included to make the hardware usable on Linux; depending on your
jurisdiction and the device's terms of use, replicating such a mechanism may carry legal
considerations. **You are responsible for ensuring your use is lawful where you live.** If you prefer
not to ship it, delete the `authenticate()` / `_compute_auth_response()` methods and the `_AUTH_K16`
constant from `hub/lens.py` — the rest of the player still runs (you would then enable the lens by
other means).

## What calibration this ships

So the player works on a fresh install, this repo ships the **numeric calibration the G90XF needs**: the
captured weave field (`hub/_fields.pkl`), the stack-correction textures (`calib/3DStackCorrection_*.png`),
and the camera/stereo calibration (`calib/intrinsics.yml`, `extrinsics.yml`, `Tracker2DisplayTransform.ini`,
`calib/baseline_scale.json`, `calib/eye_offset.json`). The weave field + corrections were **captured for
interoperability** from the panel's weave pipeline; the stereo set is a one-off calibration of the Samsung
3D webcam. They ship as the standard set because the G90XF is the same across units, so no per-monitor
capture is needed.

This is **calibration data, not code** — no vendor binaries, firmware, source, models, logos, or trademarks
are included. Replicating captured calibration data may carry legal considerations depending on your
jurisdiction; **you are responsible for lawful use where you live**, and I will remove anything a
rights-holder asks me to. To ship/run nothing captured, set **`SBS3D_NOMINAL=1`** for the synthetic weave
(derived from panel geometry alone — lower quality, but captured-free).

## Safety

Glasses-free 3D can cause eye strain, fatigue, or discomfort for some people. Take breaks. Don't use it
if it makes you unwell.
