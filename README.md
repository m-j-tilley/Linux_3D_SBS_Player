# Linux 3D SBS Player

Play **side-by-side (SBS) 3D videos glasses-free** on the **Samsung Odyssey 3D (G90XF)** under Linux.

This is an unofficial player, built by me (with plenty of help from Claude Code, so proceed with some caution). It drives the monitor's lenticular 3D lens, tracks your
eyes with the built-in webcam, and "weaves" the left/right halves of an SBS video to each eye in real
time (zero-copy libmpv → GPU, locked to the panel refresh, up to 4K @ 120 Hz). Samsung ships no Linux
software for this display, this fills that gap for people who own one.

> ⚠️ **Unofficial & not affiliated with Samsung. Use at your own risk.** It controls your monitor's
> hardware over a reverse-engineered interface. Please read [`DISCLAIMER.md`](DISCLAIMER.md).

---

## What you need

- A **Samsung Odyssey 3D (G90XF)** connected over **DisplayPort/HDMI 2.1**, with its **USB cable**
  plugged in (the player needs the built-in **webcam** for eye-tracking and the **lens serial board**).
- **Ubuntu** (22.04+ recommended) on an **X11 / Xorg** login session — **not Wayland**.
- An **NVIDIA GPU** with the **proprietary driver** installed (`nvidia-smi` works). The fast video path
  uses NVDEC.
- An **SBS 3D video** (full or half side-by-side). Most "3D SBS" clips on YouTube work once downloaded.

## Install (Ubuntu)

```bash
git clone https://github.com/m-j-tilley/Linux_3D_SBS_Player.git sbs3d-player
cd sbs3d-player
bash setup_linux.sh          # apt + python deps, head-tracking model, lens + scaler udev rules + groups
```

Then **log out and back in** (or reboot) so the new `dialout`/`input`/`plugdev` groups apply.

`setup_linux.sh` creates a local `.venv`, installs the Python deps, downloads Google's MediaPipe
face model, adds udev rules for the lens and scaler boards, and installs the right-click "Open With"
entry. (To do it by hand instead, see the commands inside that script.)

## Run

```bash
./hub/play_sbs.sh /path/to/your_sbs_video.mp4
```

The panel switches to 3D, the lens turns on, and the video is woven to your eyes. Sit roughly centred,
~60 cm away, and it tracks your head as you move.

### Hotkeys

Global (work without focusing the window), all `Ctrl+Alt+…`:

| Keys | Action |
|------|--------|
| `Ctrl+Alt+3` | toggle 3D on/off |
| `Ctrl+Alt+F` | swap left/right eye |
| `Ctrl+Alt+V` | flip image vertically |
| `Ctrl+Alt+H` | toggle the on-screen HUD |
| `Ctrl+Alt+S` | switch the panel 60 ↔ 120 Hz |
| `Ctrl+Alt+,` / `Ctrl+Alt+.` | less / more eye-prediction lead |
| `Ctrl+Alt+←/→/↑/↓` | nudge the sweet-spot (alignment) |
| `Ctrl+Alt+0` | recenter (reset alignment) |
| `Ctrl+Alt+Q` | quit (restores the panel) |

Video playback (when the weave window is focused): `Space` play/pause · `←/→` seek ±10 s ·
`↑/↓` seek ±60 s · `N`/`P` next/previous video in the folder. Click the panel to summon the on-screen
controls, then click **PREV / PLAY / NEXT**, drag the progress bar to seek, or **CLOSE** to dismiss.

### Right-click "Open With → 3D SBS Player"

`setup_linux.sh` already installs this. To (re)install by hand:

```bash
sed "s#__REPO__#$(pwd)#g" packaging/sbs3d-player.desktop \
  > ~/.local/share/applications/sbs3d-player.desktop
update-desktop-database ~/.local/share/applications 2>/dev/null || true
```

Now right-click any SBS video → **Open With → 3D SBS Player**.

## Calibration

The player ships the **standard Samsung Odyssey 3D (G90XF) calibration** — the captured weave field, the
stack-correction textures, and the camera/stereo calibration (see **[`DISCLAIMER.md`](DISCLAIMER.md)** for
exactly what that includes and why). The G90XF is the same across units, so this gives clean glasses-free
3D **out of the box — no per-monitor capture required**. (A doubled image is almost always the panel's 3D
mode not engaging — see Troubleshooting, not a calibration issue.)

To fine-tune for your seating: change the depth via `bscale` in `calib/baseline_scale.json`, and the
sweet-spot offset in `calib/eye_offset.json`. See **[`CALIBRATION.md`](CALIBRATION.md)**.

## Troubleshooting

- **Doubled / ghosted image (you see two overlapping pictures)** → the panel's 3D interleave mode isn't
  engaging. The player drives it through the monitor's scaler over a SCSI-generic node (`/dev/sg*`); if
  the startup log shows `[scaler] *** 3D PANEL MODE FAILED ***` (a `Permission denied` on `/dev/sg*`),
  run `bash setup_linux.sh` (it installs the scaler udev rule and adds you to `plugdev`) and then **log
  out and back in**. Confirm with `groups | grep plugdev` and look for `[scaler] 3D ON` in the log.
- **Picture upside down** → `GPU_CAPTURE_MPV_FLIPY=0 ./hub/play_sbs.sh file.mp4`
- **No 3D effect / lens won't turn on** → confirm the monitor's USB cable is connected and you're in
  the `dialout` group (`groups | grep dialout`); re-log-in after setup. On a cold boot the lens may need
  authenticating — see [`DISCLAIMER.md`](DISCLAIMER.md) on the lens auth.
- **Wrong screen / nothing on the panel** → set your panel's output name:
  `SBS3D_PANEL_OUTPUT=DP-4 ./hub/play_sbs.sh file.mp4` (find it with `xrandr`).
- **No sound, or video won't start** → make sure you're on **Xorg, not Wayland**, and that `mpv`,
  `libmpv` and `ffmpeg` are installed (the setup script does this).
- **No head tracking** → `models/face_landmarker.task` is missing; re-run `setup_linux.sh` or download
  it (see `models/README.md`). The weave still runs, just pinned to a fixed viewpoint.

## How it works (short version)

`play_sbs.sh` → `hub/screen_weave.py`: libmpv (NVDEC) decodes the SBS frame straight into a GL texture;
`exact_weaver.py` + `linux_oracle.py` compute the lenticular weave; `tracker.py`/`stereo_tracker.py`
find your eyes via the webcam; `lens.py` turns the physical 3D lens on over USB serial. Calibration
lives in `calib/`. There are **no external network calls** at runtime.

## Acknowledgments

- Head tracking uses Google's [MediaPipe](https://github.com/google-ai-edge/mediapipe) FaceLandmarker, licensed under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).
- Thanks to people over at [r/Odyssey3D](https://www.reddit.com/r/Odyssey3D/) for getting me into this glasses-free 3D-monitor stuff.

## License

[MIT](LICENSE). See [`DISCLAIMER.md`](DISCLAIMER.md) for the interoperability / no-warranty notice.
