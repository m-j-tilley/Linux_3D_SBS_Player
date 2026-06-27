#!/usr/bin/env bash
# One-shot Ubuntu setup for the Linux 3D SBS Player.
#   bash setup_linux.sh        # run from the repo root
# Assumes: an NVIDIA proprietary driver is already installed (nvidia-smi works) and you are in an
# X11 / Xorg login session (NOT Wayland — pick "Ubuntu on Xorg" at the login screen).
set -e
cd "$(dirname "$0")"
chmod +x hub/play_sbs.sh hub/play3d_app.sh 2>/dev/null || true   # ensure launchers are executable (a copy/clone may drop the bit)

echo "== session check =="
if [ "${XDG_SESSION_TYPE:-x11}" = "wayland" ]; then
  echo "WARNING: Wayland session detected. Global hotkeys + GL vsync need X11 — log out and choose"
  echo "         'Ubuntu on Xorg', then re-run. Continuing anyway."
fi

echo "== apt packages (need sudo) =="
sudo apt-get update
# libmpv: newer Ubuntu ships libmpv2, older ships libmpv1 — install whichever is available.
sudo apt-get install -y python3-venv python3-pip libglfw3 libgl1 ffmpeg mpv \
  v4l-utils fonts-dejavu-core curl || true
sudo apt-get install -y libmpv2 || sudo apt-get install -y libmpv1 || true

echo "== python venv + deps (.venv) =="
python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "== head-tracking model (Google MediaPipe FaceLandmarker, ~3.6 MB) =="
mkdir -p models
if [ ! -f models/face_landmarker.task ]; then
  curl -L -o models/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
fi

echo "== device access: lens serial (dialout) + global hotkeys (input) + scaler (plugdev) =="
sudo usermod -aG dialout,input,plugdev "$USER"

echo "== udev rules: lens serial node + panel scaler SG device (both needed for 3D) =="
sudo tee /etc/udev/rules.d/99-sbs3d.rules >/dev/null <<'EOF'
# Samsung Odyssey 3D (G90XF) — device access for the glasses-free 3D player. Same for every G90XF unit.
# Lens FPC board (USB CDC-ACM): group access, stable symlink, keep ModemManager off it.
SUBSYSTEM=="tty", ATTRS{idVendor}=="354b", ATTRS{idProduct}=="0116", MODE="0660", GROUP="dialout", SYMLINK+="sbs3d-fpc", ENV{ID_MM_DEVICE_IGNORE}="1"
# Panel scaler (MStar "GCREADER" SCSI device): the player sets the panel's 3D INTERLEAVE mode over SG_IO.
# WITHOUT this the lens turns on but the panel stays in 2D processing -> a doubled / ghosted image.
SUBSYSTEM=="scsi_generic", ATTRS{idVendor}=="1b20", ATTRS{idProduct}=="0300", MODE="0660", GROUP="plugdev", TAG+="uaccess", SYMLINK+="sbs3d-scaler"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger

echo "== right-click handler: install the 'Open With > 3D SBS Player' desktop entry =="
REPO="$(cd "$(dirname "$0")" && pwd)"
APPS="$HOME/.local/share/applications"
mkdir -p "$APPS"
sed "s#__REPO__#$REPO#g" packaging/sbs3d-player.desktop > "$APPS/sbs3d-player.desktop"
update-desktop-database "$APPS" 2>/dev/null || true
echo "   installed -> $APPS/sbs3d-player.desktop  (right-click a video -> Open With -> 3D SBS Player)"

echo
echo "== DONE. Log out/in (or reboot) so the dialout/input/plugdev groups take effect. =="
echo "Then play a side-by-side 3D video on the panel:"
echo "    ./hub/play_sbs.sh /path/to/your_sbs_video.mp4"
echo
echo "Calibration is OPTIONAL: the player ships with the standard G90XF panel constants (the same for"
echo "every unit) and should give clean 3D out of the box. Fine-tune the sweet-spot with the align"
echo "hotkeys if needed -- see CALIBRATION.md."
