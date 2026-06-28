#!/usr/bin/env bash
# play_sbs.sh — play a SIDE-BY-SIDE 3D video as glasses-free 3D via the ZERO-COPY libmpv path (locked to the panel refresh, 120fps @ 4K).
# No Xephyr, no screen capture: libmpv (NVDEC) renders the SBS frame straight into the weave's GL texture (cap.glo).
#   ./play_sbs.sh /path/to/sbs_4k.mp4          (a side-by-side 3D video file, or an http(s):// URL)
#   GPU_CAPTURE_MPV_FLIPY=0 ./play_sbs.sh <f>  # if the picture is upside down
#   SBS_W=1920 SBS_H=1080 ./play_sbs.sh <f>    # smaller render target
# Hotkeys (global): Ctrl+Alt+3 toggle 3D · Ctrl+Alt+F swap L/R · Ctrl+Alt+H hud · Ctrl+Alt+Q quit.
set -u
cd "$(dirname "$0")"
# Auto-detect the Odyssey panel = the connected 4K (3840x2160) output, so it survives cable/port swaps; override via SBS3D_PANEL_OUTPUT.
: "${SBS3D_PANEL_OUTPUT:=$(xrandr 2>/dev/null | awk '/ connected/{o=$1} /^[[:space:]]+3840x2160/{print o; exit}')}"
: "${SBS3D_PANEL_OUTPUT:=DP-0}"
export SBS3D_PANEL_OUTPUT
SRC="${1:-${SBS3D_VIDEO:-}}"
[ -n "$SRC" ] || { echo "usage: ./play_sbs.sh <side-by-side-3d-video.mp4 | http(s)://url>" >&2; exit 2; }
ORIG_SRC="$SRC"                               # the real file (before any VP9->cache swap) -> drives the folder playlist (next/prev)
case "$SRC" in http://*|https://*) : ;; *) [ -f "$SRC" ] || { echo "[play_sbs] file not found: $SRC" >&2; exit 1; };; esac

# --- VP9 sources trip NVDEC's 10-bit block-artifact bug -> transparently play a cached HEVC transcode.
#     Cache hit = instant+clean; miss = play the original now AND build the clean copy in the background. ---
case "$SRC" in http://*|https://*) : ;; *)
  if command -v ffprobe >/dev/null 2>&1; then
    VCODEC="$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nokey=1:noprint_wrappers=1 "$SRC" 2>/dev/null)"
    CDIR="$HOME/.cache/sbs3d-player"; mkdir -p "$CDIR"
    CACHE="$CDIR/$(printf '%s' "$SRC" | sha1sum | cut -c1-16)-$(stat -c%s "$SRC" 2>/dev/null).mp4"
    case "$VCODEC" in
      h264|hevc|av1|mpeg2video|vc1|mpeg4) : ;;   # NVDEC-decodable -> the zero-copy weave plays it directly
      vp9)
        # VP9 decodes on NVDEC but trips a 10-bit block-artifact bug -> play now, build a clean HEVC cache in bg.
        if [ -f "$CACHE" ]; then echo "[play_sbs] VP9 source -> using cached clean HEVC: $CACHE"; SRC="$CACHE"
        elif ! pgrep -f "hevc_nvenc.*$(basename "$CACHE")" >/dev/null 2>&1; then
          echo "[play_sbs] VP9 source -> building clean HEVC in background (artifacts this play; clean next open)"
          command -v notify-send >/dev/null && notify-send "3D SBS Player" "VP9 file — building a clean HEVC copy in the background. Reopen when it's done for artifact-free 3D."
          ( ffmpeg -y -loglevel error -i "$SRC" -c:v hevc_nvenc -preset p5 -rc vbr -cq 21 -b:v 0 -pix_fmt p010le -profile:v main10 -c:a copy "$CACHE.tmp.mp4" \
              && mv "$CACHE.tmp.mp4" "$CACHE" \
              && { command -v notify-send >/dev/null && notify-send "3D SBS Player" "Clean copy ready — reopen the file for artifact-free 3D."; } ) >/tmp/sbs3d_transcode.log 2>&1 &
        fi ;;
      "") echo "[play_sbs] WARN: couldn't probe video codec — playing as-is" >&2 ;;
      *)
        # NVDEC CANNOT decode this codec (utvideo/ffv1/prores/huffyuv/rawvideo/dnxhd/...). Playing it would WEDGE the
        # weave (CPU-decoding a 4K intermediate hangs the load). So do NOT launch -> use a cached HEVC copy, or build
        # one in the background and bail with a clear message instead of hanging.
        if [ -f "$CACHE" ]; then
          echo "[play_sbs] '$VCODEC' is not GPU-decodable -> using cached HEVC: $CACHE"; SRC="$CACHE"
        else
          echo "[play_sbs] '$VCODEC' is NOT GPU-decodable (would hang the weave) -> converting to a playable HEVC copy in the background." >&2
          command -v notify-send >/dev/null && notify-send "3D SBS Player" "This file is '$VCODEC' — the GPU can't decode it. Converting to a playable copy in the background; reopen the file when the 'ready' notification appears."
          if ! pgrep -f "hevc_nvenc.*$(basename "$CACHE")" >/dev/null 2>&1; then
            ( ffmpeg -y -loglevel error -i "$SRC" -c:v hevc_nvenc -preset p5 -rc vbr -cq 21 -b:v 0 -pix_fmt p010le -profile:v main10 -c:a aac -b:a 192k "$CACHE.tmp.mp4" \
                && mv "$CACHE.tmp.mp4" "$CACHE" \
                && { command -v notify-send >/dev/null && notify-send "3D SBS Player" "Converted copy ready — reopen the file to watch it in 3D."; } ) >/tmp/sbs3d_transcode.log 2>&1 &
          fi
          echo "[play_sbs] not launching the weave on an unplayable source; reopen when conversion finishes." >&2
          exit 0
        fi ;;
    esac
  fi
;; esac

# Python: prefer $SBS3D_PY, then a repo-local venv (../.venv), then ~/.venvs/sbs3d, else python3 on PATH.
# NB: we already cd'd into hub/ above, so the repo venv is at ../.venv regardless of how we were invoked
# (an earlier `$(dirname "$0")/../.venv` form broke when launched as `bash hub/play_sbs.sh` from the repo root).
PY="${SBS3D_PY:-}"
if [ -z "$PY" ]; then
  for c in "../.venv/bin/python" "$HOME/.venvs/sbs3d/bin/python" "$(command -v python3)"; do
    [ -n "$c" ] && [ -x "$c" ] && { PY="$c"; break; }
  done
fi
[ -n "$PY" ] || { echo "[play_sbs] no python interpreter found (set SBS3D_PY or create a venv)" >&2; exit 1; }
SIZE_ENV=""                                  # size auto-detects from the video; set SBS_W+SBS_H to force (e.g. downscale for headroom)
[ -n "${SBS_W:-}" ] && [ -n "${SBS_H:-}" ] && SIZE_ENV="GPU_CAPTURE_SBS_W=$SBS_W GPU_CAPTURE_SBS_H=$SBS_H"
echo "[play_sbs] $SRC -> zero-copy libmpv weave (auto-sized${SIZE_ENV:+ -> forced ${SBS_W}x${SBS_H}}, no Xephyr, no capture)"
# Run as the USER, not root. The lens serial (/dev/ttyACM0) is now user-accessible (udev uaccess tag + dialout
# group; live ACL this session) so root is no longer required -- and root MUST NOT be used: it cannot reach the
# user's PipeWire socket (`ao/pulse Init failed: Timeout`), and with video_sync=audio that froze the picture.
# WEAVE_SUDO=1 forces the legacy root path (only if the lens grant is ever missing).
WEAVE_RUN=""; [ "${WEAVE_SUDO:-0}" = "1" ] && WEAVE_RUN="sudo -E"
_UID="$(id -u)"
# Resolve X auth without assuming gdm: inherited XAUTHORITY, else the running Xorg's -auth, else ~/.Xauthority.
_xauth(){
  local p f
  for p in $(pgrep -x Xorg 2>/dev/null) $(pgrep -x X 2>/dev/null); do
    f=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | grep -A1 -x -- '-auth' | tail -1)
    [ -n "$f" ] && [ -r "$f" ] && { echo "$f"; return; }
  done
  [ -r "$HOME/.Xauthority" ] && { echo "$HOME/.Xauthority"; return; }
  echo "/run/user/$_UID/gdm/Xauthority"
}
exec $WEAVE_RUN env DISPLAY="${DISPLAY:-:1}" XAUTHORITY="${XAUTHORITY:-$(_xauth)}" \
  XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$_UID}" PULSE_SERVER="${PULSE_SERVER:-unix:/run/user/$_UID/pulse/native}" \
  GPU_CAPTURE_MPV="$SRC" SBS3D_PLAYLIST_DIR="$(dirname "$ORIG_SRC")" SBS3D_PLAYLIST_CUR="$ORIG_SRC" $SIZE_ENV \
  GPU_CAPTURE_MPV_FLIPY=${GPU_CAPTURE_MPV_FLIPY:-1} GPU_CAPTURE_MPV_FLIP=${GPU_CAPTURE_MPV_FLIP:-1} \
  GPU_CAPTURE_MPV_HWDEC=${GPU_CAPTURE_MPV_HWDEC:-auto} \
  __GL_SYNC_TO_VBLANK=1 __GL_SYNC_DISPLAY_DEVICE="${PANEL_SYNC:-${SBS3D_PANEL_OUTPUT}}" __GL_YIELD=USLEEP \
  "$PY" -u screen_weave.py
