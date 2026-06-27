#!/usr/bin/env bash
# play3d_app.sh — "Open With > 3D SBS Player" handler. Plays a SIDE-BY-SIDE 3D video on the Odyssey panel via
# the zero-copy libmpv weave (play_sbs.sh, locked to panel refresh, 120fps @4K). Nautilus invokes this with the file path as $1.
#   play3d_app.sh /path/to/sbs.mp4
set -u
HUB="$(cd "$(dirname "$0")" && pwd)"            # this script's own dir (the hub/)
_note(){ command -v notify-send >/dev/null && notify-send "3D SBS Player" "$1"; echo "$1"; }
cd "$HUB" || { _note "hub not found: $HUB"; exit 1; }
_UID="$(id -u)"
# When launched from the file manager there's often no X env. Resolve the running server's auth instead of
# assuming gdm: keep an inherited XAUTHORITY, else the active Xorg's own -auth file, else ~/.Xauthority.
_xauth(){
  local p f
  for p in $(pgrep -x Xorg 2>/dev/null) $(pgrep -x X 2>/dev/null); do
    f=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | grep -A1 -x -- '-auth' | tail -1)
    [ -n "$f" ] && [ -r "$f" ] && { echo "$f"; return; }
  done
  [ -r "$HOME/.Xauthority" ] && { echo "$HOME/.Xauthority"; return; }
  echo "/run/user/$_UID/gdm/Xauthority"
}
export DISPLAY="${DISPLAY:-:1}" XAUTHORITY="${XAUTHORITY:-$(_xauth)}"
LOG=/tmp/play3d_app.log

F="${1:-}"
[ -z "$F" ] && { _note "No file given."; exit 1; }
case "$F" in file://*) F=$(python3 -c "import sys,urllib.parse,urllib.request;print(urllib.request.url2pathname(urllib.parse.urlparse(sys.argv[1]).path))" "$F" 2>/dev/null || echo "$F");; esac
[ -f "$F" ] || { _note "Not a file: $F"; exit 1; }
# reject non-video (e.g. a PNG): libmpv would load a still image and sit on it forever, hanging the panel
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$F" 2>/dev/null)
if ! awk -v d="$DUR" 'BEGIN{exit !(d+0 > 1.0)}'; then
  _note "$(basename "$F") isn't a playable video — the 3D player only handles SBS video files."
  exit 1
fi

# one weave owns the panel at a time
if pgrep -f 'screen_[w]eave.py' >/dev/null 2>&1; then
  _note "A 3D session is already running (Ctrl+Alt+Q on the panel to quit), then reopen."
  exit 0
fi
# no live weave -> clear a stale eye-tracker orphan from a crashed previous run (else it holds the camera busy)
for p in $(pgrep -f eye_tracker_proc 2>/dev/null); do kill -9 "$p" 2>/dev/null; done

_note "Playing $(basename "$F") on the Odyssey panel… (Ctrl+Alt+Q to quit)"
exec ./play_sbs.sh "$F" >"$LOG" 2>&1
