"""
screen_weave.py — LIVE zero-copy GPU window-capture glasses-free 3D. Captures a fullscreen SBS source
window (YouTube 3D SBS, a game, a player) via GLX_EXT_texture_from_pixmap (XComposite-redirect the
window -> NameWindowPixmap -> glXBindTexImageEXT -> a GL texture the weave samples DIRECTLY), splits
L/R ON THE GPU in the shader (left half = one eye, right half = the other), and weaves it head-tracked
fullscreen on the Odyssey panel. NO CPU copy, NO feedback (we read the source's own backing store, not
the screen, so the fullscreen weave on top is never filmed). Global Ctrl+3 toggles 3D on/off.

  Normally launched via ./play_sbs.sh <video.mp4> (it sets DISPLAY/XAUTHORITY/audio + GL-sync env for you).
  (4K@120 is the DEFAULT now; pass --no-hi120 to force 4K@60. On exit the LAUNCH rate is restored, not a blind 60.)
"""
from __future__ import annotations
import os, sys, time, threading, json, argparse, subprocess
import numpy as np, cv2, glfw, moderngl
from Xlib import display as xdisplay, X

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import exact_weaver as ew
from linux_oracle import LinuxOracle, OPTPOS
import lens as lensmod
from gpu_capture import make_capture, find_fullscreen, MpvSbsCapture
try:
    import scaler as scaler_mod
except Exception:
    scaler_mod = None

W, H = 3840, 2160
CALIB = os.path.join(HERE, "..", "calib")

# panel-mode: 4K@120 is the DEFAULT (--no-hi120 forces 60); exit restores the LAUNCH rate, not a blind 60. DISPLAY/XAUTHORITY from env.
def _detect_panel_output():
    """The Odyssey panel's CURRENT xrandr output: the connected output offering a 3840x2160 mode (the G90XF is
       the only 4K display in a typical setup). Auto-detected so it survives cable/port swaps; override via env."""
    try:
        out = subprocess.run(["xrandr", "--query"], capture_output=True, text=True, timeout=8).stdout
        cur = None
        for ln in out.splitlines():
            if ln and not ln[0].isspace():
                cur = ln.split()[0] if " connected" in ln else None
            elif cur and ln.lstrip().startswith("3840x2160 "):
                return cur
    except Exception:
        pass
    return "DP-0"
PANEL_OUTPUT = os.environ.get("SBS3D_PANEL_OUTPUT") or _detect_panel_output()   # auto-detect; override via env
MODE_120 = ("3840x2160", "119.88")   # an xrandr-listed mode for DP-2 (recon: 60 / 119.88 / 164.98)
MODE_60 = ("3840x2160", "60.00")


def _xrandr(args):
    env = dict(os.environ)
    return subprocess.run(["xrandr", "--output", PANEL_OUTPUT] + args, env=env,
                          capture_output=True, text=True)


FFCP_SCRIPT = os.environ.get("SBS3D_FFCP_SCRIPT", os.path.expanduser("~/.local/bin/sbs3d-ffcp.sh"))  # optional: re-applies ForceFullCompositionPipeline (tear-free) if present

def _reassert_ffcp():
    """An `xrandr --rate` switch can drop the metamode's ForceFullCompositionPipeline token -> tearing returns.
    Re-apply it (best-effort) after any rate change so 120Hz stays tear-free."""
    try:
        if os.path.exists(FFCP_SCRIPT):
            subprocess.run(["bash", FFCP_SCRIPT], timeout=25, capture_output=True,
                           env=dict(os.environ, DISPLAY=os.environ.get("DISPLAY", ":1")))
    except Exception:
        pass

def set_panel_mode(mode, rate):
    r = _xrandr(["--mode", mode, "--rate", rate])
    ok = (r.returncode == 0)
    print(f"[panel] {mode}@{rate} ->", "OK" if ok else f"FAIL: {r.stderr.strip()[:120]}", flush=True)
    if ok: _reassert_ffcp()                                # a rate switch drops FFCP -> re-pin tear-free
    return ok

def get_panel_rate():
    """The (mode, rate) currently ACTIVE on the panel (the '*'-marked DP-2 line) or None. Lets us avoid a
    needless switch when already at target, and restore the LAUNCH rate on exit instead of a blind 4K@60."""
    try:
        out = subprocess.run(["xrandr", "--query"], timeout=5, capture_output=True, text=True,
                             env=dict(os.environ, DISPLAY=os.environ.get("DISPLAY", ":1"))).stdout
    except Exception:
        return None
    in_panel = False
    for line in out.splitlines():
        if line and not line[0].isspace():                 # output header, e.g. "DP-2 connected ..."
            in_panel = line.startswith(PANEL_OUTPUT + " ")
        elif in_panel and "*" in line:                     # active mode row: "  3840x2160  119.88*+ 60.00 ..."
            parts = line.split()
            for tok in parts[1:]:
                if "*" in tok:
                    return (parts[0], tok.rstrip("*+"))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hi120", action=argparse.BooleanOptionalAction, default=True,
                    help="4K@120 panel mode (DEFAULT on; gives a 120fps weave); use --no-hi120 to force 4K@60")
    ap.add_argument("--win-id", default=None, help="capture this X window id (hex/dec) instead of auto-find")
    a = ap.parse_args()

    # 120Hz is the DEFAULT. Capture the rate active at LAUNCH so exit restores exactly that (never a blind 60),
    # and only switch (and arm a revert) if we are not already at the target rate.
    incoming_mode = get_panel_rate()                 # ("3840x2160","119.88") | ("3840x2160","60.00") | None
    target_mode = MODE_120 if a.hi120 else MODE_60
    panel_switched = False
    if incoming_mode is None or tuple(incoming_mode) != tuple(target_mode):
        if set_panel_mode(*target_mode):             # VERIFY the lens/scaler still engage 3D above 60Hz
            panel_switched = (incoming_mode is not None)   # arm revert only if we know the launch rate to restore

    lens = None; scaler_on = False
    try:
        lens = lensmod.Lens().open()
        if not lens.authenticated(): print("[lens] auth:", lens.authenticate())
        print("[lens] auth?", lens.authenticated(), "ON:", lens.on())
    except Exception as e: print("[lens]", e)
    if scaler_mod:
        try: scaler_mod.connect(); scaler_mod.set_flag(1); scaler_on = True; print("[scaler] 3D ON")
        except Exception as e:
            # CRITICAL: without the scaler the panel stays in 2D processing -> the woven sub-pixel
            # columns don't line up with the lenslets and you get a DOUBLED / ghosted image. Almost
            # always a permissions problem on the GCREADER /dev/sg* node -> run setup_linux.sh (it adds
            # the udev rule + plugdev group), then log out/in.
            print("[scaler] *** 3D PANEL MODE FAILED -> EXPECT A DOUBLE IMAGE ***:", e)
            print("[scaler]     fix: run ./setup_linux.sh (installs the scaler udev rule), then re-login.")

    # head tracker (panel-clock 60/120 fps; capture rate is decoupled)
    # tune: beta 0.4->0.8 (position opens up faster during motion) + dc 1.0->4.0 (velocity derivative tracks
    # motion-onset fast so the lead actually compensates). lead CUT 20ms->8ms: once the raw pipeline hit 60 eye-fps
    # + 2.6ms detect the real lag fell to ~7ms, so 20ms lead OVERSHOT (negative lag, err nearly doubled at 1.5Hz);
    # 8ms brings lag ~0 with no err penalty. mc=1.0 keeps rest-stability. (sim: latency_sim.py)
    state = ew.TrackState(); tune = {"mc": 1.0, "beta": 0.8, "lead": 0.008, "dc": 4.0, "bscale": ew.load_baseline()}
    # Tracker in its OWN PROCESS (GIL isolation) so the weave's 120fps Python loop can't starve it
    # (recovers ~30 -> ~51-60 eye-fps -> lag ~15 -> ~6-8ms). Falls back to the in-process thread if it can't start.
    trk_thread = None; _track_proc = None; _track_shm = None
    if os.environ.get("WEAVE_TRACK_PROC", "1") == "1":
        try:
            _track_proc, _track_shm = ew.start_eye_tracker_proc(state, tune)
            print("[screen] eye tracker = separate process (GIL-isolated)", flush=True)
        except Exception as _e:
            print("[screen] proc tracker failed -> in-process thread:", _e, flush=True)
    if _track_proc is None:
        trk_thread = threading.Thread(target=ew.tracker_thread, args=(state, tune), daemon=True); trk_thread.start()
    try:
        with open(os.path.join(CALIB, "eye_offset.json")) as f:
            eo = json.load(f); eoff = [float(eo.get("dx", 0)), float(eo.get("dy", 0))]
    except Exception: eoff = [0.0, 0.0]
    print("[screen] eye_offset", [round(v, 2) for v in eoff])

    # Wait for the GPU eye-tracker's EGL context to come up BEFORE we create the weave's GLX context — NVIDIA needs
    # EGL-before-GLX or MediaPipe's GPU delegate fails eglMakeCurrent and silently falls back to slow CPU tracking.
    if not state.gpu_ready.wait(timeout=12):
        print("[screen] tracker GPU-ready wait timed out; proceeding", flush=True)

    # GL context (override-redirect borderless fullscreen ON THE PANEL — bypass Mutter's work-area shrink).
    # Place the window at the primary monitor's ACTUAL origin, not screen (0,0): with the CRT/HDMI also
    # connected the Odyssey panel (DP-2) is offset (e.g. +1540+0), and (0,0) is now a different monitor.
    glfw.init(); mon = glfw.get_primary_monitor()
    for _m in glfw.get_monitors():          # target the Odyssey panel BY NAME (PANEL_OUTPUT), not just X-primary
        _n = glfw.get_monitor_name(_m); _n = _n.decode() if isinstance(_n, bytes) else _n
        if _n == PANEL_OUTPUT: mon = _m; break
    vm = glfw.get_video_mode(mon)
    _mx, _my = glfw.get_monitor_pos(mon)
    print(f"[screen] panel target: {PANEL_OUTPUT} @ {vm.size.width}x{vm.size.height} origin=({_mx},{_my})", flush=True)
    glfw.window_hint(glfw.DECORATED, False); glfw.window_hint(glfw.VISIBLE, False)
    win = glfw.create_window(vm.size.width, vm.size.height, "screen_weave", None, None)
    # SIGTERM/SIGINT -> graceful exit: the handler only SETS A FLAG (the main loop calls glfw.set_window_should_close
    # from the MAIN thread, so glfw is never touched from signal context), then the finally-block runs (lens off,
    # panel restore, shm + camera freed). An unattended kill -TERM (systemd / test harness / Ctrl+C) is clean, not a
    # hard abort mid-frame. A SECOND signal force-exits in case teardown ever wedges.
    import signal as _signal
    _sig_exit = [False]; _sigcount = [0]
    def _graceful(_sig, _frm):
        _sigcount[0] += 1
        if _sigcount[0] >= 2: os._exit(1)
        _sig_exit[0] = True
    for _s in (_signal.SIGTERM, _signal.SIGINT):
        try: _signal.signal(_s, _graceful)
        except Exception: pass
    _d = xdisplay.Display(); _xw = _d.create_resource_object('window', glfw.get_x11_window(win))
    _xw.change_attributes(override_redirect=True); _d.sync()
    # Force Mutter to UN-REDIRECT us for a direct NVIDIA page-flip (tear-free vsync), deterministically
    # rather than via the OR heuristic. Keep the depth-24 visual (no ALPHA_BITS) or this gets re-redirected.
    try:
        from Xlib import Xatom
        _xw.change_property(_d.intern_atom("_NET_WM_BYPASS_COMPOSITOR"), Xatom.CARDINAL, 32, [2]); _d.sync()
    except Exception as _e:
        print("[screen] bypass-compositor hint failed:", _e)
    # INPUT PASSTHROUGH (for weaving a GAME on :1 underneath): make the weave window click-through (empty XShape
    # input region) and DON'T steal keyboard focus, so kbd+mouse reach the game. The weave's hotkeys are global
    # (RECORD backend), so they still work. Gated by GPU_CAPTURE_INPUT_PASSTHROUGH=1 (set by game3d.sh).
    _passthrough = os.environ.get("GPU_CAPTURE_INPUT_PASSTHROUGH") == "1"
    glfw.show_window(win); glfw.set_window_pos(win, _mx, _my)
    if _passthrough:
        try:
            from Xlib.ext import shape
            _xw.shape_rectangles(shape.SO.Set, shape.SK.Input, X.YXBanded, 0, 0, [])   # empty input region = click-through
            _d.sync(); print("[screen] input PASSTHROUGH on (game keeps kbd/mouse; weave hotkeys via RECORD)", flush=True)
        except Exception as _e:
            print("[screen] input-passthrough shape failed:", _e, flush=True)
    else:
        _xw.set_input_focus(X.RevertToParent, X.CurrentTime); _d.sync()
    glfw.make_context_current(win); glfw.swap_interval(1)
    ctx = moderngl.create_context()

    # SCREEN-MODE weave program: ONE captured texture (uSrc) split L/R on the GPU (no CPU split).
    prog = ctx.program(vertex_shader=ew.VERT, fragment_shader=ew.FRAG_SCREEN)
    for k, v in (("uSrc", 0), ("uCorrA", 2), ("uCorrB", 3)):
        prog[k] = v
    fbw, fbh = glfw.get_framebuffer_size(win); prog["uRes"].value = (float(fbw), float(fbh))
    for k, v in (("uFS", 10.0), ("uXTalkFac", 0.012854), ("uXTalkDyn", 0.0), ("uContrast", 1.0),
                 ("uCorrAScale", 1.0), ("uConv", 0.0), ("uConvMin", 0.0), ("uConvMax", 0.0),
                 ("uTrackOk", 1), ("uWeave", 1), ("uSrcFlip", 0), ("uSrcBGR", 0), ("uSrcSwapLR", 0)):
        try: prog[k].value = v
        except Exception: pass

    # GPU capture: construct AFTER the GL context exists (the texture lives in THIS context).
    # make_capture() picks the live path automatically (SHM here; GLX on Mesa; CPU last resort).
    _mpv_src = os.environ.get("GPU_CAPTURE_MPV")
    if _mpv_src:                                 # ZERO-COPY: libmpv renders the SBS video straight into our texture (no Xephyr/XShm)
        cap = MpvSbsCapture(ctx, _mpv_src, glfw.get_proc_address)
    else:
        win_id = None
        if a.win_id is not None:
            win_id = int(a.win_id, 0)
        elif not (os.environ.get("GPU_CAPTURE_DISPLAY") or os.environ.get("GPU_CAPTURE_SBS_SHM")
                  or os.environ.get("GPU_CAPTURE_RGBD_SHM")):
            try: win_id = find_fullscreen(_d)    # prefer the fullscreen source; else make_capture finds the browser
            except Exception: win_id = None
        cap = make_capture(ctx, win_id=win_id)   # GPU_CAPTURE_DISPLAY=:N -> captures that nested display's whole root
    print(f"[screen] capturing {cap.size[0]}x{cap.size[1]} src via {type(cap).__name__} "
          f"(flip={cap.flipped} bgr={cap.bgr} rgba={cap.rgba_fmt})", flush=True)
    # uSrcFlip/uSrcBGR are driven BY THE CAPTURE IMPL (not guesswork): SHM=top-origin BGRA -> flip+bgr;
    # GLX=FBConfig Y_INVERTED + RGBA; CPU=already RGB row-0-bottom -> neither. Env can still override.
    prog["uSrcFlip"].value = 1 if cap.flipped else 0
    prog["uSrcBGR"].value = 1 if cap.bgr else 0

    # wrap the captured GL texture object as a moderngl external texture (NO copy). Rebuilt on resize.
    def wrap_src():
        w = ctx.external_texture(cap.glo, cap.size, (4 if cap.rgba_fmt else 3), 0, 'f1')
        w.repeat_x = False; w.repeat_y = False; w.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return w
    src_wrap = [wrap_src()]

    # last-good frame / source-gone fallback: a black texture bound to unit 0 when capture is dead.
    blk = np.zeros((H, W, 3), np.uint8)
    blk_tex = ctx.texture((W, H), 3, blk.tobytes()); blk_tex.repeat_x = False; blk_tex.repeat_y = False

    # correction textures (unchanged)
    def tex(arr, unit, filt=moderngl.LINEAR):
        t = ctx.texture((arr.shape[1], arr.shape[0]), 3, np.ascontiguousarray(arr).tobytes())
        t.filter = (filt, filt); t.repeat_x = False; t.repeat_y = False; t.use(unit); return t
    cA = ew.load_correction(os.path.join(CALIB, "3DStackCorrection_A.png"), 0)    # neutral (corrA=0) if absent
    cB = ew.load_correction(os.path.join(CALIB, "3DStackCorrection_B.png"), 128)  # neutral (corrB~0.5) if absent
    def corr(im):
        if im.dtype == np.uint16: im = (im / 257).astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(im.astype(np.uint8))
    tA = tex(corr(cA), 2); tB = tex(corr(cB), 3)

    corners = [(-1., 1., 0., 0.), (1., 1., 1., 0.), (-1., -1., 0., 1.), (1., -1., 1., 1.)]
    ndc = [(-1., -1.), (1., -1.), (-1., 1.), (1., 1.)]
    vbo = ctx.buffer(reserve=4 * 11 * 4, dynamic=True); ibo = ctx.buffer(np.array([0, 1, 2, 1, 3, 2], "i4").tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "2f 2f 2f 2f 3f", "pos", "uv", "av2", "av3", "av4")], ibo)
    oracle = LinuxOracle(W, H, parametric=True)

    shown = [True]; need_retarget = [False]
    def _set_3d_flag(on):                        # x_input_forward gates on /tmp/.weave_3don (forward only when 3D-on)
        try: open("/tmp/.weave_3don", "w").write("1" if on else "0")
        except Exception: pass
    _set_3d_flag(True)
    def toggle():
        shown[0] = not shown[0]
        _set_3d_flag(shown[0])
        if shown[0]: need_retarget[0] = True     # entering 3D -> (re)capture whatever is fullscreen NOW
        (glfw.show_window if shown[0] else glfw.hide_window)(win)
        # CLEAN 2D<->3D switch: the lens + scaler follow the mode, so "2D/browse" is crisp 2D (not lenticular
        # with the lens left on). Board stays authenticated across SR+LENS=1/0, so this relights cleanly.
        try:
            if lens: (lens.on() if shown[0] else lens.off())
        except Exception as e:
            print("[lens] toggle:", e, flush=True)
        try:
            if scaler_on: scaler_mod.set_flag(1 if shown[0] else 0)
        except Exception: pass
        if shown[0] and not _passthrough:
            try: _xw.set_input_focus(X.RevertToParent, X.CurrentTime); _d.sync()
            except Exception: pass
        print("[screen] 3D", "ON (lens on)" if shown[0] else "OFF (browse, lens off)", flush=True)

    swap = [0 if os.environ.get("GPU_CAPTURE_SWAPLR", "1") == "0" else 1]   # default 1: parallel SBS orthoscopic
    prog["uSrcSwapLR"].value = swap[0]
    def flip_lr():
        swap[0] ^= 1; prog["uSrcSwapLR"].value = swap[0]
        print(f"[screen] L/R swap = {swap[0]}", flush=True)
    vflip = [1 if cap.flipped else 0]
    def flip_v():
        vflip[0] ^= 1; prog["uSrcFlip"].value = vflip[0]
        print(f"[screen] vertical flip = {vflip[0]}", flush=True)

    # on-screen HUD (lens-robust; auto-hides after 8s, any key / Ctrl+Alt+H re-reveals)
    the_hud = None
    try:
        import hud as hudmod
        the_hud = hudmod.HUD(ctx, (fbw, fbh))
        print("[hud] ready", flush=True)
    except Exception as e:
        print("[hud] disabled:", e, flush=True)
    hud_show = [True]; hud_deadline = [time.perf_counter() + 8.0]; hud_sticky = [False]
    def poke_hud():                          # brief flash (auto-hides) -> for actions: pause, align, track change
        hud_show[0] = True; hud_sticky[0] = False; hud_deadline[0] = time.perf_counter() + 8.0
    def show_hud():                          # sticky: stays up until dismissed -> click-to-summon the controls
        hud_show[0] = True; hud_sticky[0] = True
    def hide_hud():
        hud_show[0] = False; hud_sticky[0] = False
    def toggle_hud():
        hide_hud() if hud_visible() else show_hud()
        print("[screen] HUD", "on" if hud_show[0] else "off", flush=True)
    def hud_visible():
        return hud_show[0] and (hud_sticky[0] or time.perf_counter() < hud_deadline[0])

    # --- folder playlist: next/prev video in the source's directory (libmpv player path) ---
    _VID_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".ts", ".m2ts", ".wmv", ".flv")
    def _scan_playlist():
        base = os.environ.get("SBS3D_PLAYLIST_DIR") or (os.path.dirname(os.path.abspath(_mpv_src)) if _mpv_src else "")
        cur  = os.environ.get("SBS3D_PLAYLIST_CUR") or (os.path.abspath(_mpv_src) if _mpv_src else "")
        if not base or not os.path.isdir(base):
            return [], 0
        fs = sorted((os.path.join(base, f) for f in os.listdir(base)
                     if f.lower().endswith(_VID_EXTS) and os.path.isfile(os.path.join(base, f))),
                    key=lambda p: os.path.basename(p).lower())
        try: i = fs.index(os.path.abspath(cur))
        except ValueError: i = 0
        return fs, i
    _pl_files, _pl_i = _scan_playlist()
    pl = {"files": _pl_files, "i": _pl_i}
    if pl["files"]:
        print(f"[playlist] {len(pl['files'])} videos in folder (current {pl['i']+1}/{len(pl['files'])})", flush=True)
    def play_index(i):
        if not pl["files"] or not hasattr(cap, "load_file"):
            return
        i %= len(pl["files"])
        if cap.load_file(pl["files"][i]):
            pl["i"] = i; poke_hud()
            print(f"[playlist] {i+1}/{len(pl['files'])}: {os.path.basename(pl['files'][i])}", flush=True)
    def next_video():
        if pl["files"]: play_index(pl["i"] + 1)
    def prev_video():
        if pl["files"]: play_index(pl["i"] - 1)
    def _hud_src():
        if hasattr(cap, "src_path") and cap.src_path:
            name = os.path.basename(cap.src_path)
            return f"{pl['i']+1}/{len(pl['files'])} {name}" if pl["files"] else name
        return f"live {cap.size[0]}x{cap.size[1]}"

    rate120 = [bool(a.hi120)]
    def toggle_rate():
        nonlocal panel_switched
        rate120[0] = not rate120[0]
        if set_panel_mode(*(MODE_120 if rate120[0] else MODE_60)) and incoming_mode:
            panel_switched = True    # a mid-session toggle moved us off the launch rate -> exit must restore it
    def quit_app():
        glfw.set_window_should_close(win, True)

    _lead_env = os.environ.get("WEAVE_LEAD_MS", "auto").strip().lower()   # "auto" (default) or a fixed value in ms
    lead_auto = [_lead_env in ("", "auto")]                              # auto: derive the lead from the measured frame period
    pred_lead = [0.020 if lead_auto[0] else float(_lead_env) / 1000.0]   # eye-prediction lead (s); masks serve latency
    def lead_up():     # a manual nudge takes over from auto (restart to return to auto)
        lead_auto[0] = False; pred_lead[0] = min(0.10, pred_lead[0] + 0.005); print(f"[lead] {pred_lead[0]*1000:.0f} ms (manual)", flush=True)
    def lead_down():
        lead_auto[0] = False; pred_lead[0] = max(0.0, pred_lead[0] - 0.005); print(f"[lead] {pred_lead[0]*1000:.0f} ms (manual)", flush=True)
    # live sweet-spot alignment: nudge the eye-offset (cm) added to the tracked eye; persisted so it survives a restart.
    EOFF_NUDGE = 0.5
    def save_eoff():
        try:
            with open(os.path.join(CALIB, "eye_offset.json"), "w") as f:
                json.dump({"dx": eoff[0], "dy": eoff[1]}, f)
        except Exception as _e:
            print("[eoff] save failed:", _e, flush=True)
    def eoff_nudge(dx, dy):
        eoff[0] += dx; eoff[1] += dy; save_eoff()
        print(f"[eoff] dx={eoff[0]:+.1f} dy={eoff[1]:+.1f} cm", flush=True)
    def recenter():
        eoff[0] = 0.0; eoff[1] = 0.0; save_eoff()
        print("[eoff] recentered (dx=0 dy=0)", flush=True)

    def do_retarget():
        """[MAIN THREAD] switch capture to the current fullscreen window — the source 3D should follow."""
        nonlocal cap
        if (os.environ.get("GPU_CAPTURE_DISPLAY") or os.environ.get("GPU_CAPTURE_SBS_SHM")
                or os.environ.get("GPU_CAPTURE_RGBD_SHM") or os.environ.get("GPU_CAPTURE_MPV")):
            return                                  # nested-display / SBS-shm / RGBD-shm / libmpv capture is fixed (no retarget)
        try: new_id = find_fullscreen(_d)
        except Exception: new_id = None
        if not new_id or new_id == getattr(cap, "win_id", None):
            return                                  # nothing fullscreen, or already capturing it
        try:
            newcap = make_capture(ctx, new_id)      # build BEFORE closing old, so a failure leaves cap intact
        except Exception as e:
            print("[screen] retarget: capture failed:", e, flush=True); return
        old = cap; cap = newcap
        try: old.close()
        except Exception: pass
        print(f"[screen] retargeted to fullscreen source {new_id:#x}", flush=True)

    # GLOBAL hotkeys (work without window focus). backend="record" survives gnome-shell's active grab.
    hk = None
    try:
        from hotkeys import Hotkey
        hk = Hotkey({
            "ctrl+alt+3": lambda: (toggle(), poke_hud()),          # 3D on/off
            "ctrl+alt+f": lambda: (flip_lr(), poke_hud()),          # swap L/R (parallel vs cross-eyed source)
            "ctrl+alt+h": toggle_hud,                               # show/hide HUD
            "ctrl+alt+q": quit_app,
        }, display_name=os.environ.get("DISPLAY"), backend="record")
        if getattr(hk, "failed", None):
            print("[hotkey] not registered:", hk.failed, flush=True)
        print("[hotkey] global hotkeys via", hk.backend, flush=True)
    except Exception as e:
        print("[hotkey] global hotkeys disabled:", e, flush=True)

    def on_key(w, k, sc, act, mods):
        if act not in (glfw.PRESS, glfw.REPEAT): return
        # media controls (libmpv SBS player). Arrow SEEK is intentionally HUD-silent for snappy scrubbing.
        if hasattr(cap, "seek"):
            if k == glfw.KEY_SPACE:
                if act == glfw.PRESS: cap.toggle_pause(); poke_hud()   # flash play/pause state
                return
            elif k == glfw.KEY_LEFT:  cap.seek(-10); return            # no HUD on seek
            elif k == glfw.KEY_RIGHT: cap.seek(10);  return
            elif k == glfw.KEY_UP:    cap.seek(60);  return
            elif k == glfw.KEY_DOWN:  cap.seek(-60); return
            elif k == glfw.KEY_N:
                if act == glfw.PRESS: next_video()
                return
            elif k == glfw.KEY_P:
                if act == glfw.PRESS: prev_video()
                return
        if act != glfw.PRESS: return
        if k == glfw.KEY_ESCAPE: glfw.set_window_should_close(w, True)
        elif k == glfw.KEY_3: toggle(); poke_hud()
        elif k == glfw.KEY_F: flip_lr(); poke_hud()
        elif k == glfw.KEY_V: flip_v(); poke_hud()
        elif k == glfw.KEY_H: toggle_hud()
    glfw.set_key_callback(win, on_key)

    def on_mouse(w, button, action, mods):
        if button != glfw.MOUSE_BUTTON_LEFT or action != glfw.PRESS: return
        if not hasattr(cap, "seek"): return                   # only the libmpv SBS player has playback
        cx, cy = glfw.get_cursor_pos(w)
        ww, wh = glfw.get_window_size(w); fw, fh = glfw.get_framebuffer_size(w)
        hx = cx * (fw / max(ww, 1)); hy = cy * (fh / max(wh, 1))            # window px -> HUD/framebuffer px
        if not hud_visible():
            show_hud(); return                                            # first click summons the controls
        def _hit(r): return bool(r) and r[0] <= hx <= r[0] + r[2] and r[1] <= hy <= r[1] + r[3]
        btns = getattr(the_hud, "btns", None) or {}
        if _hit(btns.get("prev")):  prev_video(); show_hud(); return    # keep controls up (sticky)
        if _hit(btns.get("next")):  next_video(); show_hud(); return
        if _hit(btns.get("play")):  cap.toggle_pause(); show_hud(); return
        if _hit(btns.get("close")): hide_hud(); return
        br = getattr(the_hud, "bar_rect", None)
        if br:
            bx, by, bw, bh = br
            if bx <= hx <= bx + bw and (by - 20) <= hy <= (by + bh + 20):
                cap.seek_frac((hx - bx) / max(bw, 1.0)); show_hud(); return   # seek; keep controls up
        hide_hud()                                                        # click empty space -> dismiss the HUD
    glfw.set_mouse_button_callback(win, on_mouse)
    print("[screen] running. Global Ctrl+Alt: 3=3D  F=swap  H=hud  Q=quit.", flush=True)
    print("[screen] player (focus the window): SPACE play/pause | <>=seek10s | ^v=60s | N/P prev/next video | H hud | "
          "click summons HUD then PREV/PLAY/NEXT + bar=seek, click empty=hide.", flush=True)

    import struct
    from collections import deque
    _PACK = struct.Struct("<44f"); _vbo_buf = bytearray(44 * 4)   # 4 verts * 11 floats, reused
    _last_eye = [1e9, 1e9, 1e9]; EYE_EPS = 0.05                   # cm; below this the corner attrs don't change
    _ema = [0.0]; _ring = deque(maxlen=600); _t_prev = [time.perf_counter()]; _last_upl = [0]
    nf = 0; _cfail = [0]
    _run_secs = float(os.environ.get("WEAVE_RUN_SECONDS", "0"))   # >0: auto-quit after N sec (headless benchmarking)
    _t_start = time.perf_counter()
    try:
        while not glfw.window_should_close(win):
            glfw.poll_events()
            if _sig_exit[0]:
                glfw.set_window_should_close(win, True)  # SIGTERM/SIGINT flagged -> graceful quit (glfw on main thread)
            if _run_secs > 0 and time.perf_counter() - _t_start > _run_secs:
                glfw.set_window_should_close(win, True)  # WEAVE_RUN_SECONDS elapsed -> clean quit + teardown (benchmarking)
            try:                                         # per-frame guard: a transient GL/HUD/capture hiccup logs
                if need_retarget[0]:                     # + continues instead of unwinding the whole run (ESC/quit
                    need_retarget[0] = False; do_retarget()   # stays outside via window_should_close above)
                _now = time.perf_counter(); _ft = (_now - _t_prev[0]) * 1000.0; _t_prev[0] = _now
                _ema[0] = _ft if _ema[0] == 0.0 else _ema[0] * 0.92 + _ft * 0.08
                _ring.append(_ft)
                if lead_auto[0]:        # auto lead ~= 1.5 measured frame-periods (mode-correct: 60Hz~25ms / 120Hz~12ms), bounded
                    pred_lead[0] = min(0.045, max(0.008, 1.5 * _ema[0] / 1000.0))
                # ---- per-frame refresh of the captured window texture (uploads only on real new content) ----
                ok = cap.refresh()
                if cap.tex_changed:                          # glo/size changed (resize/rebind) -> re-wrap
                    try: src_wrap[0].release()
                    except Exception: pass
                    src_wrap[0] = wrap_src()
                    prog["uSrcFlip"].value = 1 if cap.flipped else 0
                    prog["uSrcBGR"].value = 1 if cap.bgr else 0
                    print(f"[screen] src texture rebuilt {cap.size[0]}x{cap.size[1]} "
                          f"(flip={cap.flipped} bgr={cap.bgr})", flush=True)
                if shown[0]:
                    _eye, _vel, _t = state.snap   # atomic snapshot: eye+vel+t consistent (no torn read mid-motion)
                    dt = min(max(time.perf_counter() - _t, 0.0), 0.033) + pred_lead[0]
                    ex = _eye[0] + _vel[0] * dt + eoff[0]; ey = _eye[1] + _vel[1] * dt + eoff[1]; ez = _eye[2] + _vel[2] * dt
                    # recompute the 4 corner attrs only when the eye moved (skips fill+pack+vbo.write on a still head)
                    if (abs(ex - _last_eye[0]) > EYE_EPS or abs(ey - _last_eye[1]) > EYE_EPS
                            or abs(ez - _last_eye[2]) > EYE_EPS):
                        _last_eye[0], _last_eye[1], _last_eye[2] = ex, ey, ez
                        oracle.set_eye(ex, ey, ez)
                        fa = [oracle.fill(nx, ny) for (nx, ny) in ndc]
                        v4min = float(int(min(min(f[2]) for f in fa)))
                        flat = []
                        for i, (v2, v3, v4) in enumerate(fa):
                            px, py, uvx, uvy = corners[i]
                            flat += [px, py, uvx, uvy, v2[0], v2[1], v3[0], v3[1],
                                     v4[0] - v4min, v4[1] - v4min, v4[2] - v4min]
                        _PACK.pack_into(_vbo_buf, 0, *flat)
                        vbo.write(bytes(_vbo_buf))
                    prog["uTrackOk"].value = 1 if state.ok else 0
                    ctx.screen.use(); ctx.clear(0, 0, 0)
                    if ok and cap.alive:
                        src_wrap[0].use(0)
                    else:
                        blk_tex.use(0)                       # source went away -> black (last wrap kept for re-acquire)
                    tA.use(2); tB.use(3); vao.render()
                    nf += 1
                    if nf % 120 == 0:
                        _s = sorted(_ring)
                        _hi = _s[min(len(_s) - 1, len(_s) * 99 // 100)] if _s else 0.0   # worst-1% frame time
                        _uc = getattr(cap, "_upload_count", 0); _upl = _uc - _last_upl[0]; _last_upl[0] = _uc
                        print(f"[perf] render {1000.0/max(_ema[0],1e-3):5.1f}fps (1%low {1000.0/max(_hi,1e-3):5.1f}) "
                              f"| capture {cap.fps:.0f}fps upl/120={_upl} | eye=({ex:+.1f},{ey:+.1f},{ez:+.1f}) ok={state.ok}",
                              flush=True)
                    # HUD visibility is user-controlled (click / Ctrl+Alt+H); not force-shown while paused (easy dismiss)
                    # ---- HUD overlay (blended flat pass, on top of the weave) ----
                    if the_hud is not None and hud_visible():
                        the_hud.draw({
                            "mode3d": shown[0],
                            "src": (_hud_src() if cap.alive else "** SOURCE LOST - waiting **"),
                            "cap_fps": cap.fps, "render_fps": 1000.0 / max(_ema[0], 1e-3),
                            "swap": swap[0], "vflip": vflip[0], "track_ok": state.ok,
                            "ex": ex, "ey": ey, "ez": ez, "lead_ms": pred_lead[0] * 1000.0,
                            "lead_auto": lead_auto[0],
                            "media": ({"pos": cap.time_pos, "dur": cap.duration, "paused": cap.paused}
                                      if hasattr(cap, "time_pos") else None),
                        })
                glfw.swap_buffers(win); ctx.finish()         # finish pins CPU to the vblank flip (jitter sd 1.06->0.76)
                _cfail[0] = 0
            except Exception as _fe:
                _cfail[0] += 1
                if _cfail[0] <= 3 or _cfail[0] % 120 == 0:
                    print(f"[weave] frame error #{_cfail[0]}: {type(_fe).__name__}: {_fe}", flush=True)
                if _cfail[0] > 600:                       # ~10s of solid failures -> give up gracefully
                    print("[weave] too many consecutive frame errors -> exiting", flush=True); break
                try: time.sleep(0.004)
                except Exception: pass
    finally:
        state.run = False
        try: open("/tmp/.weave_3don", "w").write("0")   # tell the forwarder to stop forwarding
        except Exception: pass
        if _track_proc is not None:                  # GIL-isolated tracker process: stop + free its shm
            try: _track_proc.terminate(); _track_proc.wait(timeout=2)
            except Exception: pass
            try: _track_shm.close(); _track_shm.unlink()
            except Exception: pass
        try:
            if trk_thread is not None: trk_thread.join(timeout=1.5)   # in-process fallback: let it finish its frame
        except Exception: pass
        try:
            if hk: hk.close()                    # stop the hotkey RECORD thread
        except Exception: pass
        try: cap.close()                         # joins the grab thread / child before freeing shm + GL
        except Exception: pass
        try:
            if scaler_on: scaler_mod.set_flag(0); scaler_mod.close()
        except Exception: pass
        try:
            if lens: lens.off(); lens.close()
        except Exception: pass
        try:
            if panel_switched and incoming_mode:
                set_panel_mode(*incoming_mode)           # restore the rate active at launch (never a blind 4K@60)
        except Exception as _pe:
            print("[panel] restore FAILED (a manual xrandr may be needed):", _pe, flush=True)
        glfw.terminate(); print("[screen] done")


if __name__ == "__main__":
    main()
