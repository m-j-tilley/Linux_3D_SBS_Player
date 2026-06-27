"""
exact_weaver.py — a TRUE clone of the Odyssey/Dimenco weave, reverse-engineered end-to-end.

Per frame:  tracked eye (stereo) -> display frame -> SetGlobalParameters(eye) -> FillAttributes() at the
4 screen corners (DimencoWeaving.dll oracle) -> upload per-vertex (v2.xy, v3.zw, v4.xyz) -> the EXACT
disassembled pixel shader (SimulatedRealityDirectX.dll_1) in GLSL, sampling the per-unit correction
textures (3DStackCorrection_A/B) -> weave.  Verified mapping (numerical + disasm, 0.035 cyc vs ground truth):
  base = v2.x*rsqrt(1+dot(v3.zw,v3.zw)) + 2*v2.y*(corrB-0.5) - corrA ;  phase_c = frac(base + v4[c] + 0.25)
No phase_field, no calibration knobs — every constant comes from this unit's DimencoWeaving.dll.
Windows for now (FillAttributes is the only DLL tie); the per-vertex terms can later be derived for Linux.
"""
from __future__ import annotations
import os, sys, time, math, struct, ctypes, threading, platform, argparse
import numpy as np, cv2, moderngl, glfw

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import sbs as sbsmod
import lens as lensmod
try:
    import scaler as scaler_mod
except Exception:
    scaler_mod = None
from tracker import open_stereo_cam, split_lr
from stereo_tracker import StereoEyeTracker

CALIB = os.path.join(HERE, "..", "calib")
SERIAL = b""   # per-unit panel serial (only used by the Windows DimencoWeaving oracle); read from the device if needed


def load_correction(path, fill):
    """Load a 3DStackCorrection_* weave texture. If the per-unit file is absent (it is unit-specific and
    not redistributed), return a NEUTRAL constant so the geometric weave still runs (uncalibrated/nominal):
    fill=0 for A (corrA=0) and fill=128 for B (corrB~0.5) make the correction terms vanish. Drop your unit's
    PNGs in calib/ for a calibrated weave -- see CALIBRATION.md."""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        print(f"[calib] {os.path.basename(path)} missing -> neutral correction (uncalibrated; see CALIBRATION.md)", flush=True)
        return np.full((64, 64), fill, np.uint8)
    return im
OPTPOS = (0.0, 8.391599655151367, 65.11128997802734)   # display-frame reference eye (cm), from DimencoWeaving
USER_IPD = 60.0                                         # user's real IPD (mm); triangulation auto-scaled so measured IPD -> this

# ---------------- DimencoWeaving oracle (ctypes) ----------------
BIN = r"C:\Program Files\LeiaSR\Platform\bin"
_k = []
class F2(ctypes.Structure): _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]
class F3(ctypes.Structure): _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]
class F4(ctypes.Structure): _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float), ("w", ctypes.c_float)]

def _mk_string(s):
    b = (ctypes.c_ubyte * 32)(); _k.append(b)
    if len(s) < 16:
        for i, ch in enumerate(s): b[i] = ch
        b[len(s)] = 0; struct.pack_into('<q', b, 16, len(s)); struct.pack_into('<q', b, 24, 15)
    else:
        h = ctypes.create_string_buffer(s, len(s) + 1); _k.append(h)
        struct.pack_into('<Q', b, 0, ctypes.addressof(h)); struct.pack_into('<q', b, 16, len(s)); struct.pack_into('<q', b, 24, len(s))
    return b

def _mk_vec1(sb):
    v = (ctypes.c_ubyte * 24)(); _k.append(v); f = ctypes.addressof(sb)
    struct.pack_into('<QQQ', v, 0, f, f + 32, f + 32); return v

class Oracle:
    """DimencoWeaving FillAttributes oracle: SetGlobalParameters(eye) then fill(corner)->(v2,v3,v4)."""
    def __init__(self, W, H):
        self.W, self.H = float(W), float(H)
        os.add_dll_directory(BIN)
        self.dll = ctypes.CDLL(os.path.join(BIN, "DimencoWeaving.dll"))
        c = self._call
        c("?SetInstallPath@Weaver@Dimenco@@SAXAEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z",
          None, [ctypes.c_void_p])(ctypes.addressof(_mk_string(rb"C:\Program Files\LeiaSR\Platform")))
        sn = _mk_string(SERIAL)
        c("?SetDeviceSerialNumbers@Weaver@Dimenco@@SAXAEBV?$vector@V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@V?$allocator@V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@2@@std@@@Z",
          None, [ctypes.c_void_p])(ctypes.addressof(_mk_vec1(sn)))
        c("?ReconfigureWeaver@Weaver@Dimenco@@SAXXZ")()
        self._setgp = c("?SetGlobalParameters@Weaver@Dimenco@@SAXUFLOAT3@@@Z", None, [F3])
        self._fa = c("?FillAttributes@Weaver@Dimenco@@SAXAEBUFLOAT2@@AEBUFLOAT4@@AEAU4@2AEAU3@3@Z", None,
                     [ctypes.POINTER(F2), ctypes.POINTER(F4), ctypes.POINTER(F4), ctypes.POINTER(F4), ctypes.POINTER(F3), ctypes.POINTER(F3)])

    def _call(self, m, rt=None, at=()):
        fn = self.dll[m]; fn.restype = rt; fn.argtypes = list(at); return fn

    def set_eye(self, ex, ey, ez):
        self._setgp(F3(float(ex), float(ey), float(ez)))

    def fill(self, ndc_x, ndc_y):
        a = F2(self.W, self.H); b = F4(float(ndc_x), float(-ndc_y), 0.0, 0.0)
        c = F4(); d = F4(); e = F3(); f = F3()
        self._fa(ctypes.byref(a), ctypes.byref(b), ctypes.byref(c), ctypes.byref(d), ctypes.byref(e), ctypes.byref(f))
        # v2=(f.x,f.y), v3.zw=(d.z,d.w), v4=(c.x,c.y,c.z)
        return (f.x, f.y), (d.z, d.w), (c.x, c.y, c.z)


# ---------------- exact shaders (ported from the disassembly) ----------------
VERT = """
#version 330
in vec2 pos; in vec2 uv; in vec2 av2; in vec2 av3; in vec3 av4;
out vec2 fuv; out vec2 v2; out vec2 v3; out vec3 v4;
void main(){ gl_Position = vec4(pos, 0.0, 1.0); fuv = uv; v2 = av2; v3 = av3; v4 = av4; }
"""
FRAG = """
#version 330
uniform sampler2D uL, uR, uCorrA, uCorrB;
uniform vec2 uRes; uniform float uFS, uXTalkFac, uXTalkDyn, uContrast, uCorrAScale, uConv, uConvMin, uConvMax; uniform int uTrackOk, uWeave;
uniform vec3 uSlidVal, uSlidDef; uniform int uSlidShow;   // tuning sliders: normalized current value + default tick, per param
in vec2 fuv; in vec2 v2; in vec2 v3; in vec3 v4;
out vec4 o;
void main(){
    if(uWeave==0){ o = vec4(texture(uL, fuv).rgb, 1.0); return; }   // 2D passthrough (left view) when weave off
    if(uConvMax > uConvMin){                                        // convergence bar overlay (bottom-centre)
        float bw=300.0, bx0=uRes.x*0.5-bw, bx1=uRes.x*0.5+bw, by0=28.0, by1=52.0;
        if(gl_FragCoord.x>bx0 && gl_FragCoord.x<bx1 && gl_FragCoord.y>by0 && gl_FragCoord.y<by1){
            float p=(uConv-uConvMin)/(uConvMax-uConvMin); float mx=bx0+p*(bx1-bx0); float cx=bx0+0.5*(bx1-bx0);
            vec3 col=vec3(0.12);
            if(abs(gl_FragCoord.x-cx)<1.0) col=vec3(0.35);                     // centre tick (conv=0)
            if(abs(gl_FragCoord.x-mx)<4.0) col=vec3(0.1,1.0,0.2);              // current-value marker
            if(gl_FragCoord.x<bx0+2.0||gl_FragCoord.x>bx1-2.0||gl_FragCoord.y<by0+2.0||gl_FragCoord.y>by1-2.0) col=vec3(0.6); // border
            o=vec4(col,1.0); return;
        }
    }
    if(gl_FragCoord.x < 36.0 && gl_FragCoord.y > uRes.y-36.0){ o = (uTrackOk==1)?vec4(0,1,0,1):vec4(1,0,0,1); return; }
    if(uSlidShow==1){   // coarse tuning sliders (drawn identical to both eyes -> flat/readable through the lens)
        float bx0=80.0, bw=1100.0, bx1=bx0+bw, bh=46.0, gap=28.0;
        for(int s=0;s<3;s++){
            float by1=uRes.y-90.0-float(s)*(bh+gap); float by0=by1-bh;
            if(gl_FragCoord.x>bx0-6.0 && gl_FragCoord.x<bx1+6.0 && gl_FragCoord.y>by0 && gl_FragCoord.y<by1){
                vec3 c = (s==0)?vec3(0.1,1.0,1.0):((s==1)?vec3(1.0,1.0,0.1):vec3(1.0,0.3,1.0));   // cyan/yellow/magenta
                float vx=bx0+clamp(uSlidVal[s],0.0,1.0)*bw, dx=bx0+clamp(uSlidDef[s],0.0,1.0)*bw;
                vec3 col=vec3(0.08);                                              // track
                if(gl_FragCoord.x>bx0 && gl_FragCoord.x<vx) col=c*0.5;            // filled = current value (length is lens-robust)
                if(abs(gl_FragCoord.x-dx)<3.0) col=vec3(0.9);                     // white default tick
                if(abs(gl_FragCoord.x-vx)<6.0) col=c;                             // current marker
                if(gl_FragCoord.x<bx0||gl_FragCoord.x>bx1||gl_FragCoord.y<by0+3.0||gl_FragCoord.y>by1-3.0) col=vec3(0.45); // border
                o=vec4(col,1.0); return;
            }
        }
    }
    vec2 suv = vec2(gl_FragCoord.x/uRes.x, 1.0 - gl_FragCoord.y/uRes.y);   // screen UV (flip GL bottom-up) for correction textures
    float corrA = texture(uCorrA, suv).r * uCorrAScale;
    float corrB = texture(uCorrB, suv).r;
    float rsq = inversesqrt(1.0 + dot(v3, v3));
    float base = v2.x*rsq - 2.0*v2.y*(corrB - 0.5) + corrA;        // ground-truth sign form (medFE 0.0355 vs 0.250 for the old +..-corrA)
    vec3 L = texture(uR, fuv).rgb, R = texture(uL, fuv + vec2(uConv, 0.0)).rgb;   // uL/uR swapped -> match Hub eye order (was pseudoscopic); uConv=convergence
    L = (L-0.5)*uContrast + 0.5; R = (R-0.5)*uContrast + 0.5;
    vec3 dRL = R - L;                                              // exact per-pixel crosstalk (disasm L46-60)
    float xt = ((1.0 + dot(v3, v3)) * v2.y*v2.y) * uXTalkDyn + uXTalkFac;
    xt = xt/(1.0 - xt);
    vec3 Lp = L - xt*dRL; vec3 Rp = R + xt*dRL;                    // exact symmetric crosstalk pre-distortion (disasm)
    vec3 outc;
    for(int c=0;c<3;c++){
        float phase = fract(base + v4[c] - 0.25);                  // view ordering -> matches hardware capture (NCC +0.99)
        float t = 2.0*phase - 1.0;
        float w = clamp(-abs(t)*uFS + uFS*0.5 + 0.5, 0.0, 1.0);     // FilterSlope triangle
        outc[c] = mix(Rp[c], Lp[c], w);                            // lerp(view1', view0', w)
    }
    o = vec4(outc, 1.0);
}
"""


# ---------------- SCREEN-MODE fragment shader (zero-copy GPU capture) ----------------
# Identical weave math to FRAG, but the two SBS views come from ONE captured texture (uSrc = the full
# SBS window) split on the GPU: left half -> uv.x in [0,0.5], right half -> uv.x in [0.5,1.0]. This means
# NO CPU L/R split and NO second texture upload. The eye order matches FRAG (L view <- RIGHT half,
# R view <- LEFT half, the validated Hub orientation). uSrcFlip flips the sample uv.y when the capture
# texture is pixmap-y-flipped (texture_from_pixmap is usually top-origin). uSrcBGR swaps R/B if the
# captured pixmap is BGRA rather than RGBA. The correction-texture UV (suv) and all weave math below the
# split are byte-for-byte the same as FRAG.
FRAG_SCREEN = """
#version 330
uniform sampler2D uSrc, uCorrA, uCorrB;
uniform vec2 uRes; uniform float uFS, uXTalkFac, uXTalkDyn, uContrast, uCorrAScale, uConv, uConvMin, uConvMax;
uniform int uTrackOk, uWeave, uSrcFlip, uSrcBGR, uSrcSwapLR;
in vec2 fuv; in vec2 v2; in vec2 v3; in vec3 v4;
out vec4 o;
vec3 srcHalf(vec2 uv, float xlo){            // sample one SBS half; xlo=0.0 left half, 0.5 right half
    vec2 s = vec2(xlo + uv.x*0.5, uv.y);
    if(uSrcFlip==1) s.y = 1.0 - s.y;         // pixmap y-flip (texture_from_pixmap top-origin)
    vec3 c = texture(uSrc, s).rgb;
    return (uSrcBGR==1) ? c.bgr : c;
}
void main(){
    if(uWeave==0){ o = vec4(srcHalf(fuv, 0.0), 1.0); return; }   // 2D passthrough = left SBS half
    if(gl_FragCoord.x < 36.0 && gl_FragCoord.y > uRes.y-36.0){ o = (uTrackOk==1)?vec4(0,1,0,1):vec4(1,0,0,1); return; }
    vec2 suv = vec2(gl_FragCoord.x/uRes.x, 1.0 - gl_FragCoord.y/uRes.y);   // screen UV for correction textures
    float corrA = texture(uCorrA, suv).r * uCorrAScale;
    float corrB = texture(uCorrB, suv).r;
    float rsq = inversesqrt(1.0 + dot(v3, v3));
    float base = v2.x*rsq - 2.0*v2.y*(corrB - 0.5) + corrA;
    float lx = (uSrcSwapLR==1) ? 0.0 : 0.5;                      // which SBS half feeds each woven view;
    float rx = (uSrcSwapLR==1) ? 0.5 : 0.0;                      // toggle (key F) to fix flipped/pseudoscopic depth
    vec3 L = srcHalf(fuv, lx);
    vec3 R = srcHalf(fuv + vec2(uConv, 0.0), rx);
    L = (L-0.5)*uContrast + 0.5; R = (R-0.5)*uContrast + 0.5;
    vec3 dRL = R - L;
    float xt = ((1.0 + dot(v3, v3)) * v2.y*v2.y) * uXTalkDyn + uXTalkFac;
    xt = xt/(1.0 - xt);
    vec3 Lp = L - xt*dRL; vec3 Rp = R + xt*dRL;
    vec3 outc;
    for(int c=0;c<3;c++){
        float phase = fract(base + v4[c] - 0.25);
        float t = 2.0*phase - 1.0;
        float w = clamp(-abs(t)*uFS + uFS*0.5 + 0.5, 0.0, 1.0);
        outc[c] = mix(Rp[c], Lp[c], w);
    }
    o = vec4(outc, 1.0);
}
"""


class TrackState:
    def __init__(self):
        self.eye = list(OPTPOS); self.vel = [0.0, 0.0, 0.0]; self.t = 0.0; self.ok = False; self.run = True
        self.snap = (list(OPTPOS), [0.0, 0.0, 0.0], 0.0)   # atomic (eye,vel,t) — render reads this to avoid torn reads
        self.gpu_ready = threading.Event()   # set once the tracker's GPU(EGL) ctx exists; weave waits before its GLX ctx (EGL-before-GLX)


class OneEuro:
    """Low-jitter, low-lag 1D filter; returns filtered value + velocity (for a predictive lead)."""
    def __init__(self, mincutoff=0.8, beta=0.6, dcutoff=1.0):
        self.mc, self.beta, self.dc = mincutoff, beta, dcutoff
        self.xp = None; self.dxp = 0.0; self.tp = None
    def _a(self, cut, dt): tau = 1.0/(2*math.pi*cut); return 1.0/(1.0 + tau/dt)
    def filt(self, x, t):
        if self.xp is None: self.xp = x; self.tp = t; return x, 0.0
        dt = max(t - self.tp, 1e-3); self.tp = t
        dx = (x - self.xp)/dt; ad = self._a(self.dc, dt); dxh = ad*dx + (1-ad)*self.dxp; self.dxp = dxh
        a = self._a(self.mc + self.beta*abs(dxh), dt); xh = a*x + (1-a)*self.xp; self.xp = xh
        return xh, dxh
    def reset(self): self.xp = None; self.dxp = 0.0; self.tp = None


def load_cam2display():
    """Tracker2DisplayTransform.ini (per-unit, fetched from the FPC) -> absolute camera->display transform.
       eye_display_cm = Rcd @ eye_cam_cm + tcd.  Camera faces user: mirror X, flip Y(down->up), keep Z (S),
       then the small mounting rotation; translation is the camera-origin offset in display coords."""
    import configparser
    cfg = configparser.ConfigParser(); cfg.read(os.path.join(CALIB, "Tracker2DisplayTransform.ini"))
    t = cfg["Transform"]
    rx, ry, rz = np.radians([float(t["xrot_deg"]), float(t["yrot_deg"]), float(t["zrot_deg"])])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    Rcd = Rz @ Ry @ Rx @ np.diag([-1.0, 1.0, 1.0])   # mirror X only; Y-flip REMOVED (was the off-axis inversion bug)
    # Oracle Y is TOP-edge-origin, +DOWN -- same sense as camera +y(down), so NO Y flip. X,Z = camera-origin offset
    # from the .ini. Y: anchor to optPos so the sweet spot is preserved while off-axis Y now tracks the right way.
    # Derive the sweet cam-midpoint from the legacy Y-flip mapping (which DID hit optPos at the sweet spot by
    # coincidence), then re-emit its Y under the corrected Rcd.
    Rcd_legacy = Rz @ Ry @ Rx @ np.diag([-1.0, -1.0, 1.0])
    tcd_legacy = np.array([float(t["xoff_cm"]), -float(t["yoff_cm"]), float(t["zoff_cm"])])
    p_sweet = np.linalg.inv(Rcd_legacy) @ (np.array(OPTPOS) - tcd_legacy)
    tcd = np.array([float(t["xoff_cm"]), float(OPTPOS[1] - (Rcd @ p_sweet)[1]), float(t["zoff_cm"])])
    print(f"[cam2display] tcd={tcd.round(3)} (cm, oracle frame; Y un-inverted)")
    return Rcd, tcd


# Z_CAL removed: eye-z phase sensitivity is ~0.0008 cyc/cm (a 4cm z error = 0.003 cyc, invisible), so depth needs
# NO correction; bscale is the sole distance scalar. Do NOT reintroduce a z-only term (the old -5.5 affine offset
# was an artifact of an earlier wrong baseline and was never applied anyway).

def cam_to_display(mid_cm, Rcd, tcd, bscale=1.0):
    """camera(left-cam,cm) -> absolute display-frame eye (cm). bscale = stereo-baseline scale, the ONE true unknown:
       z (and x,y) scale linearly with the camera separation, so scaling the triangulated point uniformly == fixing
       the baseline, correcting all distances by pure trig with a single scalar."""
    return Rcd @ (np.asarray(mid_cm, float) * bscale) + tcd


BASELINE_FILE = os.path.join(CALIB, "baseline_scale.json")   # the ONE shared scalar (camera separation), persisted

def load_baseline():
    import json
    try:
        with open(BASELINE_FILE) as f: return float(json.load(f)["bscale"])
    except Exception: return 1.0

def save_baseline(v):
    import json
    try:
        with open(BASELINE_FILE, "w") as f: json.dump({"bscale": float(v)}, f)
    except Exception: pass


def tracker_thread(state, tune=None, grace=12):
    """stereo eye (cam frame, cm) -> ABSOLUTE display frame via Tracker2DisplayTransform.ini (no reference / no R).
       One-Euro on x + grace period (hold through brief detection drops)."""
    cap = None
    try:
        # CPU is the DEFAULT for tracking: the weave is GPU-bound (4K decode + lenticular @120fps), so the GPU
        # delegate gets STARVED and detect balloons to ~50ms vs ~16ms on the idle CPU cores. GPU only wins on an
        # idle GPU (measured 1.3ms standalone). WEAVE_GPU_TRACK=1 to force GPU (e.g. a light/static source).
        _gpu_track = os.environ.get("WEAVE_GPU_TRACK", "0") == "1"
        cap = open_stereo_cam(raw_mjpg=True); et = StereoEyeTracker(gpu=_gpu_track)   # raw MJPG -> 1.4ms gray decode -> 60fps
        print(f"[track] StereoEyeTracker gpu={getattr(et, '_gpu', False)}", flush=True)
        if getattr(et, "_gpu", False):           # establish the EGL ctx NOW (before the weave's GLX) via a dummy detect
            try: _d0 = np.zeros((480, 640), np.uint8); et.detect(_d0, _d0)
            except Exception: pass
        state.gpu_ready.set()                     # release the weave to create its GL context (EGL-before-GLX ordering)
        # BUFFERSIZE=2 is the fix for the 30fps cap: with ONE buffer the driver has nowhere to queue the next frame
        # while this single-threaded loop processes the current one (read+decode+detect), so it drops every other
        # frame -> 30fps. A 2-deep queue gives the driver a buffer to fill while we hold one -> a clean 60 eye-fps.
        # 2 (not 3+) keeps worst-case staleness to a single frame (~16ms) if the loop ever briefly stalls.
        _bufsz = int(os.environ.get("WEAVE_CAM_BUFSIZE", "2"))
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, _bufsz)
        except Exception: pass
        print(f"[track] cam BUFFERSIZE={_bufsz}", flush=True)
        Rcd, tcd = load_cam2display()
        # SINGLE-THREADED read+decode+detect: a separate grabber thread serializes with this loop on the GIL
        # (OpenCV's V4L2 cap.read holds it during the frame-wait) -> halves the rate to ~30fps. One thread +
        # BUFFERSIZE=1 (freshest frame) avoids that -> ~60fps. Raw read is fast; imdecode GRAY ~1.4ms; detect ~10ms.
        oe = OneEuro(); smy = smz = None; last = None; miss = 0
        psmy = None; pt = None; vyf = 0.0                  # Y-velocity: finite-diff of smoothed Y, then EMA-smoothed
        psmz = None; vzf = 0.0                             # Z-velocity (depth), same scheme as Y; both gated by predict_yz
        predict_yz = os.environ.get("WEAVE_PREDICT_YZ", "0") == "1"   # gate: publish vel[1]/vel[2] (Y/Z) for render extrapolation
        _ti_n = 0; _ti_det = 0.0; _ti_t0 = time.perf_counter()        # instrumentation: eye-sample fps + detect-ms
        _readfail = 0; _last_reopen = 0.0; _reopen_iv = 1.0    # camera-loss resilience: USB/DPMS drop -> release + re-open
        while state.run:                                       #   with EXP backoff (1->2->4..30s) so a permanent outage
            if tune is not None: oe.mc = tune.get("mc", oe.mc); oe.beta = tune.get("beta", oe.beta); oe.dc = tune.get("dc", oe.dc)  # doesn't hammer the driver
            try: okg, fr = cap.read()                          # BUFFERSIZE=2 -> freshest frame; single-threaded (no grabber thread)
            except Exception: okg, fr = False, None
            if not okg or fr is None:                          # transient glitch OR a sustained USB/DPMS camera drop
                _readfail += 1
                if _readfail >= 30 and (time.perf_counter() - _last_reopen) > _reopen_iv:   # sustained -> re-open (backed off)
                    _last_reopen = time.perf_counter(); state.ok = False
                    print(f"[track] camera read failing ({_readfail}x) -> re-opening", flush=True)
                    try: cap.release()
                    except Exception: pass
                    try:
                        cap = open_stereo_cam(raw_mjpg=True)   # re-probe indices; re-enables IR + manual exposure
                        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, _bufsz)
                        except Exception: pass
                        _readfail = 0; _reopen_iv = 1.0; print("[track] camera re-opened OK", flush=True)
                    except Exception as _re:
                        _reopen_iv = min(30.0, _reopen_iv * 2.0)   # back off so a permanent outage won't spam the driver
                        print(f"[track] camera re-open failed: {_re} (retry in {_reopen_iv:.0f}s)", flush=True)
                else:
                    time.sleep(0.005)
                continue
            _readfail = 0
            if fr.ndim == 2 and fr.shape[0] == 1:              # raw MJPG bytes (CONVERT_RGB=0) -> fast gray decode (~1.4ms)
                frame = cv2.imdecode(fr.reshape(-1), cv2.IMREAD_GRAYSCALE)
                if frame is None: continue
            else:
                frame = fr
            L, Rv = split_lr(frame)
            gL = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY) if L.ndim == 3 else L
            gR = cv2.cvtColor(Rv, cv2.COLOR_BGR2GRAY) if Rv.ndim == 3 else Rv
            _td0 = time.perf_counter()
            r = et.detect(gL, gR); now = time.perf_counter()
            _ti_n += 1; _ti_det += (now - _td0) * 1000.0             # instrumentation
            if now - _ti_t0 >= 2.0:
                print(f"[track] {_ti_n/(now-_ti_t0):.1f} eye-fps | detect {_ti_det/max(_ti_n,1):.1f} ms | gpu={et._gpu} ok={state.ok}", flush=True)
                _ti_n = 0; _ti_det = 0.0; _ti_t0 = now
            # STABILITY: accept ONLY clean STEREO triangulation — reject the MONO fallback (its baseline-fudged x
            # was ~30% of frames and the dominant glitch) and reject physically-impossible per-frame jumps (>10cm).
            # Anything rejected -> hold the last good eye (grace lets a sustained real move re-acquire).
            accepted = False
            if r is not None and r.get("mode") == "stereo":
                ed = cam_to_display(r["mid_cm"], Rcd, tcd, (tune or {}).get("bscale", load_baseline()))
                x, y, z = float(ed[0]), float(ed[1]), float(ed[2])
                if last is None or miss >= grace or abs(x - last[0]) <= 10.0:
                    accepted = True
            if accepted:
                if last is None or miss >= grace:        # snap on (re)acquire
                    oe.reset(); smy = y; smz = z; psmy = None; vyf = 0.0; psmz = None; vzf = 0.0
                last = (x, y, z); miss = 0
                xf, vx = oe.filt(x, now)
                smy += 0.25*(y - smy); smz += 0.20*(z - smz)
                if predict_yz and psmy is not None and now > pt:
                    vyf += 0.3 * ((smy - psmy) / (now - pt) - vyf)   # finite-diff Y vel, EMA-smoothed (raw diff is noisy)
                    vzf += 0.3 * ((smz - psmz) / (now - pt) - vzf)   # same for Z (depth) so lean-forward/back also gets lead
                psmy = smy; psmz = smz; pt = now
                _vy = max(-25.0, min(25.0, vyf)) if predict_yz else 0.0
                _vz = max(-25.0, min(25.0, vzf)) if predict_yz else 0.0
                _eye = [xf, smy, smz]                     # absolute — no optPos offset, no reference
                _vel = [max(-25.0, min(25.0, vx)), _vy, _vz]   # clamp x/y/z-velocity; y/z gated by WEAVE_PREDICT_YZ (else 0)
                state.snap = (_eye, _vel, now)            # ATOMIC publish (one rebind) — render reads this; no torn read
                state.eye = _eye; state.vel = _vel; state.t = now   # compat shims
                state.ok = True
            else:
                miss += 1
                if miss > grace: state.ok = False    # hold last eye + stay green through brief detection drops/rejects
    except Exception as e:
        print("[track] error:", e); state.ok = False
        try: state.gpu_ready.set()             # don't strand the weave waiting if the tracker failed to start
        except Exception: pass
    finally:
        try:
            if cap is not None: cap.release()    # always free the camera, even if the loop raised (no leaked /dev/video0)
        except Exception: pass


# ---- SEPARATE-PROCESS eye tracker (GIL isolation) ---------------------------------------------------------
# The tracker thread shares Python's GIL with the weave's 120fps render loop, which starves it to ~30fps (vs
# ~51-60 standalone). Running it in its OWN process gives it its own GIL -> full detection-limited rate. It runs
# the SAME tracker_thread() and mirrors each (eye,vel,t,ok) into a tiny double-buffered shm the weave reads.
# (time.perf_counter is CLOCK_MONOTONIC = system-wide on Linux, so the cross-process timestamp/age math is valid.)
EYE_SHM_FLOATS = 1 + 2 * 8     # [0]=active slot; then 2 slots of [eye3, vel3, t, ok]

class _ShmSink:
    """TrackState-shaped object whose snap/ok mirror into the eye shm. Used INSIDE the tracker process."""
    def __init__(self, arr):
        self._arr = arr; self.run = True
        self.gpu_ready = threading.Event(); self.gpu_ready.set()
        self.eye = list(OPTPOS); self.vel = [0.0, 0.0, 0.0]; self.t = 0.0; self._ok = False
    def _write(self):
        a = self._arr; s = 1 - int(a[0]); b = 1 + s * 8
        a[b], a[b+1], a[b+2] = self.eye; a[b+3], a[b+4], a[b+5] = self.vel
        a[b+6] = self.t; a[b+7] = 1.0 if self._ok else 0.0
        a[0] = float(s)                    # single-float flip = lock-free publish
    @property
    def snap(self): return (self.eye, self.vel, self.t)
    @snap.setter
    def snap(self, v): self.eye = list(v[0]); self.vel = list(v[1]); self.t = float(v[2]); self._ok = True; self._write()
    @property
    def ok(self): return self._ok
    @ok.setter
    def ok(self, v): self._ok = bool(v); self._write()

def _eye_shm_reader(state, arr):
    """Weave-side: cheaply mirror shm -> state.snap/ok (~500Hz, tiny GIL cost); the render reads state.snap."""
    state.gpu_ready.set()
    while state.run:
        s = int(arr[0]); b = 1 + s * 8
        state.snap = ([float(arr[b]), float(arr[b+1]), float(arr[b+2])],
                      [float(arr[b+3]), float(arr[b+4]), float(arr[b+5])], float(arr[b+6]))
        state.ok = arr[b+7] > 0.5
        time.sleep(0.002)

def _spawn_tracker_child(env):
    import subprocess
    return subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "eye_tracker_proc.py")], env=env)


class _TrackerHandle:
    """Holds the CURRENT tracker child; the watchdog swaps in a fresh one on death. Exposes terminate/wait/poll/pid
       so the weave teardown drives it exactly like a Popen (no screen_weave change needed). terminate() is a FULL
       safe shutdown: stop watchdog+reader, SIGTERM->SIGKILL-escalate the child, then JOIN the shm-reader so the
       caller can free the shm without a use-after-free."""
    def __init__(self, proc, state=None, reader=None):
        self.proc = proc; self._stop = False; self.state = state; self.reader = reader
    def terminate(self):
        self._stop = True                                    # stop the watchdog respawning
        if self.state is not None:
            try: self.state.run = False                      # stop the shm-reader loop
            except Exception: pass
        p = self.proc
        try:
            if p is not None:
                p.terminate()
                try: p.wait(timeout=2)
                except Exception:                            # child ignored SIGTERM (e.g. stuck in a cv2 C call) -> SIGKILL
                    try: p.kill(); p.wait(timeout=2)
                    except Exception: pass
        except Exception: pass
        if self.reader is not None:                          # JOIN the shm-reader BEFORE the caller frees the shm (no UAF)
            try: self.reader.join(timeout=1.5)
            except Exception: pass
    def wait(self, timeout=None):
        try: return self.proc.wait(timeout=timeout) if self.proc is not None else None
        except Exception: return None
    def poll(self):
        try: return self.proc.poll() if self.proc is not None else None
        except Exception: return None
    @property
    def pid(self):
        return self.proc.pid if self.proc is not None else None


def _eye_tracker_watchdog(state, env, handle, poll_s=0.25):
    """Respawn the tracker child if it DIES (crash/OOM/exit) so an unattended run self-heals. Reuses the SAME shm
       (the new child re-attaches by name; the weave still owns + unlinks it), so the shm-reader + render keep
       reading without a hitch. Exp backoff 0.5/1/2/4/8/10s (cap 10s). Stops when state.run is False or handle terminated."""
    fails = 0; spawn_t = time.perf_counter()
    while getattr(state, "run", True) and not handle._stop:
        time.sleep(poll_s)
        if not getattr(state, "run", True) or handle._stop: break
        if handle.poll() is None:                            # child alive
            if time.perf_counter() - spawn_t > 30.0: fails = 0   # a long healthy run -> reset the backoff
            continue
        rc = handle.poll(); fails += 1
        backoff = min(10.0, 0.5 * (2 ** min(fails - 1, 5)))  # 0.5/1/2/4/8/10s: a persistent failure (camera gone) slows
        print(f"[track-wd] tracker process exited (rc={rc}) -> respawn #{fails} in {backoff:.1f}s", flush=True)   # down, not busy-loops
        time.sleep(backoff)
        if not getattr(state, "run", True) or handle._stop: break
        try:
            handle.proc = _spawn_tracker_child(env); spawn_t = time.perf_counter()
            if handle._stop:                                 # teardown raced us -> don't leave the fresh child orphaned
                try: handle.proc.terminate()
                except Exception: pass
                break
            print(f"[track-wd] respawned tracker pid {handle.pid}", flush=True)
        except Exception as e:
            print("[track-wd] respawn failed:", e, flush=True)


def start_eye_tracker_proc(state, tune):
    """Spawn the tracker in its own process + start the cheap shm-reader + a watchdog that respawns the child if it
       dies. Returns (_TrackerHandle, SharedMemory) to clean up (handle.terminate()/wait() behave like a Popen)."""
    import json
    from multiprocessing import shared_memory
    name = f"sbs3d_eye_{os.getpid()}"
    try: shared_memory.SharedMemory(name=name).unlink()      # clear any stale one
    except Exception: pass
    shm = shared_memory.SharedMemory(name=name, create=True, size=EYE_SHM_FLOATS * 8)
    arr = np.ndarray((EYE_SHM_FLOATS,), dtype=np.float64, buffer=shm.buf); arr[:] = 0.0
    arr[1:4] = OPTPOS                                         # seed a sane eye before the first detection
    env = dict(os.environ, WEAVE_EYE_SHM=name,
               WEAVE_TUNE_JSON=json.dumps({k: tune[k] for k in ("mc", "beta", "dc", "lead") if k in tune}))
    handle = _TrackerHandle(_spawn_tracker_child(env), state=state)
    rdr = threading.Thread(target=_eye_shm_reader, args=(state, arr), daemon=True); rdr.start()
    handle.reader = rdr                                      # so terminate() can join it before the shm is freed
    if os.environ.get("WEAVE_TRACK_WATCHDOG", "1") == "1":   # respawn-on-death (set 0 to disable)
        threading.Thread(target=_eye_tracker_watchdog, args=(state, env, handle), daemon=True).start()
    return handle, shm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image"); ap.add_argument("--video"); ap.add_argument("--layout", default="half")
    ap.add_argument("--no-track", action="store_true"); ap.add_argument("--no-lens", action="store_true")
    ap.add_argument("--no-scaler", action="store_true")
    ap.add_argument("--exp-ipd", action="store_true")   # EXPERIMENTAL: scale triangulation so measured IPD -> USER_IPD
    a = ap.parse_args()

    vcap = None; vfps = 30.0
    if a.video:
        vcap = cv2.VideoCapture(a.video); vfps = vcap.get(cv2.CAP_PROP_FPS) or 30.0
        okv, fr = vcap.read()
        if not okv: raise RuntimeError("cannot read video: " + a.video)
        hw = fr.shape[1] // 2
        L = cv2.cvtColor(fr[:, :hw], cv2.COLOR_BGR2RGB); R = cv2.cvtColor(fr[:, hw:2 * hw], cv2.COLOR_BGR2RGB)
        print(f"[video] {a.video} {fr.shape[1]}x{fr.shape[0]} @ {vfps:.0f}fps (SBS split in half)")
    else:
        L, R = sbsmod.load_sbs(a.image, a.layout)            # full-res per eye, RGB
    L = np.ascontiguousarray(L); R = np.ascontiguousarray(R)
    H, W = L.shape[0], L.shape[1]

    lens = None; svc_stopped = False; scaler_on = False
    state = TrackState()   # defined up-front so the finally-block (state.run=False) never NameErrors if setup fails
    if platform.system() == "Windows" and not a.no_lens:
        try: lensmod.windows_free_port(); svc_stopped = True
        except Exception: pass
    try:
        if not a.no_lens:
            lens = lensmod.Lens().open()
            if not lens.authenticated():
                print("[lens] authenticating (native SR+AUTH)...", lens.authenticate())  # cold-boot native auth
            print("[lens] auth?", lens.authenticated(), "ON:", lens.on())
        if scaler_mod and not a.no_scaler:
            try: scaler_mod.connect(); print("[scaler] 3D ON ->", scaler_mod.set_flag(1)); scaler_on = True
            except Exception as e: print("[scaler]", e)

        if platform.system() == "Windows":
            oracle = Oracle(3840, 2160)                  # DimencoWeaving.dll FillAttributes (Windows)
        else:
            from linux_oracle import LinuxOracle          # pure-Python FillAttributes, no DLL (Linux)
            oracle = LinuxOracle(3840, 2160, parametric=not a.no_track)
        tune = {"mc": 1.0, "beta": 0.4, "lead": 0.02, "bscale": load_baseline()}; csA = [1.0]; fsv = [10.0]; xtd = [0.012854]   # mc/beta/lead tuned for low movement-jitter (audit); xtd=dynamic-crosstalk (off-center ghost fix; UP/DOWN keys)
        conv = [0.0]   # view-centering offset (display-x cm). ANCHOR TEST (2026-06-11) proved set_eye=MIDPOINT is
        # CORRECT: weaving at E puts the dead-zone at E and the two view-centers at E +/- period/4 (=3.1cm), so a
        # head whose midpoint is fed lands BOTH eyes on the view-centers. So conv MUST be 0; a nonzero conv shifts
        # the eyes off the centers into the dead-zone (that was my mis-diagnosis from a sign-flipped metric). The
        # real bug is eye-POSITION ACCURACY: the view-centers are only +/-3.1cm wide, so a >1.3cm eye-x error bleeds.
        # 9/0 keys nudge conv only for experimentation; leave at 0. Eye-x CALIBRATION (key K) is the actual fix.
        import json as _json
        EYE_OFF_FILE = os.path.join(CALIB, "eye_offset.json")   # persisted eye-x/y tracker offset (one-key calibration)
        try:
            with open(EYE_OFF_FILE) as _f: _eo = _json.load(_f); eoff = [float(_eo.get("dx", 0.0)), float(_eo.get("dy", 0.0))]
        except Exception:
            eoff = [0.0, 0.0]
        print("[eye-offset] loaded", [round(v, 2) for v in eoff], "(press K while centered at the sweet spot to recalibrate)")
        if not a.no_track:
            threading.Thread(target=tracker_thread, args=(state, tune), daemon=True).start()

        glfw.init(); mon = glfw.get_primary_monitor(); vm = glfw.get_video_mode(mon)
        glfw.window_hint(glfw.RED_BITS, vm.bits.red); glfw.window_hint(glfw.REFRESH_RATE, vm.refresh_rate)
        if platform.system() == "Windows":
            win = glfw.create_window(vm.size.width, vm.size.height, "exact weave", mon, None)
        else:
            # Linux/X11: Mutter shrinks a true-fullscreen window to its work area (3786x2091), which breaks the
            # pixel-exact sub-pixel weave (edge cut-off + ghosting). Bypass the WM with an OVERRIDE-REDIRECT
            # borderless window -> raw 3840x2160 at (0,0). Set input focus since the WM won't manage it.
            glfw.window_hint(glfw.DECORATED, False); glfw.window_hint(glfw.VISIBLE, False)
            win = glfw.create_window(vm.size.width, vm.size.height, "exact weave", None, None)
            try:
                from Xlib import display as _xd, X as _X
                _d = _xd.Display(); _xw = _d.create_resource_object('window', glfw.get_x11_window(win))
                _xw.change_attributes(override_redirect=True); _d.sync()
                glfw.show_window(win); glfw.set_window_pos(win, 0, 0); glfw.set_window_size(win, vm.size.width, vm.size.height)
                _xw.set_input_focus(_X.RevertToParent, _X.CurrentTime); _xw.configure(x=0, y=0); _d.sync()
            except Exception as e:
                print("[exact] override-redirect failed (edges may clip):", e); glfw.show_window(win)
            glfw.set_window_pos(win, 0, 0)
        fbw, fbh = glfw.get_framebuffer_size(win)
        glfw.make_context_current(win); glfw.swap_interval(1)
        ctx = moderngl.create_context()
        prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        prog["uL"] = 0; prog["uR"] = 1; prog["uCorrA"] = 2; prog["uCorrB"] = 3
        prog["uRes"].value = (float(fbw), float(fbh))
        prog["uFS"].value = 10.0; prog["uXTalkFac"].value = 0.012853500433266163; prog["uXTalkDyn"].value = float(xtd[0])   # dynamic crosstalk ENABLED (off-center ghost fix)
        # tuning sliders: (lo, hi, default) per param -> cyan=xtalk_dyn(UP/DOWN), yellow=baseline(LEFT/RIGHT), magenta=FilterSlope(,/.)
        SLR = [(0.0, 0.05, 0.012854), (0.7, 1.3, 1.0), (4.0, 12.0, 10.0)]; slid_show = [0]   # sliders HIDDEN (they tune crosstalk/slope, not the eye-x position that's the real issue)
        def _slnorm(v, lo, hi): return min(1.0, max(0.0, (v - lo) / (hi - lo)))
        prog["uSlidShow"].value = 1
        prog["uSlidDef"].value = tuple(_slnorm(d, lo, hi) for (lo, hi, d) in SLR)
        prog["uSlidVal"].value = (0.0, 0.0, 0.0)
        prog["uContrast"].value = 1.0; prog["uCorrAScale"].value = 1.0; prog["uWeave"].value = 1
        prog["uConv"].value = 0.0; prog["uConvMin"].value = 0.0; prog["uConvMax"].value = 0.0   # NCC-validated config: no convergence shift, bar hidden (don't rely on GL zero-init)

        def tex(arr, unit, filt=moderngl.LINEAR):
            t = ctx.texture((arr.shape[1], arr.shape[0]), 3, np.ascontiguousarray(arr).tobytes()); t.filter = (filt, filt)
            t.repeat_x = False; t.repeat_y = False   # CLAMP_TO_EDGE: no wrap artifact at the screen edge
            t.use(unit); return t
        tl = tex(L, 0); tr = tex(R, 1)
        cA = load_correction(os.path.join(CALIB, "3DStackCorrection_A.png"), 0)    # neutral (corrA=0) if absent
        cB = load_correction(os.path.join(CALIB, "3DStackCorrection_B.png"), 128)  # neutral (corrB~0.5) if absent
        def corr_rgb(im):
            if im.dtype == np.uint16: im = (im / 257).astype(np.uint8)
            if im.ndim == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            elif im.shape[2] == 4: im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGB)
            else: im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            return np.ascontiguousarray(im.astype(np.uint8))   # uint8 -> GL normalizes to 0..1 on sample
        tA = tex(corr_rgb(cA), 2); tB = tex(corr_rgb(cB), 3)

        # quad: 4 corners. per-vertex (pos, uv, v2, v3, v4) — v2/v3/v4 refreshed each frame from FillAttributes.
        corners = [(-1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 1.0, 0.0), (-1.0, -1.0, 0.0, 1.0), (1.0, -1.0, 1.0, 1.0)]  # pos.x,pos.y,uv.x,uv.y
        ndc = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]   # FillAttributes y is screen-DOWN: matches pos TL,TR,BL,BR
        vbo = ctx.buffer(reserve=4 * 11 * 4, dynamic=True)   # 11 floats/vertex: pos2 uv2 v2 v3 v4(3)
        ibo = ctx.buffer(np.array([0, 1, 2, 1, 3, 2], dtype="i4").tobytes())
        vao = ctx.vertex_array(prog, [(vbo, "2f 2f 2f 2f 3f", "pos", "uv", "av2", "av3", "av4")], ibo)

        weave_on = [True]; cur_eye = [0.0, 0.0, 0.0]
        def on_key(w, k, sc, act, mods):
            if act not in (glfw.PRESS, glfw.REPEAT): return
            if k == glfw.KEY_ESCAPE: glfw.set_window_should_close(w, True)
            elif k == glfw.KEY_3: weave_on[0] = not weave_on[0]; print("[weave]", "ON (3D)" if weave_on[0] else "OFF (2D)")
            elif k == glfw.KEY_LEFT: tune["bscale"] = max(0.5, tune["bscale"] - 0.005); save_baseline(tune["bscale"]); print("[baseline]", round(70.0*tune["bscale"], 1), "mm  x", round(tune["bscale"], 3), flush=True)
            elif k == glfw.KEY_RIGHT: tune["bscale"] += 0.005; save_baseline(tune["bscale"]); print("[baseline]", round(70.0*tune["bscale"], 1), "mm  x", round(tune["bscale"], 3), flush=True)
            elif k == glfw.KEY_L: print(f"CALIB LEFT-leak  eye=({cur_eye[0]:+6.1f},{cur_eye[1]:+6.1f},{cur_eye[2]:+6.1f})", flush=True)
            elif k == glfw.KEY_R: print(f"CALIB RIGHT-leak eye=({cur_eye[0]:+6.1f},{cur_eye[1]:+6.1f},{cur_eye[2]:+6.1f})", flush=True)
            elif k == glfw.KEY_SPACE: print(f"CALIB CLEAN      eye=({cur_eye[0]:+6.1f},{cur_eye[1]:+6.1f},{cur_eye[2]:+6.1f})", flush=True)
            elif k == glfw.KEY_UP: xtd[0] = min(0.05, xtd[0] + 0.002); print("[xtalk_dyn]", round(xtd[0], 4), flush=True)
            elif k == glfw.KEY_DOWN: xtd[0] = max(0.0, xtd[0] - 0.002); print("[xtalk_dyn]", round(xtd[0], 4), flush=True)
            elif k == glfw.KEY_COMMA: fsv[0] = max(2.0, fsv[0] - 0.5); print("[filterslope]", round(fsv[0], 1), flush=True)
            elif k == glfw.KEY_PERIOD: fsv[0] = min(16.0, fsv[0] + 0.5); print("[filterslope]", round(fsv[0], 1), flush=True)
            elif k == glfw.KEY_TAB: slid_show[0] = 0 if slid_show[0] else 1; print("[sliders]", "on" if slid_show[0] else "off", flush=True)
            elif k == glfw.KEY_9: conv[0] -= 0.2; print("[conv view-centering]", round(conv[0], 2), "cm", flush=True)
            elif k == glfw.KEY_0: conv[0] += 0.2; print("[conv view-centering]", round(conv[0], 2), "cm", flush=True)
            elif k == glfw.KEY_8: conv[0] = -conv[0]; print("[conv view-centering] flipped ->", round(conv[0], 2), "cm (orthoscopic<->pseudoscopic)", flush=True)
            elif k == glfw.KEY_K:   # EYE-X CALIBRATION: declare "I am centered at the sweet spot now" -> this pose := optPos
                eoff[0] = OPTPOS[0] - state.eye[0]; eoff[1] = OPTPOS[1] - state.eye[1]
                try:
                    with open(EYE_OFF_FILE, "w") as _f: _json.dump({"dx": eoff[0], "dy": eoff[1]}, _f)
                except Exception: pass
                print(f"[eye-offset] CALIBRATED to your centered pose: dx={eoff[0]:+.2f} dy={eoff[1]:+.2f} cm (saved). Now hold still / move slowly.", flush=True)
        glfw.set_key_callback(win, on_key)
        print(f"[exact] {W}x{H} per-eye {fbw}x{fbh}. SLIDERS: cyan=xtalk_dyn(UP/DOWN) yellow=baseline(LEFT/RIGHT) magenta=FilterSlope(,/.) | white tick=default | Tab=hide | L/R/SPACE=leak-log | 3=2D/3D ESC=quit")

        nf = 0; vint = 1.0 / max(vfps, 1.0); vt = time.perf_counter()
        while not glfw.window_should_close(win):
            glfw.poll_events()
            if vcap is not None and time.perf_counter() - vt >= vint:           # paced video frame
                okv, fr = vcap.read()
                if not okv: vcap.set(cv2.CAP_PROP_POS_FRAMES, 0); okv, fr = vcap.read()   # loop
                if okv:
                    hw = fr.shape[1] // 2
                    tl.write(np.ascontiguousarray(cv2.cvtColor(fr[:, :hw], cv2.COLOR_BGR2RGB)).tobytes())
                    tr.write(np.ascontiguousarray(cv2.cvtColor(fr[:, hw:2 * hw], cv2.COLOR_BGR2RGB)).tobytes())
                vt += vint
            dt = min(max(time.perf_counter() - state.t, 0.0), 0.033) + tune["lead"]   # cap measurement-age THEN add lead (audit: kills async-dt spikes + over-lead)
            ex = state.eye[0] + state.vel[0]*dt + conv[0] + eoff[0]; ey = state.eye[1] + eoff[1]; ez = state.eye[2]   # +eoff: one-key eye-x/y calibration (K)
            cur_eye[0], cur_eye[1], cur_eye[2] = ex, ey, ez
            oracle.set_eye(ex, ey, ez)
            fa = [oracle.fill(nx, ny) for (nx, ny) in ndc]
            v4min = float(int(min(min(f[2]) for f in fa)))     # reduce v4 magnitude -> sharper float32 phase (frac unchanged)
            data = []
            for i, (v2, v3, v4) in enumerate(fa):
                px, py, uvx, uvy = corners[i]
                data += [px, py, uvx, uvy, v2[0], v2[1], v3[0], v3[1], v4[0]-v4min, v4[1]-v4min, v4[2]-v4min]
            dbg_v2 = fa[0][0]
            vbo.write(np.array(data, dtype="f4").tobytes())
            nf += 1
            if nf % 60 == 0:
                print(f"[dbg] ok={state.ok} eye=({ex:+.1f},{ey:+.1f},{ez:+.1f}) corner0 v2.x={dbg_v2[0]:+.4f} v2.y={dbg_v2[1]:+.4f}", flush=True)
                try:   # HOT-RELOAD the eye-offset (so the calibration can be applied LIVE without relaunch)
                    with open(EYE_OFF_FILE) as _hf: _he = _json.load(_hf); eoff[0] = float(_he.get("dx", eoff[0])); eoff[1] = float(_he.get("dy", eoff[1]))
                except Exception: pass
            prog["uTrackOk"].value = 1 if state.ok else 0; prog["uWeave"].value = 1 if weave_on[0] else 0
            prog["uCorrAScale"].value = float(csA[0]); prog["uFS"].value = float(fsv[0]); prog["uXTalkDyn"].value = float(xtd[0])
            prog["uSlidShow"].value = slid_show[0]
            prog["uSlidVal"].value = (_slnorm(xtd[0], 0.0, 0.05), _slnorm(tune["bscale"], 0.7, 1.3), _slnorm(fsv[0], 4.0, 12.0))
            ctx.screen.use(); ctx.clear(0, 0, 0); tl.use(0); tr.use(1); tA.use(2); tB.use(3)
            vao.render(); glfw.swap_buffers(win)
    finally:
        state.run = False
        try:
            if scaler_on: time.sleep(0.15); scaler_mod.set_flag(0); scaler_mod.close()
        except Exception: pass
        try:
            if lens: lens.off(); lens.close()
        except Exception: pass
        try: glfw.terminate()
        except Exception: pass
        if platform.system() == "Windows" and not a.no_lens: lensmod.windows_restore_service()
        print("[done]")


if __name__ == "__main__":
    main()
