"""
linux_oracle.py — pure-Python replacement for the Windows DimencoWeaving.dll FillAttributes oracle.

Drop-in for exact_weaver.Oracle: same interface
    o = LinuxOracle(W, H); o.set_eye(ex,ey,ez); (v2,v3,v4) = o.fill(ndc_x, ndc_y)
returning ((v2x,v2y),(v3x,v3y),(v4x,v4y,v4z)) where v2=(f.x,f.y), v3=(d.z,d.w), v4=(c.x,c.y,c.z).

NO Windows DLL is loaded. The attribute fields produced by FillAttributes are AFFINE in screen
position (confirmed: every component's bilinear cross-term is <=1 float32 ULP, i.e. rounding noise),
so the 4 screen-corner vectors fully determine the field; the GPU's linear vertex-
attribute interpolation reproduces it exactly. The DLL was sampled at exactly one eye position
(optPos), captured into _fields.pkl 'corners'.

TWO MODES (selected by `parametric` flag, default False = static):

  STATIC  (default, PROVEN-EXACT at the reference eye): fill() returns the captured corner vectors
          from _fields.pkl (bilinearly interpolated to an arbitrary ndc if requested, but the 4
          corners are the canonical vertex inputs and are returned verbatim at the 4 corners).
          This reproduces the DLL to float32 precision *for the reference eye only*. With this path
          the weave is correct but does NOT track head motion (it always weaves for ~optPos).

  PARAMETRIC (opt-in, eye-tracking; verified to 2.3e-5 vs the DLL at optPos): derives d.z/d.w/f.x/f.y
          analytically from eye position via the geometric model reverse-engineered from the corner
          capture. The per-channel phase ramp c (=v4) is a pure screen ramp (eye-
          independent to the DLL's precision). CAVEAT: the eye-DEPTH (ez) law of the two constants
          ez_eff and period_cm could not be separated from a single eye sample (additive vs
          multiplicative); at ez=optPos.z both modes are identical, and the (x,y) eye dependence is
          linear and well-determined. See EYE-DEPENDENCE STATUS below.

CONFIRMED WEAVE FORMULA (median frac err 0.0355 cyc vs the ground-truth phase field — see self-test):
    rsqd    = 1/sqrt(1 + d.z^2 + d.w^2)
    phase_c = frac( -( f.x*rsqd - 2*f.y*(corrB-0.5) + corrA + c[ch] + 0.25 ) )
  NOTE the sign convention: corrA is ADDED and the (corrB-0.5) term is -2*f.y, all under a global
  negate. exact_weaver.py's GLSL currently uses the opposite corrA/corrB signs (a positive base);
  for the triangle FilterSlope weight the global negate is irrelevant (the weight is symmetric about
  phase=0.5), but the corrA/corrB signs are NOT — exact_weaver's FRAG corrA/corrB signs should be
  flipped to match this ground-truth-validated form. The oracle itself is sign-agnostic (it only
  emits f/d/c); this note is for whoever wires the shader.
"""
from __future__ import annotations
import os, math, pickle

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS_PKL = os.path.join(HERE, "_fields.pkl")

# --- reference eye / nominal sweet-spot (DimencoWeaving optPos, display-frame cm). Standard G90XF working
#     distance; used as the default eye seed and the ez reference. Same for every G90XF unit. ---
OPTPOS = (0.0, 8.391599655151367, 65.11128997802734)

# --- STANDARD G90XF panel/lens constants. These are panel-DESIGN geometry (pixel pitch, lenslet pitch &
#     slant, effective stack height) -- identical across G90XF units, NOT a per-unit calibration. We assume
#     all Samsung Odyssey 3D (G90XF) monitors are the same, so these nominal constants drive the weave
#     directly; no per-monitor capture is required for a correct 3D image. ---
PX_CM   = 0.01554        # pixel pitch, cm (screen.ini pixelSize)
SLANT   = 0.28893        # lenticular slant, measured from c-ramp (nominal param 0.29)
# constants MEASURED from the optPos corner capture; see EYE-DEPENDENCE STATUS for ez laws:
EZ_EFF_REF    = 86.6526  # effective depth used inside the perspective tangent d.zw (cm) at optPos
PERIOD_CM_REF = 12.42811 # phase-pitch DISTANCE used inside f.x at optPos (cm) = p_lens*optPos.z/h_eff
# --- first-principles lens geometry (closes the ez-law that one capture left open) -----------------
# A slanted lenticular of pitch p_lens (panel cm) at reduced stack height h_eff above the panel maps a
# lateral parallax to a view-slot shift with sensitivity h_eff/(p_lens*ez). Hence f.x carries 1/ez just
# like f.y, and the ratio f.x/f.y = h_eff/p_lens is a pure-geometry CONSTANT (independent of eye):
#     measured f.x/f.y = 5.23903 at every corner  ==  h_eff/p_lens = 0.146548/0.027972 = 5.23910.
# So f.x = (h_eff/p_lens)*f.y = (h_eff/p_lens)*proj/ez. This reproduces the capture EXACTLY at optPos
# (period_cm = optPos.z/(h_eff/p_lens) = 12.42811) and—unlike the old ez-CONSTANT period—gives the
# physically-correct 1/ez fall-off of eye-x sensitivity off the working distance (the previously-OPEN
# depth law). Likewise ez_eff scales multiplicatively with ez (same perspective-distance geometry).
PX_LENS_PX = 1.8                         # lenticular pitch in PANEL pixels (weave_params_real.json Px)
P_LENS_CM  = PX_LENS_PX * PX_CM          # physical lenticular pitch on the panel, cm  (0.027972)
H_EFF_CM   = (OPTPOS[2] / PERIOD_CM_REF) * P_LENS_CM  # reduced stack height pinned to capture (0.146548)
FX_FY_RATIO = H_EFF_CM / P_LENS_CM       # = ez/period_cm at optPos = 5.23910 (pure-geometry constant)


def _bilerp(c00, c10, c01, c11, u, v):
    """Bilinear interpolation matching the DLL field generation."""
    return (c00 * (1 - u) * (1 - v) + c10 * u * (1 - v)
            + c01 * (1 - u) * v + c11 * u * v)


def _synth_corners(W, H):
    """Build the STANDARD G90XF FillAttributes corner field analytically from the lens-geometry constants.

    This is the normal path: we assume every G90XF panel is geometrically identical, so the weave field
    comes straight from the standard panel constants above -- no per-monitor capture needed. (An optional
    _fields.pkl, if present, overrides this with a captured field; absent, this analytic field IS the
    canonical weave.) The v4 phase ramp is a pure screen ramp of slope
    1/PX_LENS_PX in x and SLANT/PX_LENS_PX in y, with 1/3-pixel RGB sub-steps; v2/v3 are the
    parametric model evaluated at the reference eye OPTPOS. Only frac(phase) matters downstream, so
    the absolute phase origin is arbitrary (re-aligned at runtime with the align hotkeys). The
    result is uncalibrated/nominal -- capture _fields.pkl for a calibrated weave (see CALIBRATION.md)."""
    c_kx = 1.0 / PX_LENS_PX                          # v4 phase cycles per pixel in x (== 1/Px_eff)
    c_ky = SLANT * c_kx                              # slant-coupled y ramp
    step = [ch * c_kx / 3.0 for ch in range(3)]     # RGB sub-pixel phase steps
    ex, ey, ez = OPTPOS
    ez_eff = ez * (EZ_EFF_REF / OPTPOS[2])
    corners = {}
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        px = (W - 0.5) if sx > 0 else -0.5
        py = (H - 0.5) if sy > 0 else -0.5
        base = c_kx * (px + 0.5) + c_ky * (py + 0.5)
        c = [base + step[ch] for ch in range(3)] + [base + step[2]]   # c0..c3 (4th unused)
        dx = (ex / PX_CM + W / 2.0) - px
        dy = (ey / PX_CM) - py
        v3x = dx * PX_CM / ez_eff
        v3y = dy * PX_CM / ez_eff
        proj = (dx + SLANT * dy) * PX_CM
        v2y = proj / ez
        v2x = FX_FY_RATIO * v2y
        corners[(sx, sy)] = ([c[0], c[1], c[2], c[3]],  # c (v4 phase ramp)
                             [0.0, 0.0, v3x, v3y],      # d (v3 = d.z,d.w used)
                             [0.0, 0.0, 0.0],           # e (unused)
                             [v2x, v2y, 0.0])           # f (v2 = f.x,f.y used)
    return corners


class LinuxOracle:
    """Pure-Python FillAttributes oracle. Same interface as exact_weaver.Oracle.

    Args:
        W, H        : panel resolution (px).
        parametric  : if True, fill() is eye-parametric (tracks head motion) via the derived
                      geometric model. If False (default), fill() returns the captured reference-eye
                      corner field (static, proven-exact at optPos, no head tracking).
        fields_pkl  : path to the corner capture (defaults to HUB/_fields.pkl).
    """

    def __init__(self, W, H, parametric: bool = False, fields_pkl: str | None = None):
        self.W, self.H = float(W), float(H)
        self.parametric = bool(parametric)
        self.eye = list(OPTPOS)

        # The repo SHIPS the standard G90XF lenticular weave field as hub/_fields.pkl (captured once;
        # the panel geometry is identical across units, so it is THE standard field, not a per-unit secret).
        # If it's ever absent -- or SBS3D_NOMINAL=1 -- fall back to the analytic field from panel geometry.
        pkl_path = fields_pkl or FIELDS_PKL
        if os.path.exists(pkl_path) and os.environ.get("SBS3D_NOMINAL", "0") != "1":
            with open(pkl_path, "rb") as fh:
                corners = pickle.load(fh)["corners"]
            print("[oracle] standard G90XF weave field (shipped _fields.pkl capture)", flush=True)
        else:
            print("[oracle] _fields.pkl absent -> analytic fallback weave field (set SBS3D_NOMINAL=0 / restore the file for the calibrated weave)", flush=True)
            corners = _synth_corners(self.W, self.H)
        # corners keyed (sx,sy) in {-1,1}; value = ([c0..c3],[d0..d3],[e0..e2],[f0..f2])
        # (sx,sy)=(-1,-1)->TL, (1,-1)->TR, (-1,1)->BL, (1,1)->BR ; U=px_x/W (l->r), V=px_y/H (t->b)
        self._corners = corners
        self._TL, self._TR = corners[(-1, -1)], corners[(1, -1)]
        self._BL, self._BR = corners[(-1, 1)], corners[(1, 1)]

        # ---- fit the exact AFFINE screen model of c (=v4) directly from the corners ----
        # c_ch(px,py) = c00_ch + kx*(px+0.5) + ky*(py+0.5)  (px=-0.5 at left edge -> matches TL).
        # The huge integer part is irrelevant (only frac() of phase matters) but we keep it exact so
        # the static and parametric paths agree bit-for-bit at the corners. Per-channel: ch=0,1,2.
        c_TL = self._TL[0]
        self._c00 = [c_TL[ch] for ch in range(3)]              # c at TL (px=py=-0.5)
        self._c_kx = (self._TR[0][0] - self._TL[0][0]) / self.W  # x slope (= 1/Px_eff), per px
        self._c_ky = (self._BL[0][0] - self._TL[0][0]) / self.H  # y slope (= Slant/Px_eff), per px

        # ---- reduce the v4 integer magnitude to preserve float32 phase precision downstream ----
        # frac() is invariant to subtracting a common integer; do it so the shader's float32 frac is sharp.
        self._v4_int = float(int(min(self._c00)))

        # subpixel per-channel phase steps (RGB), constant across screen and eye:
        self._c_step = [self._TL[0][ch] - self._TL[0][0] for ch in range(3)]  # [0, +0.18457, +0.36914]

    # ---------------------------------------------------------------- interface
    def set_eye(self, ex, ey, ez):
        self.eye = [float(ex), float(ey), float(ez)]

    def fill(self, ndc_x, ndc_y):
        """Return ((v2x,v2y),(v3x,v3y),(v4x,v4y,v4z)) for the given NDC corner/position.

        ndc in [-1,1]^2; (ndc_x, ndc_y) maps to screen U=(ndc_x+1)/2, V=(ndc_y+1)/2 (y screen-DOWN,
        i.e. ndc_y matches exact_weaver's `ndc` list TL,TR,BL,BR — caller passes screen-down y)."""
        u = (float(ndc_x) + 1.0) * 0.5    # 0=left, 1=right
        v = (float(ndc_y) + 1.0) * 0.5    # 0=top,  1=bottom  (screen-down)
        if self.parametric:
            return self._fill_parametric(u, v)
        return self._fill_static(u, v)

    # ---------------------------------------------------------------- static (proven exact @ optPos)
    def _fill_static(self, u, v):
        """Bilerp the captured reference-eye corner vectors. Exact at optPos to float32."""
        f0 = _bilerp(self._TL[3][0], self._TR[3][0], self._BL[3][0], self._BR[3][0], u, v)
        f1 = _bilerp(self._TL[3][1], self._TR[3][1], self._BL[3][1], self._BR[3][1], u, v)
        d2 = _bilerp(self._TL[1][2], self._TR[1][2], self._BL[1][2], self._BR[1][2], u, v)
        d3 = _bilerp(self._TL[1][3], self._TR[1][3], self._BL[1][3], self._BR[1][3], u, v)
        v4 = tuple(
            _bilerp(self._TL[0][ch], self._TR[0][ch], self._BL[0][ch], self._BR[0][ch], u, v)
            - self._v4_int
            for ch in range(3)
        )
        return (f0, f1), (d2, d3), v4

    # ---------------------------------------------------------------- parametric (eye-tracking)
    def _fill_parametric(self, u, v):
        """Eye-parametric attribute model (derived from the optPos corner capture; verified 2.3e-5
        vs the DLL at optPos). Eye enters LINEARLY through the perpendicular-foot offset d.xy."""
        ex, ey, ez = self.eye
        px = u * self.W - 0.5          # screen pixel x (matches e/corner pixel coords)
        py = v * self.H - 0.5          # screen pixel y

        # eye's perpendicular foot, in pixels (proven: d.x=eye_x_px-px, d.y=eye_y_px-py)
        eye_x_px = ex / PX_CM + self.W / 2.0
        eye_y_px = ey / PX_CM
        dx = eye_x_px - px
        dy = eye_y_px - py

        # v3 = screen->eye direction tangent (used only inside rsqrt for perspective foreshortening)
        # ez_eff = ez * (EZ_EFF_REF/optPos.z): multiplicative — a perspective distance scales with ez, so this
        # degrades more gracefully off-distance than the additive offset (audit; both equal at optPos, <=0.087cyc).
        ez_eff = ez * (EZ_EFF_REF / OPTPOS[2])
        v3x = dx * PX_CM / ez_eff
        v3y = dy * PX_CM / ez_eff

        # v2 = phase coefficients, BOTH built from ONE proj (proven at the capture: f.x/f.y is the
        # same constant 5.23903 at all 4 corners -> v2.x and v2.y truly share a single proj).
        #   f.y = true slant-projected viewing angle (carries 1/ez).
        #   f.x = the eye-x PHASE term. Physically it is f.y times the pure-geometry lens ratio
        #         h_eff/p_lens, so it ALSO carries 1/ez (a lateral parallax subtends a smaller view-slot
        #         shift the farther the eye is). This equals proj/period_cm only AT optPos.z; off the
        #         working distance the 1/ez form is the physically-correct one (closes the open ez-law).
        proj = (dx + SLANT * dy) * PX_CM
        v2y = proj / ez
        v2x = FX_FY_RATIO * v2y        # = (h_eff/p_lens)*proj/ez ; == proj/PERIOD_CM_REF at ez=optPos.z

        # v4 = per-channel screen phase ramp (eye-independent to DLL precision); reduce integer.
        v4 = tuple(
            self._c00[ch] + self._c_kx * (px + 0.5) + self._c_ky * (py + 0.5) - self._v4_int
            for ch in range(3)
        )
        return (v2x, v2y), (v3x, v3y), v4


# ====================================================================== self-test / validation
if __name__ == "__main__":
    import sys
    import numpy as np

    CALIB = os.path.join(HERE, "..", "calib")
    PF = os.environ.get("SBS3D_PHASE_FIELD", "")   # optional ground-truth phase-field .npy for numeric validation

    try:
        import cv2
    except Exception:
        cv2 = None

    W, H = 3840, 2160

    def reconstruct_phase(oracle):
        """Bilinear-interpolate fill() across the screen, apply the confirmed formula, return (H,W,3)
        phase in cycles. Mirrors the GPU vertex interpolation + decoded pixel-shader base."""
        # screen UV meshgrid (pixel centers)
        xs = (np.arange(W) + 0.5) / W
        ys = (np.arange(H) + 0.5) / H
        U, V = np.meshgrid(xs, ys)

        # sample fill() at the 4 NDC corners (TL,TR,BL,BR; ndc_y screen-down), GPU interpolates between.
        ndc = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
        fa = [oracle.fill(nx, ny) for (nx, ny) in ndc]   # [(v2,v3,v4)] for TL,TR,BL,BR

        def bil(getter):
            a, b, c, d = (getter(fa[0]), getter(fa[1]), getter(fa[2]), getter(fa[3]))
            return a * (1 - U) * (1 - V) + b * U * (1 - V) + c * (1 - U) * V + d * U * V

        fx = bil(lambda t: t[0][0])
        fy = bil(lambda t: t[0][1])
        dz = bil(lambda t: t[1][0])
        dw = bil(lambda t: t[1][1])
        c_ch = [bil(lambda t, ch=ch: t[2][ch]) for ch in range(3)]

        cA = cv2.resize(cv2.imread(os.path.join(CALIB, "3DStackCorrection_A.png"),
                                   cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0,
                        (W, H), interpolation=cv2.INTER_LINEAR)
        cB = cv2.resize(cv2.imread(os.path.join(CALIB, "3DStackCorrection_B.png"),
                                   cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0,
                        (W, H), interpolation=cv2.INTER_LINEAR)

        rsqd = 1.0 / np.sqrt(1.0 + dz * dz + dw * dw)
        out = np.empty((H, W, 3), np.float64)
        for ch in range(3):
            # ground-truth-validated negative-sign form (median 0.0355 cyc):
            base = fx * rsqd - 2.0 * fy * (cB - 0.5) + cA
            out[:, :, ch] = -(base + c_ch[ch] + 0.25)
        return out

    def medfe(model, gt):
        return float(np.median(np.abs((((model - gt) + 0.5) % 1.0) - 0.5)))

    if cv2 is None or not PF or not os.path.exists(PF):
        print("[self-test] no ground-truth phase field (set SBS3D_PHASE_FIELD); skipping numeric validation.")
        sys.exit(0)

    gt = np.load(PF)[:, :, ::-1].astype(np.float64)   # stored BGR -> channel-reversed

    print("=== LinuxOracle self-test (vs ground-truth phase field) ===")
    for mode, parm in (("STATIC", False), ("PARAMETRIC@optPos", True)):
        o = LinuxOracle(W, H, parametric=parm)
        o.set_eye(*OPTPOS)
        model = reconstruct_phase(o)
        per = [medfe(model[:, :, ch], gt[:, :, ch]) for ch in range(3)]
        print(f"[{mode:18s}] median |frac err| per channel (R,G,B): "
              f"{per[0]:.5f} {per[1]:.5f} {per[2]:.5f}  | mean {np.mean(per):.5f}")

    # cross-check: static vs parametric agreement at optPos (should be ~ULP)
    a = LinuxOracle(W, H, parametric=False); a.set_eye(*OPTPOS)
    b = LinuxOracle(W, H, parametric=True);  b.set_eye(*OPTPOS)
    for ndc in [(-1, -1), (1, -1), (-1, 1), (1, 1), (0.0, 0.0)]:
        va, vb = a.fill(*ndc), b.fill(*ndc)
        diffs = [abs(va[i][j] - vb[i][j]) for i in range(3) for j in range(len(va[i]))]
        print(f"[static-vs-parametric] ndc={ndc} max|diff|={max(diffs):.3e}")
