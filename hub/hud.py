"""
hud.py — minimal, lens-robust on-screen HUD for screen_weave.

A monospace bitmap-font atlas (built once, cached to assets/) + instanced glyph quads + flat panels,
drawn as a post-weave BLENDED pass. Everything is drawn as FLAT pixels (identical for both eye-views), so
it reads cleanly through the lenticular lens (same trick as the existing track-dot bypass). Preallocated
instance buffers (no per-frame allocs) so it's cheap even at 120/165 Hz.

Headless self-test (no panel, no lens):  DISPLAY=:1 python hud.py   ->  writes assets/hud_validate.png
"""
import os, math, json
import numpy as np
import moderngl

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
]
GLYPH_PX = 32
FIRST, LAST = 32, 126        # printable ASCII space..~
NCOLS = 16
MAXG = 600                   # max glyphs per draw_text call (preallocated instance buffer)
try:
    from linux_oracle import OPTPOS
except Exception:
    OPTPOS = (0.0, 8.39, 65.11)


# ---------------------------------------------------------------- atlas
def build_atlas(px=GLYPH_PX, cols=NCOLS):
    from PIL import Image, ImageFont, ImageDraw
    fp = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)
    if fp is None:
        raise RuntimeError("no DejaVuSansMono font found; install fonts-dejavu-core")
    font = ImageFont.truetype(fp, px)
    asc, desc = font.getmetrics()
    cw = int(math.ceil(font.getlength("M")))     # monospace advance == cell width
    ch = asc + desc
    n = LAST - FIRST + 1
    rows = (n + cols - 1) // cols
    img = Image.new("L", (cw * cols, ch * rows), 0)
    d = ImageDraw.Draw(img)
    for i in range(n):
        cx, cy = (i % cols) * cw, (i // cols) * ch
        d.text((cx, cy), chr(FIRST + i), fill=255, font=font, anchor="la")
    return np.asarray(img, np.uint8), cw, ch, cols, rows


def load_or_build_atlas():
    """Cache the atlas PNG + metrics in assets/; rebuild only if missing/corrupt."""
    png = os.path.join(ASSETS, "hud_atlas.png"); meta = os.path.join(ASSETS, "hud_atlas.json")
    try:
        from PIL import Image
        if os.path.exists(png) and os.path.exists(meta):
            a = np.asarray(Image.open(png).convert("L"), np.uint8)
            m = json.load(open(meta))
            return a, m["CW"], m["CH"], m["COLS"], m["ROWS"]
    except Exception:
        pass
    a, cw, ch, cols, rows = build_atlas()
    try:
        os.makedirs(ASSETS, exist_ok=True)
        from PIL import Image
        Image.fromarray(a).save(png)
        json.dump({"CW": cw, "CH": ch, "COLS": cols, "ROWS": rows, "FIRST": FIRST, "LAST": LAST}, open(meta, "w"))
    except Exception:
        pass
    return a, cw, ch, cols, rows


# ---------------------------------------------------------------- shaders
HUD_VERT = """
#version 330
in vec2 corner; in vec4 rect; in vec2 cell;
uniform vec2 uRes; uniform vec2 uCell;
out vec2 fuv;
void main(){
    vec2 px = rect.xy + corner * rect.zw;
    gl_Position = vec4(px.x/uRes.x*2.0-1.0, 1.0-px.y/uRes.y*2.0, 0.0, 1.0);
    fuv = (cell + corner) * uCell;
}
"""
HUD_FRAG = """
#version 330
uniform sampler2D uAtlas; uniform vec3 uColor; uniform float uAlpha;
in vec2 fuv; out vec4 o;
void main(){ o = vec4(uColor, texture(uAtlas, fuv).r * uAlpha); }
"""
PANEL_VERT = """
#version 330
in vec2 corner; uniform vec4 rect; uniform vec2 uRes;
void main(){ vec2 px = rect.xy + corner*rect.zw;
  gl_Position = vec4(px.x/uRes.x*2.0-1.0, 1.0-px.y/uRes.y*2.0, 0.0, 1.0); }
"""
PANEL_FRAG = """
#version 330
uniform vec4 uFill; out vec4 o;
void main(){ o = uFill; }
"""


class HUD:
    def __init__(self, ctx, res, scale=2.0):
        self.ctx = ctx; self.W, self.H = res
        a, self.CW, self.CH, self.COLS, self.ROWS = load_or_build_atlas()
        self.GW = self.CW * scale; self.GH = self.CH * scale; self.PAD = 28.0
        self.atlas = ctx.texture((a.shape[1], a.shape[0]), 1, np.ascontiguousarray(a).tobytes(), dtype="f1")
        self.atlas.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.atlas.repeat_x = False; self.atlas.repeat_y = False
        self.atlas.swizzle = "RRRR"
        self.tprog = ctx.program(vertex_shader=HUD_VERT, fragment_shader=HUD_FRAG)
        self.tprog["uRes"].value = (float(self.W), float(self.H))
        self.tprog["uCell"].value = (self.CW / float(a.shape[1]), self.CH / float(a.shape[0]))
        self.tprog["uAlpha"].value = 1.0
        self.pprog = ctx.program(vertex_shader=PANEL_VERT, fragment_shader=PANEL_FRAG)
        self.pprog["uRes"].value = (float(self.W), float(self.H))
        self.unit = ctx.buffer(np.array([0, 0, 1, 0, 0, 1, 1, 1], "f4").tobytes())
        self.rb = ctx.buffer(reserve=MAXG * 16, dynamic=True)   # rect: 4 f32/glyph
        self.cb = ctx.buffer(reserve=MAXG * 8, dynamic=True)    # cell: 2 f32/glyph
        self.tvao = ctx.vertex_array(self.tprog, [(self.unit, "2f", "corner"),
                                                  (self.rb, "4f/i", "rect"), (self.cb, "2f/i", "cell")])
        self.pvao = ctx.vertex_array(self.pprog, [(self.unit, "2f", "corner")])
        self.bar_rect = None      # (x,y,w,h) of the seek bar last drawn, in framebuffer px (for click hit-testing); None if hidden
        self.btns = {}            # name -> (x,y,w,h) of clickable control buttons last drawn (for hit-testing)

    def panel(self, x, y, w, h, fill):
        self.pprog["rect"].value = (float(x), float(y), float(w), float(h))
        self.pprog["uFill"].value = fill
        self.pvao.render(moderngl.TRIANGLE_STRIP)

    def button(self, name, label, x, y, w, h, active=False):
        """Draw a clickable button (panel + centred label) and register its rect under `name` in self.btns."""
        self.panel(x, y, w, h, (0.12, 0.46, 0.62, 0.95) if active else (0.10, 0.12, 0.16, 0.90))
        self.panel(x, y, w, 2, (0.30, 0.85, 1.0, 0.6))                 # subtle top accent
        tw = len(label) * self.GW
        self.text([label], x + (w - tw) * 0.5, y + (h - self.GH) * 0.5, (0.92, 0.97, 1.0))
        self.btns[name] = (x, y, w, h)

    def text(self, lines, x0, y0, color):
        rects = []; cells = []
        for li, line in enumerate(lines):
            for ci, chx in enumerate(line):
                o = ord(chx)
                if o < FIRST or o > LAST or chx == " ":
                    continue
                gi = o - FIRST
                rects.append((x0 + ci * self.GW, y0 + li * self.GH, self.GW, self.GH))
                cells.append((gi % self.COLS, gi // self.COLS))
        n = len(rects)
        if n == 0:
            return
        n = min(n, MAXG)
        self.rb.write(np.asarray(rects[:n], "f4").tobytes())
        self.cb.write(np.asarray(cells[:n], "f4").tobytes())
        self.atlas.use(0); self.tprog["uAtlas"].value = 0; self.tprog["uColor"].value = color
        self.tvao.render(moderngl.TRIANGLE_STRIP, instances=n)

    def _sweet(self, ex):
        d = abs(ex - OPTPOS[0])
        if d < 1.3:
            return "CENTER OK"
        return "move RIGHT >>" if ex < OPTPOS[0] else "<< move LEFT"

    def draw(self, st):
        """st keys: mode3d, src, cap_fps, render_fps, swap, vflip, track_ok, ex, ey, ez. Self-blends."""
        ctx = self.ctx
        self.btns = {}                       # re-registered each frame (for click hit-testing)
        mode = "3D ON" if st.get("mode3d", True) else "2D (browse)"
        lines = [
            f"{mode:<16}(C-A-3 toggle)",
            f"SRC   {str(st.get('src',''))[:28]}",
            f"CAP   {st.get('cap_fps',0):3.0f} fps    RENDER {st.get('render_fps',0):3.0f} fps",
            f"L/R   swap:{'ON ' if st.get('swap') else 'OFF'}   vflip:{'ON ' if st.get('vflip') else 'OFF'}",
            f"TRACK {'OK  ' if st.get('track_ok') else 'LOST'}     eye {st.get('ex',0):+.1f},{st.get('ey',0):+.1f},{st.get('ez',0):+.1f}",
            f"LEAD  {st.get('lead_ms',20.0):3.0f} ms    {'(auto)' if st.get('lead_auto', False) else '(C-A- ./, tune)'}",
            f"SWEET {self._sweet(st.get('ex',0.0))}",
            "",
            "C-A: 3 F V H S Q  arrows align    player: SPACE pause  </> seek  N/P video    click: PREV PLAY NEXT  empty=hide",
        ]
        ncols = max(len(l) for l in lines)
        pw = ncols * self.GW + self.PAD * 2; ph = len(lines) * self.GH + self.PAD * 2
        x0, y0 = 60.0, 60.0
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.panel(x0, y0, pw, ph, (0.02, 0.03, 0.05, 0.62))
        self.panel(x0, y0, 6, ph, (0.10, 1.0, 0.30, 0.9) if st.get("mode3d", True) else (1.0, 0.70, 0.10, 0.9))
        self.text([lines[0]], x0 + self.PAD, y0 + self.PAD, (0.30, 1.0, 0.45))
        self.text(lines[1:7], x0 + self.PAD, y0 + self.PAD + self.GH, (0.80, 0.92, 1.0))   # body: SRC..SWEET (6 lines)
        self.text([lines[8]], x0 + self.PAD, y0 + self.PAD + 8 * self.GH, (0.55, 0.62, 0.70))  # legend
        dot = (0.10, 1.0, 0.20, 1.0) if st.get("track_ok") else (1.0, 0.20, 0.20, 1.0)
        self.panel(x0 + pw - self.PAD - 18, y0 + self.PAD + 4 * self.GH + 4, 16, 16, dot)
        cw = 5 * self.GW + 24; ch = self.GH + 10              # CLOSE button (top-right) -> click to hide the HUD
        self.button("close", "CLOSE", x0 + pw - cw - 12, y0 + 12, cw, ch)
        # ---- media seek bar (libmpv SBS player): flat = identical per eye = comfortable screen-depth ----
        self.bar_rect = None
        m = st.get("media")
        if m and m.get("dur", 0) > 0.5:
            W, H = self.W, self.H
            bw = W * 0.5; bh = 16.0; bx = (W - bw) * 0.5; by = H - 200.0   # room below for the transport buttons
            self.bar_rect = (bx, by, bw, bh)
            frac = max(0.0, min(1.0, m.get("pos", 0.0) / m["dur"]))
            self.panel(bx - 6, by - self.GH - 16, bw + 12, self.GH + bh + 30, (0.02, 0.03, 0.05, 0.55))  # backing
            self.panel(bx, by, bw, bh, (0.22, 0.24, 0.28, 0.95))                        # track
            self.panel(bx, by, bw * frac, bh, (0.20, 0.85, 1.0, 0.97))                  # progress fill
            self.panel(bx + bw * frac - 3, by - 6, 6, bh + 12, (1.0, 1.0, 1.0, 1.0))    # playhead
            def _fmt(s):
                s = int(max(0, s))
                return f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}" if s >= 3600 else f"{s//60}:{s%60:02d}"
            label = ("[PAUSED]  " if m.get("paused") else "") + f"{_fmt(m.get('pos',0))} / {_fmt(m['dur'])}"
            self.text([label], bx, by - self.GH - 8, (0.85, 0.95, 1.0))
            # transport buttons (flat = comfortable screen depth), centred just below the seek bar
            specs = [("prev", "PREV"), ("play", "PLAY" if m.get("paused") else "PAUSE"), ("next", "NEXT")]
            bhc = self.GH + 18; gap = 20.0
            ws = [max(len(l) * self.GW + 44, 150.0) for _, l in specs]
            total = sum(ws) + gap * (len(specs) - 1); xc = (W - total) * 0.5
            yc = min(by + bh + 26, H - bhc - 24)            # clamp so the row stays on-screen at any res/atlas
            for (nm, lab), wbtn in zip(specs, ws):
                self.button(nm, lab, xc, yc, wbtn, bhc, active=(nm == "play" and not m.get("paused")))
                xc += wbtn + gap
        ctx.disable(moderngl.BLEND)


def validate():
    """Headless: render the HUD over a fake backdrop to an offscreen FBO and save a PNG. No panel/lens."""
    ctx = moderngl.create_context(standalone=True, backend="egl")
    print("GL_RENDERER:", ctx.info.get("GL_RENDERER"))
    W, H = 3840, 2160
    fbo = ctx.framebuffer(color_attachments=[ctx.texture((W, H), 4, dtype="f1")])
    fbo.use(); ctx.clear(0.10, 0.12, 0.15, 1.0)
    hud = HUD(ctx, (W, H))
    hud.draw({"mode3d": True, "src": "YouTube - Avatar 3D SBS", "cap_fps": 60, "render_fps": 60,
              "swap": True, "vflip": False, "track_ok": True, "ex": -0.4, "ey": 8.4, "ez": 65.1,
              "lead_ms": 25.0, "lead_auto": True})
    ctx.finish()
    img = np.frombuffer(fbo.read(components=4), np.uint8).reshape(H, W, 4)[::-1]
    crop = img[40:760, 40:1500]
    from PIL import Image
    out = os.path.join(ASSETS, "hud_validate.png"); os.makedirs(ASSETS, exist_ok=True)
    Image.fromarray(crop, "RGBA").save(out)
    nonblack = int((crop[:, :, :3].max(2) > 20).sum())
    print(f"wrote {out} {crop.shape} nonblack_px={nonblack}")
    assert nonblack > 1000, "HUD rendered blank!"
    print("HUD validate PASS")


if __name__ == "__main__":
    validate()
