"""
gpu_capture.py — fast, no-feedback window capture for the glasses-free-3D weave.

  >>> DECISION (measured live on THIS box, RTX 3090 / driver 590.48.01 / GNOME Shell, 2026-06-11):
      The brief's planned path — GLX_EXT_texture_from_pixmap "true GPU zero-copy" — is DEAD on the
      real target. The extension is advertised and works on OUR OWN windows, but glXCreatePixmap
      returns BadMatch(11) on every config for the COMPOSITOR-OWNED browser window (and BadValue(8)
      on depth-mismatched configs). Root cause is structural: GNOME Shell already owns the window's
      redirect (owns _NET_WM_CM_S0); NVIDIA refuses a second client's GLXPixmap over a compositor-
      owned NameWindowPixmap. No attribute fixes it. Verified by tfp_bind_test.py on 0xa00004:
      "no FBConfig+format produced a valid glXCreatePixmap+bind (all BadMatch)".

      THEREFORE the PRIMARY path is MIT-SHM (XShmGetImage): the X server DMAs the window's OWN
      backing pixels into a shared-memory segment we mmap — NO 33MB X-socket transfer (that was the
      ~1fps python-xlib bottleneck), NO feedback (we read the SOURCE window's store, never the
      screen, so the fullscreen weave composited on top is never captured). Measured here:
        XShmGetImage @ 2506x1406 = 459.8 fps (1380/1380 ok)   [xshm_bench.py]
        moderngl texture.write of a 4K RGBA frame + glFinish = 409.7 fps (2.44 ms/frame)
      So the chained pipe (XShm grab -> shared-mem numpy view -> PBO upload) clears 165 fps at 4K
      with large headroom, lossless, no feedback. This is "zero-copy server->client + one PBO
      upload", NOT literal GPU-zero-copy (that route is blocked by the compositor as proven above),
      but it is the fast path that actually delivers content on this hardware.

  CONTRACT (unchanged — screen_weave.py codes against this; all impls satisfy it):
        cap = make_capture(ctx, win_id=None)       # ctx = LIVE moderngl.Context, current
        cap.size         (w, h) full SBS window pixel size
        cap.glo          int GL texture-object name the weave samples (0 if not ready)
        cap.alive        bool — source window still exists & mapped
        cap.flipped      bool — texture top-origin vs GL bottom-origin (XShm path: True; see Y-FLIP)
        cap.rgba_fmt     bool — texture has 4 components (True for XShm depth-32 / GLX RGBA)
        cap.bgr          bool — sampled texels are B,G,R,(A) not R,G,B,(A)  (XShm depth-32 is BGRA)
        cap.fps          float — measured capture rate
        cap.refresh()    -> bool, per-frame on the GL thread; updates cap.glo's content
        cap.tex_changed  bool — True for ONE refresh() after glo/size changed (re-wrap externally)
        cap.use(unit)    bind the GL texture to a sampler unit
        cap.close()      teardown; safe twice. (alias .release)

  IMPLEMENTATIONS:
    ShmWindowCapture   PRIMARY. XShmGetImage into shared mem on a BACKGROUND thread, main thread
                       does the moderngl PBO upload (texture.write). depth-32 -> BGRA -> rgba_fmt=True,
                       bgr=True, flipped=True. This is what runs on this box.
    GLXWindowCapture   GLX_EXT_texture_from_pixmap zero-copy. Correct code; works on Mesa (AMD/Intel)
                       and any host where glXCreatePixmap succeeds on the target window. On this
                       NVIDIA+GNOME box it self-detects the BadMatch/black-texture no-op and raises so
                       the factory falls through. Force with GPU_CAPTURE_FORCE=glx.
    CpuWindowCapture   last-resort pure python-xlib get_image (~1fps@4K). Force GPU_CAPTURE_FORCE=cpu.

    make_capture(ctx, win_id) tries:  GLX (only if forced) -> SHM -> CPU.
    Default order is SHM first (GLX is proven dead here; don't pay its probe cost every launch).
    Env:
      GPU_CAPTURE_FORCE = shm | glx | cpu     pin one impl
      GPU_CAPTURE_BGR   = 0|1                  override auto BGR detection (shader uSrcBGR)
      GPU_CAPTURE_FLIP  = 0|1                  override auto y-flip (shader uSrcFlip)

  Y-FLIP: X11 images/pixmaps are top-left origin; GL texture coords are bottom-left. So a texture
  uploaded row-0-first reads upside-down in GL -> cap.flipped=True and the screen FRAG flips uv.y.
  (GLX path reads the FBConfig's GLX_Y_INVERTED_EXT instead of hardcoding.)
"""
from __future__ import annotations
import os, time, threading, ctypes

# ---------------------------------------------------------------------------
# library loading (this box ships only versioned .so for the X libs)
# ---------------------------------------------------------------------------
def _load(*names):
    last = None
    for n in names:
        try:
            return ctypes.CDLL(n, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            last = e
    raise OSError(f"cannot load any of {names}: {last}")


libX11 = _load("libX11.so.6", "libX11.so")
# THREAD SAFETY (critical): libX11 is shared by GLFW (main thread, every poll_events) AND our SHM grab
# thread (XShmGetImage). Without XInitThreads() libX11's global state is NOT thread-safe -> the two
# threads race and SIGSEGV intermittently (this was the screen_weave crash). Must run before ANY X
# connection opens; this module is imported before glfw.init(), so module-load is the right place.
try:
    libX11.XInitThreads.restype = ctypes.c_int
    if not libX11.XInitThreads():
        print("[gpu_capture] WARNING: XInitThreads() returned 0 (libX11 not thread-safe)")
except Exception as _e:
    print("[gpu_capture] XInitThreads failed:", _e)
libXext = _load("libXext.so.6", "libXext.so")
libc = _load("libc.so.6", "libc.so")
libXcomp = None  # lazy (only the GLX path needs it)
libGL = None     # lazy (only the GLX path needs it)

GL_TEXTURE_2D = 0x0DE1
GL_LINEAR = 0x2601


# ---------------------------------------------------------------------------
# shared window-find helper (same heuristic as capture_probe / xshm_bench)
# ---------------------------------------------------------------------------
def find_browser(d):
    """Largest viewable browser window -> (xlib_window, w, h). None if not found."""
    from Xlib import X
    BROWSERS = ("firefox", "chrome", "chromium", "mozilla", "navigator", "brave", "edge")
    root = d.screen().root
    out = []
    def walk(win):
        try:
            for c in win.query_tree().children:
                try:
                    g = c.get_geometry()
                    if g.width > 300 and g.height > 200 and c.get_attributes().map_state == X.IsViewable:
                        cls = c.get_wm_class(); nm = c.get_wm_name() or ""
                        if (cls and any(b in str(cls).lower() for b in BROWSERS)) or \
                           any(b in nm.lower() for b in ("youtube", "mozilla", "chrome", "firefox")):
                            out.append((c, g.width, g.height))
                except Exception: pass
                walk(c)
        except Exception: pass
    walk(root)
    return max(out, key=lambda w: w[1] * w[2]) if out else None


def find_fullscreen(d):
    """Window id of the current _NET_WM_STATE_FULLSCREEN window (largest), or None. This is the source the
    weave should 3D — fullscreen media (a fullscreened video), not a windowed tab. The OR weave window itself
    is override-redirect and never sets this state, so it's naturally excluded; so are the desktop/guard."""
    from Xlib import X
    try:
        FS = d.intern_atom('_NET_WM_STATE_FULLSCREEN'); ST = d.intern_atom('_NET_WM_STATE')
    except Exception:
        return None
    root = d.screen().root; out = []
    def walk(w):
        try:
            for c in w.query_tree().children:
                try:
                    if c.get_attributes().map_state == X.IsViewable:
                        p = c.get_full_property(ST, 0)
                        if p and FS in p.value:
                            g = c.get_geometry()
                            if g.width > 300 and g.height > 200:
                                out.append((c.id, g.width * g.height))
                except Exception: pass
                walk(c)
        except Exception: pass
    walk(root)
    return max(out, key=lambda t: t[1])[0] if out else None


# ===========================================================================
# PRIMARY: MIT-SHM XShmGetImage capture (the path that runs on this box)
# ===========================================================================
# ctypes signatures lifted verbatim from the proven xshm_bench.py (459 fps live).
libX11.XOpenDisplay.restype = ctypes.c_void_p
libX11.XOpenDisplay.argtypes = [ctypes.c_char_p]
libX11.XCloseDisplay.argtypes = [ctypes.c_void_p]
libX11.XDefaultScreen.restype = ctypes.c_int
libX11.XDefaultScreen.argtypes = [ctypes.c_void_p]
libX11.XDefaultVisual.restype = ctypes.c_void_p
libX11.XDefaultVisual.argtypes = [ctypes.c_void_p, ctypes.c_int]
libX11.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
libX11.XGetGeometry.restype = ctypes.c_int
libX11.XGetGeometry.argtypes = [
    ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint),
    ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]

# Non-fatal X error handler: a dying source window (BadWindow/BadDrawable on XShmGetImage) must
# NOT abort the weave. Install once.
_XErrorHandler = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
libX11.XSetErrorHandler.argtypes = [_XErrorHandler]
libX11.XSetErrorHandler.restype = _XErrorHandler
_xerr = [0]
@_XErrorHandler
def _on_xerror(dpy, ev):
    _xerr[0] += 1
    return 0
libX11.XSetErrorHandler(_on_xerror)


class _ShmSeg(ctypes.Structure):
    _fields_ = [("shmseg", ctypes.c_ulong), ("shmid", ctypes.c_int),
                ("shmaddr", ctypes.c_void_p), ("readOnly", ctypes.c_int)]


libXext.XShmQueryExtension.restype = ctypes.c_int
libXext.XShmQueryExtension.argtypes = [ctypes.c_void_p]
libXext.XShmCreateImage.restype = ctypes.c_void_p
libXext.XShmCreateImage.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_int,
                                    ctypes.c_char_p, ctypes.POINTER(_ShmSeg), ctypes.c_uint, ctypes.c_uint]
libXext.XShmAttach.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ShmSeg)]
libXext.XShmAttach.restype = ctypes.c_int
libXext.XShmDetach.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ShmSeg)]
libXext.XShmGetImage.restype = ctypes.c_int
libXext.XShmGetImage.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
libX11.XDestroyImage = getattr(libX11, "XDestroyImage", None)

libc.shmget.restype = ctypes.c_int
libc.shmget.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_int]
libc.shmat.restype = ctypes.c_void_p
libc.shmat.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
libc.shmdt.argtypes = [ctypes.c_void_p]
libc.shmctl.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

_ZPixmap = 2
_IPC_CREAT = 0o1000
_IPC_RMID = 0
# XImage int-field offsets on 64-bit (probed live): data ptr @16, depth @40, bytes_per_line @44.
_OFF_DATA = 16
_OFF_BPL = 44


_GRAB_HZ = float(os.environ.get("GPU_CAPTURE_GRAB_HZ", "130"))   # grab ~display rate, not the wasted 280fps


class ShmWindowCapture:
    """XShmGetImage -> shared mem (bg thread) -> moderngl PBO upload (GL thread). The live path."""

    def __init__(self, ctx, win_id=None, display_name=None, src_x=0, src_y=0):
        import numpy as np
        self.np = np
        self.ctx = ctx
        self._src_x = int(src_x); self._src_y = int(src_y)

        # Dedicated raw X connection for the SHM grab (its own socket; independent of the GL conn).
        # display_name grabs a NESTED X server (Xephyr :N) or another display = feedback-free isolation.
        self.dpy = libX11.XOpenDisplay(display_name.encode() if isinstance(display_name, str) else None)
        if not self.dpy:
            raise RuntimeError(f"XOpenDisplay({display_name!r}) failed (DISPLAY/XAUTHORITY?)")
        if not libXext.XShmQueryExtension(self.dpy):
            raise RuntimeError("MIT-SHM extension not available on this X server")
        self.scr = libX11.XDefaultScreen(self.dpy)
        self.vis = ctypes.c_void_p(libX11.XDefaultVisual(self.dpy, self.scr))

        # read-only python-xlib connection ONLY for window-find / liveness
        from Xlib import display as xdisplay
        self._xd = xdisplay.Display(display_name)
        if win_id == "root" or (win_id is None and display_name is not None):
            win_id = self._xd.screen().root.id        # capture the (nested) display's whole root
        elif win_id is None:
            b = find_browser(self._xd)
            if not b:
                raise RuntimeError("no browser window found — open your SBS video and play it")
            win_id = b[0].id
        self.win_id = int(win_id); self.win = ctypes.c_ulong(self.win_id)
        try:                                  # remember name/class so we can re-find the source after a loss/restart
            _w = self._xd.create_resource_object('window', int(win_id))
            self._wm_name = _w.get_wm_name() or ""
            self._wm_class = _w.get_wm_class()
        except Exception:
            self._wm_name = ""; self._wm_class = None
        print(f"[capture] source name={self._wm_name!r} class={self._wm_class}", flush=True)

        w, h, depth = self._geometry()
        self.depth = depth
        # depth-32 source -> 4 components, BGRA byte order (X stores ARGB little-endian = B,G,R,A).
        # depth-24 -> still padded to 4 bytes/pixel by the server; treat as BGRX (4 comp, bgr, drop A).
        self.rgba_fmt = True                # always upload 4 components (XShm pads to 32bpp)
        self.bgr = (os.environ.get("GPU_CAPTURE_BGR", "1") == "1")   # depth-32/24 X images are BGRA
        self.flipped = (os.environ.get("GPU_CAPTURE_FLIP", "0") == "1")  # SHM data top-origin AND weave UV top-origin -> NO flip (default 0; =1 was upside down)

        self.size = (0, 0)
        self.glo = 0
        self.alive = True
        self.tex_changed = True
        self.fps = 0.0
        self._tex = None
        self._ximg = None
        self._shm = _ShmSeg()
        self._shmaddr = None
        self._bpl = 0
        self._lock = threading.Lock()
        self._pending = None         # (w, h, bytes) freshest grabbed frame
        self._run = True
        self._n = 0; self._t0 = time.time()
        self._sig = None           # content signature -> skip duplicate-frame uploads
        self._copy = None          # private reusable buffer so tobytes() is OUT of the lock
        self._upload_count = 0     # bumped in refresh() on a real upload (perf line reads this)
        self._resize_cand = None; self._resize_n = 0   # debounce: resize only after a NEW size is stable N grabs

        self._build(w, h)
        self._thread = threading.Thread(target=self._grab_loop, daemon=True); self._thread.start()

    # -- internals ----------------------------------------------------------
    def _geometry(self):
        root = ctypes.c_ulong(0); x = ctypes.c_int(0); y = ctypes.c_int(0)
        ww = ctypes.c_uint(0); hh = ctypes.c_uint(0); bw = ctypes.c_uint(0); dp = ctypes.c_uint(0)
        ok = libX11.XGetGeometry(self.dpy, self.win.value, ctypes.byref(root),
                                 ctypes.byref(x), ctypes.byref(y), ctypes.byref(ww), ctypes.byref(hh),
                                 ctypes.byref(bw), ctypes.byref(dp))
        if not ok or not ww.value or not hh.value:
            raise RuntimeError(f"XGetGeometry failed for {self.win.value:#x}")
        return ww.value, hh.value, dp.value

    def _alloc_shm(self, w, h):
        """Create an XShmImage backed by a fresh shared-memory segment sized for w*h*4."""
        # Create the XImage at depth-24/ZPixmap (server pads to 32bpp); XShmGetImage reads the
        # drawable's own format regardless of this nominal depth (proven in xshm_bench).
        ximg = libXext.XShmCreateImage(self.dpy, self.vis, 24, _ZPixmap, None,
                                       ctypes.byref(self._shm), w, h)
        if not ximg:
            raise RuntimeError("XShmCreateImage failed")
        bpl = ctypes.cast(ximg + _OFF_BPL, ctypes.POINTER(ctypes.c_int))[0]  # bytes_per_line (stride)
        if bpl < w * 4:
            bpl = w * 4
        size = bpl * h
        shmid = libc.shmget(0, size, _IPC_CREAT | 0o666)   # 0o666 (not 0o600): we run as root for lens/camera, but the X SERVER is the user's session — it must be able to attach to this segment, so it can't be root-only
        if shmid < 0:
            raise RuntimeError("shmget failed")
        addr = libc.shmat(shmid, None, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            libc.shmctl(shmid, _IPC_RMID, None)
            raise RuntimeError("shmat failed")
        self._shm.shmid = shmid; self._shm.shmaddr = addr; self._shm.readOnly = 0
        ctypes.cast(ximg + _OFF_DATA, ctypes.POINTER(ctypes.c_void_p))[0] = addr  # XImage->data = shm
        libXext.XShmAttach(self.dpy, ctypes.byref(self._shm))
        libX11.XSync(self.dpy, 0)
        # Mark the segment for deletion now: it stays alive while attached, auto-freed on detach/exit.
        libc.shmctl(shmid, _IPC_RMID, None)
        self._ximg = ximg
        self._shmaddr = addr
        self._bpl = bpl

    def _free_shm(self):
        try:
            if self._ximg:
                libXext.XShmDetach(self.dpy, ctypes.byref(self._shm)); libX11.XSync(self.dpy, 0)
        except Exception: pass
        try:
            if self._shmaddr:
                libc.shmdt(self._shmaddr)
        except Exception: pass
        # XDestroyImage frees the XImage struct (data ptr is shm, already detached -> don't double free).
        self._ximg = None; self._shmaddr = None

    def _build(self, w, h):
        """(Re)allocate the shm image + the moderngl texture for size (w,h). [GL-CONTEXT: texture()]
        LOCKED: the grab thread must not be reading the shm while we free/realloc it, else use-after-free
        -> SIGSEGV (this was the resize crash). _build is never called while already holding _lock."""
        if (w, h) == self.size and self._tex is not None:
            return                                # no-op when size unchanged (don't realloc shm+texture for a no-change)
        with self._lock:
            self._free_shm()
            self._alloc_shm(w, h)
            if self._tex is not None:
                try: self._tex.release()
                except Exception: pass
            # 4-component f1 texture; the screen FRAG samples .rgb and swizzles via uSrcBGR.
            self._tex = self.ctx.texture((w, h), 4, dtype="f1")        # [NEEDS LIVE GL CONTEXT]
            self._tex.repeat_x = False; self._tex.repeat_y = False
            self._tex.filter = (moderngl_LINEAR(), moderngl_LINEAR())
            self.size = (w, h)
            self.glo = int(self._tex.glo)
            self.tex_changed = True

    def _grab_loop(self):
        """Background: XShmGetImage into shm at ~_GRAB_HZ; publish ONLY when content changed (skips
        duplicate-frame PBO uploads). tobytes() is done OUT of the lock so the lock-hold stays tiny;
        the SHM grab+copy stay UNDER the lock (else _build's realloc -> use-after-free SIGSEGV)."""
        np = self.np
        period = 1.0 / max(_GRAB_HZ, 1.0)
        next_t = time.perf_counter()
        while self._run:
            now = time.perf_counter()
            if now < next_t:
                time.sleep(min(period, next_t - now)); continue
            next_t += period
            if now - next_t > 0.25:           # fell badly behind (stall) -> resync, don't spiral
                next_t = now + period
            try:
                w, h, _ = self._geometry()
                if (w, h) != self.size:
                    if (w, h) == self._resize_cand: self._resize_n += 1
                    else: self._resize_cand = (w, h); self._resize_n = 1
                    if self._resize_n >= 3:        # debounce: resize only after a new size is stable 3 grabs (ignore 1px flaps)
                        with self._lock:
                            self._pending = ("RESIZE", w, h)
                        self._resize_cand = None; self._resize_n = 0
                    time.sleep(0.002); continue
                self._resize_cand = None; self._resize_n = 0   # at current size -> any transient flap resolved
                got = None
                with self._lock:              # lock ONLY across the SHM grab + copy, NOT the 4.5ms tobytes
                    if (w, h) == self.size:
                        bpl = self._bpl
                        r = libXext.XShmGetImage(self.dpy, self.win, self._ximg, self._src_x, self._src_y, 0xffffffff)
                        libX11.XSync(self.dpy, 0)
                        if r:
                            buf = (ctypes.c_char * (bpl * h)).from_address(self._shmaddr)
                            arr = np.frombuffer(buf, np.uint8).reshape(h, bpl)[:, :w * 4]
                            if self._copy is None or self._copy.shape != arr.shape:
                                self._copy = np.empty_like(arr)
                            self._copy[:] = arr            # one copy out of the volatile shm
                            got = self._copy
                if got is None:
                    try: self._geometry()                       # alive but grab failed transiently -> quick retry
                    except Exception:
                        self.alive = False; time.sleep(0.3)     # source gone -> mark lost, slow retry (re-acquire if it returns)
                    time.sleep(0.005); continue
                if not self.alive:                              # got a frame after a loss -> re-acquired
                    self.alive = True; print("[capture] source re-acquired", flush=True)
                # change-detect (~0.03ms): only publish new content -> refresh() skips dup uploads.
                # force a publish every 30 grabs (guards a slow pan that misses the sparse stride samples).
                sig = int(got.reshape(-1)[::4099].sum())
                if sig != self._sig or (self._n % 30 == 0):
                    self._sig = sig
                    data = got.tobytes()      # 4.5ms but ONLY on real new content, OUT of the lock
                    with self._lock:
                        self._pending = data
                self._n += 1
                t = time.time()
                if t - self._t0 >= 2.0:
                    self.fps = self._n / (t - self._t0); self._t0 = t; self._n = 0
            except Exception:
                try: self._geometry()
                except Exception:
                    self.alive = False; time.sleep(0.3)        # don't kill the grab thread on a dead window -> retry/re-acquire
                time.sleep(0.005)

    # -- public (contract) --------------------------------------------------
    def refresh(self):
        """GL thread, per frame: upload the freshest grabbed frame into the texture (PBO). [GL CTX]"""
        self.tex_changed = False
        with self._lock:
            p = self._pending; self._pending = None
        if p is None:
            return self.alive and self.glo != 0
        if isinstance(p, tuple) and p and p[0] == "RESIZE":      # background saw a resize
            _, w, h = p
            try:
                self._build(w, h)                                # [NEEDS LIVE GL CONTEXT]
            except Exception:
                self.alive = False
            return self.alive and self.glo != 0
        try:
            self._tex.write(p); self._upload_count += 1          # [GL CTX] ~5ms — but only when content changed
        except Exception:
            return self.alive and self.glo != 0
        return True

    def resize_if_needed(self):
        try:
            w, h, _ = self._geometry()
        except Exception:
            self.alive = False; return False
        if (w, h) == self.size:
            return False
        try:
            self._build(w, h); return True
        except Exception:
            self.alive = False; return False

    @property
    def texture(self):
        return self._tex

    def use(self, unit):
        self._tex.use(unit)

    def close(self):
        self._run = False
        try:                                  # JOIN the grab thread BEFORE freeing its shm / X conn (else use-after-free -> SIGSEGV on exit)
            t = getattr(self, "_thread", None)
            if t is not None: t.join(timeout=1.5)
        except Exception: pass
        try: self._free_shm()
        except Exception: pass
        try:
            if self._tex is not None: self._tex.release()
        except Exception: pass
        self._tex = None; self.glo = 0
        try:
            if self.dpy: libX11.XCloseDisplay(self.dpy)
        except Exception: pass
        self.dpy = None
        try: self._xd.close()
        except Exception: pass
    release = close

    def __del__(self):
        try: self.close()
        except Exception: pass


def moderngl_LINEAR():
    import moderngl
    return moderngl.LINEAR


# ===========================================================================
# PORTABLE zero-copy: GLX_EXT_texture_from_pixmap (works on Mesa; dead on NVIDIA+GNOME here)
# ===========================================================================
CompositeRedirectAutomatic = 0
GLX_DRAWABLE_TYPE = 0x8010
GLX_PIXMAP_BIT = 0x00000002
GLX_BIND_TO_TEXTURE_RGB_EXT = 0x20D0
GLX_BIND_TO_TEXTURE_RGBA_EXT = 0x20D1
GLX_BIND_TO_TEXTURE_TARGETS_EXT = 0x20D3
GLX_TEXTURE_2D_BIT_EXT = 0x00000002
GLX_Y_INVERTED_EXT = 0x20D4
GLX_TEXTURE_FORMAT_EXT = 0x20D5
GLX_TEXTURE_TARGET_EXT = 0x20D6
GLX_TEXTURE_FORMAT_RGB_EXT = 0x20D8
GLX_TEXTURE_FORMAT_RGBA_EXT = 0x20D9
GLX_TEXTURE_2D_EXT = 0x20DC
GLX_FRONT_LEFT_EXT = 0x20DE
GLX_BUFFER_SIZE = 2
GLX_ALPHA_SIZE = 11
GLX_RED_SIZE = 8
GLX_GREEN_SIZE = 9
GLX_BLUE_SIZE = 10
GLX_DOUBLEBUFFER = 5
GL_TEXTURE_MIN_FILTER = 0x2801
GL_TEXTURE_MAG_FILTER = 0x2800
GL_TEXTURE_WRAP_S = 0x2802
GL_TEXTURE_WRAP_T = 0x2803
GL_CLAMP_TO_EDGE = 0x812F
X_None = 0
Display_p = ctypes.c_void_p
GLXFBConfig = ctypes.c_void_p
GLXPixmap = ctypes.c_ulong
Pixmap = ctypes.c_ulong
Window = ctypes.c_ulong


def _lazy_glx():
    """Load libGL + libXcomposite and bind GLX/Composite signatures once (GLX path only)."""
    global libGL, libXcomp
    if libGL is not None and libXcomp is not None:
        return
    libGL = _load("libGL.so.1", "libGL.so")
    libXcomp = _load("libXcomposite.so.1", "libXcomposite.so")
    libX11.XFree.argtypes = [ctypes.c_void_p]; libX11.XFree.restype = ctypes.c_int
    libX11.XFreePixmap.argtypes = [Display_p, Pixmap]; libX11.XFreePixmap.restype = ctypes.c_int
    libXcomp.XCompositeQueryExtension.argtypes = [Display_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    libXcomp.XCompositeQueryExtension.restype = ctypes.c_int
    libXcomp.XCompositeRedirectWindow.argtypes = [Display_p, Window, ctypes.c_int]; libXcomp.XCompositeRedirectWindow.restype = None
    libXcomp.XCompositeUnredirectWindow.argtypes = [Display_p, Window, ctypes.c_int]; libXcomp.XCompositeUnredirectWindow.restype = None
    libXcomp.XCompositeNameWindowPixmap.argtypes = [Display_p, Window]; libXcomp.XCompositeNameWindowPixmap.restype = Pixmap
    libGL.glXChooseFBConfig.argtypes = [Display_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    libGL.glXChooseFBConfig.restype = ctypes.POINTER(GLXFBConfig)
    libGL.glXGetFBConfigAttrib.argtypes = [Display_p, GLXFBConfig, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    libGL.glXGetFBConfigAttrib.restype = ctypes.c_int
    libGL.glXCreatePixmap.argtypes = [Display_p, GLXFBConfig, Pixmap, ctypes.POINTER(ctypes.c_int)]
    libGL.glXCreatePixmap.restype = GLXPixmap
    libGL.glXDestroyPixmap.argtypes = [Display_p, GLXPixmap]; libGL.glXDestroyPixmap.restype = None
    libGL.glXGetProcAddressARB.argtypes = [ctypes.c_char_p]; libGL.glXGetProcAddressARB.restype = ctypes.c_void_p
    libGL.glGenTextures.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
    libGL.glDeleteTextures.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
    libGL.glBindTexture.argtypes = [ctypes.c_uint, ctypes.c_uint]
    libGL.glTexParameteri.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_int]
    libGL.glGetError.argtypes = []; libGL.glGetError.restype = ctypes.c_uint
    libX11.XDefaultScreen.argtypes = [Display_p]; libX11.XDefaultScreen.restype = ctypes.c_int


_BindProto = ctypes.CFUNCTYPE(None, Display_p, GLXPixmap, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
_ReleaseProto = ctypes.CFUNCTYPE(None, Display_p, GLXPixmap, ctypes.c_int)
_glXBindTexImageEXT = None
_glXReleaseTexImageEXT = None


def _glx_procs():
    global _glXBindTexImageEXT, _glXReleaseTexImageEXT
    if _glXBindTexImageEXT:
        return
    def proc(name, proto):
        addr = libGL.glXGetProcAddressARB(name.encode())
        if not addr:
            raise OSError(f"glXGetProcAddressARB({name}) NULL — GLX_EXT_texture_from_pixmap missing")
        return proto(addr)
    _glXBindTexImageEXT = proc("glXBindTexImageEXT", _BindProto)
    _glXReleaseTexImageEXT = proc("glXReleaseTexImageEXT", _ReleaseProto)


def _attr(dpy, cfg, a):
    v = ctypes.c_int(0); libGL.glXGetFBConfigAttrib(dpy, cfg, a, ctypes.byref(v)); return v.value


def _resolve_glx_display(display=None):
    if display is not None:
        return ctypes.c_void_p(int(display))
    import glfw
    if not glfw.get_current_context():
        raise RuntimeError("no current GLFW context; pass display=glfw.get_x11_display()")
    return ctypes.c_void_p(int(glfw.get_x11_display()))


def _choose_fbconfig(dpy, want_rgba):
    screen = libX11.XDefaultScreen(dpy)
    def attrs(rgba):
        a = [GLX_DRAWABLE_TYPE, GLX_PIXMAP_BIT, GLX_BIND_TO_TEXTURE_TARGETS_EXT, GLX_TEXTURE_2D_BIT_EXT,
             GLX_DOUBLEBUFFER, 0, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8]
        a += ([GLX_BIND_TO_TEXTURE_RGBA_EXT, 1, GLX_ALPHA_SIZE, 8, GLX_BUFFER_SIZE, 32] if rgba
              else [GLX_BIND_TO_TEXTURE_RGB_EXT, 1, GLX_BUFFER_SIZE, 24])
        a += [X_None]
        return (ctypes.c_int * len(a))(*a)
    for rgba in (bool(want_rgba), not bool(want_rgba)):
        n = ctypes.c_int(0)
        cfgs = libGL.glXChooseFBConfig(dpy, screen, attrs(rgba), ctypes.byref(n))
        if cfgs and n.value > 0:
            chosen = None
            for i in range(n.value):
                c = cfgs[i]
                if (_attr(dpy, c, GLX_DRAWABLE_TYPE) & GLX_PIXMAP_BIT) and \
                   (_attr(dpy, c, GLX_BIND_TO_TEXTURE_TARGETS_EXT) & GLX_TEXTURE_2D_BIT_EXT):
                    chosen = c; break
            if chosen is None: chosen = cfgs[0]
            y_inv = bool(_attr(dpy, chosen, GLX_Y_INVERTED_EXT))
            libX11.XFree(cfgs)
            return chosen, rgba, y_inv
        if cfgs: libX11.XFree(cfgs)
    raise RuntimeError("no texture-from-pixmap FBConfig")


class GLXWindowCapture:
    """GLX_EXT_texture_from_pixmap zero-copy. Correct on Mesa; self-detects the NVIDIA no-op here."""

    def __init__(self, ctx, win_id=None, display=None):
        _lazy_glx(); _glx_procs()
        self.ctx = ctx
        self.dpy = _resolve_glx_display(display)
        from Xlib import display as xdisplay
        self._xd = xdisplay.Display()
        if win_id is None:
            b = find_browser(self._xd)
            if not b: raise RuntimeError("no browser window found")
            win_id = b[0].id
        self.win = Window(int(win_id))

        ev, er = ctypes.c_int(0), ctypes.c_int(0)
        if not libXcomp.XCompositeQueryExtension(self.dpy, ctypes.byref(ev), ctypes.byref(er)):
            raise RuntimeError("XComposite not available")
        libXcomp.XCompositeRedirectWindow(self.dpy, self.win, CompositeRedirectAutomatic)
        libX11.XSync(self.dpy, 0); self._redirected = True

        w, h, depth = self._geometry()
        self.depth = depth
        self.fbconfig, self.rgba_fmt, y_inv = _choose_fbconfig(self.dpy, depth >= 32)
        self.flipped = not y_inv                 # TFP top-origin vs GL bottom (NVIDIA: True)
        self.bgr = False                         # GLX delivers in FBConfig channel order (RGBA)
        self.size = (0, 0); self.glo = 0
        self._glo = ctypes.c_uint(0); self._glxpixmap = GLXPixmap(0); self._pixmap = Pixmap(0)
        self._bound = False; self.alive = True; self.tex_changed = True; self.fps = 0.0
        self._n = 0; self._t0 = time.time(); self._wrap = None
        self._build(w, h)

        # NVIDIA+GNOME: glXCreatePixmap raises BadMatch on the compositor-owned window (proven), and
        # even when it returns a handle the bound RGB is all-black. Probe; raise so factory falls back.
        if os.environ.get("GPU_CAPTURE_SKIP_PROBE", "0") != "1":
            if _xerr[0] or not self._probe_has_content():
                raise RuntimeError("GLX texture_from_pixmap unusable on this window "
                                   "(BadMatch/black — compositor-owned; known NVIDIA+GNOME limit)")

    def _geometry(self):
        root = ctypes.c_ulong(0); x = ctypes.c_int(0); y = ctypes.c_int(0)
        w = ctypes.c_uint(0); h = ctypes.c_uint(0); bw = ctypes.c_uint(0); dp = ctypes.c_uint(0)
        ok = libX11.XGetGeometry(self.dpy, self.win.value, ctypes.byref(root), ctypes.byref(x),
                                 ctypes.byref(y), ctypes.byref(w), ctypes.byref(h), ctypes.byref(bw), ctypes.byref(dp))
        if not ok or not w.value or not h.value:
            raise RuntimeError(f"XGetGeometry failed for {self.win.value:#x}")
        return w.value, h.value, dp.value

    def _pixmap_attribs(self):
        fmt = GLX_TEXTURE_FORMAT_RGBA_EXT if self.rgba_fmt else GLX_TEXTURE_FORMAT_RGB_EXT
        a = [GLX_TEXTURE_TARGET_EXT, GLX_TEXTURE_2D_EXT, GLX_TEXTURE_FORMAT_EXT, fmt, X_None]
        return (ctypes.c_int * len(a))(*a)

    def _build(self, w, h):
        self._pixmap = Pixmap(libXcomp.XCompositeNameWindowPixmap(self.dpy, self.win))
        libX11.XSync(self.dpy, 0)
        if not self._pixmap.value:
            raise RuntimeError("XCompositeNameWindowPixmap returned None")
        self._glxpixmap = GLXPixmap(libGL.glXCreatePixmap(self.dpy, self.fbconfig, self._pixmap, self._pixmap_attribs()))
        libX11.XSync(self.dpy, 0)
        if not self._glxpixmap.value or _xerr[0]:
            raise RuntimeError("glXCreatePixmap failed (BadMatch on compositor-owned window)")
        libGL.glGenTextures(1, ctypes.byref(self._glo))
        libGL.glBindTexture(GL_TEXTURE_2D, self._glo.value)
        for p, v in ((GL_TEXTURE_MIN_FILTER, GL_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
                     (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)):
            libGL.glTexParameteri(GL_TEXTURE_2D, p, v)
        _glXBindTexImageEXT(self.dpy, self._glxpixmap, GLX_FRONT_LEFT_EXT, None)
        self._bound = True; self.size = (w, h); self.glo = int(self._glo.value); self.tex_changed = True

    def _probe_has_content(self):
        try:
            import numpy as _np, moderngl as _mgl
            dst = self.ctx.texture((4, 4), 4, dtype="f1")
            fbo = self.ctx.framebuffer(color_attachments=[dst])
            prog = self.ctx.program(
                vertex_shader="#version 330\nin vec2 p;out vec2 uv;void main(){uv=p*0.5+0.5;gl_Position=vec4(p,0,1);}",
                fragment_shader="#version 330\nuniform sampler2D s;in vec2 uv;out vec4 o;void main(){o=texture(s,uv);}")
            quad = self.ctx.buffer(_np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes())
            vao = self.ctx.vertex_array(prog, [(quad, "2f", "p")])
            self.texture.use(0); prog["s"] = 0; fbo.use(); self.ctx.clear(0, 0, 0)
            vao.render(_mgl.TRIANGLE_STRIP); self.ctx.finish()
            px = _np.frombuffer(fbo.read(components=4), _np.uint8).reshape(-1, 4)
            rgb_max = int(px[:, :3].max())
            for o in (dst, fbo, quad, vao, prog):
                try: o.release()
                except Exception: pass
            return rgb_max > 0
        except Exception:
            return True

    def refresh(self):
        self.tex_changed = False
        if not self.alive: return self.glo != 0
        try:
            w, h, _ = self._geometry()
        except Exception:
            self.alive = False; return self.glo != 0
        if (w, h) != self.size:
            try: self._rebuild(w, h)
            except Exception: self.alive = False; return self.glo != 0
        try:
            if self._bound: _glXReleaseTexImageEXT(self.dpy, self._glxpixmap, GLX_FRONT_LEFT_EXT)
            libGL.glBindTexture(GL_TEXTURE_2D, self._glo.value)
            _glXBindTexImageEXT(self.dpy, self._glxpixmap, GLX_FRONT_LEFT_EXT, None)
            self._bound = True
        except Exception:
            return self.glo != 0
        self._n += 1; dt = time.time() - self._t0
        if dt >= 2.0: self.fps = self._n / dt; self._t0 = time.time(); self._n = 0
        return True

    def resize_if_needed(self):
        try:
            w, h, _ = self._geometry()
        except Exception:
            self.alive = False; return False
        if (w, h) == self.size: return False
        try: self._rebuild(w, h); return True
        except Exception: self.alive = False; return False

    def _rebuild(self, w, h):
        old = (self._glo, self._glxpixmap, self._pixmap, self._bound)
        self._glo = ctypes.c_uint(0); self._glxpixmap = GLXPixmap(0); self._pixmap = Pixmap(0); self._bound = False
        self._build(w, h)
        og, ogx, op, ob = old
        try:
            if ob and ogx.value: _glXReleaseTexImageEXT(self.dpy, ogx, GLX_FRONT_LEFT_EXT)
        except Exception: pass
        for fn, arg in ((libGL.glXDestroyPixmap, ogx), (libX11.XFreePixmap, op)):
            try:
                if arg.value: fn(self.dpy, arg)
            except Exception: pass
        try:
            if og.value: libGL.glDeleteTextures(1, ctypes.byref(og))
        except Exception: pass

    @property
    def texture(self):
        if self._wrap is None or self._wrap_glo != self.glo or self._wrap_size != self.size:
            if self._wrap is not None:
                try: self._wrap.release()
                except Exception: pass
            self._wrap = self.ctx.external_texture(self.glo, self.size, 4 if self.rgba_fmt else 3, 0, "f1")
            self._wrap_glo = self.glo; self._wrap_size = self.size
        return self._wrap

    def use(self, unit):
        try: self.texture.use(unit)
        except Exception: libGL.glBindTexture(GL_TEXTURE_2D, self._glo.value)

    def close(self):
        try:
            if self._bound and self._glxpixmap.value:
                _glXReleaseTexImageEXT(self.dpy, self._glxpixmap, GLX_FRONT_LEFT_EXT)
        except Exception: pass
        self._bound = False
        if self._wrap is not None:
            try: self._wrap.release()
            except Exception: pass
            self._wrap = None
        if self._glxpixmap.value:
            try: libGL.glXDestroyPixmap(self.dpy, self._glxpixmap)
            except Exception: pass
            self._glxpixmap = GLXPixmap(0)
        if self._pixmap.value:
            try: libX11.XFreePixmap(self.dpy, self._pixmap)
            except Exception: pass
            self._pixmap = Pixmap(0)
        if self._glo.value:
            try: libGL.glDeleteTextures(1, ctypes.byref(self._glo))
            except Exception: pass
            self._glo = ctypes.c_uint(0)
        self.glo = 0
        if getattr(self, "_redirected", False):
            try:
                libXcomp.XCompositeUnredirectWindow(self.dpy, self.win, CompositeRedirectAutomatic)
                libX11.XSync(self.dpy, 0)
            except Exception: pass
            self._redirected = False
        try: self._xd.close()
        except Exception: pass
    release = close

    def __del__(self):
        try: self.close()
        except Exception: pass


# ===========================================================================
# LAST RESORT: pure python-xlib get_image (~1fps@4K). Only if SHM is unavailable.
# ===========================================================================
class CpuWindowCapture:
    def __init__(self, ctx, win_id=None):
        import numpy as np
        from Xlib import display as xdisplay
        self.np = np; self.ctx = ctx; self.d = xdisplay.Display()
        if win_id is not None:
            self.win = self.d.create_resource_object('window', win_id)
        else:
            b = find_browser(self.d)
            if not b: raise RuntimeError("no browser window found")
            self.win = b[0]
        g = self.win.get_geometry()
        self.size = (int(g.width), int(g.height))
        self.flipped = False; self.bgr = False; self.rgba_fmt = False
        self.alive = True; self.fps = 0.0; self.tex_changed = True
        import moderngl
        self._tex = ctx.texture(self.size, 3, dtype='f1')
        self._tex.repeat_x = False; self._tex.repeat_y = False; self._tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.glo = int(self._tex.glo)
        self._lock = threading.Lock(); self._pending = None; self._run = True
        self._n = 0; self._t0 = time.time(); self._upload_count = 0
        threading.Thread(target=self._grab_loop, daemon=True).start()

    def _grab_loop(self):
        from Xlib import X
        np = self.np
        while self._run:
            try:
                g = self.win.get_geometry(); w, h = int(g.width), int(g.height)
                raw = self.win.get_image(0, 0, w, h, X.ZPixmap, 0xffffffff)
                img = np.frombuffer(raw.data, np.uint8).reshape(h, w, 4)[:, :, :3]
                img = np.ascontiguousarray(img[:, :, ::-1])
                with self._lock: self._pending = (w, h, img.tobytes())
                self._n += 1
                if time.time() - self._t0 >= 2.0:
                    self.fps = self._n / (time.time() - self._t0); self._t0 = time.time(); self._n = 0
            except Exception:
                try: self.win.get_geometry()
                except Exception: self.alive = False
                time.sleep(0.05)
            time.sleep(0.003)

    def refresh(self):
        self.tex_changed = False
        with self._lock: p = self._pending; self._pending = None
        if p is None: return self.alive and self.glo != 0
        import moderngl
        w, h, data = p
        if (w, h) != self.size:
            self.size = (w, h)
            try: self._tex.release()
            except Exception: pass
            self._tex = self.ctx.texture((w, h), 3, dtype='f1')
            self._tex.repeat_x = False; self._tex.repeat_y = False; self._tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.glo = int(self._tex.glo); self.tex_changed = True
        self._tex.write(data); self._upload_count += 1; return True

    def resize_if_needed(self): return False
    @property
    def texture(self): return self._tex
    def use(self, unit): self._tex.use(unit)
    def close(self):
        self._run = False
        try: self._tex.release()
        except Exception: pass
        try: self.d.close()
        except Exception: pass
    release = close


# Ergonomic alias the brief requested (ctor takes ctx, window_id). Resolves to the live path.
def GpuWindowCapture(ctx, window_id, display=None):
    return make_capture(ctx, win_id=window_id, display=display)


class SharedMemSBSCapture:
    """Reads SBS stereo frames published by twod3d_source.py (Proc A, the 2D->3D depth source) from a named
    POSIX shm (/dev/shm/<name>) and uploads them to a GL texture (cap.glo). Matches the ShmWindowCapture
    interface so screen_weave weaves the synthesized stereo unchanged. shm = 6 int64 header
    [magic, seq, slot, W, H, alive] + two W*H*3 BGR slots; refresh() dup-skips on unchanged seq."""
    MAGIC = 0x5342

    def __init__(self, ctx, shm_name, w=None, h=None):
        import numpy as np, mmap
        self.np = np; self.ctx = ctx; self._name = shm_name
        path = "/dev/shm/" + shm_name
        for _ in range(200):                                  # wait for Proc A to create + size it
            try:
                if os.path.exists(path) and os.path.getsize(path) >= 48:
                    break
            except Exception:
                pass
            time.sleep(0.05)
        self._fd = os.open(path, os.O_RDONLY)
        self._mm = mmap.mmap(self._fd, os.fstat(self._fd).st_size, prot=mmap.PROT_READ)
        self._hdr = np.frombuffer(self._mm, dtype=np.int64, count=6, offset=0)
        if int(self._hdr[0]) != self.MAGIC:
            raise RuntimeError(f"SBS shm {shm_name}: bad magic {int(self._hdr[0]):#x}")
        W = int(w or self._hdr[3]); H = int(h or self._hdr[4])
        self._W, self._H, self._slot_bytes = W, H, W * H * 3
        self.size = (W, H); self.bgr = True; self.rgba_fmt = False
        self.flipped = (os.environ.get("GPU_CAPTURE_FLIP", "0") == "1")
        self.glo = 0; self.alive = True; self.tex_changed = True; self.fps = 0.0
        self._tex = None; self._upload_count = 0; self._last_seq = -1
        self._n = 0; self._t0 = time.time()
        self._build(W, H)
        print(f"[capture] SBS shm '{shm_name}' {W}x{H} (2D->3D source via Proc A)", flush=True)

    def _build(self, w, h):
        if self._tex is not None:
            try: self._tex.release()
            except Exception: pass
        self._tex = self.ctx.texture((w, h), 3, dtype="f1")   # 3-comp BGR; FRAG samples .rgb, swizzles via uSrcBGR
        self._tex.repeat_x = False; self._tex.repeat_y = False
        self._tex.filter = (moderngl_LINEAR(), moderngl_LINEAR())
        self.size = (w, h); self.glo = int(self._tex.glo); self.tex_changed = True

    def refresh(self):
        self.tex_changed = False                          # one-shot: SBS glo is stable -> don't re-wrap every frame
        if not self.alive:
            return self.glo != 0
        try:
            if int(self._hdr[5]) == 0:                        # Proc A signalled done
                self.alive = False; return self.glo != 0
            seq = int(self._hdr[1])
            if seq == self._last_seq:                         # no new frame -> skip upload
                return self.glo != 0
            self._last_seq = seq
            off = 48 + (int(self._hdr[2]) & 1) * self._slot_bytes
            self._tex.write(self._mm[off:off + self._slot_bytes])
            self._upload_count += 1; self._n += 1
            dt = time.time() - self._t0
            if dt >= 1.0:
                self.fps = self._n / dt; self._n = 0; self._t0 = time.time()
        except Exception as e:
            print("[capture] SBS refresh error:", e, flush=True); self.alive = False
        return self.glo != 0

    def use(self, unit=0):
        if self._tex is not None:
            self._tex.use(unit)

    def close(self):
        try:
            if self._tex is not None: self._tex.release()
        except Exception: pass
        self._tex = None; self.glo = 0
        try: self._mm.close()
        except Exception: pass
        try: os.close(self._fd)
        except Exception: pass
    release = close

    def __del__(self):
        try: self.close()
        except Exception: pass


# ---- DIBR shader (the GLSL backward-warp validated headless in gl_dibr_test.py vs cv2, 0.15 mean-diff) ----
_DIBR_VERT = """#version 330
in vec2 pos; out vec2 uv;
void main(){ uv = pos*0.5+0.5; gl_Position = vec4(pos,0.0,1.0); }"""
_DIBR_FRAG = """#version 330
uniform sampler2D srcTex; uniform sampler2D depthTex;
uniform float maxdisp; uniform float conv; uniform float w;
in vec2 uv; out vec4 f;
void main(){
    bool isL = uv.x < 0.5;
    vec2 euv = vec2(isL ? uv.x*2.0 : (uv.x-0.5)*2.0, uv.y);
    float d = texture(depthTex, euv).r;
    float disp = (d - conv) * maxdisp / w;          // pixels -> uv
    vec2 suv = vec2(euv.x + (isL ? 0.5 : -0.5)*disp, euv.y);
    f = texture(srcTex, clamp(suv, vec2(0.0), vec2(1.0)));
}"""


class SharedMemRGBDCapture:
    """GPU-DIBR path (gated GPU_DIBR=1, off by default). Reads RGB + 1ch depth planes published by twod3d_source.py
    (Proc A skips cv2.remap) from a named POSIX shm, uploads both to GL, and runs the validated DIBR shader as an
    internal FBO pre-pass -> an SBS texture. Presents the SAME interface as SharedMemSBSCapture (size=(2W,H), .glo,
    .bgr, .refresh(), .use()), so screen_weave weaves it UNCHANGED. shm = 6 int64 header [magic, seq, slot, W, H,
    alive] + two slots of (W*H*3 BGR  then  W*H*4 float32 depth). Moves the L/R warp off Proc A's CPU onto the GPU."""
    MAGIC = 0x5244        # 'RD'

    def __init__(self, ctx, shm_name, w=None, h=None, maxdisp=None, conv=None):
        import numpy as np, mmap, moderngl
        self.np = np; self.ctx = ctx; self._name = shm_name
        self._maxdisp = float(maxdisp if maxdisp is not None else os.environ.get("WEAVE_MAXDISP", "35"))
        self._conv = float(conv if conv is not None else os.environ.get("WEAVE_CONV", "0.5"))
        path = "/dev/shm/" + shm_name
        for _ in range(200):                                  # wait for Proc A to create + size it
            try:
                if os.path.exists(path) and os.path.getsize(path) >= 48:
                    break
            except Exception:
                pass
            time.sleep(0.05)
        self._fd = os.open(path, os.O_RDONLY)
        self._mm = mmap.mmap(self._fd, os.fstat(self._fd).st_size, prot=mmap.PROT_READ)
        self._hdr = np.frombuffer(self._mm, dtype=np.int64, count=6, offset=0)
        if int(self._hdr[0]) != self.MAGIC:
            raise RuntimeError(f"RGBD shm {shm_name}: bad magic {int(self._hdr[0]):#x} (expected {self.MAGIC:#x})")
        W = int(w or self._hdr[3]); H = int(h or self._hdr[4])
        self._W, self._H = W, H
        self._rgb_bytes = W * H * 3; self._depth_bytes = W * H * 4
        self._slot_bytes = self._rgb_bytes + self._depth_bytes
        self.size = (2 * W, H); self.bgr = True; self.rgba_fmt = False
        # Headless parity test (test_gpu_dibr.py) shows the DIBR-FBO SBS texture is byte-identical in orientation to the
        # SBS direct-write path, so the default flip MATCHES the SBS path (False). Override with GPU_CAPTURE_RGBD_FLIP.
        self.flipped = (os.environ.get("GPU_CAPTURE_RGBD_FLIP", "0") == "1")
        self.glo = 0; self.alive = True; self.tex_changed = True; self.fps = 0.0
        self._rgb_tex = self._depth_tex = self._sbs_tex = self._fbo = self._vao = None
        self._upload_count = 0; self._last_seq = -1; self._n = 0; self._t0 = time.time()
        self._prog = ctx.program(vertex_shader=_DIBR_VERT, fragment_shader=_DIBR_FRAG)
        self._prog["srcTex"] = 0; self._prog["depthTex"] = 1
        self._prog["maxdisp"].value = self._maxdisp; self._prog["conv"].value = self._conv
        self._prog["w"].value = float(W)
        self._quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self._build(W, H)
        print(f"[capture] RGBD shm '{shm_name}' {W}x{H} -> GPU-DIBR SBS {2*W}x{H} "
              f"(maxdisp={self._maxdisp} conv={self._conv} flip={self.flipped})", flush=True)

    def _build(self, w, h):
        import moderngl
        for t in (self._rgb_tex, self._depth_tex, self._sbs_tex, self._fbo, self._vao):
            try:
                if t is not None: t.release()
            except Exception: pass
        lin = moderngl_LINEAR()
        self._rgb_tex = self.ctx.texture((w, h), 3, dtype="f1"); self._rgb_tex.filter = (lin, lin)
        self._rgb_tex.repeat_x = False; self._rgb_tex.repeat_y = False
        self._depth_tex = self.ctx.texture((w, h), 1, dtype="f4"); self._depth_tex.filter = (lin, lin)
        self._depth_tex.repeat_x = False; self._depth_tex.repeat_y = False
        self._sbs_tex = self.ctx.texture((2 * w, h), 3, dtype="f1"); self._sbs_tex.filter = (lin, lin)
        self._sbs_tex.repeat_x = False; self._sbs_tex.repeat_y = False
        self._fbo = self.ctx.framebuffer(color_attachments=[self._sbs_tex])
        self._vao = self.ctx.vertex_array(self._prog, [(self._quad, "2f", "pos")])
        self._W, self._H = w, h; self.size = (2 * w, h)
        self.glo = int(self._sbs_tex.glo); self.tex_changed = True

    def _dibr(self):
        import moderngl
        self._rgb_tex.use(0); self._depth_tex.use(1)
        self._fbo.use()                                   # screen_weave re-binds ctx.screen before its weave render
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def refresh(self):
        self.tex_changed = False                          # one-shot: SBS-out glo is stable -> don't re-wrap every frame
        if not self.alive:
            return self.glo != 0
        try:
            if int(self._hdr[5]) == 0:                        # Proc A signalled done
                self.alive = False; return self.glo != 0
            seq = int(self._hdr[1])
            if seq == self._last_seq:                         # no new frame -> skip upload+DIBR
                return self.glo != 0
            self._last_seq = seq
            off = 48 + (int(self._hdr[2]) & 1) * self._slot_bytes
            self._rgb_tex.write(self._mm[off:off + self._rgb_bytes])
            self._depth_tex.write(self._mm[off + self._rgb_bytes:off + self._slot_bytes])
            self._dibr()                                      # GPU backward-warp -> SBS texture (self.glo)
            self._upload_count += 1; self._n += 1
            dt = time.time() - self._t0
            if dt >= 1.0:
                self.fps = self._n / dt; self._n = 0; self._t0 = time.time()
        except Exception as e:
            print("[capture] RGBD refresh error:", e, flush=True); self.alive = False
        return self.glo != 0

    def use(self, unit=0):
        if self._sbs_tex is not None:
            self._sbs_tex.use(unit)

    def close(self):
        for t in (self._rgb_tex, self._depth_tex, self._sbs_tex, self._fbo, self._vao, self._quad, self._prog):
            try:
                if t is not None: t.release()
            except Exception: pass
        self._rgb_tex = self._depth_tex = self._sbs_tex = self._fbo = self._vao = None
        self.glo = 0
        try: self._mm.close()
        except Exception: pass
        try: os.close(self._fd)
        except Exception: pass
    release = close

    def __del__(self):
        try: self.close()
        except Exception: pass


def _pulse_serving(timeout=2.0):
    """True only if the audio server actually RESPONDS (does a round-trip). A bare socket connect succeeds even when
    the daemon is wedged (-> mpv 'Server proto: 4294967295' / 'Init failed: Timeout'), so probe with a real client
    tool: pactl (PulseAudio) or wpctl/pw-cli (PipeWire) -- whichever exists. A hung daemon makes the tool time out ->
    we return False and play silent instead of freezing. Inherits XDG_RUNTIME_DIR/PULSE_SERVER from the launch env."""
    import subprocess, shutil
    for cmd in (["pactl", "info"], ["wpctl", "status"], ["pw-cli", "info", "0"]):
        if not shutil.which(cmd[0]):
            continue
        try:
            return subprocess.run(cmd, capture_output=True, timeout=timeout).returncode == 0
        except Exception:
            return False                                      # tool exists but hung/errored -> server not serving
    import socket as _s                                       # no probe tool at all -> weak socket-connect fallback
    p = os.environ.get("PULSE_SERVER", ""); p = p[5:] if p.startswith("unix:") else p
    if not p: p = os.path.join(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"), "pulse", "native")
    try:
        c = _s.socket(_s.AF_UNIX, _s.SOCK_STREAM); c.settimeout(timeout); c.connect(p); c.close(); return True
    except Exception:
        return False


class MpvSbsCapture:
    """ZERO-COPY native-SBS source (gated GPU_CAPTURE_MPV=<file>). libmpv (NVDEC) decodes + renders the SBS frame
    DIRECTLY into an FBO whose color texture is cap.glo -- no Xephyr, no XShm, no ~33MB/frame CPU upload. Presents the
    SAME interface as SharedMemSBSCapture (.glo/.size/.refresh()/.use()/.bgr/.flipped/.rgba_fmt/.close()) so the weave
    shader is UNCHANGED (it splits L|R from the SBS texture). MUST be constructed on the GL thread with the context
    current; get_proc_address comes from the GL toolkit (glfw). The render is driven every weave frame (advanced_control
    keeps it non-blocking); mpv composites the latest decoded frame into the FBO."""

    def __init__(self, ctx, src, get_proc_address, w=None, h=None):
        import mpv, numpy as np, time as _time
        self.np = np; self.ctx = ctx; self._src = src
        self.bgr = False; self.rgba_fmt = True                 # mpv renders RGBA, top-origin
        self.flipped = (os.environ.get("GPU_CAPTURE_MPV_FLIP", "1") == "1")   # weave uSrcFlip; default 1 = right-side-up (live-confirmed)
        self._flip_y = (os.environ.get("GPU_CAPTURE_MPV_FLIPY", "1") == "1")  # mpv FBO render origin
        self.glo = 0; self.alive = True; self.tex_changed = True; self.fps = 0.0
        self._upload_count = 0; self._n = 0; self._t0 = time.time()
        self._hwdec = os.environ.get("GPU_CAPTURE_MPV_HWDEC", "auto")
        self._dump = os.environ.get("GPU_CAPTURE_MPV_DUMP") or None   # one-shot FBO dump (headless fill/SBS validation)
        self._dump_at = int(os.environ.get("GPU_CAPTURE_MPV_DUMP_AT", "90"))
        # mpv + render context first (on the current GL thread/context).
        # video-sync default 'audio' (mpv's default) is CORRECT for this render-API path: the weave already
        # presents every decoded frame vblank-locked (swap_interval(1) + update()-gated render at 120Hz), so we
        # don't need mpv to pace to the display. 'display-resample' would be WRONG here — report_swap() fires
        # per DECODED frame (gated), not per vblank, so mpv's display-rate estimate is wrong and it would
        # resample AUDIO off it (pitch/speed artifacts, esp. if toggled to 60 with a 120fps clip) for no visual
        # gain. Override via GPU_CAPTURE_MPV_VSYNC only if report_swap() is first made per-swap. A 120fps clip
        # still shows at 120 because the weave swaps at 120 and grabs the freshest decoded frame each vblank.
        self._vsync = os.environ.get("GPU_CAPTURE_MPV_VSYNC", "audio")
        # video_sync='audio' paces the picture to the AUDIO clock, so a DEAD audio device (pipewire-pulse not serving
        # -> 'ao/pulse Init failed: Timeout') FREEZES the video, not just the sound. Probe pulse; if it isn't actually
        # serving, disable mpv audio so the 3D ALWAYS plays (silent) rather than freezing. GPU_CAPTURE_MPV_AUDIO=yes|no|auto.
        _amode = os.environ.get("GPU_CAPTURE_MPV_AUDIO", "auto")
        _audio_on = (_amode == "yes") or (_amode == "auto" and _pulse_serving())
        _mkw = dict(vo="libmpv", hwdec=self._hwdec, loop_file="inf", terminal=False,
                    osc=False, input_default_bindings=False, video_sync=self._vsync)
        if not _audio_on:
            _mkw["audio"] = "no"
            print("[capture] mpv AUDIO OFF (pulse not serving or =no) -> video plays silent, never freezes on dead audio", flush=True)
        self._mpv = mpv.MPV(**_mkw)
        self._proc_cb = mpv.MpvGlGetProcAddressFn(lambda _c, name: get_proc_address(name.decode("utf-8")))  # keep ref (GC -> segfault)
        self._rctx = mpv.MpvRenderContext(self._mpv, "opengl",
                                          opengl_init_params={"get_proc_address": self._proc_cb},
                                          advanced_control=True)
        self._mpv.play(src)
        # FBO size: explicit override (arg/env) wins (e.g. downscale a 4K SBS to 1080p for >60fps headroom);
        # else AUTO-DETECT the source resolution from mpv (works for any SBS clip); else 4K fallback.
        W = int(w or os.environ.get("GPU_CAPTURE_SBS_W", "0") or 0)
        H = int(h or os.environ.get("GPU_CAPTURE_SBS_H", "0") or 0)
        if not (W and H):
            _deadline = _time.time() + float(os.environ.get("GPU_CAPTURE_MPV_LOADTIMEOUT", "6"))
            while _time.time() < _deadline:
                try:
                    vw = self._mpv.dwidth or self._mpv.width; vh = self._mpv.dheight or self._mpv.height
                    if vw and vh: W, H = int(vw), int(vh); break
                except Exception: pass
                _time.sleep(0.05)
        if not (W and H):                                      # load timed out -> 4K fallback
            W, H = 3840, 2160
            print(f"[capture] MpvSbsCapture: dim auto-detect timed out -> fallback {W}x{H}", flush=True)
        self.size = (W, H)
        lin = moderngl_LINEAR()
        self._tex = ctx.texture((W, H), 4, dtype="f1"); self._tex.filter = (lin, lin)
        self._tex.repeat_x = False; self._tex.repeat_y = False
        self._fbo = ctx.framebuffer(color_attachments=[self._tex])
        self.glo = int(self._tex.glo)
        print(f"[capture] MpvSbsCapture '{src}' -> {W}x{H} FBO (libmpv render, hwdec={self._hwdec}, "
              f"flip_y={self._flip_y} flip={self.flipped})", flush=True)

    def refresh(self):
        self.tex_changed = False                               # glo is stable -> one-shot (don't re-wrap each frame)
        if not self.alive:
            return self.glo != 0
        try:
            new_frame = False
            try: new_frame = self._rctx.update()               # True ONLY when mpv has a freshly-decoded frame ready
            except Exception: pass
            if new_frame or self._upload_count == 0:            # render only a COMPLETE frame (never sample mid-decode)
                self._rctx.render(flip_y=self._flip_y, block_for_target_time=False,
                                  opengl_fbo={"w": self.size[0], "h": self.size[1], "fbo": int(self._fbo.glo)})
                self.ctx.screen.use()                          # mpv bound its own FBO/state -> restore moderngl default
                try: self._rctx.report_swap()                  # advanced_control contract: report the frame was consumed
                except Exception: pass
                self._upload_count += 1; self._n += 1
                dt = time.time() - self._t0
                if dt >= 1.0:
                    self.fps = self._n / dt; self._n = 0; self._t0 = time.time()
            if self._dump and self._upload_count >= self._dump_at:
                try:
                    import cv2
                    d = self.np.frombuffer(self._fbo.read(components=3), self.np.uint8).reshape(self.size[1], self.size[0], 3)
                    cv2.imwrite(self._dump, d[:, :, ::-1])     # RGB(FBO) -> BGR(cv2)
                    print(f"[capture] mpv FBO dump -> {self._dump} (frame {self._upload_count})", flush=True)
                except Exception as _de:
                    print("[capture] mpv dump err:", _de, flush=True)
                self._dump = None
        except Exception as e:
            print("[capture] mpv refresh error:", e, flush=True); self.alive = False
        return self.glo != 0

    # ---- playback controls (libmpv player path) ----
    def toggle_pause(self):
        try: self._mpv.pause = not self._mpv.pause
        except Exception: pass
    def seek(self, secs):
        try: self._mpv.command("seek", float(secs), "relative")
        except Exception: pass
    def seek_frac(self, frac):
        try:
            d = self.duration
            if d > 0: self._mpv.command("seek", max(0.0, min(1.0, float(frac))) * d, "absolute")
        except Exception: pass
    def load_file(self, path):
        """Switch playback to another file (next/prev video). Reuses the same FBO/texture — mpv just
        decodes the new source into it, so the weave is uninterrupted. Returns True on success."""
        try:
            self._mpv.play(path); self._src = path
            try: self._mpv.pause = False
            except Exception: pass
            return True
        except Exception as e:
            print("[capture] load_file failed:", e, flush=True); return False
    @property
    def src_path(self):
        return getattr(self, "_src", None)
    @property
    def paused(self):
        try: return bool(self._mpv.pause)
        except Exception: return False
    @property
    def time_pos(self):
        try: return float(self._mpv.time_pos or 0.0)
        except Exception: return 0.0
    @property
    def duration(self):
        try: return float(self._mpv.duration or 0.0)
        except Exception: return 0.0

    def use(self, unit=0):
        if self._tex is not None:
            self._tex.use(unit)

    def close(self):
        try:
            if getattr(self, "_rctx", None) is not None: self._rctx.free()
        except Exception: pass
        try:
            if getattr(self, "_mpv", None) is not None: self._mpv.terminate()
        except Exception: pass
        for t in (getattr(self, "_fbo", None), getattr(self, "_tex", None)):
            try:
                if t is not None: t.release()
            except Exception: pass
        self._tex = self._fbo = self._rctx = self._mpv = None; self.glo = 0
    release = close

    def __del__(self):
        try: self.close()
        except Exception: pass


def make_capture(ctx, win_id=None, display=None):
    """Factory. Order: SBS-shm (2D->3D) -> DISPLAY-root (Xephyr) -> PROC -> GLX -> SHM -> CPU."""
    rgbd_shm = os.environ.get("GPU_CAPTURE_RGBD_SHM")      # GPU-DIBR path (gated): RGB+depth planes, warp on the GPU
    if rgbd_shm:
        return SharedMemRGBDCapture(ctx, rgbd_shm,
                                    w=int(os.environ.get("GPU_CAPTURE_SBS_W", "0")) or None,
                                    h=int(os.environ.get("GPU_CAPTURE_SBS_H", "0")) or None)
    sbs_shm = os.environ.get("GPU_CAPTURE_SBS_SHM")        # read synthesized SBS from Proc A (twod3d_source.py)
    if sbs_shm:
        return SharedMemSBSCapture(ctx, sbs_shm,
                                   w=int(os.environ.get("GPU_CAPTURE_SBS_W", "0")) or None,
                                   h=int(os.environ.get("GPU_CAPTURE_SBS_H", "0")) or None)
    cap_display = os.environ.get("GPU_CAPTURE_DISPLAY")     # ":3" -> grab that NESTED display's root (feedback-free isolation)
    if cap_display:
        _wid = win_id if win_id is not None else "root"
        _sx = int(os.environ.get("GPU_CAPTURE_SRC_X", "0")); _sy = int(os.environ.get("GPU_CAPTURE_SRC_Y", "0"))
        if os.environ.get("GPU_CAPTURE_PROC") == "1":      # GIL-free child grab -> 60fps on the nested root
            from proc_capture import ProcWindowCapture
            return ProcWindowCapture(ctx, win_id=_wid, display_name=cap_display, src_x=_sx, src_y=_sy)
        return ShmWindowCapture(ctx, win_id=_wid, display_name=cap_display, src_x=_sx, src_y=_sy)
    if os.environ.get("GPU_CAPTURE_PROC") == "1":
        from proc_capture import ProcWindowCapture        # separate-process GIL-free XShm grab
        return ProcWindowCapture(ctx, win_id)
    force = os.environ.get("GPU_CAPTURE_FORCE", "").lower()
    if force == "cpu":
        return CpuWindowCapture(ctx, win_id)
    if force == "glx":
        return GLXWindowCapture(ctx, win_id, display=display)   # raises on failure (forced)
    if force == "shm":
        return ShmWindowCapture(ctx, win_id)                    # raises on failure (forced)
    # AUTO: SHM is the proven live path here; try it first, CPU as last resort.
    try:
        return ShmWindowCapture(ctx, win_id)
    except Exception as e:
        print(f"[gpu_capture] SHM path failed ({e}); falling back to CPU get_image (~1fps)")
        return CpuWindowCapture(ctx, win_id)


# ---------------------------------------------------------------------------
def _selftest(window_id=None):
    os.environ.setdefault("DISPLAY", ":1")
    import glfw, moderngl, time as _t
    if not glfw.init(): raise SystemExit("glfw init failed")
    glfw.window_hint(glfw.VISIBLE, False)
    w = glfw.create_window(64, 64, "gpu_capture_selftest", None, None)
    if not w: raise SystemExit("glfw create_window failed (DISPLAY/XAUTHORITY?)")
    glfw.make_context_current(w)
    ctx = moderngl.create_context()
    print("GL_RENDERER:", ctx.info.get("GL_RENDERER"))
    cap = make_capture(ctx, window_id)
    print("impl:", type(cap).__name__, "size:", cap.size, "glo:", cap.glo,
          "rgba:", cap.rgba_fmt, "bgr:", cap.bgr, "flip:", cap.flipped)
    n = 0; t0 = _t.time()
    while _t.time() - t0 < 3.0:
        cap.refresh(); ctx.finish(); n += 1
    print(f"refresh+finish loop: {n/3.0:.1f} fps, capture-thread fps={cap.fps:.1f}, alive={cap.alive}, xerr={_xerr[0]}")
    # sample a center pixel to prove non-black content
    import numpy as np
    fbo_t = ctx.texture((1, 1), 4, dtype="f1"); fbo = ctx.framebuffer(color_attachments=[fbo_t])
    prog = ctx.program(vertex_shader="#version 330\nin vec2 p;out vec2 uv;void main(){uv=vec2(0.5);gl_Position=vec4(p,0,1);}",
                       fragment_shader="#version 330\nuniform sampler2D s;in vec2 uv;out vec4 o;void main(){o=texture(s,uv);}")
    quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes())
    vao = ctx.vertex_array(prog, [(quad, "2f", "p")]); cap.use(0); prog["s"] = 0
    fbo.use(); ctx.clear(0, 0, 0); vao.render(moderngl.TRIANGLE_STRIP); ctx.finish()
    px = np.frombuffer(fbo.read(components=4), np.uint8)
    print("center pixel (sampled):", px.tolist(), "-> non-black" if int(px[:3].max()) > 0 else "-> BLACK(!)")
    cap.close(); glfw.terminate()


if __name__ == "__main__":
    import sys
    wid = int(sys.argv[1], 0) if len(sys.argv) > 1 else None
    _selftest(wid)
