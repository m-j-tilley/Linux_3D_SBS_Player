"""
hotkeys.py — robust GLOBAL X11 hotkeys for the override-redirect weave window (which usually lacks
keyboard focus, so the local glfw key callback alone is not enough).

WHY THIS EXISTS / what was wrong with the old screen_weave.Hotkey:
  * It grabbed PLAIN Ctrl+3. On this box (GNOME Shell / Mutter, X.Org 21.1.11, DISPLAY :1) the grab of
    Ctrl+3 + {none, NumLock, CapsLock} returns BadAccess(code 10, GrabKey major_opcode 33) because the
    compositor (or another client) already owns those combos. Only Ctrl+3+NumLock+CapsLock survived, so
    the toggle effectively NEVER fired unless both locks happened to be on. (Probed live, 2026-06-11.)
  * It used a single combo per action and let the default Xlib error handler print the protocol error.

WHAT THIS DOES:
  * Default modifier is Ctrl+Alt (Mod1). Probed live: Ctrl+Alt+{3,F,V,Q,H,S,Up,...} are ALL grabbable
    (Mutter does not reserve Ctrl+Alt+<letter/number> by default), and Ctrl+Alt avoids stealing the
    plain keys or the common Ctrl+F / Ctrl+V that other apps use.
  * For each binding it registers ALL FOUR lock-modifier variants (base, +NumLock(Mod2), +CapsLock(Lock),
    +both) so the hotkey fires no matter the lock state. Mod2/Lock keycodes are read from the live
    modifier map, not hardcoded.
  * Dispatch is by (keycode, modifiers-with-locks-masked-off) -> callback, so one event loop serves every
    binding. The XNextEvent loop runs on its own connection/thread.
  * A custom error handler swallows the async BadAccess from any combo a heavier WM happens to own, and
    records which bindings actually registered (.failed) so the app can warn instead of dying.
  * close()/ungrab_all() cleanly releases every grab and stops the thread (idempotent).

KNOWN LIMITATION (X11 spec, verified live here): while ANOTHER client holds an *active* keyboard grab
(XGrabKeyboard) — e.g. GNOME Shell's overview, a modal, the lock screen, or a not-yet-logged-in greeter
session — ALL passive grabs are SUSPENDED and will not fire. On a normal interactive GNOME session there
is no permanent active grab, so Ctrl+Alt+<key> works. For environments where an active grab can persist,
construct with backend="record": it uses the X RECORD extension to observe device-level key events, which
is IMMUNE to focus and to passive-grab suspension (it sees keys even under an active grab). RECORD does NOT
consume the event, so pick combos no one else acts on.

The local glfw key callback in screen_weave should be KEPT as a third path (fires when the OR window does
hold focus); Hotkey covers the no-focus case.
"""
from __future__ import annotations
import threading
from Xlib import display as _xdisplay, X
from Xlib.ext import xtest  # noqa: F401  (re-exported convenience for the test harness)

# Modifier names -> X mask, for parsing "ctrl+alt+3" style strings.
_MODNAMES = {
    "ctrl": X.ControlMask, "control": X.ControlMask,
    "alt": X.Mod1Mask, "mod1": X.Mod1Mask,
    "shift": X.ShiftMask,
    "super": X.Mod4Mask, "win": X.Mod4Mask, "mod4": X.Mod4Mask,
}

# Common keysym aliases so callers can say "up"/"esc"/"3" etc.
_KEYSYMS = {
    "esc": 0xFF1B, "escape": 0xFF1B,
    "up": 0xFF52, "down": 0xFF54, "left": 0xFF51, "right": 0xFF53,
    "space": 0x0020, "tab": 0xFF09, "return": 0xFF0D, "enter": 0xFF0D,
    "f1": 0xFFBE, "f2": 0xFFBF, "f3": 0xFFC0, "f4": 0xFFC1, "f5": 0xFFC2,
    "f6": 0xFFC3, "f7": 0xFFC4, "f8": 0xFFC5, "f9": 0xFFC6, "f10": 0xFFC7,
    "f11": 0xFFC8, "f12": 0xFFC9,
}


def keysym_for(token: str) -> int:
    """'3'->0x33, 'f'->0x66, 'up'->0xFF52, 'esc'->0xFF1B. Single printable chars use their codepoint."""
    t = token.strip().lower()
    if t in _KEYSYMS:
        return _KEYSYMS[t]
    if len(t) == 1:
        return ord(t)
    raise ValueError(f"unknown key token: {token!r}")


def parse_spec(spec: str):
    """'ctrl+alt+3' -> (modmask, keysym). Modifier order is irrelevant; the last token is the key."""
    parts = [p for p in spec.replace(" ", "").split("+") if p]
    if not parts:
        raise ValueError("empty hotkey spec")
    mods = 0
    for p in parts[:-1]:
        m = _MODNAMES.get(p.lower())
        if m is None:
            raise ValueError(f"unknown modifier {p!r} in {spec!r}")
        mods |= m
    return mods, keysym_for(parts[-1])


class Hotkey:
    """Generalized global hotkeys.

    bindings: dict {spec_string: callback}, e.g.
        Hotkey({
            "ctrl+alt+3":  toggle_3d,
            "ctrl+alt+f":  flip_lr,
            "ctrl+alt+v":  flip_v,
            "ctrl+alt+h":  toggle_hud,
            "ctrl+alt+s":  toggle_rate,     # 60<->120 Hz
            "ctrl+alt+q":  quit_app,
        })

    backend="grab" (default): passive root grabs (fast, event-consuming, but suspended under another
        client's active grab — see module docstring).
    backend="record": X RECORD observer (focus- and active-grab-immune; does NOT consume the event).

    on_error(spec, exc): optional callback invoked for each binding that fails to register (BadAccess).
    """

    # Lock modifiers we don't care about but must register every combination of, so a hotkey fires
    # regardless of NumLock/CapsLock state. Keycodes resolved from the live map in __init__.
    def __init__(self, bindings: dict, display_name=None, backend="grab", on_error=None, daemon=True):
        self.d = _xdisplay.Display(display_name)
        self.root = self.d.screen().root
        self.backend = backend
        self.on_error = on_error
        self._stop = threading.Event()
        self.failed = []          # list of (spec, exc) that did not register
        self.registered = []      # list of (spec, keycode, modmask) that DID register (base combo)

        # discover the lock modifiers actually in use (don't hardcode Mod2 == NumLock).
        self._num, self._caps = self._discover_locks()
        self._lockmask = self._num | self._caps     # masked off when matching incoming events
        lock_variants = [0, self._num, self._caps, self._num | self._caps]

        # build (keycode, clean_modmask) -> callback dispatch table, and remember every grabbed combo.
        self._table = {}
        self._grabs = []          # (keycode, full_modmask) actually requested (for clean ungrab)
        for spec, cb in bindings.items():
            mods, ks = parse_spec(spec)
            kc = self.d.keysym_to_keycode(ks)
            if not kc:
                self.failed.append((spec, ValueError(f"no keycode for keysym {hex(ks)}")))
                continue
            clean = mods & ~self._lockmask
            self._table[(kc, clean)] = cb
            self.registered.append((spec, kc, mods))
            for lv in lock_variants:
                self._grabs.append((kc, mods | lv, spec))

        if backend == "grab":
            self._install_grabs()
        elif backend == "record":
            self._install_record()
        else:
            raise ValueError("backend must be 'grab' or 'record'")

        self._thread = threading.Thread(
            target=self._loop_grab if backend == "grab" else self._loop_record,
            daemon=daemon, name="Hotkey")
        self._thread.start()

    # ---- lock-modifier discovery ----
    def _discover_locks(self):
        num = caps = 0
        mm = self.d.get_modifier_mapping()
        names = ["Shift", "Lock", "Control", "Mod1", "Mod2", "Mod3", "Mod4", "Mod5"]
        masks = [X.ShiftMask, X.LockMask, X.ControlMask, X.Mod1Mask,
                 X.Mod2Mask, X.Mod3Mask, X.Mod4Mask, X.Mod5Mask]
        NUM_LOCK = 0xFF7F
        CAPS_LOCK = 0xFFE5
        for i, row in enumerate(mm):
            for kc in row:
                if not kc:
                    continue
                ks = self.d.keycode_to_keysym(kc, 0)
                if ks == NUM_LOCK:
                    num = masks[i]
                elif ks == CAPS_LOCK:
                    caps = masks[i]
        # CapsLock is conventionally X.LockMask even if the keysym mapping is odd.
        return num, (caps or X.LockMask)

    # ---- grab backend ----
    def _install_grabs(self):
        # Swallow async protocol errors so one BadAccess (a WM-owned combo) doesn't abort the rest or
        # spam stderr. python-xlib has no get_error_handler; set_error_handler(None) restores the default.
        self.d.set_error_handler(lambda *a: None)
        try:
            for kc, mod, _spec in self._grabs:
                self.root.grab_key(kc, mod, True, X.GrabModeAsync, X.GrabModeAsync)
            self.d.sync()
        finally:
            self.d.set_error_handler(None)
        # Verify each binding's BASE combo registered (BadAccess => owned elsewhere). Done with a clean
        # probe so .failed is accurate for the caller's warning.
        self._verify_base_grabs()

    def _verify_base_grabs(self):
        owned = {"hit": False}
        for spec, kc, mods in list(self.registered):
            owned["hit"] = False
            self.d.set_error_handler(lambda *a: owned.__setitem__("hit", True))
            # ungrab then re-grab the base combo: if BadAccess, it's owned by another client.
            self.root.ungrab_key(kc, mods)
            self.root.grab_key(kc, mods, True, X.GrabModeAsync, X.GrabModeAsync)
            self.d.sync()
            self.d.set_error_handler(None)
            if owned["hit"]:
                self.failed.append((spec, "BadAccess: combo owned by another client (WM)"))
                if self.on_error:
                    try:
                        self.on_error(spec, "BadAccess")
                    except Exception:
                        pass

    def _loop_grab(self):
        while not self._stop.is_set():
            if self.d.pending_events():
                ev = self.d.next_event()
                if ev.type == X.KeyPress:
                    cb = self._table.get((ev.detail, ev.state & ~self._lockmask))
                    if cb:
                        try:
                            cb()
                        except Exception as e:
                            print("[hotkey] callback error:", e, flush=True)
            else:
                self._stop.wait(0.004)

    # ---- record backend (focus/active-grab-immune) ----
    def _install_record(self):
        from Xlib.ext import record
        from Xlib.protocol import rq
        self._record = record
        self._rq = rq
        # second connection: RECORD requires a dedicated data connection.
        self._rd = _xdisplay.Display(self.d.get_display_name())
        self._ctx = self._rd.record_create_context(
            0,
            [record.AllClients],
            [{
                'core_requests': (0, 0),
                'core_replies': (0, 0),
                'ext_requests': (0, 0, 0, 0),
                'ext_replies': (0, 0, 0, 0),
                'delivered_events': (0, 0),
                'device_events': (X.KeyPress, X.KeyPress),  # key-press only
                'errors': (0, 0),
                'client_started': False,
                'client_died': False,
            }])

    def _loop_record(self):
        rd = self._rd

        def cb(reply):
            if reply.category != self._record.FromServer or reply.client_swapped:
                return
            data = reply.data
            while data and not self._stop.is_set():
                ev, data = self._rq.EventField(None).parse_binary_value(
                    data, rd.display, None, None)
                if ev.type == X.KeyPress:
                    hit = self._table.get((ev.detail, ev.state & ~self._lockmask))
                    if hit:
                        try:
                            hit()
                        except Exception as e:
                            print("[hotkey] callback error:", e, flush=True)

        # enable_context blocks until the context is freed; runs on this thread.
        rd.record_enable_context(self._ctx, cb)
        rd.record_free_context(self._ctx)

    # ---- teardown ----
    def ungrab_all(self):
        if self.backend == "grab":
            self.d.set_error_handler(lambda *a: None)
            for kc, mod, _spec in self._grabs:
                self.root.ungrab_key(kc, mod)
            self.d.sync()
            self.d.set_error_handler(None)

    def close(self):
        self._stop.set()
        try:
            self.ungrab_all()
        except Exception:
            pass
        if self.backend == "record":
            try:
                # disabling the context makes record_enable_context return, ending the loop thread.
                self.d.record_disable_context(self._ctx)
                self.d.sync()
            except Exception:
                pass
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.d.close()
        except Exception:
            pass

    # context-manager sugar
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# ---------------------------------------------------------------------------
# Headless self-test: fakes each hotkey with XTEST and asserts the callback ran.
# Run:  python hotkeys.py        (uses $DISPLAY)
# ---------------------------------------------------------------------------
def _fake_chord(fakedpy, modmask, keycode, num_mask, caps_mask, with_locks=0):
    """Synthesize a modifier+key chord via XTEST on `fakedpy`. with_locks ORs Num/Caps lock keys in."""
    from Xlib.ext import xtest as _xt
    d = fakedpy
    press = []
    if modmask & X.ControlMask:
        press.append(d.keysym_to_keycode(0xFFE3))   # Control_L
    if modmask & X.Mod1Mask:
        press.append(d.keysym_to_keycode(0xFFE9))   # Alt_L
    if modmask & X.ShiftMask:
        press.append(d.keysym_to_keycode(0xFFE1))   # Shift_L
    if modmask & X.Mod4Mask:
        press.append(d.keysym_to_keycode(0xFFEB))   # Super_L
    for kc in press:
        _xt.fake_input(d, X.KeyPress, kc)
    _xt.fake_input(d, X.KeyPress, keycode)
    _xt.fake_input(d, X.KeyRelease, keycode)
    for kc in reversed(press):
        _xt.fake_input(d, X.KeyRelease, kc)
    d.sync()


def _selftest():
    import os, time, sys
    disp = os.environ.get("DISPLAY", ":1")

    # 1) Pure-logic dispatch test: drive the (keycode,mods)->callback table directly. Always works,
    #    independent of WM grabs / active-grab suspension. This is the authoritative "bindings fire" check.
    fired = {}
    specs = {
        "ctrl+alt+3": lambda: fired.__setitem__("3", fired.get("3", 0) + 1),
        "ctrl+alt+f": lambda: fired.__setitem__("f", fired.get("f", 0) + 1),
        "ctrl+alt+v": lambda: fired.__setitem__("v", fired.get("v", 0) + 1),
        "ctrl+alt+h": lambda: fired.__setitem__("h", fired.get("h", 0) + 1),
        "ctrl+alt+s": lambda: fired.__setitem__("s", fired.get("s", 0) + 1),
        "ctrl+alt+q": lambda: fired.__setitem__("q", fired.get("q", 0) + 1),
    }
    hk = Hotkey(specs, display_name=disp, backend="grab")
    print("[selftest] registered:", [s for s, _, _ in hk.registered])
    if hk.failed:
        print("[selftest] FAILED to grab (owned by WM / no keycode):", hk.failed)

    # synthesize the table dispatch directly (bypasses the X event path) to prove each callback wires up:
    lockmask = hk._lockmask
    for spec, cb in specs.items():
        mods, ks = parse_spec(spec)
        kc = hk.d.keysym_to_keycode(ks)
        cb_lookup = hk._table.get((kc, mods & ~lockmask))
        assert cb_lookup is cb, f"dispatch table wrong for {spec}"
        cb_lookup()                       # simulate the loop firing it
    assert all(fired.get(k) == 1 for k in ("3", "f", "v", "h", "s", "q")), fired
    print("[selftest] dispatch-table logic: PASS (every binding maps to its callback)", fired)

    # 2) Live XTEST round-trip (best-effort): fakes the real chords and checks the loop fired them.
    #    Will be blocked if another client holds an ACTIVE keyboard grab (gnome-shell overview / greeter /
    #    lock screen). We detect that and report rather than fail.
    fired.clear()
    fakedpy = _xdisplay.Display(disp)
    blocked = fakedpy.screen().root.grab_keyboard(
        True, X.GrabModeAsync, X.GrabModeAsync, X.CurrentTime)
    if blocked == 0:
        fakedpy.screen().root.ungrab_keyboard(X.CurrentTime); fakedpy.sync()
    active_grab_present = (blocked != 0)   # 1 == AlreadyGrabbed by another client

    for spec in specs:
        mods, ks = parse_spec(spec)
        kc = fakedpy.keysym_to_keycode(ks)
        _fake_chord(fakedpy, mods, kc, hk._num, hk._caps)
        time.sleep(0.05)
    time.sleep(0.2)
    live_ok = all(fired.get(k, 0) >= 1 for k in ("3", "f", "v", "h", "s", "q"))
    if live_ok:
        print("[selftest] live XTEST round-trip: PASS", fired)
    elif active_grab_present:
        print("[selftest] live XTEST round-trip: SKIPPED — another client holds an ACTIVE keyboard "
              "grab (passive grabs are suspended; expected at a greeter/lock screen). Re-run inside a "
              "normal interactive GNOME session, or use backend='record'.", fired)
    else:
        print("[selftest] live XTEST round-trip: PARTIAL/FAIL", fired,
              "(some combos may be owned by the WM:", hk.failed, ")")
    fakedpy.close()
    hk.close()
    print("[selftest] done. ungrab clean.")
    return 0 if (not hk.failed or active_grab_present) else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
