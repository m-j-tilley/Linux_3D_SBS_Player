"""
ir_control.py — SAFE UVC Extension-Unit (XU) tool for the Samsung "3D Stereo WebCam"
(USB 04e8:20d7) IR illuminator, on Linux.

Goal: turn the IR illuminator STEADY-ON so eye tracking works in all lighting. The camera
exposes its IR/illuminator control only through two vendor XUs (NOT via standard V4L2 ctrls):
  * UnitID 3  GUID {0fb885c3-68c2-4547-90f7-8f47579d95fc}  bmControls 0x1f      -> selectors 1..5
  * UnitID 4  GUID {63610682-5070-49ab-b8cc-b3855e8d221d}  bmControls ff ff 77 07 -> sels 1-19,21,22,23,25,26,27

This module does THREE things:
  1. SAFE DEFAULT (`--probe`, also the no-arg default): read-only GET_* dump of the full XU map
     (unit/selector/len/info-caps/min/max/res/def/cur). Issues ONLY GET_* (0x81..0x87). Never writes.
  2. OPT-IN single write (`--set UNIT:SELECTOR=HEXBYTES`): issues exactly ONE SET_CUR, printing the
     CUR before and after so the change is visible and reversible. Requires `--i-have-a-phone-camera`
     as an explicit confirmation flag (the phone camera is the near-IR on/off oracle).
  3. Importable API: probe_all(), get_cur(), set_cur(), and ir_on() / ir_off() helpers that
     tracker.open_stereo_cam can call once the correct (unit, selector, payload) is confirmed.

IMPORTANT — current device state (as of investigation): the camera's control endpoint is GATED.
Every GET_* to units 3 and 4 — and even standard Gain/Exposure and VIDIOC_STREAMON — time out
(-110 ETIMEDOUT) until the host runs the Windows-side BLINKEYE/SREyeTracker bring-up and the lens
auth on /dev/ttyACM0. So `--probe` will currently report errno/ETIMEDOUT for every selector. Once the
device is initialised (lens auth + stream up), re-run `--probe` and the values become readable; only
THEN should a `--set` be attempted.

Run as a user with rw access to /dev/video0 (camera ACL / video group) or under sudo. All ioctls are guarded.
"""
from __future__ import annotations
import argparse
import ctypes
import fcntl
import os
import sys

DEV_DEFAULT = "/dev/video0"

# ---- Known XU topology for 04e8:20d7 (from lsusb -v) ----
UNIT3_GUID = "{0fb885c3-68c2-4547-90f7-8f47579d95fc}"
UNIT4_GUID = "{63610682-5070-49ab-b8cc-b3855e8d221d}"
UNIT3_SELECTORS = list(range(1, 6))                       # bmControls 0x1f -> 1..5
# bmControls ff ff 77 07 -> present bits: 1-8, 9-16, (17,18,19,21,22,23), (25,26,27)
UNIT4_SELECTORS = [b for b in range(1, 33) if b not in (20, 24, 28, 29, 30, 31, 32)]


# ---- UVCIOC_CTRL_QUERY = _IOWR('u', 0x21, struct uvc_xu_control_query) ----
# struct uvc_xu_control_query { __u8 unit; __u8 selector; __u8 query; __u16 size; __u8 *data; }
class uvc_xu_control_query(ctypes.Structure):
    _fields_ = [
        ("unit", ctypes.c_uint8),
        ("selector", ctypes.c_uint8),
        ("query", ctypes.c_uint8),
        ("size", ctypes.c_uint16),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
    ]


def _IOC(direction, typ, nr, size):
    return (direction << 30) | (size << 16) | (typ << 8) | nr


_IOC_WRITE = 1
_IOC_READ = 2
UVCIOC_CTRL_QUERY = _IOC(_IOC_READ | _IOC_WRITE, ord("u"), 0x21,
                         ctypes.sizeof(uvc_xu_control_query))  # = 0xc0107521

SET_CUR = 0x01
GET_CUR = 0x81
GET_MIN = 0x82
GET_MAX = 0x83
GET_RES = 0x84
GET_LEN = 0x85
GET_INFO = 0x86
GET_DEF = 0x87
QNAME = {0x01: "SET", 0x81: "CUR", 0x82: "MIN", 0x83: "MAX",
         0x84: "RES", 0x85: "LEN", 0x86: "INFO", 0x87: "DEF"}


# ---------------------------------------------------------------------------
# low-level
# ---------------------------------------------------------------------------
def _query(fd, unit, selector, q, size, data_in=None):
    """Issue one UVCIOC_CTRL_QUERY. Returns (bytes|None, OSError|None).

    For GET_* leave data_in None. For SET_CUR pass data_in (bytes) of exactly `size`.
    """
    buf = (ctypes.c_uint8 * size)()
    if data_in is not None:
        if len(data_in) != size:
            return None, ValueError(f"data_in len {len(data_in)} != size {size}")
        for i, b in enumerate(data_in):
            buf[i] = b
    cq = uvc_xu_control_query()
    cq.unit = unit
    cq.selector = selector
    cq.query = q
    cq.size = size
    cq.data = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8))
    try:
        fcntl.ioctl(fd, UVCIOC_CTRL_QUERY, cq)
        return bytes(buf), None
    except (OSError, ValueError) as e:
        return None, e


def _le(b):
    if b is None:
        return None
    v = 0
    for i, x in enumerate(b):
        v |= x << (8 * i)
    return v


def _hx(b):
    return "-" if b is None else " ".join(f"{x:02x}" for x in b)


def _decode_info(info):
    if info is None:
        return "?", []
    caps = []
    if info & 0x01: caps.append("GET")
    if info & 0x02: caps.append("SET")
    if info & 0x04: caps.append("DISABLED")
    if info & 0x08: caps.append("AUTOUPD")
    if info & 0x10: caps.append("ASYNC")
    return f"0x{info:02x}", caps


def get_len(fd, unit, selector):
    data, err = _query(fd, unit, selector, GET_LEN, 2)
    if err:
        return None, err
    return data[0] | (data[1] << 8), None


def get_info(fd, unit, selector):
    data, err = _query(fd, unit, selector, GET_INFO, 1)
    if err:
        return None, err
    return data[0], None


def get_cur(fd, unit, selector, size=None):
    """Read GET_CUR. If size is None, learn it via GET_LEN first."""
    if size is None:
        size, err = get_len(fd, unit, selector)
        if err:
            return None, err
        if not size:
            return None, OSError("GET_LEN returned 0")
    return _query(fd, unit, selector, GET_CUR, size)


def set_cur(fd, unit, selector, payload):
    """Issue exactly one SET_CUR(unit, selector, payload-bytes). payload len must match GET_LEN.

    Returns (ok: bool, err). This is the ONLY function that writes. Callers must be explicit.
    """
    size, err = get_len(fd, unit, selector)
    if err:
        return False, err
    if size != len(payload):
        return False, ValueError(f"payload len {len(payload)} != control len {size}")
    _, err = _query(fd, unit, selector, SET_CUR, size, data_in=bytes(payload))
    return (err is None), err


# ---------------------------------------------------------------------------
# probe (read-only)
# ---------------------------------------------------------------------------
def probe_selector(fd, unit, selector):
    """Read-only: returns a dict describing one selector (or its failure)."""
    row = {"unit": unit, "sel": selector, "len": None, "info": None,
           "caps": [], "cur": None, "min": None, "max": None,
           "res": None, "def": None, "error": None}
    length, lerr = get_len(fd, unit, selector)
    if lerr is not None:
        row["error"] = f"GET_LEN: {_errstr(lerr)}"
        return row
    row["len"] = length
    info, ierr = get_info(fd, unit, selector)
    if ierr is None:
        row["info"] = info
        _, row["caps"] = _decode_info(info)
    if length and length > 0 and info is not None and (info & 0x01):
        for q, key in ((GET_CUR, "cur"), (GET_MIN, "min"), (GET_MAX, "max"),
                       (GET_RES, "res"), (GET_DEF, "def")):
            d, e = _query(fd, unit, selector, q, length)
            row[key] = d
    return row


def probe_all(fd):
    rows = []
    for sel in UNIT3_SELECTORS:
        rows.append(probe_selector(fd, 3, sel))
    for sel in UNIT4_SELECTORS:
        rows.append(probe_selector(fd, 4, sel))
    return rows


def _errstr(e):
    if isinstance(e, OSError) and e.errno is not None:
        return f"errno {e.errno} ({os.strerror(e.errno)})"
    return str(e)


def print_probe(rows):
    cur_unit = None
    any_ok = False
    candidates = []
    for r in rows:
        if r["unit"] != cur_unit:
            cur_unit = r["unit"]
            guid = UNIT3_GUID if cur_unit == 3 else UNIT4_GUID
            print(f"\n===== EXTENSION UNIT {cur_unit}  GUID {guid} =====")
        if r["error"]:
            print(f"  sel {r['sel']:2d}: {r['error']}")
            continue
        any_ok = True
        info_s, caps = _decode_info(r["info"])
        caps_s = "|".join(caps) if caps else "-"
        print(f"  sel {r['sel']:2d}: len={r['len']} info={info_s} [{caps_s}]")
        print(f"          CUR=[{_hx(r['cur'])}] (={_le(r['cur'])})"
              f"  MIN=[{_hx(r['min'])}] (={_le(r['min'])})"
              f"  MAX=[{_hx(r['max'])}] (={_le(r['max'])})"
              f"  RES=[{_hx(r['res'])}]"
              f"  DEF=[{_hx(r['def'])}] (={_le(r['def'])})")
        # heuristic flag for IR-enable candidates
        ln, curv, mnv, mxv = r["len"], _le(r["cur"]), _le(r["min"]), _le(r["max"])
        flags = []
        settable = "SET" in caps
        if ln == 8 and r["def"] is not None:                       # MS IR-Torch shape
            mode_def = _le(r["def"][:4])
            if mode_def in (0, 2, 4):
                flags.append(f"MS-IR-TORCH-SHAPE(def-mode={mode_def})")
        if ln == 1 and mnv == 0 and mxv == 1:
            flags.append("BOOL(0/1)")
        if ln in (1, 2, 4) and mxv is not None and mxv and 1 <= mxv <= 8:
            flags.append(f"ENUM(0..{mxv})")
        if ln in (1, 2, 4) and mxv is not None and mxv > 8:
            flags.append(f"INTENSITY-LIKE(max={mxv})")
        if flags and settable:
            candidates.append((r, flags))

    print("\n===== CANDIDATE IR CONTROLS (settable + IR-shaped) =====")
    if not any_ok:
        print("  (no selectors readable — control endpoint is GATED/uninitialised; see module docstring)")
    elif not candidates:
        print("  (readable, but none matched the IR-enable/intensity heuristic)")
    else:
        for r, flags in candidates:
            print(f"  unit {r['unit']} sel {r['sel']}: len={r['len']}"
                  f" CUR={_le(r['cur'])} MIN={_le(r['min'])} MAX={_le(r['max'])}"
                  f" DEF={_le(r['def'])}  -> {','.join(flags)}")


# ---------------------------------------------------------------------------
# importable helpers for tracker.open_stereo_cam
# ---------------------------------------------------------------------------
# Fill these in ONCE the read-probe + phone-camera oracle confirm the right control.
# Confirmed on the Samsung Odyssey 3D (G90XF): the IR illuminator enable is
# extension UnitID 3 (GUID {0fb885c3-68c2-4547-90f7-8f47579d95fc}), selector 2, a 1-byte bool.
# Verified live on the webcam feed: 3:2=01 visibly illuminates the scene (steady, default power is enough);
# 3:2=00 = off. No separate intensity write needed.
IR_UNIT = 3
IR_SELECTOR = 2
IR_ON_PAYLOAD = bytes([0x01])
IR_OFF_PAYLOAD = bytes([0x00])


def _open(dev=DEV_DEFAULT):
    return os.open(dev, os.O_RDWR)


def ir_on(dev=DEV_DEFAULT):
    """Turn IR steady-on. No-op + warning until IR_UNIT/SELECTOR/PAYLOAD are confirmed."""
    if IR_UNIT is None or IR_SELECTOR is None or IR_ON_PAYLOAD is None:
        print("[ir_control] ir_on(): IR control not yet confirmed — refusing to write. "
              "Run ir_control.py --probe (after device init) and the phone-camera oracle first.",
              file=sys.stderr)
        return False
    fd = _open(dev)
    try:
        ok, err = set_cur(fd, IR_UNIT, IR_SELECTOR, IR_ON_PAYLOAD)
        if not ok:
            print(f"[ir_control] ir_on() failed: {_errstr(err)}", file=sys.stderr)
        return ok
    finally:
        os.close(fd)


def ir_off(dev=DEV_DEFAULT):
    if IR_UNIT is None or IR_SELECTOR is None or IR_OFF_PAYLOAD is None:
        print("[ir_control] ir_off(): IR control not yet confirmed — refusing to write.",
              file=sys.stderr)
        return False
    fd = _open(dev)
    try:
        ok, err = set_cur(fd, IR_UNIT, IR_SELECTOR, IR_OFF_PAYLOAD)
        if not ok:
            print(f"[ir_control] ir_off() failed: {_errstr(err)}", file=sys.stderr)
        return ok
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_set(spec):
    """'UNIT:SELECTOR=HEXBYTES' -> (unit, selector, bytes). HEXBYTES may have spaces/colons."""
    try:
        lhs, rhs = spec.split("=", 1)
        unit_s, sel_s = lhs.split(":", 1)
        unit, selector = int(unit_s), int(sel_s)
        clean = rhs.replace(" ", "").replace(":", "").replace("0x", "").replace(",", "")
        if len(clean) % 2 != 0:
            raise ValueError("hex byte string must have an even number of nibbles")
        payload = bytes(int(clean[i:i + 2], 16) for i in range(0, len(clean), 2))
        return unit, selector, payload
    except Exception as e:
        raise SystemExit(f"bad --set spec {spec!r}: {e}\n"
                         f"  expected UNIT:SELECTOR=HEXBYTES, e.g. 3:1=02000000ff000000")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Safe UVC XU probe / IR illuminator control for Samsung 3D Stereo WebCam (04e8:20d7)")
    ap.add_argument("--dev", default=DEV_DEFAULT, help=f"video device (default {DEV_DEFAULT})")
    ap.add_argument("--probe", action="store_true",
                    help="read-only: dump the full XU map (DEFAULT if no --set)")
    ap.add_argument("--get", metavar="UNIT:SELECTOR",
                    help="read-only: read GET_LEN/INFO/CUR/MIN/MAX/DEF for one selector")
    ap.add_argument("--set", metavar="UNIT:SELECTOR=HEXBYTES", dest="set_spec",
                    help="WRITE one SET_CUR (opt-in). Prints CUR before+after. Requires --i-have-a-phone-camera")
    ap.add_argument("--i-have-a-phone-camera", action="store_true",
                    help="required confirmation for --set: a phone camera (near-IR oracle) is aimed at the bezel")
    a = ap.parse_args(argv)

    try:
        fd = os.open(a.dev, os.O_RDWR)
    except OSError as e:
        raise SystemExit(f"cannot open {a.dev}: {_errstr(e)}\n"
                         f"  (need camera ACL or sudo; is anything else streaming the camera?)")
    try:
        if a.set_spec:
            if not a.i_have_a_phone_camera:
                raise SystemExit("--set refused: pass --i-have-a-phone-camera to confirm the near-IR "
                                 "oracle is aimed at the bezel before writing a vendor control.")
            unit, selector, payload = _parse_set(a.set_spec)
            # show before
            before, berr = get_cur(fd, unit, selector)
            length, lerr = get_len(fd, unit, selector)
            if lerr is not None:
                raise SystemExit(f"GET_LEN(unit {unit} sel {selector}) failed: {_errstr(lerr)} "
                                 f"-> control endpoint likely still GATED; do not write yet.")
            print(f"[set] unit {unit} sel {selector} len={length}")
            print(f"[set] CUR before = [{_hx(before)}]  ({_errstr(berr) if berr else 'ok'})")
            print(f"[set] writing    = [{_hx(payload)}]")
            ok, err = set_cur(fd, unit, selector, payload)
            if not ok:
                raise SystemExit(f"[set] SET_CUR FAILED: {_errstr(err)} (device unchanged as far as we can tell)")
            after, aerr = get_cur(fd, unit, selector)
            print(f"[set] CUR after  = [{_hx(after)}]  ({_errstr(aerr) if aerr else 'ok'})")
            print("[set] DONE. Look at the phone camera now: a steady white/purple glow on the bezel = IR steady-ON.")
            if before is not None:
                print(f"[set] TO REVERT: --set {unit}:{selector}={_hx(before).replace(' ', '')} "
                      f"--i-have-a-phone-camera")
            return 0

        if a.get:
            unit_s, sel_s = a.get.split(":", 1)
            unit, selector = int(unit_s), int(sel_s)
            print_probe([probe_selector(fd, unit, selector)])
            return 0

        # default: read-only full probe
        print_probe(probe_all(fd))
        return 0
    finally:
        os.close(fd)


if __name__ == "__main__":
    sys.exit(main())
