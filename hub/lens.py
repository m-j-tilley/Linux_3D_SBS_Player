"""
lens.py — 3D-mode (lenticular lens) control for the Samsung Odyssey 3D.

The monitor's "SR Lens Switch Board" (FPC) is a USB CDC-ACM serial device
(VID 0x354B / PID 0x0116) that speaks an ASCII "SR+" command protocol, CR-terminated.
Responses look like  '\\r\\n<value>\\r\\nOK\\r\\n'  or  '\\r\\nERROR <reason>\\r\\n'.

Verbs we use:  SR+LENS=1 (3D on) | SR+LENS=0 (3D off) | SR+LENS (state) | SR+INFO (status).

AUTH: the firmware gates SR+LENS behind an SR+NONCE / SR+AUTH challenge/response. The board reports
`Authenticated: 1` once authenticated, and that state PERSISTS until the monitor is power-cycled.
  * Windows: Samsung's "SR Service" authenticates at startup. Call windows_free_port() to stop it
    (releasing the COM port; the board stays authenticated) so this app can take over.
  * Linux:   no SR Service -> board boots unauthenticated -> SR+LENS=1 returns ERROR NOT AUTHENTICATED
             until the handshake is performed by authenticate(). (.on() raises LensAuthError otherwise.)

Cross-platform: identical code on Windows (COMx) and Linux (/dev/ttyACM0); the port is auto-detected.
"""
from __future__ import annotations
import time
import platform
import secrets

try:
    import serial
    from serial.tools import list_ports
except ImportError as e:  # pragma: no cover
    raise SystemExit("pyserial is required: pip install pyserial") from e

FPC_VID = 0x354B
FPC_PID = 0x0116
SERVICE_NAME = "SR Service"  # Samsung/LeiaSR Windows service that authenticates the board


class LensAuthError(PermissionError):
    """Raised when the FPC rejects a lens command because the board isn't authenticated."""


def find_port() -> str | None:
    """Return the serial device path for the FPC, or None if not present."""
    for p in list_ports.comports():
        if (p.vid, p.pid) == (FPC_VID, FPC_PID):
            return p.device
    return None


class Lens:
    def __init__(self, port: str | None = None, baud: int = 115200, timeout: float = 0.8):
        self.port = port or find_port()
        if not self.port:
            raise RuntimeError("FPC serial port (VID 0x354B / PID 0x0116) not found")
        self.baud, self.timeout = baud, timeout
        self.ser: serial.Serial | None = None

    # --- lifecycle ---
    def open(self) -> "Lens":
        self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        self.ser.dtr = True
        self.ser.rts = True
        time.sleep(0.2)
        return self

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # --- protocol ---
    def cmd(self, c: str, wait: float = 1.0) -> str:
        """Send 'c' + CR, collect the reply until OK/ERROR or timeout. Returns raw text."""
        assert self.ser is not None
        self.ser.reset_input_buffer()
        self.ser.write((c + "\r").encode("ascii"))
        deadline = time.time() + wait
        buf = b""
        while time.time() < deadline:
            n = self.ser.in_waiting
            if n:
                buf += self.ser.read(n)
                if b"OK\r\n" in buf or b"ERROR" in buf:
                    break
            else:
                time.sleep(0.02)
        return buf.decode("ascii", "replace")

    @staticmethod
    def _value(resp: str) -> str:
        """Extract the value lines of an 'SR+' response (everything before the trailing OK)."""
        lines = [l for l in resp.replace("\r", "\n").split("\n") if l.strip()]
        if lines and lines[-1].strip() == "OK":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    # --- high level ---
    def info(self) -> dict:
        """Parse SR+INFO into a dict (Hardware, Firmware, ChipID, Authenticated, Lens enabled, ...)."""
        out: dict[str, str] = {}
        for line in self._value(self.cmd("SR+INFO")).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
        return out

    def authenticated(self) -> bool:
        return self.info().get("Authenticated") == "1"

    def state(self) -> bool:
        """True if the lens (3D) is currently ON."""
        return self._value(self.cmd("SR+LENS")) == "1"

    def on(self) -> bool:
        r = self.cmd("SR+LENS=1")
        if "NOT AUTHENTICATED" in r:
            raise LensAuthError(
                "FPC refused SR+LENS=1: board not authenticated. "
                "Windows: run windows_free_port() so SR Service authenticates first. "
                "Linux: the auth handshake is not yet implemented."
            )
        return "OK" in r

    def off(self) -> bool:
        return "OK" in self.cmd("SR+LENS=0")

    # --- auth (challenge/response) — perform the SR+AUTH handshake so SR+LENS works natively on Linux ---

    def get_chip(self) -> dict:
        """Identity/chip values that may feed the auth response (from SR+INFO)."""
        info = self.info()
        return {k: info.get(k) for k in ("ChipID", "SEChipID", "Serial", "Ser1", "Ser2", "Ser3", "Ser4")}

    def capture_handshake(self, host_nonce: bytes | None = None) -> dict:
        """Read the SR+NONCE / SR+AUTH exchange values without authenticating (diagnostic; non-destructive)."""
        hn = host_nonce or secrets.token_bytes(64)
        set_resp = self.cmd("SR+NONCE=" + hn.hex().upper())          # host -> device nonce
        challenge = self.cmd("SR+AUTH")                              # device challenge/MAC (read; no '=')
        device_nonce = self.cmd("SR+NONCE")                          # device nonce (read)
        return {"host_nonce": hn.hex().upper(),
                "set_nonce_resp": set_resp.strip(),
                "device_challenge_raw": challenge.strip(),
                "device_nonce_raw": device_nonce.strip(),
                "chip": self.get_chip()}

    # Fixed protocol constant for the SR+AUTH response (the same for all units of this device class).
    _AUTH_K16 = bytes.fromhex("dba444fb61c906f2e74c45a0f9b36d8d")

    def _compute_auth_response(self, host_nonce: bytes, device_nonce: bytes, chip: dict) -> bytes:
        """Compute the SR+AUTH response from the device nonce and the chip identity (read from SR+INFO)."""
        import hmac, hashlib
        chipid = bytes.fromhex(chip["ChipID"])
        sechip = bytes.fromhex(chip["SEChipID"])
        hdr = sechip[6:14] + b"\xff" * 16 + b"\x00" * 16
        s0 = hmac.new(self._AUTH_K16, hdr + device_nonce[0:32] + bytes([0x80, 0x07, 0x00]), hashlib.sha3_256).digest()
        s1 = hmac.new(s0, hdr + device_nonce[32:64] + bytes([0x00, 0x07, 0x00]), hashlib.sha3_256).digest()
        mask = chipid + chipid
        return bytes(a ^ b for a, b in zip(s1, mask))

    def authenticate(self) -> bool:
        """Run the SR+AUTH handshake so SR+LENS works on a cold Linux boot."""
        if self.authenticated():
            return True
        hn = secrets.token_bytes(64)
        self.cmd("SR+NONCE=" + hn.hex().upper())                     # send host nonce
        self.cmd("SR+AUTH")                                          # read device challenge
        dn_hex = (self._value(self.cmd("SR+NONCE")).split() or [""])[0]
        if dn_hex.lower().endswith("aa01"):                          # strip trailing framing
            dn_hex = dn_hex[:-4]
        dn = bytes.fromhex(dn_hex[:128]) if dn_hex else b""          # 64-byte device nonce
        resp = self._compute_auth_response(hn, dn, self.get_chip())
        self.cmd("SR+AUTH=" + resp.hex().upper())
        return self.authenticated()


# --- Windows SR Service bootstrap (authenticate the board + free the COM port) ---

def windows_free_port(verbose: bool = True) -> None:
    """Stop Samsung 'SR Service' so we can open the FPC COM port.

    Assumes SR Service has already authenticated the board this power-cycle (it does so at startup).
    The board stays authenticated after the service stops. No-op on non-Windows.
    """
    if platform.system() != "Windows":
        return
    import subprocess
    subprocess.run(["sc", "stop", SERVICE_NAME], capture_output=True, text=True)
    time.sleep(2)
    if verbose:
        print(f"[lens] stopped '{SERVICE_NAME}' to free the port (board stays authenticated)")


def windows_restore_service(verbose: bool = True) -> None:
    """Restart Samsung 'SR Service' (restores normal Reality Hub operation). No-op on non-Windows."""
    if platform.system() != "Windows":
        return
    import subprocess
    subprocess.run(["sc", "start", SERVICE_NAME], capture_output=True, text=True)
    if verbose:
        print(f"[lens] restarted '{SERVICE_NAME}'")


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Odyssey 3D lens (FPC) control")
    ap.add_argument("action", choices=["info", "state", "on", "off", "blink", "auth-capture", "auth"])
    ap.add_argument("--free-port", action="store_true",
                    help="(Windows) stop SR Service first to free the COM port")
    ap.add_argument("--restore", action="store_true",
                    help="(Windows) restart SR Service when done")
    a = ap.parse_args()
    if a.free_port:
        windows_free_port()
    try:
        with Lens() as L:
            print("[lens] port:", L.port)
            if a.action == "info":
                for k, v in L.info().items():
                    print(f"  {k}: {v}")
            elif a.action == "state":
                print("  lens on?", L.state())
            elif a.action == "on":
                print("  on ok?", L.on())
            elif a.action == "off":
                print("  off ok?", L.off())
            elif a.action == "blink":
                print("  authenticated?", L.authenticated())
                print("  on:", L.on())
                time.sleep(3)
                print("  off:", L.off())
            elif a.action == "auth-capture":
                import json
                print("  authenticated (before)?", L.authenticated())
                hs = L.capture_handshake()
                print(json.dumps(hs, indent=2))
                print("  (non-destructive: no response sent; board stays unauthenticated)")
            elif a.action == "auth":
                print("  authenticated?", L.authenticate())
    finally:
        if a.restore:
            windows_restore_service()


if __name__ == "__main__":
    _main()
