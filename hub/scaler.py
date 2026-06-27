r"""scaler.py — OS-dispatch wrapper for the monitor scaler 3D flag. Same API on both platforms:
connect(), set_flag(0|1), close(). Windows -> scaler_scsi (IOCTL_SCSI_PASS_THROUGH on \\.\E:);
Linux -> scaler_linux (SG_IO on the GCREADER /dev/sgN)."""
import platform

if platform.system() == "Windows":
    from scaler_scsi import connect, set_flag, close
else:
    from scaler_linux import connect, set_flag, close
