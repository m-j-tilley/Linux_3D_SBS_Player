"""
scaler_linux.py — Linux equivalent of scaler_scsi.py: set the MediaTek scaler 3D flag via the SG_IO ioctl
on the GCREADER SCSI generic device (/dev/sgN). Same reverse-engineered command as Windows:
  CDB(16) = E9 61 00 00 00 00 00 00 02 00 00 00 00 00 00 00 ; DIR = host->device ; 512-byte data
  flag=1 -> [01 00 00 00 D2 ...], flag=0 -> [00 00 00 00 D2 ...]
No external deps (pure ctypes + fcntl). Needs read/write access to the sg device (often root or the 'disk'
group / a udev rule). find_device() locates the GCREADER sg node automatically.
"""
import os, ctypes, fcntl, glob

SG_IO = 0x2285
SG_DXFER_TO_DEV = -2
CDB_SET = bytes([0xE9, 0x61, 0, 0, 0, 0, 0, 0, 0x02, 0, 0, 0, 0, 0, 0, 0])


class sg_io_hdr(ctypes.Structure):
    _fields_ = [
        ("interface_id", ctypes.c_int), ("dxfer_direction", ctypes.c_int),
        ("cmd_len", ctypes.c_ubyte), ("mx_sb_len", ctypes.c_ubyte), ("iovec_count", ctypes.c_ushort),
        ("dxfer_len", ctypes.c_uint), ("dxferp", ctypes.c_void_p), ("cmdp", ctypes.c_void_p),
        ("sbp", ctypes.c_void_p), ("timeout", ctypes.c_uint), ("flags", ctypes.c_uint),
        ("pack_id", ctypes.c_int), ("usr_ptr", ctypes.c_void_p),
        ("status", ctypes.c_ubyte), ("masked_status", ctypes.c_ubyte), ("msg_status", ctypes.c_ubyte),
        ("sb_len_wr", ctypes.c_ubyte), ("host_status", ctypes.c_ushort), ("driver_status", ctypes.c_ushort),
        ("resid", ctypes.c_int), ("duration", ctypes.c_uint), ("info", ctypes.c_uint),
    ]


def find_device():
    """Return the /dev/sgN whose SCSI vendor is GCREADER (the Odyssey scaler), or None."""
    for sg in sorted(glob.glob("/sys/class/scsi_generic/sg*")):
        try:
            vendor = open(os.path.join(sg, "device", "vendor")).read().strip()
            if "GCREADER" in vendor.upper():
                return "/dev/" + os.path.basename(sg)
        except OSError:
            continue
    return None


_fd = [None]; _dev = [None]


def connect(dev=None):
    _dev[0] = dev or find_device()
    if not _dev[0]:
        raise OSError("GCREADER sg device not found (is the monitor's USB connected? need sg access)")
    _fd[0] = os.open(_dev[0], os.O_RDWR)
    return _dev[0]


def set_flag(flag):
    if _fd[0] is None:
        connect()
    data = bytearray(512); data[0] = 1 if flag else 0; data[4] = 0xD2
    cdb_buf = (ctypes.c_ubyte * 16).from_buffer_copy(CDB_SET)
    data_buf = (ctypes.c_ubyte * 512).from_buffer(data)
    sense = (ctypes.c_ubyte * 32)()
    h = sg_io_hdr()
    h.interface_id = ord('S'); h.dxfer_direction = SG_DXFER_TO_DEV
    h.cmd_len = 16; h.mx_sb_len = 32; h.dxfer_len = 512
    h.dxferp = ctypes.cast(data_buf, ctypes.c_void_p)
    h.cmdp = ctypes.cast(cdb_buf, ctypes.c_void_p)
    h.sbp = ctypes.cast(sense, ctypes.c_void_p)
    h.timeout = 5000
    fcntl.ioctl(_fd[0], SG_IO, h)
    return (h.status == 0), h.status, h.host_status


def close():
    if _fd[0] is not None:
        os.close(_fd[0]); _fd[0] = None


if __name__ == "__main__":
    import time, sys
    dev = connect(sys.argv[1] if len(sys.argv) > 1 else None)
    print("scaler device:", dev)
    print("set3d(1) ->", set_flag(1)); time.sleep(3)
    print("set3d(0) ->", set_flag(0))
    close()
