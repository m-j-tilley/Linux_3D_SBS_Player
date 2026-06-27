#!/usr/bin/env python3
"""eye_tracker_proc.py — the head tracker in its OWN process (GIL isolation).

The tracker as a thread inside the weave shares Python's GIL with the weave's 120fps render loop, which starves
it to ~30 eye-fps (vs ~51-60 standalone). Here it runs in a separate process (its own GIL) at the full
detection-limited rate, running the EXACT same exact_weaver.tracker_thread() loop (camera -> FaceMesh iris ->
triangulate -> OneEuro/EMA filter) and mirroring each (eye,vel,t,ok) into the eye shm the weave reads.

Spawned by exact_weaver.start_eye_tracker_proc(); not run by hand. Env: WEAVE_EYE_SHM (shm name, created by the
weave), WEAVE_TUNE_JSON (mc/beta/dc/lead). Honors WEAVE_GPU_TRACK (default 0=CPU; GPU contends with the weave).
"""
import os, sys, json, signal
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exact_weaver as ew
from multiprocessing import shared_memory


def main():
    name = os.environ["WEAVE_EYE_SHM"]
    shm = shared_memory.SharedMemory(name=name)                 # attach (the weave created it)
    # Python resource_tracker footgun: an ATTACHING process registers the shm and UNLINKS it when it exits/dies,
    # destroying it for the weave (creator) AND any watchdog-respawned tracker (-> FileNotFoundError on re-attach).
    # Unregister here so ONLY the weave ever unlinks it.
    try:
        from multiprocessing import resource_tracker
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass
    arr = np.ndarray((ew.EYE_SHM_FLOATS,), dtype=np.float64, buffer=shm.buf)
    sink = ew._ShmSink(arr)
    signal.signal(signal.SIGTERM, lambda *_: setattr(sink, "run", False))   # clean exit when the weave terminates us
    signal.signal(signal.SIGINT, lambda *_: setattr(sink, "run", False))
    tune = {"mc": 1.0, "beta": 0.8, "dc": 4.0, "lead": 0.008}   # 8ms lead: with 60fps + 2.6ms detect the raw lag is
    try: tune.update(json.loads(os.environ.get("WEAVE_TUNE_JSON", "{}")))
    except Exception: pass
    tune["bscale"] = ew.load_baseline()
    print(f"[eye-proc] tracker pid {os.getpid()} tune={tune}", flush=True)
    try:
        ew.tracker_thread(sink, tune)        # the real camera+detect+filter loop; publishes via sink -> shm
    finally:
        try: shm.close()
        except Exception: pass


if __name__ == "__main__":
    main()
