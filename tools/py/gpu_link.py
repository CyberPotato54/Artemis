#!/usr/bin/env python3
"""Read the RX 9070 XT's live PCIe generation, clocks and temperatures through AMD's user-mode ADL (atiadlxx.dll).

Why: on 2026-10-31 every binary ran about 22% slow because the GPU's PCIe link had dropped to half rate (Gen3; 14 GB/s to
VRAM against the usual 28). Expert reads are bound by that link. GPU clocks, VRAM, CPU and the process list all looked
normal, so the link is the first thing to check when everything is slow. `artemis snapshot-times` measures the rate
itself but loads the model; this script asks the driver and never touches the GPU's work.

    python3 -I tools/py/gpu_link.py            one sample
    python3 -I tools/py/gpu_link.py 24 3       24 samples, 3 s apart
    python3 -I tools/py/gpu_link.py watch      log every change of the link sensors (and a heartbeat every 5 min) until killed
    python3 -I tools/py/gpu_link.py check [--name 9070] [--min-gen 4]
                                               print one JSON line {"adapter", "pcie_gen", "ok", "why"}; exit 0 when the link
                                               is at --min-gen or better, 1 when it is lower, 2 when it cannot be read

Recovery when the link reads gen 3 and `snapshot-times` says ~14 GB/s: from an elevated shell,
`pnputil /restart-device "<the 9070 XT's PnP instance id>"` retrains it (gen 4, 28 GB/s) without a reboot.

The sensor ids come from ADL's PMLog table; the ones used here were checked against measured behaviour on the 9070 XT
with driver 32.0.31041.1004: id 1 gfx clock MHz, id 2 memory clock MHz, id 8 memory temperature, id 14 fan rpm, id 15 fan
percent, id 19 memory activity percent, id 27 gpu temperature, id 40 PCIe generation (3 while slow, 4 after the retrain),
id 73 board power W. Windows only. Read-only.
"""

from __future__ import annotations

import ctypes
import sys
import time
from ctypes import CFUNCTYPE, POINTER, Structure, byref, c_int, c_void_p, sizeof

NAMES = {
    1: "gfx_mhz",
    2: "mem_mhz",
    8: "t_mem",
    14: "fan_rpm",
    15: "fan_pct",
    19: "mem_act_pct",
    27: "t_gpu",
    40: "pcie_gen",
    73: "power_w",
}
WATCH = (40, 41, 58)  # 40 is the PCIe generation; 41 and 58 moved with it (58: 4 slow, 5 good) and are logged alongside


class AdapterInfo(Structure):
    _fields_ = [
        ("iSize", c_int),
        ("iAdapterIndex", c_int),
        ("strUDID", ctypes.c_char * 256),
        ("iBusNumber", c_int),
        ("iDeviceNumber", c_int),
        ("iFunctionNumber", c_int),
        ("iVendorID", c_int),
        ("strAdapterName", ctypes.c_char * 256),
        ("strDisplayName", ctypes.c_char * 256),
        ("iPresent", c_int),
        ("iExist", c_int),
        ("strDriverPath", ctypes.c_char * 256),
        ("strDriverPathExt", ctypes.c_char * 256),
        ("strPNPString", ctypes.c_char * 256),
        ("iOSDisplayIndex", c_int),
    ]


class SensorData(Structure):
    _fields_ = [("supported", c_int), ("value", c_int)]


class PMLogDataOutput(Structure):
    _fields_ = [("size", c_int), ("sensors", SensorData * 256)]


class LinkUnreadable(Exception):
    """ADL is missing, the adapter is not found or the sensor query failed."""


MALLOC = CFUNCTYPE(c_void_p, c_int)
_buffers: list = []


def _malloc(n: int) -> int:
    buf = ctypes.create_string_buffer(n)
    _buffers.append(buf)
    return ctypes.addressof(buf)


def open_adl(name_part: bytes = b"9070"):
    try:
        adl = ctypes.WinDLL("atiadlxx.dll")
    except (OSError, AttributeError) as e:
        raise LinkUnreadable(f"AMD's ADL library (atiadlxx.dll) cannot be loaded: {e}")
    callback = MALLOC(_malloc)
    _buffers.append(callback)
    ctx = c_void_p()
    if adl.ADL2_Main_Control_Create(callback, 1, byref(ctx)) != 0:
        raise LinkUnreadable("ADL2_Main_Control_Create failed")
    n = c_int()
    adl.ADL2_Adapter_NumberOfAdapters_Get(ctx, byref(n))
    infos = (AdapterInfo * n.value)()
    adl.ADL2_Adapter_AdapterInfo_Get(ctx, infos, sizeof(infos))
    for a in infos:
        if a.iPresent and name_part.lower() in a.strAdapterName.lower():
            return adl, ctx, a.iAdapterIndex, a.strAdapterName.decode(errors="replace")
    names = sorted({a.strAdapterName.decode(errors="replace") for a in infos})
    raise LinkUnreadable(f"no adapter named like {name_part.decode(errors='replace')!r}; adapters: {names}")


def sample(adl, ctx, idx) -> dict[int, int] | None:
    out = PMLogDataOutput()
    if adl.ADL2_New_QueryPMLogData_Get(ctx, idx, byref(out)) != 0:
        return None
    return {i: out.sensors[i].value for i in range(256) if out.sensors[i].supported}


def line(s: dict[int, int] | None) -> str:
    if s is None:
        return "query failed"
    return " ".join(f"{NAMES[i]}={s[i]}" for i in NAMES if i in s)


def judge_link(sample: dict[int, int] | None, min_gen: int = 4) -> tuple[bool, int | None, str]:
    """(ok, PCIe generation or None, why) for one sensor sample. Sensor 40 is the generation; a sample that failed or
    lacks it is unreadable (generation None, not ok)."""
    if sample is None:
        return False, None, "the driver's sensor query failed, so the PCIe link could not be read"
    if 40 not in sample:
        return False, None, "the driver reports no sensor 40, so the PCIe generation could not be read"
    gen = sample[40]
    if gen >= min_gen:
        return True, gen, f"PCIe gen {gen}"
    return (
        False,
        gen,
        f"the PCIe link is at gen {gen}, below gen {min_gen}: expect about 22% slower runs; from an elevated shell, "
        "pnputil /restart-device \"<the GPU's PnP instance id>\" retrains it",
    )


def _read_link(name: str) -> tuple[str, dict[int, int] | None]:
    adl, ctx, idx, adapter = open_adl(name.encode())
    try:
        return adapter, sample(adl, ctx, idx)
    finally:
        adl.ADL2_Main_Control_Destroy(ctx)


def check(argv: list[str], reader=_read_link) -> int:
    """The `check` mode: one JSON line, exit 0 (ok), 1 (below --min-gen) or 2 (unreadable). `reader(name)` returns the
    adapter's name and its sensor sample; tests pass a stub."""
    import argparse
    import json

    ap = argparse.ArgumentParser(prog="gpu_link.py check")
    ap.add_argument("--name", default="9070", help="part of the adapter's name (case-insensitive)")
    ap.add_argument("--min-gen", type=int, default=4)
    args = ap.parse_args(argv[1:])
    try:
        adapter, smp = reader(args.name)
    except LinkUnreadable as e:
        print(json.dumps({"adapter": None, "pcie_gen": None, "ok": False, "why": str(e)}), flush=True)
        return 2
    ok, gen, why = judge_link(smp, args.min_gen)
    print(json.dumps({"adapter": adapter, "pcie_gen": gen, "ok": ok, "why": why}), flush=True)
    if ok:
        return 0
    return 2 if gen is None else 1


def main(argv: list[str]) -> None:
    adl, ctx, idx, name = open_adl()
    print(f"# {name} adapter {idx}", flush=True)
    if argv[1:2] == ["watch"]:
        interval = float(argv[2]) if len(argv) > 2 else 5.0
        last, beat = None, 0.0
        while True:
            s = sample(adl, ctx, idx)
            cur = None if s is None else tuple(s.get(i) for i in WATCH)
            now = time.time()
            if cur != last or now - beat >= 300:
                tag = "CHANGE" if cur != last else "beat"
                print(time.strftime("%F %T"), tag, f"watch{WATCH}={cur}", line(s), flush=True)
                last, beat = cur, now
            time.sleep(interval)
    samples = int(argv[1]) if len(argv) > 1 else 1
    interval = float(argv[2]) if len(argv) > 2 else 2.0
    for i in range(samples):
        print(time.strftime("%H:%M:%S"), line(sample(adl, ctx, idx)), flush=True)
        if i + 1 < samples:
            time.sleep(interval)
    adl.ADL2_Main_Control_Destroy(ctx)


if __name__ == "__main__":
    if sys.argv[1:2] == ["check"]:
        sys.exit(check(sys.argv[1:]))
    try:
        main(sys.argv)
    except LinkUnreadable as e:
        sys.exit(str(e))
