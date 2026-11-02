"""The llama.cpp speed comparison (reported, not gating): the speed prompts through llama.cpp
b11509's llama-server without speculation, with the speed prompts' sampling. The default configuration (plan 2a's) is
16 GB, `--n-cpu-moe 24 --moe-cache-mib 1500`, q8_0 KV and a 32,768-token context; --config-json replaces it with a
searched `best.json`. Writes each run's decode speed and their median as JSON, with the device and the CPU affinity.

Usage: python -I tools/py/speed_llama.py <repo root> <model.gguf> <oracle bin dir> --out FILE --gpu-name NAME
           [--affinity HEX] [--config-json FILE] [--ctx N]

The GPU is chosen by name from `llama-server --list-devices`, never by index (the iGPU can be Vulkan0): no device with
that name, or two, stops it with what llama.cpp lists. This process sets its own CPU affinity (--affinity; default
every CPU in the system's mask, whatever this process is pinned to) and the server inherits it; the server's mask is
read once it is ready and again just before it stops, and a mask other than the one asked for fails the run.

It refuses to start if anything accepts a connection on its port, and never touches production's port 8080. If a
response carries no timings, or the server fails to load, it writes the runs that completed to --out (with an "error"
and no median) and exits non-zero.
"""
import argparse
import json
import pathlib
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_speed_prompts as speed  # noqa: E402

PORT = 8095
# Plan 2a's configuration. No --device: the device is chosen by name and put first by server_args.
BASE_ARGS = ["-ngl", "99", "--fit", "off", "-c", "32768", "--parallel", "1", "--jinja", "-fa", "on",
             "-ctk", "q8_0", "-ctv", "q8_0", "--load-mode", "none", "-b", "2048", "-ub", "2048", "--n-cpu-moe", "24",
             "--moe-cache-mib", "1500"]
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
READY_SECONDS = 600

# `  Vulkan1: AMD Radeon RX 9070 XT (16304 MiB, 15416 MiB free)` under "Available devices:". The id is letters then
# digits; the `ggml_vulkan: 1 = ...` log lines match neither the indent nor the id.
DEVICE_LINE = re.compile(r"^\s+([A-Za-z]+\d+):\s+(.+?)(?:\s+\(\d+ MiB(?:, \d+ MiB free)?\))?\s*$")


def parse_devices(text: str) -> list[tuple[str, str]]:
    """`(id, name)` for each device line under "Available devices" in `llama-server --list-devices`."""
    devices = []
    listing = False
    for line in text.splitlines():
        if line.strip().lower().startswith("available devices"):
            listing = True
            continue
        if not listing:
            continue
        found = DEVICE_LINE.match(line)
        if found:
            devices.append((found.group(1), found.group(2)))
    return devices


def _listed(devices: list[tuple[str, str]]) -> str:
    return ", ".join(f"{i}: {n}" for i, n in devices) if devices else "(nothing)"


def pick_device(devices: list[tuple[str, str]], name: str) -> tuple[str, str]:
    """The one device named `name` (case ignored): an exact match first, else a unique name containing it. No match,
    or more than one at the deciding rule, is a SystemExit that lists what llama.cpp found."""
    want = name.strip().lower()
    if not want:
        raise SystemExit("the GPU name is blank; name the llama.cpp device to use")
    for same in (lambda n: n.lower() == want, lambda n: want in n.lower()):
        found = [d for d in devices if same(d[1])]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise SystemExit(f"{name} matches more than one llama.cpp device: {_listed(found)}")
    raise SystemExit(f"no llama.cpp device named {name}; it lists: {_listed(devices)}")


def list_devices(oracle_bin: pathlib.Path) -> list[tuple[str, str]]:
    """Runs `llama-server --list-devices` (60 s timeout) and parses it."""
    exe = pathlib.Path(oracle_bin) / "llama-server.exe"
    try:
        done = subprocess.run([str(exe), "--list-devices"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"could not list llama.cpp's devices with {exe}: {type(e).__name__}")
    return parse_devices(done.stdout + "\n" + done.stderr)


def server_args(device_id: str, args: list[str]) -> list[str]:
    return ["--device", device_id, *args]


def with_ctx(args: list[str], ctx: int) -> list[str]:
    """`args` with the value after `-c` replaced by ctx (appended when there is no `-c`)."""
    out = list(args)
    if "-c" in out:
        out[out.index("-c") + 1] = str(ctx)
        return out
    return out + ["-c", str(ctx)]


def mask_hex(mask: int) -> str:
    return f"0x{mask:x}"


def parse_mask(text) -> int:
    """A hex mask such as 0xffff0000 (or an int, as a JSON file may hold it); SystemExit if it names no CPU."""
    if isinstance(text, int):
        mask = text
    else:
        try:
            mask = int(str(text).strip(), 16)
        except ValueError:
            raise SystemExit(f"{text!r} is not a hex affinity mask such as 0xffff0000")
    if mask <= 0:
        raise SystemExit(f"affinity mask {text!r} names no CPU")
    return mask


def _kernel32():
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                         ctypes.POINTER(ctypes.c_size_t)]
    k.GetProcessAffinityMask.restype = wintypes.BOOL
    k.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    k.SetProcessAffinityMask.restype = wintypes.BOOL
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.OpenProcess.restype = wintypes.HANDLE
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    return k


def _masks(handle=None) -> tuple[int, int]:
    """`(process mask, system mask)` of a process handle (this process when None)."""
    import ctypes
    k = _kernel32()
    proc, system = ctypes.c_size_t(), ctypes.c_size_t()
    target = handle if handle is not None else k.GetCurrentProcess()
    if not k.GetProcessAffinityMask(target, ctypes.byref(proc), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    return proc.value, system.value


def system_affinity() -> int:
    """Every CPU the system offers (kernel32's lpSystemAffinityMask), whatever this process is pinned to. Never derived
    from a CPU count: a process pinned to one CCD sees only its own CPUs there."""
    return _masks()[1]


def set_affinity(mask: int) -> int:
    """Pins this process to `mask` (children inherit it), which may be wider than its current mask, and returns the
    mask read back."""
    import ctypes
    k = _kernel32()
    if not k.SetProcessAffinityMask(k.GetCurrentProcess(), mask):
        raise ctypes.WinError(ctypes.get_last_error())
    return _masks()[0]


def process_affinity(pid: int) -> int:
    """The affinity mask of process `pid`, read fresh; OSError if it can't be opened (it has exited, say)."""
    import ctypes
    k = _kernel32()
    handle = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return _masks(handle)[0]
    finally:
        k.CloseHandle(handle)


def config_from_best(best: dict) -> tuple[list[str], int]:
    """The `(args, affinity)` of a `best.json` (`best.args`, `best.affinity`)."""
    b = best.get("best") if isinstance(best, dict) else None
    if not isinstance(b, dict) or not isinstance(b.get("args"), list) or "affinity" not in b:
        raise SystemExit("the config file holds no best.args and best.affinity")
    return [str(x) for x in b["args"]], parse_mask(b["affinity"])


def port_in_use() -> bool:
    """Whether anything accepts a connection on PORT, whatever it would answer."""
    try:
        socket.create_connection(("127.0.0.1", PORT), timeout=2).close()
        return True
    except OSError:
        return False


def healthy() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
            return r.status == 200
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return False


def _read_mask(pid: int):
    """`(mask, None)`, or `(None, why)` when the process's mask can't be read."""
    try:
        return process_affinity(pid), None
    except OSError as e:
        return None, f"could not read the server's affinity ({type(e).__name__})"


def run_six(oracle_bin: pathlib.Path, model: str, device: tuple[str, str], args: list[str], affinity: int,
            repo: pathlib.Path) -> dict:
    """One llama-server on `device` with `args`, the six speed runs (write, edit and explain at the speed seeds), and
    the server's affinity read once it is ready and again just before it stops. The caller has already pinned this
    process to `affinity` (the server inherits it); a server mask that differs at either read is an error. Returns
    `{"device", "affinity", "affinity_at_stop", "affinity_asked", "args", "runs", "median"}`, or the same with an
    "error" in place of the median (and whatever completed in `runs`). Never exits the process: a load
    failure is an "error" too."""
    root = pathlib.Path(repo)
    out = {"device": {"id": device[0], "name": device[1]}, "affinity": None, "affinity_at_stop": None,
           "affinity_asked": mask_hex(affinity), "args": list(args), "runs": []}
    if port_in_use():
        out["error"] = f"something already answers on port {PORT}; stop it first"
        return out
    defs = speed.load(root)
    messages = {p["id"]: speed.message(root, defs, p) for p in defs["prompts"]}
    s = defs["sampling"]
    log = pathlib.Path(tempfile.gettempdir()) / "artemis-speed-llama-server.log"
    with open(log, "w") as lf:
        proc = subprocess.Popen([str(pathlib.Path(oracle_bin) / "llama-server.exe"), "-m", model, "--host",
                                 "127.0.0.1", "--port", str(PORT), *server_args(device[0], args)],
                                stdout=subprocess.DEVNULL, stderr=lf)
    failed = None
    try:
        deadline = time.monotonic() + READY_SECONDS
        while not failed:
            # Our server must be alive before a healthy answer on the port counts as its.
            if proc.poll() is not None:
                failed = f"llama-server exited with {proc.returncode}; see {log}"
            elif healthy():
                break
            elif time.monotonic() > deadline:
                failed = f"llama-server wasn't ready within {READY_SECONDS} s; see {log}"
            else:
                time.sleep(2)
        if not failed:
            mask, why = _read_mask(proc.pid)
            out["affinity"] = mask_hex(mask) if mask is not None else None
            if why or mask != affinity:
                failed = why or f"the server's affinity is {mask_hex(mask)}, not the {mask_hex(affinity)} asked for"
        for seed in s["seeds"]:
            if failed:
                break
            for p in defs["prompts"]:
                body = {"messages": [{"role": "user", "content": messages[p["id"]]}], "max_tokens": s["max_tokens"],
                        "temperature": s["temp"], "top_p": s["top_p"], "top_k": s["top_k"], "min_p": s["min_p"],
                        "seed": seed, "chat_template_kwargs": {"enable_thinking": False}}
                req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=600) as r:
                        t = json.loads(r.read()).get("timings", {})
                except (OSError, ValueError) as e:
                    failed = f"{p['id']} seed {seed}: the request failed ({type(e).__name__})"
                    break
                if not t.get("predicted_per_second"):
                    failed = f"{p['id']} seed {seed}: the response has no decode timings"
                    break
                run = {"prompt": p["id"], "seed": seed, "decoded": t.get("predicted_n"),
                       "tok_s": t["predicted_per_second"]}
                print(run, flush=True)
                out["runs"].append(run)
        if out["affinity"] is not None:
            mask, why = _read_mask(proc.pid)
            out["affinity_at_stop"] = mask_hex(mask) if mask is not None else None
            if not failed and (why or mask != affinity):
                failed = why or (f"the server's affinity at stop is {mask_hex(mask)}, not the {mask_hex(affinity)} "
                                 f"asked for")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if not failed and not out["runs"]:
        failed = "no speed runs were made"
    if failed:
        out["error"] = failed
    else:
        out["median"] = statistics.median(r["tok_s"] for r in out["runs"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("model")
    ap.add_argument("oracle_bin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-name", required=True, help="the llama.cpp device's name, e.g. 'AMD Radeon RX 9070 XT'")
    ap.add_argument("--affinity", help="hex CPU mask; default every CPU in the system's mask")
    ap.add_argument("--config-json", help="a best.json whose best.args and best.affinity replace the defaults")
    ap.add_argument("--ctx", type=int, help="the context (replaces -c); default 32768")
    a = ap.parse_args()
    out_path = pathlib.Path(a.out)

    def refuse(why: str) -> None:
        # The script's own refusals are written as a result with an "error", so the runner records the cause.
        out_path.write_text(json.dumps({"runs": [], "error": why}, indent=1) + "\n", encoding="utf-8",
                            newline="\n")
        sys.exit(why)

    try:
        if port_in_use():
            raise SystemExit(f"something already answers on port {PORT}; stop it first")
        device = pick_device(list_devices(pathlib.Path(a.oracle_bin)), a.gpu_name)
        if a.config_json:
            best = json.loads(pathlib.Path(a.config_json).read_text(encoding="utf-8"))
            args, affinity = config_from_best(best)
        else:
            args = list(BASE_ARGS)
            affinity = parse_mask(a.affinity) if a.affinity else system_affinity()
        if a.ctx:
            args = with_ctx(args, a.ctx)
        got = set_affinity(affinity)
        if got != affinity:
            raise SystemExit(f"asked for affinity {mask_hex(affinity)}, but it reads back as {mask_hex(got)}")
    except SystemExit as e:
        refuse(str(e.code))
    except (OSError, ValueError) as e:
        refuse(f"{type(e).__name__}: {e}")
    print(f"device {device[0]}: {device[1]}; affinity {mask_hex(affinity)}", flush=True)
    out = run_six(pathlib.Path(a.oracle_bin), a.model, device, args, affinity, pathlib.Path(a.repo))
    out["config"] = "llama.cpp b11509 llama-server " + " ".join(server_args(device[0], args))
    out_path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    if "error" in out:
        sys.exit(f"{out['error']}; the runs that completed are in {a.out}")
    print(f"median {out['median']:.1f} tok/s")


if __name__ == "__main__":
    main()
