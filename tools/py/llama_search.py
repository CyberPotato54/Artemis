"""A coordinate-descent search for llama.cpp b11509's best non-speculative configuration on the card (reported, not
gating). The start is speed_llama's plan 2a configuration at every CPU in the system's mask; each axis (CPU set and
threads, batch sizes, KV type, flash attention, expert placement, the MoE cache, layers on the GPU, load mode) is tried
in turn on top of the best so far. Every variant is screened with one request (the write prompt, seed 1234, 256 tokens,
the speed sampling, thinking off) on its own llama-server; the top two screened variants plus the start then each go
through speed_llama's run_six, outside the budget, and the best median of six is written to best.json. The context stays
32,768 and speculation stays off.

Usage: python -I tools/py/llama_search.py <repo root> <model.gguf> <oracle bin dir> --gpu-name NAME --out DIR
           [--budget-min 75] [--driver TEXT] [--llama-build TEXT]

One llama-server at a time, on port 8095: each variant's server starts, is screened and stops before the next. A busy
port is refused and port 8080 is never touched. Writes DIR\\search.json (every variant) and DIR\\best.json
(`{"best": {"args", "affinity", "median", "runs"}, "device", "finalists", "driver", "llama_build"}`).
"""
import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_speed_prompts as speed  # noqa: E402
import speed_llama as sl  # noqa: E402

SCREEN_TOKENS = 256
SCREEN_SEED = 1234
SCREEN_PROMPT = "write"
SCREEN_TIMEOUT_S = 180
TAIL_LINES = 20

OT_24 = r"blk\.(\d|1\d|2[0-3])\.ffn_(gate|up)_exps\.=CPU"
OT_32 = r"blk\.(\d|1\d|2\d|3[01])\.ffn_(gate|up)_exps\.=CPU"


class NoFit(Exception):
    """The server did not answer /health with 200 within the ready time."""


class Refused(Exception):
    """The server exited (llama.cpp refused the configuration); carries the last lines of its stderr."""

    def __init__(self, stderr_tail: str = ""):
        super().__init__(stderr_tail)
        self.stderr_tail = stderr_tail


class TooSlow(Exception):
    """The screening request ran past its time limit."""


class RunError(Exception):
    """Anything else that went wrong with a variant (a busy port, a response with no timings, a moved affinity)."""


def patch(label: str, set=None, drop=(), affinity=None) -> dict:
    """A change to a variant: `set` is `{flag: value}`, `drop` the flags to remove (with their values), `affinity` a
    mask."""
    return {"label": label, "set": dict(set or {}), "drop": list(drop), "affinity": affinity}


def split_halves(system_mask: int) -> tuple[int, int]:
    """`(low, high)`: the lower and upper half of the set bits of the system's affinity mask, 0xffffffff giving
    0x0000ffff and 0xffff0000. Never derived from a CPU count: this process may be pinned to one CCD."""
    bits = [b for b in range(system_mask.bit_length()) if (system_mask >> b) & 1]
    if len(bits) < 2:
        raise SystemExit(f"the system affinity mask {sl.mask_hex(system_mask)} has fewer than two CPUs to split")
    half = len(bits) // 2
    low = sum(1 << b for b in bits[:half])
    high = sum(1 << b for b in bits[half:])
    return low, high


def axes_for(system_mask: int) -> list[tuple[str, list[dict]]]:
    """The search axes, in the order they are searched, for a machine whose system mask is `system_mask`."""
    low, high = split_halves(system_mask)
    cpu = [patch("cpu all default threads", drop=["-t"], affinity=system_mask)]
    cpu += [patch(f"cpu all -t {t}", {"-t": str(t)}, affinity=system_mask) for t in (8, 16, 32)]
    cpu += [patch(f"cpu low half -t {t}", {"-t": str(t)}, affinity=low) for t in (8, 16)]
    cpu += [patch(f"cpu high half -t {t}", {"-t": str(t)}, affinity=high) for t in (8, 16)]
    sizes = ((2048, 2048), (2048, 512), (512, 512))
    batch = [patch(f"batch {b}/{ub}", {"-b": str(b), "-ub": str(ub)}) for b, ub in sizes]
    kv = [patch(f"kv {t}", {"-ctk": t, "-ctv": t}) for t in ("q8_0", "f16")]
    # llama.cpp refuses a quantized V cache without flash attention, so -fa off always runs with an f16 cache.
    fa = [patch("fa on", {"-fa": "on"}), patch("fa off", {"-fa": "off", "-ctk": "f16", "-ctv": "f16"})]
    experts = [patch(f"n-cpu-moe {n}", {"--n-cpu-moe": str(n)}, drop=["-ot"]) for n in (16, 20, 24, 28, 32, 40)]
    experts += [patch("ot gate+up of 24 layers on cpu", {"-ot": OT_24}, drop=["--n-cpu-moe"]),
                patch("ot gate+up of 32 layers on cpu", {"-ot": OT_32}, drop=["--n-cpu-moe"])]
    cache = [patch("moe-cache none", drop=["--moe-cache-mib"])]
    cache += [patch(f"moe-cache {n}", {"--moe-cache-mib": str(n)}) for n in (1500, 3000)]
    ngl = [patch(f"ngl {n}", {"-ngl": str(n)}) for n in (99, 36)]
    load = [patch("load-mode none", {"--load-mode": "none"}), patch("load-mode absent", drop=["--load-mode"])]
    return [("cpu", cpu), ("batch", batch), ("kv", kv), ("flash_attn", fa), ("experts", experts),
            ("moe_cache", cache), ("ngl", ngl), ("load_mode", load)]


def _pairs(args: list[str]) -> list[list[str]]:
    """The arguments as `[flag, value]` groups (a flag with no value, such as --jinja, is a group of one)."""
    out = []
    for tok in args:
        if tok.startswith("-") and len(tok) > 1 and not tok[1:].isdigit():
            out.append([tok])
        elif out:
            out[-1].append(tok)
        else:
            out.append([tok])
    return out


def apply_patch(variant: dict, p: dict) -> dict:
    """A new variant: `variant` with the patch applied (a set flag replaces its value in place, else is appended)."""
    pairs = _pairs(variant["args"])
    for flag in p["drop"]:
        pairs = [g for g in pairs if g[0] != flag]
    for flag, value in p["set"].items():
        for g in pairs:
            if g[0] == flag:
                g[1:] = [value]
                break
        else:
            pairs.append([flag, value])
    return {"args": [t for g in pairs for t in g],
            "affinity": p["affinity"] if p["affinity"] is not None else variant["affinity"], "label": p["label"]}


def canon(variant: dict) -> tuple:
    """What makes two variants the same server: its flags (order aside) and its affinity."""
    return (tuple(sorted(tuple(g) for g in _pairs(variant["args"]))), variant["affinity"])


def flags_used(axes) -> set[str]:
    return {flag for _, patches in axes for p in patches for flag in p["set"]}


def check_flags(help_text: str, axes, extra=()) -> None:
    """SystemExit naming every flag the axes use (and `extra`) that `llama-server --help` does not list."""
    import re
    missing = sorted(f for f in flags_used(axes) | set(extra)
                     if not re.search(r"(?<![\w-])" + re.escape(f) + r"(?![\w-])", help_text))
    if missing:
        raise SystemExit("this llama-server does not list " + ", ".join(missing) + " in --help; the search would "
                         "fail on every variant that uses it")


def _record(axis: str, v: dict, tok_s, outcome: str, tail: str = "", error: str = "") -> dict:
    """`stderr_tail` is the server's last lines (a Refused); `error` is the reason of an `error` outcome."""
    return {"axis": axis, "label": v["label"], "args": list(v["args"]), "affinity": v["affinity"], "tok_s": tok_s,
            "outcome": outcome, "stderr_tail": tail, "error": error}


def search(measure: Callable[[dict], float], start: dict, axes, clock: Callable[[], float], budget_s: float,
           on_record: Callable[[dict], None] = None) -> dict:
    """Coordinate descent from `start`: for each axis in order, each value is a patch on the best so far. The budget is
    checked before each variant starts. A variant the same as one already measured is not run again. `measure` returns
    decode tok/s or raises NoFit, Refused, TooSlow or RunError, which are recorded and the search goes on; any other
    exception is recorded as an `error` outcome too. `on_record(result so far)` is called after each variant, so a
    caller can keep the record on disk."""
    t0 = clock()
    variants = []
    seen = set()
    best = {"args": list(start["args"]), "affinity": start["affinity"], "label": start.get("label", "start"),
            "tok_s": None}
    budget_hit = False

    def run(axis: str, v: dict) -> bool:
        """Measures `v`; False when the budget is spent (nothing was started)."""
        nonlocal best
        if clock() - t0 >= budget_s:
            return False
        seen.add(canon(v))
        try:
            tok_s, outcome, tail, why = float(measure(v)), "ok", "", ""
        except NoFit:
            tok_s, outcome, tail, why = None, "no_fit", "", ""
        except Refused as e:
            tok_s, outcome, tail, why = None, "refused", e.stderr_tail, ""
        except TooSlow:
            tok_s, outcome, tail, why = None, "too_slow", "", ""
        except RunError as e:
            tok_s, outcome, tail, why = None, "error", "", str(e)
        except Exception as e:  # noqa: BLE001 - one bad variant must not lose the record of the rest
            tok_s, outcome, tail, why = None, "error", "", f"{type(e).__name__}: {e}"
        variants.append(_record(axis, v, tok_s, outcome, tail, why))
        if tok_s is not None and (best["tok_s"] is None or tok_s > best["tok_s"]):
            best = {"args": list(v["args"]), "affinity": v["affinity"], "label": v["label"], "tok_s": tok_s}
        if on_record:
            on_record({"variants": variants, "best": best, "budget_hit": False})
        return True

    first = {"args": list(start["args"]), "affinity": start["affinity"], "label": start.get("label", "start")}
    if not run("start", first):
        budget_hit = True
    else:
        for axis, patches in axes:
            for p in patches:
                v = apply_patch(best, p)
                if canon(v) in seen:
                    continue
                if not run(axis, v):
                    budget_hit = True
                    break
            if budget_hit:
                break
    return {"variants": variants, "best": best, "budget_hit": budget_hit}


def finalists(result: dict, start: dict, n: int = 2) -> list[dict]:
    """The top `n` distinct screened variants (by tok/s), then `start` when it is not among them."""
    ranked = sorted((v for v in result["variants"] if v["outcome"] == "ok"), key=lambda v: -v["tok_s"])
    chosen, keys = [], set()
    for v in ranked:
        c = canon(v)
        if c in keys:
            continue
        keys.add(c)
        chosen.append({"args": list(v["args"]), "affinity": v["affinity"], "label": v["label"]})
        if len(chosen) == n:
            break
    if canon(start) not in keys:
        chosen.append({"args": list(start["args"]), "affinity": start["affinity"],
                       "label": start.get("label", "start")})
    return chosen


def stderr_tail(path: pathlib.Path, n: int = TAIL_LINES) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def make_measure(oracle_bin: pathlib.Path, model: str, device: tuple[str, str], repo: pathlib.Path):
    """A `measure(variant) -> tok/s`: starts the variant's own llama-server, screens it with one request, stops it."""
    root = pathlib.Path(repo)
    defs = speed.load(root)
    prompt = next(p for p in defs["prompts"] if p["id"] == SCREEN_PROMPT)
    content = speed.message(root, defs, prompt)
    s = defs["sampling"]
    body = json.dumps({"messages": [{"role": "user", "content": content}], "max_tokens": SCREEN_TOKENS,
                       "temperature": s["temp"], "top_p": s["top_p"], "top_k": s["top_k"], "min_p": s["min_p"],
                       "seed": SCREEN_SEED, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    log = pathlib.Path(tempfile.gettempdir()) / "artemis-llama-search-server.log"

    def measure(v: dict) -> float:
        if sl.port_in_use():
            raise RunError(f"something already answers on port {sl.PORT}; stop it first")
        got = sl.set_affinity(v["affinity"])
        if got != v["affinity"]:
            raise RunError(f"asked for affinity {sl.mask_hex(v['affinity'])}, but it reads back as {sl.mask_hex(got)}")
        with open(log, "w") as lf:
            proc = subprocess.Popen([str(pathlib.Path(oracle_bin) / "llama-server.exe"), "-m", model, "--host",
                                     "127.0.0.1", "--port", str(sl.PORT), *sl.server_args(device[0], v["args"])],
                                    stdout=subprocess.DEVNULL, stderr=lf)
        try:
            deadline = time.monotonic() + sl.READY_SECONDS
            while True:
                if proc.poll() is not None:
                    raise Refused(stderr_tail(log))
                if sl.healthy():
                    break
                if time.monotonic() > deadline:
                    raise NoFit()
                time.sleep(2)
            mask, why = sl._read_mask(proc.pid)
            if why or mask != v["affinity"]:
                raise RunError(why or f"the server's affinity is {sl.mask_hex(mask)}, not the "
                                      f"{sl.mask_hex(v['affinity'])} asked for")
            req = urllib.request.Request(f"http://127.0.0.1:{sl.PORT}/v1/chat/completions", body,
                                         {"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=SCREEN_TIMEOUT_S) as r:
                    t = json.loads(r.read()).get("timings", {})
            except TimeoutError:
                raise TooSlow()
            except urllib.error.URLError as e:
                if isinstance(e.reason, TimeoutError):
                    raise TooSlow()
                raise RunError(f"the request failed ({type(e).__name__})")
            except (OSError, ValueError) as e:
                raise RunError(f"the request failed ({type(e).__name__})")
            if not t.get("predicted_per_second"):
                raise RunError("the response has no decode timings")
            mask, why = sl._read_mask(proc.pid)
            if why or mask != v["affinity"]:
                raise RunError(why or f"the server's affinity is {sl.mask_hex(mask)} after the request, not the "
                                      f"{sl.mask_hex(v['affinity'])} asked for")
            return float(t["predicted_per_second"])
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    return measure


def _hex(v: dict) -> dict:
    return {**v, "affinity": sl.mask_hex(v["affinity"])}


def run_finalists(oracle_bin, model, device, repo, picks: list[dict]) -> list[dict]:
    """Each finalist through run_six (six runs, median), pinned to its own mask first."""
    out = []
    for f in picks:
        print(f"finalist {f['label']}: affinity {sl.mask_hex(f['affinity'])} {' '.join(f['args'])}", flush=True)
        got = sl.set_affinity(f["affinity"])
        if got != f["affinity"]:
            r = {"runs": [], "error": f"asked for affinity {sl.mask_hex(f['affinity'])}, reads back {sl.mask_hex(got)}"}
        else:
            r = sl.run_six(pathlib.Path(oracle_bin), model, device, f["args"], f["affinity"], pathlib.Path(repo))
        out.append({"label": f["label"], "args": list(f["args"]), "affinity": sl.mask_hex(f["affinity"]),
                    "median": r.get("median"), "runs": r.get("runs", []), "error": r.get("error")})
    return out


def pick_best(done: list[dict]):
    """The finalist with the highest median among those that completed; None when none did."""
    ok = [f for f in done if f.get("median") is not None and not f.get("error")]
    return max(ok, key=lambda f: f["median"]) if ok else None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("model")
    ap.add_argument("oracle_bin")
    ap.add_argument("--gpu-name", required=True, help="the llama.cpp device's name, e.g. 'AMD Radeon RX 9070 XT'")
    ap.add_argument("--out", required=True, help="the directory for search.json and best.json")
    ap.add_argument("--budget-min", type=float, default=75, help="minutes for the screening (finalists run outside it)")
    ap.add_argument("--driver", default="", help="the GPU driver version, from the runner")
    ap.add_argument("--llama-build", default="", help="`llama-server --version`, from the runner")
    a = ap.parse_args(argv)
    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = pathlib.Path(a.oracle_bin)

    def write(name: str, obj: dict) -> None:
        (out_dir / name).write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8", newline="\n")

    def refuse(why: str) -> None:
        write("search.json", {"variants": [], "error": why})
        sys.exit(why)

    try:
        if sl.port_in_use():
            raise SystemExit(f"something already answers on port {sl.PORT}; stop it first")
        device = sl.pick_device(sl.list_devices(bin_dir), a.gpu_name)
        system = sl.system_affinity()
        axes = axes_for(system)
        try:
            helped = subprocess.run([str(bin_dir / "llama-server.exe"), "--help"], capture_output=True, text=True,
                                    timeout=60)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise SystemExit(f"could not run llama-server --help ({type(e).__name__})")
        check_flags(helped.stdout + "\n" + helped.stderr, axes, extra=[t for t in sl.BASE_ARGS if t.startswith("-")])
    except SystemExit as e:
        refuse(str(e.code))
    except (OSError, ValueError) as e:
        refuse(f"{type(e).__name__}: {e}")

    start = {"args": list(sl.BASE_ARGS), "affinity": system, "label": "start (plan 2a, all CPUs)"}
    print(f"device {device[0]}: {device[1]}; system affinity {sl.mask_hex(system)}", flush=True)
    measure = make_measure(bin_dir, a.model, device, pathlib.Path(a.repo))

    def logged(v: dict) -> float:
        print(f"screen {v['label']}: affinity {sl.mask_hex(v['affinity'])} {' '.join(v['args'])}", flush=True)
        tok_s = measure(v)
        print(f"  {tok_s:.1f} tok/s", flush=True)
        return tok_s

    def save(r: dict) -> None:
        write("search.json", {"variants": [_hex(v) for v in r["variants"]], "best": _hex(r["best"]),
                              "budget_hit": r["budget_hit"], "device": {"id": device[0], "name": device[1]}})

    result = search(logged, start, axes, time.monotonic, a.budget_min * 60, on_record=save)
    save(result)
    picks = finalists(result, start)
    if len(picks) == 1 and result["best"]["tok_s"] is None:
        sys.exit(f"no variant ran; see {out_dir / 'search.json'}")
    done = run_finalists(bin_dir, a.model, device, pathlib.Path(a.repo), picks)
    winner = pick_best(done)
    if winner is None:
        write("search.json", {"variants": [_hex(v) for v in result["variants"]], "budget_hit": result["budget_hit"],
                              "finalists": done, "error": "no finalist completed its six runs"})
        sys.exit(f"no finalist completed its six runs; see {out_dir / 'search.json'}")
    write("best.json", {"best": {"args": winner["args"], "affinity": winner["affinity"], "median": winner["median"],
                                 "runs": winner["runs"]},
                        "device": {"id": device[0], "name": device[1]}, "finalists": done, "driver": a.driver,
                        "llama_build": a.llama_build})
    print(f"best {winner['label']}: median {winner['median']:.1f} tok/s")


if __name__ == "__main__":
    main()
