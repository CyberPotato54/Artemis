"""How often llama.cpp b11509's own MTP drafting is right on the six speed runs.

llama-server runs with `--spec-type draft-mtp --spec-draft-n-max N --spec-draft-p-min 0`, once per N, so every step
drafts exactly N tokens. Each reply's `draft_n` and `draft_n_accepted` are pooled per N, and `depth_rates` turns the
pooled ratios into the chance that draft i is kept given drafts 1 .. i-1 were (the simulator's MTP model).

Usage: python -I tools/py/mtp_probe.py <repo root> <model.gguf> <oracle bin dir> --out FILE --gpu-name NAME
           [--n-max 1,2,3,4] [--golden]

The GPU is chosen by name (speed_llama.pick_device), never by index.

The six speed runs are write, edit and explain at seeds 1234 and 1235 (the speed prompts' sampling, 700 tokens, thinking
off). --golden runs the 20 golden prompts greedy, 256 tokens, at n_max 1 instead.

If the server fails to load (the draft context's buffers on top of --n-cpu-moe 24) it retries with 32, then 40, before
reporting a refusal; accept counts don't depend on speed, and each record names the value it ran with. It refuses a busy
port 8095 and never touches 8080. A reply without the draft counts stops it, with the runs so far written and an
"error". Counts only: no prompt or output text is written.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import speed_llama as base  # noqa: E402  (BASE_ARGS, PORT, port_in_use, healthy)

PORT = base.PORT
CPU_MOE = (24, 32, 40)
READY_SECONDS = 600


class ProbeError(Exception):
    """The probe can't go on; the message is what the output's "error" says."""


def depth_rates(counts: dict) -> list:
    """`counts[N] = (draft_n, draft_n_accepted)` pooled at n_max N = 1 .. D, every step drafting N. With a_i the chance
    draft i is kept given the ones before it were, the mean accepted per step is `N * r_N = a1 + a1*a2 + ...`, so
    `a_N = (N * r_N - sum of the shorter prefix products) / (the prefix product of a1 .. a_{N-1})`. Each is clamped to
    [0, 1]; a prefix product of 0 leaves the next rate 0 (nothing to measure it from)."""
    depth = len(counts)
    if depth == 0 or sorted(counts) != list(range(1, depth + 1)):
        raise ValueError(f"counts must hold n_max 1 .. D with no gap, got {sorted(counts)}")
    rates = []
    prefix = []  # prefix[j] = a1 * ... * a_{j+1}
    for n in range(1, depth + 1):
        drafted, accepted = counts[n]
        if drafted <= 0:
            raise ValueError(f"no drafts at n_max {n}")
        r = accepted / drafted
        before = prefix[-1] if prefix else 1.0
        a = (n * r - sum(prefix)) / before if before > 0 else 0.0
        a = min(1.0, max(0.0, a))
        rates.append(a)
        prefix.append(before * a)
    return rates


def server_args(device_id: str, n_max: int, cpu_moe: int) -> list:
    """The speed runs' llama-server arguments on `device_id` with `--n-cpu-moe cpu_moe` and MTP drafting N tokens a
    step."""
    args = list(base.BASE_ARGS)
    args[args.index("--n-cpu-moe") + 1] = str(cpu_moe)
    return base.server_args(device_id, args) + ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n_max),
                                                "--spec-draft-p-min", "0"]


def counts_of(reply: dict) -> dict:
    """A reply's draft counts, predicted tokens and decode speed; a ProbeError if any is missing."""
    t = reply.get("timings") or {}
    need = ("draft_n", "draft_n_accepted", "predicted_n", "predicted_per_second")
    missing = [k for k in need if t.get(k) is None]
    if missing:
        raise ProbeError("the response has no " + ", ".join(missing))
    return {"draft_n": int(t["draft_n"]), "draft_n_accepted": int(t["draft_n_accepted"]),
            "predicted_n": int(t["predicted_n"]), "tok_s": float(t["predicted_per_second"])}


def start_server(launch, ready, moes=CPU_MOE):
    """Starts a server with each `--n-cpu-moe` value in turn until one loads. `launch(moe)` returns a process (with
    `poll`, `terminate`, `wait`, `kill`); `ready(proc)` is True once it serves and False if it exited. Returns
    `(proc, moe)`; a ProbeError naming every value tried if none loads."""
    for moe in moes:
        proc = launch(moe)
        if ready(proc):
            return proc, moe
        stop(proc)
    raise ProbeError("llama-server refused draft-mtp at --n-cpu-moe " + ", ".join(str(m) for m in moes))


def stop(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def probe(jobs, n_maxes, launch, ready, post, in_use) -> dict:
    """Runs `jobs` (a list of `(label, body, seed)`) once per n_max. `launch(n_max, moe)` starts a server,
    `ready(proc)` waits for it, `post(body)` returns one reply. Returns the output: `runs`, the pooled `counts` per
    n_max, `depth_rates` (when every depth ran), and an `error` if it stopped early."""
    out = {"n_max": list(n_maxes), "runs": [], "pooled": {}}
    pooled = {}
    for n in n_maxes:
        if in_use():
            out["error"] = f"something already answers on port {PORT}; stop it first"
            return out
        try:
            proc, moe = start_server(lambda m, n=n: launch(n, m), ready)
        except ProbeError as e:
            out["error"] = f"n_max {n}: {e}"
            return out
        try:
            for label, body, seed in jobs:
                try:
                    c = counts_of(post(body))
                except ProbeError as e:
                    out["error"] = f"n_max {n}, {label}: {e}"
                    return out
                except (OSError, ValueError) as e:
                    # HTTPError, URLError and timeouts are OSErrors; a bad JSON body is a ValueError.
                    out["error"] = f"n_max {n}, {label}: the request failed ({type(e).__name__})"
                    return out
                run = {"n_max": n, "n_cpu_moe": moe, "prompt": label, "seed": seed, **c}
                print(run, flush=True)
                out["runs"].append(run)
                d, a = pooled.get(n, (0, 0))
                pooled[n] = (d + c["draft_n"], a + c["draft_n_accepted"])
        finally:
            stop(proc)
    out["pooled"] = {str(n): {"draft_n": d, "draft_n_accepted": a} for n, (d, a) in pooled.items()}
    if sorted(pooled) == list(range(1, len(pooled) + 1)):
        try:
            out["depth_rates"] = depth_rates(pooled)
        except ValueError as e:
            out["error"] = str(e)
    by_prompt = {}
    for r in out["runs"]:
        if r["n_max"] == 1:
            key = r["prompt"].split("-")[0]  # write-1234 -> write: both seeds pooled
            d, a = by_prompt.get(key, (0, 0))
            by_prompt[key] = (d + r["draft_n"], a + r["draft_n_accepted"])
    out["depth1_per_prompt"] = {k: (a / d if d else 0.0) for k, (d, a) in by_prompt.items()}
    return out


def chat_body(message: str, s: dict, seed: int) -> dict:
    return {"messages": [{"role": "user", "content": message}], "max_tokens": s["max_tokens"],
            "temperature": s["temp"], "top_p": s["top_p"], "top_k": s["top_k"], "min_p": s["min_p"], "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False}}


def speed_jobs(root: pathlib.Path, defs: dict, message) -> list:
    s = defs["sampling"]
    return [(f"{p['id']}-{seed}", chat_body(message(root, defs, p), s, seed), seed)
            for seed in s["seeds"] for p in defs["prompts"]]


def golden_jobs(root: pathlib.Path) -> list:
    """The 20 golden prompts' rendered text, greedy, 256 tokens, through /completion."""
    jobs = []
    for txt in sorted((root / "testdata" / "golden" / "prompts").glob("*.txt")):
        body = {"prompt": txt.read_text(encoding="utf-8"), "n_predict": 256, "temperature": 0.0, "top_k": 1,
                "cache_prompt": False}
        jobs.append((txt.stem, body, 0))
    return jobs


def main() -> None:
    import build_speed_prompts as speed
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("model")
    ap.add_argument("oracle_bin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-name", required=True, help="the llama.cpp device's name, e.g. 'AMD Radeon RX 9070 XT'")
    ap.add_argument("--n-max", default="1,2,3,4")
    ap.add_argument("--golden", action="store_true")
    a = ap.parse_args()
    if base.port_in_use():
        sys.exit(f"something already answers on port {PORT}; stop it first")
    device_id = base.pick_device(base.list_devices(pathlib.Path(a.oracle_bin)), a.gpu_name)[0]
    root = pathlib.Path(a.repo)
    n_maxes = [1] if a.golden else [int(x) for x in a.n_max.split(",")]
    if a.golden:
        jobs, path = golden_jobs(root), "/completion"
    else:
        jobs, path = speed_jobs(root, speed.load(root), speed.message), "/v1/chat/completions"
    exe = str(pathlib.Path(a.oracle_bin) / "llama-server.exe")
    def launch(n_max, moe):
        log = pathlib.Path(tempfile.gettempdir()) / f"artemis-mtp-probe-server-n{n_max}-moe{moe}.log"
        with open(log, "w") as lf:
            return subprocess.Popen([exe, "-m", a.model, "--host", "127.0.0.1", "--port", str(PORT),
                                     *server_args(device_id, n_max, moe)], stdout=subprocess.DEVNULL, stderr=lf)

    def ready(proc):
        deadline = time.monotonic() + READY_SECONDS
        while proc.poll() is None:
            if base.healthy():
                return True
            if time.monotonic() > deadline:
                return False
            time.sleep(2)
        return False

    def post(body):
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read())

    out = probe([(label, {**body}, seed) for label, body, seed in jobs], n_maxes, launch, ready, post,
                base.port_in_use)
    out["config"] = ("llama.cpp b11509 llama-server " + " ".join(server_args(device_id, n_maxes[0], CPU_MOE[0]))
                     + " (--n-cpu-moe is each run's own n_cpu_moe)")
    out["mode"] = "golden" if a.golden else "speed"
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    if "error" in out:
        sys.exit(f"{out['error']}; the runs that completed are in {a.out}")
    print("depth rates " + ", ".join(f"{r:.3f}" for r in out.get("depth_rates", [])))


if __name__ == "__main__":
    main()
