"""Runs OpenCode and Claude Code against the recording stub. Standard library only.

The clients are real programs on a machine that holds the user's real credentials. The capture must be unable to reach a
real API, a real host or those credentials, and these guards are what make it so. None of them has an off switch:

  * the child environment is built from an allowlist (`child_env`), never copied, and checked again (`verify_env`)
    before every process this script starts for a client;
  * every path the clients can read (home, config, data, cache, temp, project) is under <root>\\home or <root>\\project;
  * `C:\\Users\\<you>\\.claude\\.credentials.json` is only stat()ed (size and mtime), before and after every run and
    once a second during one; a change stops the task;
  * the stub's ABORT marker (a credential that is not the dummy key reached it) kills the client and stops the task;
  * once a second the client's process tree is joined with `netstat -ano`; a connection to anything but 127.0.0.1
    kills the client and stops the task;
  * a full run refuses to start unless a canary for that client passed, with these exact scripts, and left its marker.

Nothing here prints an environment value or a key. `--print-env-names` prints names only.

Usage: python -I tools/py/capture_clients.py --root T:\\artemis-oracle\\runs\\m3b-capture
           [--only opencode|claude-code] [--canary] [--print-env-names]

Two more modes keep every guard above and point the client at the real `artemis serve` on 127.0.0.1:8090:
  smoke --client C --root R --key-file K --server-pid P --task T [--timeout 900] --out F [--project project|project-2]
      one client, one task, in the scratch repo R\\<project>; the key comes from K, the server is the script's (its pid is P);
      writes the result to F; exit 0 = completed under the guards, 2 = a guard stopped it. It needs the smoke canary's marker.
  smoke-canary --client C --root R
      the same env and config shape against the recording stub with the capture's dummy key; a pass writes the marker
      T:\\artemis-oracle\\runs\\m3c-smoke\\canary-ok-smoke-C, which binds the seven files in SMOKE_BOUND.
  blank-opencode-key --root R
      sets apiKey to "" in OpenCode's two temporary configs under R; the smoke script runs it after a clean key scan.
  --selftest-smoke   checks of the smoke modes' own parts; starts no client.
"""

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
import traceback

DEFAULT_ROOT = "T:" + chr(92) + "artemis-oracle" + chr(92) + "runs" + chr(92) + "m3b-capture"
DUMMY_KEY = "artemis-capture-dummy-key-0000000000000000"
MODEL = "qwen3.6-35b-a3b"
PORT = 8090  # the stub's; never 8080 (the production server listens there)
STUB_HOST = f"127.0.0.1:{PORT}"
STUB_URL = f"http://{STUB_HOST}"
PROXY_URL = "http://127.0.0.1:9"  # the discard port: nothing listens, so a client that honours the proxy fails at once
NOTES = "The build uses three stages: fetch, compile, link."
CLIENTS = ("opencode", "claude-code")
FULL_TIMEOUT, CANARY_TIMEOUT, HELP_TIMEOUT = 180, 60, 60
HERE = os.path.dirname(os.path.abspath(__file__))
RECORD_PY = os.path.join(HERE, "record_clients.py")
SELF_PY = os.path.abspath(__file__)
REPO = os.path.dirname(os.path.dirname(HERE))

# The smoke modes: the clients run against the real `artemis serve` on 8090, which the PowerShell
# scripts start. Every run root is below SMOKE_BASE, and the canary's marker sits directly in it, at a fixed path.
SMOKE_BASE = "T:" + chr(92) + "artemis-oracle" + chr(92) + "runs" + chr(92) + "m3c-smoke"
# The files a smoke canary marker binds, by repo-relative path. A bound file that is missing voids the marker.
SMOKE_BOUND = ("tools/py/capture_clients.py", "tools/py/record_clients.py", "tools/smoke/opencode.ps1",
               "tools/smoke/claude-code.ps1", "tools/smoke/Smoke.psm1", "tools/ps/ArtemisServer.psm1",
               "tools/ps/Python.ps1")
SMOKE_MAX_TIMEOUT = 3600
KEY_SHAPE = re.compile(r"[A-Za-z0-9_=+/.-]{16,256}")
# `claude`'s model aliases, every one pointed at the served model, set by name like the other client variables.
CLAUDE_ALIASES = ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                  "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_SMALL_FAST_MODEL")
# OpenCode's tool permissions in the smoke. In 2.0.30 every agent starts from `{"*": "allow"}` (only a
# read of `.env` files asks), so what is not denied here is allowed, in any directory: the block denies everything first
# and allows back what the task needs. The rules are matched in order and the last match wins (`rules.findLast(...)`),
# so each catch-all comes before the rules that refine it. A tool whose last rule denies every resource is not even
# offered to the model. `external_directory` is asked by the read, edit, glob and grep tools for a path outside the
# working directory and the git root (`FileAccess.resolve`: a path inside either is not external), and denying it keeps
# the agent inside the project. `run` without `--auto` asks nothing: an `ask` is answered "reject" at once.
# The project's own `opencode.json(c)`, `.opencode/`, `.agents/` and `.claude/` would load after OPENCODE_CONFIG and be
# reloaded live, so an agent that could write them could rewrite this block (`{"permission": {"*": "allow"}}`). What stops
# that is `OPENCODE_DISABLE_PROJECT_CONFIG=1` in the smoke's environment (OpenCode then does not discover any project or
# ancestor config); the `edit` rules are a second line, not the guarantee. They are globs on the project-relative,
# `/`-separated resource, anchored at both ends and case-insensitive, `*` matching any characters including `/`: they deny
# the spellings of those paths as written (`.opencode/*` covers nested files) and `.git/*`, and `*:*` denies any resource
# with a colon (a project-relative path has none; an NTFS stream such as `opencode.json::$DATA` or
# `.git::$INDEX_ALLOCATION/x` would otherwise match only `*`). They do not interpret other Windows spellings of a path.
# `list` and `todowrite` are not tools in 2.0.30 and are not here.
OPENCODE_PERMISSION = {
    "*": "deny",
    "read": "allow",
    "glob": "allow",
    "grep": "allow",
    "edit": {
        "*": "allow",
        "*:*": "deny",
        "opencode.json": "deny",
        "opencode.jsonc": "deny",
        ".opencode/*": "deny",
        ".agents/*": "deny",
        ".claude/*": "deny",
        ".git/*": "deny",
    },
    "bash": {"*": "deny", "python -m unittest*": "allow"},
    "external_directory": "deny",
    "webfetch": "deny",
    "websearch": "deny",
    "task": "deny",
}
# Read in 2.0.30 as `config.project: !D(OPENCODE_CONFIG_PROJECT_DISABLE ?? OPENCODE_DISABLE_PROJECT_CONFIG)`; with it set,
# config discovery drops the project's and its ancestors' opencode.json(c), .opencode, .claude and .agents. The
# `--standalone` server child inherits the environment.
OPENCODE_SMOKE_VARS = {"OPENCODE_DISABLE_PROJECT_CONFIG": "1"}
# What the smoke canary asks the client to do: the stub answers with a call of the client's read tool on a sentinel file
# outside the run's root, and the canary passes only if the client refuses it.
CONFINE_PROMPT = "Open the sentinel file and tell me what it says."
# Where Claude Code runs its shell commands on Windows: Git Bash (its Bash tool). Its PowerShell tool is a preview that
# `CLAUDE_CODE_USE_POWERSHELL_TOOL=1` turns on; setting 0 (and denying the tool by name) keeps Bash the one shell tool.
CLAUDE_EXTRA_VARS = {"CLAUDE_CODE_USE_POWERSHELL_TOOL": "0"}
# Claude Code's tool rules in the smoke: a read inside the working directory needs no rule (outside it, a read asks, and
# `-p` answers an ask with a refusal); a bare `Edit` would match every path, so edits are allowed only by the two rules
# rooted at the working directory (a rule from the command line is rooted at the original cwd: `/**` as the binary's own
# advice words it, `./**` as the docs do); the one command is the test command; the network and sub-agent tools, and the
# PowerShell tool, are denied by name. `--permission-mode acceptEdits` was not used: it also auto-allows `rm`, `mv`, `cp`,
# `sed`, `mkdir` and `touch` shell commands.
CLAUDE_ALLOWED = ("Bash(python -m unittest:*)", "Edit(/**)", "Edit(./**)")
CLAUDE_DENIED = ("WebFetch", "WebSearch", "Task", "Agent", "PowerShell")

# The one file this script must never open. It stat()s it (size, mtime) and nothing else.
CREDENTIALS = os.path.join(os.environ.get("USERPROFILE") or os.path.expanduser("~"), ".claude", ".credentials.json")

# From the parent environment, by name, when set there.
SYSTEM_NAMES = ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "PATH", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS")
# Set by the script, all under <root>\home: name -> path below it.
HOME_VARS = {
    "USERPROFILE": "",
    "HOME": "",
    "APPDATA": os.path.join("AppData", "Roaming"),
    "LOCALAPPDATA": os.path.join("AppData", "Local"),
    "TEMP": "Temp",
    "TMP": "Temp",
    "XDG_CONFIG_HOME": ".config",
    "XDG_DATA_HOME": os.path.join(".local", "share"),
    "XDG_STATE_HOME": os.path.join(".local", "state"),
    "XDG_CACHE_HOME": ".cache",
}
PROXY_VARS = {
    **{n: PROXY_URL for n in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")},
    "NO_PROXY": "127.0.0.1,localhost",
    "no_proxy": "127.0.0.1,localhost",
}
# No variable with one of these in its name is copied from the parent. The client switches below are set explicitly,
# by name; they are the only variables with such a name that the child gets.
FORBIDDEN_NAME = re.compile("KEY|TOKEN|SECRET|ANTHROPIC|OPENAI|CLAUDE|OPENCODE", re.I)

PACKAGE_FETCH = re.compile(
    r"\bbun (add|install)\b|\bnpm (install|add)\b|\bnpm (error|err!)|npmjs\.|registry\.(npm|yarn)|"
    r"\binstalling (package|provider|dependenc)|releases/download|ripgrep",
    re.I,
)
# Set by name for both clients (the system gitconfig names a credential helper that could prompt or look up a login).
GIT_VARS = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}
OPENCODE_SERVICE_PORT = "45278"  # the fixed loopback port of OpenCode's background service
SURVIVOR_GRACE = 5  # seconds a process that outlives the client's root process is still watched, then killed
SWITCHES = ("OPENCODE_DISABLE_MODELS_FETCH", "OPENCODE_DISABLE_AUTOUPDATE", "OPENCODE_DISABLE_CLAUDE_CODE")
# Both clients read these from ancestors of the project directory.
ANCESTOR_NAMES = ("AGENTS.md", "CLAUDE.md", "CLAUDE.local.md", ".claude", ".opencode", "opencode.json", "opencode.jsonc")


class Stop(Exception):
    """A guard fired or a check failed: the task stops and is reported. The message never holds a secret."""


def say(msg=""):
    print(msg, flush=True)


# ---- paths ----


def home_dir(root):
    return os.path.join(root, "home")


def project_dir(root):
    return os.path.join(root, "project")


def norm(p):
    return os.path.normcase(os.path.abspath(p))


def is_under(path, base):
    """True when `path` is `base` or inside it (a pure string test on absolute paths)."""
    if not os.path.isabs(path):
        return False
    a, b = norm(path), norm(base)
    try:
        return os.path.commonpath([a, b]) == b
    except ValueError:  # another drive
        return False


def under_root_everywhere(path, base):
    """`is_under`, also after resolving links when the path exists."""
    return is_under(path, base) and (not os.path.exists(path) or is_under(os.path.realpath(path), os.path.realpath(base)))


# ---- the environment ----


def base_env(root, parent=None):
    """The allowlisted part both for clients and for the helper processes: system names, the redirected home."""
    parent = os.environ if parent is None else parent
    upper = {k.upper(): v for k, v in parent.items()}
    env = {}
    for name in SYSTEM_NAMES:
        assert not FORBIDDEN_NAME.search(name)
        if name in upper:
            env[name] = upper[name]
    home = home_dir(root)
    for name, below in HOME_VARS.items():
        env[name] = os.path.join(home, below) if below else home
    return env


def client_vars(client, root, key=DUMMY_KEY, smoke=False):
    """The client's own variables, set by name. The values here are the only places the key appears in an environment:
    the dummy key in the capture, the run's key (read from its file) in the smoke. `smoke` adds Claude Code's model aliases."""
    home = home_dir(root)
    if client == "opencode":
        out = {
            "OPENCODE_DISABLE_MODELS_FETCH": "1",
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
            "OPENCODE_DISABLE_CLAUDE_CODE": "1",
            "OPENCODE_CONFIG": os.path.join(home, "opencode.json"),
        }
        if smoke:
            out.update(OPENCODE_SMOKE_VARS)
        return out
    if client == "claude-code":
        out = {
            "CLAUDE_CONFIG_DIR": os.path.join(home, "claude"),
            "ANTHROPIC_BASE_URL": STUB_URL,
            "ANTHROPIC_AUTH_TOKEN": key,
            "ANTHROPIC_MODEL": MODEL,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "DISABLE_AUTOUPDATER": "1",
        }
        if smoke:
            out.update({name: MODEL for name in CLAUDE_ALIASES})
            out.update(CLAUDE_EXTRA_VARS)
        return out
    raise ValueError(client)


def child_env(client, root, parent=None, key=DUMMY_KEY, smoke=False):
    """The environment a client gets. Pure: reads `parent` (default os.environ), touches nothing else."""
    env = base_env(root, parent)
    env.update(PROXY_VARS)
    env.update(GIT_VARS)
    env.update(client_vars(client, root, key, smoke))
    return env


def verify_env(env, root, client, key=DUMMY_KEY, smoke=False):
    """Checks a child environment before use: only allowlisted names, every path under <root>\\home, proxies set."""
    mine = client_vars(client, root, key, smoke)
    allowed = set(SYSTEM_NAMES) | set(HOME_VARS) | set(PROXY_VARS) | set(GIT_VARS) | set(mine)
    for name in env:
        if name not in allowed:
            raise Stop("the child environment holds a variable outside the allowlist")
        if FORBIDDEN_NAME.search(name) and name not in mine:
            raise Stop("the child environment holds a forbidden-name variable")
    home = home_dir(root)
    for name in HOME_VARS:
        if name not in env or not under_root_everywhere(env[name], home):
            raise Stop(f"{name} is not under <root>\\home")
    for name in ("CLAUDE_CONFIG_DIR", "OPENCODE_CONFIG"):
        if name in mine and not under_root_everywhere(env.get(name, ""), home):
            raise Stop(f"{name} is not under <root>\\home")
    for name, value in {**PROXY_VARS, **GIT_VARS}.items():
        if env.get(name) != value:
            raise Stop("a proxy-isolation or git variable is missing or changed")
    for name, value in mine.items():
        if env.get(name) != value:
            raise Stop("a client variable is missing or changed")


def leak_check(parent, env, argv, key=None):
    """No secret-named parent variable's value may appear in the child's environment or command line, and the run's own
    key (smoke) may appear in its environment and configs but never on a command line."""
    secrets = [v for k, v in parent.items() if FORBIDDEN_NAME.search(k) and len(v) >= 12]
    haystack = list(env.values()) + list(argv)
    for s in secrets:
        if any(s in h for h in haystack):
            raise Stop("a value from a secret-named parent variable reached a child")
    if key and any(key in a for a in argv):
        raise Stop("the run's key is on a client's command line")


def print_env_names():
    say("child_env names only; no value is read out or printed")
    for client in CLIENTS:
        env = child_env(client, DEFAULT_ROOT)
        say(f"{client}:")
        for name in sorted(env, key=str.upper):
            how = "copied from the parent" if name in SYSTEM_NAMES else "set by the script"
            say(f"  {name}  ({how})")


# ---- the guards ----


def creds_stat():
    """(size, mtime_ns) of the credentials file, or None when it is absent. Never opens it."""
    try:
        st = os.stat(CREDENTIALS)
    except FileNotFoundError:
        return None
    return (st.st_size, st.st_mtime_ns)


def check_ancestors(project):
    d = os.path.dirname(os.path.abspath(project))
    while True:
        for name in ANCESTOR_NAMES:
            if os.path.exists(os.path.join(d, name)):
                raise Stop(f"{name} exists in {d}: a client would read it from an ancestor of the project")
        up = os.path.dirname(d)
        if up == d:
            return
        d = up


def script_hashes():
    out = {}
    for path in (SELF_PY, RECORD_PY):
        with open(path, "rb") as f:
            out[os.path.basename(path)] = hashlib.sha256(f.read()).hexdigest()
    return out


def marker_path(root, client):
    return os.path.join(root, f"canary-ok-{client}")


def require_canary(root, client):
    """The gate: a full run needs the marker a passing canary wrote, made with these exact scripts."""
    path = marker_path(root, client)
    if not os.path.isfile(path):
        raise Stop(f"canary-ok-{client} is missing: run --canary first; a full run does not start without it")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        raise Stop(f"canary-ok-{client} is unreadable")
    if data.get("scripts") != script_hashes():
        raise Stop(f"the scripts changed after the {client} canary: run --canary again")


def smoke_hashes(repo=REPO):
    """SHA-256 of every file the smoke marker binds, by repo-relative path. A missing file is a Stop: the marker fails closed."""
    out = {}
    for rel in SMOKE_BOUND:
        path = os.path.join(repo, *rel.split("/"))
        if not os.path.isfile(path):
            raise Stop(f"{rel} is missing: a smoke canary marker binds it, so no marker holds without it")
        with open(path, "rb") as f:
            out[rel] = hashlib.sha256(f.read()).hexdigest()
    return out


def smoke_marker_path(client, base=SMOKE_BASE):
    """The one place a smoke marker is written and read: directly in the smoke base, never under a run's root."""
    return os.path.join(base, f"canary-ok-smoke-{client}")


def require_smoke_canary(client, base=SMOKE_BASE, repo=REPO):
    """The smoke gate: a real run needs the marker a passing smoke canary wrote at the fixed path, made with these exact files."""
    path = smoke_marker_path(client, base)
    if not os.path.isfile(path):
        raise Stop(f"canary-ok-smoke-{client} is missing: run smoke-canary first; a smoke run does not start without it")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        raise Stop(f"canary-ok-smoke-{client} is unreadable")
    current = smoke_hashes(repo)  # a bound file that is missing voids the marker
    if not isinstance(data, dict) or data.get("client") != client or data.get("scripts") != current:
        raise Stop(f"the bound files changed after the {client} smoke canary: run smoke-canary again")
    return data


def require_client_version(client, found, marker_version):
    """The marker records the client version the canary ran; a client that has updated since is not the one it vouched for."""
    if not marker_version or found != marker_version:
        raise Stop(f"the {client} version ({found or 'unread'}) is not the one the smoke canary ran "
                   f"({marker_version or 'not recorded'}): run smoke-canary again")


def check_unchanged(before, hashes=smoke_hashes):
    """The bound files are what they were when the canary began (`before`): a file edited while the canary ran voids it."""
    if hashes() != before:
        raise Stop("a bound file changed while the smoke canary ran: no marker is written")


def require_creds(creds):
    """In a smoke mode the credentials file must exist: its absence would make 'unchanged' mean nothing."""
    if creds is None:
        raise Stop("the credentials file is absent, so a change to it could not be seen")


def write_smoke_marker(client, base, repo, extra, root=None, hashes=None):
    """Writes the marker at the fixed path. It binds all of SMOKE_BOUND (`hashes`, taken when the canary began, or the
    files now), so it cannot be written while one is missing."""
    path = smoke_marker_path(client, base)
    if root is not None and is_under(path, root):
        raise Stop("the smoke marker would sit under the run's root")
    data = {"client": client, "scripts": hashes if hashes is not None else smoke_hashes(repo), **extra}
    os.makedirs(base, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2)
    return path


def redact(text, *secrets):
    """Hides the dummy key, the run's own key and anything key- or token-shaped. `secrets` are values of at least 8 chars."""
    for s in secrets:
        if s and len(s) >= 8:
            text = text.replace(s, "<key>")
    text = text.replace(DUMMY_KEY, "<dummy>")
    text = re.sub(r"sk-ant-[\w-]+", "<redacted>", text)
    text = re.sub(r"sk-[\w-]{16,}", "<redacted>", text)
    text = re.sub(r"gh[pousr]_\w+", "<redacted>", text)
    return re.sub(r"(?i)\bbearer\s+\S+", "Bearer <redacted>", text)


def system32(env, *parts):
    return os.path.join(env.get("SYSTEMROOT") or env.get("WINDIR") or r"C:\Windows", "System32", *parts)


def filetime_now():
    """The current time as a FILETIME (100 ns ticks since 1601), the unit of `process_table`'s creation times."""
    return int(time.time() * 10_000_000) + 116444736000000000


def kill_pids(env, pids):
    """taskkill /F each pid, never its descendants (no /T): only Tree members, matched by creation time in the listing the
    caller just took, are killed."""
    for pid in pids:
        try:
            subprocess.run([system32(env, "taskkill.exe"), "/PID", str(pid), "/F"], env=env,
                           capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            pass


def kill_tree(proc, env):
    """The root process only. Its descendants are Tree members and are the job of `Session.kill_all`."""
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass


class Table(dict):
    """pid -> (parent pid, creation time as a FILETIME integer); `.names` maps pid -> image name (for the logs only)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.names = {}


def process_table(env):
    """A `Table` of every process that has a creation time."""
    ps = system32(env, "WindowsPowerShell", "v1.0", "powershell.exe")
    cmd = ("Get-CimInstance Win32_Process | "
           "Select-Object ProcessId,ParentProcessId,@{n='Created';e={$_.CreationDate.ToFileTimeUtc()}},Name")
    try:
        r = subprocess.run([ps, "-NoProfile", "-Command", cmd], env=env, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        raise Stop("the process listing failed, so the connection watch cannot run")
    table = Table()
    for a, b, c, name in re.findall(r"^\s*(\d+)\s+(\d+)\s+(\d{10,})\s+(\S.*?)\s*$", r.stdout, re.M):
        table[int(a)] = (int(b), int(c))
        table.names[int(a)] = name
    if r.returncode != 0 or len(table) < 10:
        raise Stop("the process listing failed, so the connection watch cannot run")
    return table


ADOPT_SLACK = 30_000_000  # 3 s in FILETIME ticks: about one poll (a listing takes a second or two)


class Tree:
    """A client's process tree, cumulative. Every (pid, creation time) ever seen below the root stays a member, so the
    children of an intermediate process that has exited are still found, and a pid reused by another process is not
    mistaken for a member (its creation time differs).

    A child of a member that is alive is adopted. A child of a member that has died is adopted only if it was created no
    later than one poll after the member was last seen alive: a process that merely inherited a reused pid, and its
    children, appear long after, so `reap` and `kill_all` never kill an unrelated process on that account."""

    def __init__(self, root_pid, launched):
        self.root = root_pid
        # pid -> creation time, dead members included. Until the first listing shows the root, its creation time is the
        # moment before it was launched, so a root that exits before the first poll still has its children adopted.
        self.members = {root_pid: launched}
        self.last_seen = {root_pid: launched + 10_000_000}  # pid -> when it was last seen alive (FILETIME)
        self.root_seen = False
        self.live = set()  # the members alive at the last successful listing
        self.prev = launched + 10_000_000  # when the last listing began (the launch, until the first)

    def update(self, table, now, strict=True):
        """Adds what `table` (from `process_table`, taken at `now`) shows below the members. Returns the live pids.
        A gap of more than `ADOPT_SLACK` since the last listing could hide an adoption, so a strict update refuses it."""
        gap = now - self.prev
        self.prev = now
        if strict and gap > ADOPT_SLACK:
            raise Stop(f"the watch fell behind: {gap / 10_000_000:.1f} s between process listings (the limit is "
                       f"{ADOPT_SLACK / 10_000_000:.0f} s), so a child could be missed")
        m = self.members
        if not self.root_seen and self.root in table:
            m[self.root] = table[self.root][1]
            self.root_seen = True
        grew = True
        while grew:
            grew = False
            for pid, (ppid, created) in table.items():
                if pid in m or ppid not in m:
                    continue
                if ppid in table:
                    if table[ppid][1] != m[ppid]:
                        continue  # the parent's pid was reused: this is not its child
                elif created > self.last_seen.get(ppid, now) + ADOPT_SLACK:
                    continue  # the parent was dead long before this process began
                if created >= m[ppid]:
                    m[pid] = created
                    grew = True
        live = {pid for pid, created in m.items() if pid in table and table[pid][1] == created}
        for pid in live:
            self.last_seen[pid] = now
        self.live = live
        return live


def tcp_rows(env):
    """(pid, local, remote, state) for every TCP row of `netstat -ano`."""
    try:
        r = subprocess.run([system32(env, "netstat.exe"), "-ano"], env=env, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        raise Stop("netstat failed, so the connection watch cannot run")
    rows = []
    for line in r.stdout.splitlines():
        p = line.split()
        if len(p) == 5 and p[0].upper() == "TCP" and p[4].isdigit():
            rows.append((int(p[4]), p[1], p[2], p[3]))
    if r.returncode != 0 or not rows:
        raise Stop("netstat returned nothing, so the connection watch cannot run")
    return rows


def split_addr(addr):
    host, _, port = addr.rpartition(":")
    return host.strip("[]"), port


WHY_LOOPBACK = "a loopback port that is not the stub's, the proxy's or the client's own"


CLOSING_STATES = ("CLOSE_WAIT", "FIN_WAIT_1", "FIN_WAIT_2", "TIME_WAIT", "LAST_ACK", "CLOSING")


def owned_endpoints(rows, live):
    """The loopback endpoints (ip:port) that members of the tree own in these rows: a socket's local end, and a listener's
    (a listener on 0.0.0.0 owns the 127.0.0.1 endpoint too). Never `[::]`, which is another address family."""
    out = set()
    for pid, local, _, _ in rows:
        if pid not in live:
            continue
        host, port = split_addr(local)
        if host in ("127.0.0.1", "0.0.0.0"):
            out.add(f"127.0.0.1:{port}")
    return out


def outsider_owner(rows, known, remote):
    """The first process outside the tree that owns the endpoint `remote` in these rows, as its row, or None: a row whose
    local address is exactly `remote`, or a LISTENING row on 127.0.0.1:P or 0.0.0.0:P with P that port. `known` is every
    pid of the tree, dead members included (netstat is read before the process listing, so a member that has just exited
    still has rows). Pid 0 (the closing sockets nobody owns any more) does not count."""
    _, port = split_addr(remote)
    for r in rows:
        if r[0] in known or r[0] == 0:
            continue
        host_l, port_l = split_addr(r[1])
        if r[1] == remote or (r[3].upper() == "LISTENING" and port_l == port and host_l in ("0.0.0.0", "127.0.0.1")):
            return r
    return None


def classify_rows(rows, live, owned=frozenset(), names=None, known=None):
    """Splits the tree's TCP rows into (accepted, refused), each a list of (row, reason).

    Accepted: no peer at all (listening); the stub, 127.0.0.1:8090; the discard proxy port, 127.0.0.1:9; and a loopback
    peer that is itself a socket of a process in the tree. Port 8080 (the production server) is refused before that last
    rule.
    An ESTABLISHED row needs its peer row (local and remote swapped) to be in the tree's rows, or else its local port to be a
    tree listener's and its remote endpoint to be in `owned` (the tree's own service, whose client just exited: an inbound
    connection from an endpoint a member owned; a foreign process connecting in is refused); a SYN_SENT row has no peer
    yet, so it needs a tree listener on that port at 0.0.0.0 or 127.0.0.1 (never `[::]`, which is another address
    family); a closing state (CLOSE_WAIT, FIN_WAIT_1/2, TIME_WAIT, LAST_ACK, CLOSING), whose peer may have exited, takes
    either of those or a remote endpoint in `owned`, the set of loopback endpoints a member of the tree has owned at any
    poll (the cumulative set kept by the Session). `owned` rescues a row only if no live process outside the tree owns that
    far end right now (a foreign process may have taken a port a member once used): `outsider_owner`, named in the refusal
    with its image from `names`. Everything else is refused: another loopback port (the clients embed
    11434 and 1234), any other address, and ::1."""
    mine = [r for r in rows if r[0] in live]
    known = live if known is None else known
    pairs = {(r[1], r[2]) for r in mine}
    listeners = {split_addr(r[1])[1] for r in mine
                 if r[3].upper() == "LISTENING" and split_addr(r[1])[0] in ("0.0.0.0", "127.0.0.1")}
    accepted, refused = [], []
    for row in mine:
        _, local, remote, state = row
        host, port = split_addr(remote)
        st = state.upper()
        peer_in_tree = (remote, local) in pairs
        foe = outsider_owner(rows, known, remote) if remote in owned else None
        if st == "LISTENING" or (host in ("0.0.0.0", "::", "*") and port in ("0", "*")):
            accepted.append((row, "listening or unconnected"))
        elif host == "127.0.0.1" and port == str(PORT):
            accepted.append((row, "the stub"))
        elif host == "127.0.0.1" and port == "9":
            accepted.append((row, "the discard proxy port"))
        elif host == "127.0.0.1" and port == "8080":
            refused.append((row, "127.0.0.1:8080 is the production server's port"))
        elif host == "127.0.0.1" and (
            (peer_in_tree or (split_addr(local)[1] in listeners and remote in owned and foe is None))
            if st == "ESTABLISHED"
            else port in listeners if st == "SYN_SENT"
            else peer_in_tree or port in listeners or (st in CLOSING_STATES and remote in owned and foe is None)
        ):
            accepted.append((row, "client-internal loopback"))
        elif host == "127.0.0.1":
            why = WHY_LOOPBACK
            if foe is not None:
                why += f" (its far end belongs to pid {foe[0]}, image {(names or {}).get(foe[0], '?')}, outside the tree)"
            refused.append((row, why))
        else:
            refused.append((row, "an address beyond 127.0.0.1"))
    return accepted, refused


def accepted_endpoints(accepted):
    """The remote endpoints of the accepted rows (from `classify_rows`) that are the server's and the discard proxy's:
    (hosts, proxy hosts). The two are reported apart: the proxy port is where a client's non-loopback attempt dies."""
    hosts = {r[2] for r, _ in accepted if r[2] == STUB_HOST}
    proxies = {r[2] for r, _ in accepted if r[2] == "127.0.0.1:9"}
    return hosts, proxies


def check_8090_owner(rows, server_pid):
    """In a smoke run the server is outside the client's tree, so the watch accepts 127.0.0.1:8090 by its port alone. The
    LISTENING rows on that port must be the server's, on 127.0.0.1, and there must be one: a squatter, a second listener or
    a replacement stops the run. Rows that are connections, not listeners, are the clients' and the server's own."""
    mine = False
    for pid, local, _, state in rows:
        host, port = split_addr(local)
        if state.upper() != "LISTENING" or port != str(PORT):
            continue
        if pid != server_pid or host != "127.0.0.1":
            raise Stop(f"port {PORT} has a listener on {local} (pid {pid}) that is not 127.0.0.1:{PORT} of the server pid {server_pid}")
        mine = True
    if not mine:
        raise Stop(f"nothing listens on 127.0.0.1:{PORT}: the server (pid {server_pid}) is not up")


def check_server_image(names, server_pid):
    """The pid the script names must be an `artemis` process (the process listing's image name)."""
    name = names.get(server_pid)
    if name is None or not name.lower().startswith("artemis"):
        raise Stop(f"the server pid {server_pid} is not an artemis process (image {name or 'absent'})")


def install_marks(home):
    """Relative paths of package-manager or tool content under `home`: node_modules, rg.exe, non-empty bin dirs."""
    hits = set()
    for dirpath, dirs, files in os.walk(home):
        for d in list(dirs):
            full = os.path.join(dirpath, d)
            if d.lower() == "node_modules":
                hits.add(os.path.relpath(full, home))
                dirs.remove(d)
            elif d.lower() == "bin" and os.listdir(full):
                hits.add(os.path.relpath(full, home))
        for f in files:
            if f.lower() == "rg.exe":
                hits.add(os.path.relpath(os.path.join(dirpath, f), home))
    return hits


# ---- the capture root ----


def safe_root(root):
    """The capture root is outside this repo and outside the user's profile."""
    if not os.path.isabs(root) or os.path.dirname(os.path.abspath(root)) == os.path.abspath(root):
        raise Stop("the capture root must be an absolute path below a drive root")
    profile = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    repo = os.path.dirname(os.path.dirname(HERE))
    for base, what in ((profile, "the user profile"), (repo, "the repository")):
        if is_under(root, base) or is_under(base, root):
            raise Stop(f"the capture root overlaps {what}")


def fresh_dir(path):
    """An empty directory at `path`; an earlier non-empty one is renamed aside, never deleted."""
    if os.path.isdir(path) and os.listdir(path):
        os.rename(path, os.path.join(os.path.dirname(path), "_old-" + time.strftime("%Y%m%d-%H%M%S-") + os.path.basename(path)))
    os.makedirs(path, exist_ok=True)
    return path


def rename_aside(path):
    """An earlier file at `path` is renamed aside, never overwritten or deleted. A name taken in the same second gets a counter."""
    if os.path.exists(path):
        stamp = os.path.join(os.path.dirname(path), "_old-" + time.strftime("%Y%m%d-%H%M%S-") + os.path.basename(path))
        target, n = stamp, 1
        while os.path.exists(target):
            n += 1
            target = f"{stamp}.{n}"
        os.rename(path, target)


def machine_config_dirs(env=None):
    """Where OpenCode 2.0.30 reads its machine-wide (managed) config on Windows: `<ProgramData>\\opencode`, where the
    child's own environment has no ProgramData and so falls back to C:\\ProgramData\\opencode. Both are checked, the
    parent's ProgramData first."""
    env = os.environ if env is None else env
    upper = {k.upper(): v for k, v in env.items()}
    out = [os.path.join(upper["PROGRAMDATA"], "opencode")] if upper.get("PROGRAMDATA") else []
    out.append("C:" + chr(92) + "ProgramData" + chr(92) + "opencode")
    return out


def check_project_files(project):
    """OpenCode is a Bun standalone build, which loads `.env*` files and `bunfig.toml` from the working directory by default
    (a bunfig preload would run code). The scratch repo has none, and the run does not start if one appeared."""
    try:
        names = os.listdir(project)
    except OSError:
        return
    for name in names:
        low = name.lower()
        if low.startswith(".env") or low == "bunfig.toml":
            raise Stop(f"{name} exists in {project}: a Bun program reads it from its working directory")


def check_machine_config(dirs=None):
    """OpenCode loads a machine-wide config last, over the home redirect and the temporary config, and nothing the script
    sets can move it. If one exists the run does not start (the same stance as `check_ancestors`)."""
    for d in machine_config_dirs() if dirs is None else dirs:
        if os.path.exists(d):
            raise Stop(f"{d} exists: OpenCode reads a machine-wide config from there after its temporary one")


def make_home_tree(root):
    home = home_dir(root)
    for below in ("", "claude", *HOME_VARS.values(), os.path.join(".config", "opencode")):
        os.makedirs(os.path.join(home, below) if below else home, exist_ok=True)


def scratch_git(root, project, paths):
    """`git init` and one commit of `paths` in `project`, unless it is already a repository."""
    if os.path.isdir(os.path.join(project, ".git")):
        return
    git = shutil.which("git")
    if not git:
        raise Stop("git is not on PATH")
    env = {**base_env(root), **GIT_VARS, "GIT_CONFIG_GLOBAL": "NUL"}
    ident = ["-c", "user.name=capture", "-c", "user.email=capture@example.invalid", "-c", "commit.gpgsign=false"]
    for args in (["init", "-q"], ["add", *paths], ["commit", "-q", "-m", "scratch repo"]):
        r = subprocess.run([git, *ident, *args], cwd=project, env=env, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise Stop(f"git {args[0]} failed in the scratch repo: {redact(r.stderr.strip())[:300]}")


def prepare_root(root):
    safe_root(root)
    make_home_tree(root)
    for name in ("raw", "logs", "project"):
        os.makedirs(os.path.join(root, name), exist_ok=True)
    key_file = os.path.join(root, "dummy-key.txt")
    with open(key_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(DUMMY_KEY + "\n")
    project = project_dir(root)
    with open(os.path.join(project, "notes.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(NOTES + "\n")
    scratch_git(root, project, ["notes.txt"])
    check_ancestors(project)
    return key_file


def prepare_smoke_root(root, project_name):
    """The smoke's home tree and logs. The scratch repo is the script's: it must exist, and is made a repository only if it
    is not one. No notes.txt and no dummy-key.txt are written."""
    safe_root(root)
    make_home_tree(root)
    os.makedirs(os.path.join(root, "logs"), exist_ok=True)
    project = os.path.join(root, project_name)
    if not os.path.isdir(project):
        raise Stop(f"the scratch repo {project_name} is missing: the script makes it before the client starts")
    scratch_git(root, project, ["-A"])
    check_ancestors(project)


def write_opencode_config(root, key=DUMMY_KEY, smoke=False):
    """OpenCode's temporary config, written to the two places it can read it from (both under <root>\\home). The key is
    the one place a key sits on disk; the smoke's adds the tool permissions (`OPENCODE_PERMISSION`)."""
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False,
        "share": "disabled",
        "model": f"artemis/{MODEL}",
        "small_model": f"artemis/{MODEL}",
        "provider": {
            "artemis": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Artemis",
                "options": {"baseURL": f"{STUB_URL}/v1", "apiKey": key},
                "models": {
                    MODEL: {"name": MODEL, "tool_call": True, "limit": {"context": 65536, "output": 16384}},
                },
            }
        },
    }
    if smoke:
        cfg["permission"] = json.loads(json.dumps(OPENCODE_PERMISSION))
    for path in opencode_config_paths(root):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")


def opencode_config_paths(root):
    """The two places OpenCode's temporary config sits, both under <root>\\home."""
    env = child_env("opencode", root)
    paths = (env["OPENCODE_CONFIG"], os.path.join(env["XDG_CONFIG_HOME"], "opencode", "opencode.json"))
    for path in paths:
        if not under_root_everywhere(path, home_dir(root)):
            raise Stop("an OpenCode config path is not under <root>\\home")
    return paths


def blank_opencode_key(root):
    """Sets `apiKey` to "" in each of OpenCode's two configs under <root>\\home that exists, and changes nothing else in
    them. The smoke calls it once the key scan has passed, so the run's key does not stay on disk in the scratch root."""

    def blank(node):
        if isinstance(node, dict):
            return {k: ("" if k == "apiKey" else blank(v)) for k, v in node.items()}
        if isinstance(node, list):
            return [blank(v) for v in node]
        return node

    for path in opencode_config_paths(root):
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(blank(cfg), f, indent=2)
            f.write("\n")


# ---- one client, guarded ----


class Smoke:
    """What the smoke modes add to a Session: the run's key (read from its file, or the canary's dummy key), the pid of the
    real server (None in the canary, whose server is the stub) and the scratch repo's directory name below the root."""

    def __init__(self, key, server_pid=None, project="project", task=None, timeout=None, sentinel=None):
        self.key, self.server_pid, self.project, self.task, self.timeout = key, server_pid, project, task, timeout
        self.sentinel = sentinel  # the canary's file outside the run's root, which the stub has the client's read tool ask for


class TimedOut(Stop):
    """The run's own timeout: a Stop (the tree is killed) that the smoke's result file names."""


class Session:
    """One client's runs against one server: the recording stub (the capture and the canaries) or, in a smoke run, the real
    `artemis serve` that a script started. Every process it starts goes through `run`, and construction is the gate."""

    def __init__(self, root, client, canary, key_file, creds_before, smoke=None):
        self.marker_version = None  # the client version the smoke canary recorded; set only for a real smoke run
        if smoke is not None and not canary:
            # the smoke's gate: the marker at the fixed path, bound to the seven files
            self.marker_version = require_smoke_canary(client).get("version")
        elif not canary:
            require_canary(root, client)  # the canary gate: no full run without it
        self.root, self.client, self.canary = root, client, canary
        self.smoke = smoke
        self.real = smoke is not None and not canary  # a smoke run against the real server: no stub
        self.key = smoke.key if smoke else DUMMY_KEY
        self.project = os.path.join(root, smoke.project if smoke else "project")
        self.key_file = key_file
        self.creds_before = creds_before
        self.label = "canary" if canary else ("smoke" if smoke else "full")
        self.parent = dict(os.environ)
        self.base = base_env(root)
        self.env = child_env(client, root, key=self.key, smoke=smoke is not None)
        verify_env(self.env, root, client, key=self.key, smoke=smoke is not None)
        self.out = os.path.join(root, "raw", f"smoke-{client}") if self.real else fresh_dir(
            os.path.join(root, "raw", f"canary-{client}" if canary else client))
        self.report = os.path.join(self.out, "report.json")
        self.logs = os.path.join(root, "logs")
        self.stub = None
        self.polls = 0
        self.started_wall = None
        self.ports_seen = set()
        self.owned = set()  # loopback endpoints a Tree member has owned at any poll (for the closing states)
        self.installs = set()
        self.runs = []  # (label, exit code)
        self.hosts, self.proxy_hosts = set(), set()  # the server's and the discard proxy's endpoints the watch accepted
        self.refused = []  # what the watch refused, as the log describes it (a refusal is also a Stop)
        self.version = ""
        self.last_seconds, self.seconds, self.exit = 0.0, 0.0, None  # of the last run; of the smoke's run and its exit code
        # The path the stub has the client's read tool ask for: notes.txt in the capture; in the smoke canary, the sentinel.
        self.read_path = smoke.sentinel if smoke is not None and smoke.sentinel else os.path.join(self.project, "notes.txt")
        self.exe = self.find_exe()

    def bind_version(self, version):
        """Records the client's version. A real smoke run Stops here, before the run, if it is not the canary's."""
        self.version = version
        if self.real:
            require_client_version(self.client, version, self.marker_version)

    def tag(self, name):
        """A real smoke run's log names carry its scratch repo's name, so attempt 2 never overwrites attempt 1's logs."""
        return f"{name}-{self.smoke.project}" if self.real else name

    def find_exe(self):
        if self.client == "opencode":
            app = self.parent.get("LOCALAPPDATA") or ""
            exe = os.path.join(app, "Programs", "@opencodedesktop", "resources", "opencode-cli.exe")
        else:
            exe = shutil.which("claude", path=self.parent.get("PATH")) or ""
        if not exe or not os.path.isfile(exe):
            raise Stop(f"the {self.client} executable was not found")
        if is_under(exe, self.root):
            raise Stop("the client executable is inside the capture root")
        return exe

    def __enter__(self):
        self.started_wall = time.time()  # files that prove the redirection must be newer than this
        if self.real:
            # The server is the script's, already up: its listener on 8090 must be the pid the script named.
            self.guard("before the run")
            self.check_server()
            self.installs = install_marks(home_dir(self.root))
            return self
        sock = socket.socket()
        try:
            sock.bind(("127.0.0.1", PORT))
        except OSError:
            raise Stop(f"127.0.0.1:{PORT} is busy")
        finally:
            sock.close()
        self.guard("before the stub starts")
        self.installs = install_marks(home_dir(self.root))
        errlog = open(os.path.join(self.logs, f"stub-{self.label}-{self.client}.log"), "wb")
        argv = [sys.executable, "-I", RECORD_PY, "--port", str(PORT), "--key-file", self.key_file, "--out", self.out,
                "--read-path", self.read_path, "--report", self.report]
        self.stub = subprocess.Popen(argv, env=self.base, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=errlog)
        errlog.close()
        for _ in range(100):
            if self.stub.poll() is not None:
                raise Stop("the stub exited at startup (see logs\\stub-*.log)")
            try:
                socket.create_connection(("127.0.0.1", PORT), timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise Stop("the stub did not come up")
        return self

    def __exit__(self, *exc):
        if self.stub is not None and self.stub.poll() is None:
            self.stub.terminate()
            try:
                self.stub.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.stub.kill()
                self.stub.wait(timeout=10)
        # The guards run once more with the stub down, whatever happened.
        if exc[0] is None:
            self.guard("after the run" if self.real else "after the stub stopped")
        return False

    def check_server(self):
        """Before a real run: 8090's only listener is 127.0.0.1:8090 of the pid the script named, and that pid is `artemis`."""
        check_8090_owner(tcp_rows(self.base), self.smoke.server_pid)
        check_server_image(getattr(process_table(self.base), "names", {}), self.smoke.server_pid)

    def read_report(self):
        try:
            with open(self.report, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {"requests": 0, "classes": {"dummy": 0, "none": 0, "other": 0}, "hosts": [], "abort": False}

    def guard(self, when):
        """Checked before and after every run: credentials untouched, env under the root, no ABORT."""
        if creds_stat() != self.creds_before:
            raise Stop(f"the credentials file's size or mtime changed ({when}); a token refresh would do that")
        verify_env(self.env, self.root, self.client, key=self.key, smoke=self.smoke is not None)
        if os.path.exists(os.path.join(self.out, "ABORT")) or self.read_report().get("abort"):
            raise Stop(f"the stub's ABORT marker is set ({when}): a credential that is not the dummy key reached it")
        check_ancestors(self.project)
        if self.client == "opencode" and self.smoke is not None:
            check_machine_config()  # before every launch of the smoke or its canary, and after
            check_project_files(self.project)

    def conn_log(self, tag, lines):
        path = os.path.join(self.logs, f"connections-{self.client}-{self.label}.txt")
        with open(path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(f"{tag}: {line}\n")

    def watch(self, proc, tree, log_path, tag):
        """One poll: the other guards, the client log for package fetches, then the whole tree's connections.
        Returns the live processes of the tree other than the root."""
        if creds_stat() != self.creds_before:
            raise Stop("the credentials file's size or mtime changed while the client ran")
        if os.path.exists(os.path.join(self.out, "ABORT")):
            raise Stop("the stub's ABORT marker appeared while the client ran: a real credential reached it")
        with open(log_path, "rb") as f:
            text = f.read().decode("utf-8", "replace")
        hit = PACKAGE_FETCH.search(text)
        if hit:
            raise Stop(f"the client output shows a package fetch or install ({hit.group(0)!r})")
        rows = tcp_rows(self.base)  # netstat first: every pid in these rows is in the listing taken next, unless it died
        if self.real:
            check_8090_owner(rows, self.smoke.server_pid)  # the watch accepts 8090 by port: it must still be the server's
        live = self.snapshot(tree)
        self.owned |= owned_endpoints(rows, live)
        accepted, refused = classify_rows(rows, live, self.owned, getattr(self.table, "names", None), self.known(tree))
        if any(why == WHY_LOOPBACK for _, why in refused):
            accepted, refused, rows = self.recheck(tree, tag, accepted, refused, rows)
        self.polls += 1
        hosts, proxies = accepted_endpoints(accepted)
        self.hosts |= hosts
        self.proxy_hosts |= proxies
        fresh = []
        for (pid, local, remote, state), why in accepted + refused:
            key = (local, remote, state)
            if key not in self.ports_seen and "127.0.0.1" in local + remote:  # every loopback endpoint seen, once
                self.ports_seen.add(key)
                fresh.append(f"pid {pid} {local} -> {remote} {state}: {why}")
        self.conn_log(tag, fresh)
        if refused:
            self.refused += [f"{r[1]} -> {r[2]} {r[3]} ({why})" for r, why in refused]
            self.conn_log(tag, [f"REFUSED ({why}): {self.describe(r, rows)}" for r, why in refused])
            raise Stop("the client's process tree made a connection the watch refuses: "
                       + "; ".join(sorted({f"{r[2]} ({why})" for r, why in refused})))
        return live - {proc.pid}

    def snapshot(self, tree, strict=True):
        """Lists the processes and updates the tree. Returns the live members. `strict` stops the run when the gap since
        the last listing exceeds `ADOPT_SLACK` (the stop path is not strict: it must not raise)."""
        now = filetime_now()
        self.table = process_table(self.base)
        return tree.update(self.table, now, strict)

    def known(self, tree):
        """Every pid the tree has had, dead members included, except a pid that another process has since reused."""
        table = getattr(self, "table", {})
        return {pid for pid, created in tree.members.items() if pid not in table or table[pid][1] == created}

    def describe(self, row, rows):
        """Both ends of a refused row for the log: pids, endpoints, the peer's parent pid, creation time and image name.
        No command line, no environment."""
        pid, local, remote, state = row
        peer = next((r for r in rows if r[1] == remote and r[2] == local), None)
        if peer is None:  # no peer socket yet (SYN_SENT): the listener on that port, if any
            port = split_addr(remote)[1]
            peer = next((r for r in rows if r[3].upper() == "LISTENING" and split_addr(r[1])[1] == port
                         and split_addr(r[1])[0] in ("0.0.0.0", "127.0.0.1", "::")), None)
        table = getattr(self, "table", {})
        names = getattr(table, "names", {})

        def proc(p):
            if p is None:
                return "none"
            ppid, created = table.get(p, ("?", "?"))
            return f"pid {p} image {names.get(p, '?')} ppid {ppid} created {created}"

        return f"{local} -> {remote} {state}; own {proc(pid)}; peer {proc(peer[0] if peer else None)}"

    def recheck(self, tree, tag, accepted, refused, rows):
        """One re-check of the refused loopback rows with a fresh netstat and a fresh process listing taken right now.
        A row is accepted only if the normal rules accept it on that data, that is, its peer socket belongs to a member
        of the tree. 8080 and non-loopback rows are refused at once and never come here. Returns the new
        (accepted, refused, rows)."""
        rows2 = tcp_rows(self.base)
        live2 = self.snapshot(tree)
        self.owned |= owned_endpoints(rows2, live2)
        acc2, _ = classify_rows(rows2, live2, self.owned, getattr(self.table, "names", None), self.known(tree))
        ok2 = {(r[1], r[2]) for r, _ in acc2}
        rescued, still = [], []
        for row, why in refused:
            if why == WHY_LOOPBACK and (row[1], row[2]) in ok2 and row[0] in live2:  # a foreign owner's refusal is final
                rescued.append((row, "client-internal loopback, peer in the tree on a fresh listing"))
            else:
                still.append((row, why))
        lines = [f"RECHECK accepted: {self.describe(r, rows2)}" for r, _ in rescued]
        lines += [f"RECHECK still refused: {self.describe(r, rows2)}" for r, why in still if why.startswith(WHY_LOOPBACK)]
        self.conn_log(tag, lines)
        return accepted + rescued, still, rows2

    def sweep(self, tree, keep):
        """Kills every live member except `keep` (by pid, creation time matched by the listing just taken), listing again
        after each kill to catch late children. Returns the members still alive afterwards."""
        for _ in range(4):
            live = self.snapshot(tree, strict=False) - {keep}
            if not live:
                return set()
            kill_pids(self.base, sorted(live))
            time.sleep(0.5)
        return self.snapshot(tree, strict=False) - {keep}

    def kill_all(self, proc, tree):
        """Stop path: the root, then every live member of the tree that was ever seen (orphans included), re-listing
        until none is left. When the listing fails, the members alive at the last poll."""
        kill_tree(proc, self.base)
        try:
            live = self.snapshot(tree, strict=False) - {proc.pid}
        except Stop:
            live = tree.live - {proc.pid}
        kill_pids(self.base, sorted(live))
        try:
            self.sweep(tree, proc.pid)
        except Stop:
            pass

    def reap(self, tree, survivors, tag):
        """Kills processes that outlived the client's root process, then proves they are gone."""
        left = self.sweep(tree, tree.root)
        say(f"[{self.client} {tag}] {len(survivors)} process(es) outlived the client for {SURVIVOR_GRACE} s and were killed")
        self.conn_log(tag, [f"reaped {len(survivors)} survivor(s) of the client's tree"])
        if left:
            raise Stop("a process of the client's tree survived taskkill")

    def check_opencode_isolation(self):
        """Before every OpenCode run: nothing outside the tree may hold OpenCode's service port, and no OpenCode process may
        be running outside the tree, so a request can never reach a service that holds the user's real configuration."""
        for pid, local, _, state in tcp_rows(self.base):
            if state.upper() == "LISTENING" and split_addr(local) in (("127.0.0.1", OPENCODE_SERVICE_PORT),
                                                                      ("0.0.0.0", OPENCODE_SERVICE_PORT)):
                raise Stop(f"pid {pid} is listening on port {OPENCODE_SERVICE_PORT}, OpenCode's service port; "
                           "a run could reach a service with real configuration")
        names = getattr(process_table(self.base), "names", {})
        running = sorted((pid, name) for pid, name in names.items() if name.lower().startswith("opencode"))
        if running:
            raise Stop("an OpenCode process is running outside the tree: "
                       + ", ".join(f"pid {pid} {name}" for pid, name in running[:5]))

    def run(self, tag, argv, timeout):
        """Starts one client process under every guard and returns its exit code."""
        self.guard(f"before {tag}")
        if self.client == "opencode":
            self.check_opencode_isolation()
        if self.real:
            self.check_server()
        leak_check(self.parent, self.env, argv, self.key if self.smoke else None)
        log_path = os.path.join(self.logs, f"{self.client}-{self.label}-{tag}.log")
        rename_aside(log_path)
        started = time.monotonic()
        launched = filetime_now() - 10_000_000  # a second before the launch, for the Tree
        with open(log_path, "wb") as log:
            proc = subprocess.Popen(argv, cwd=self.project, env=self.env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
            tree = Tree(proc.pid, launched)
            try:
                while True:  # while the root runs
                    t0 = time.monotonic()
                    self.watch(proc, tree, log_path, tag)
                    if proc.poll() is not None:
                        break
                    if t0 - started > timeout:
                        raise TimedOut(f"{tag} timed out after {timeout} s")
                    time.sleep(max(0.0, 1 - (time.monotonic() - t0)))
                grace = time.monotonic() + SURVIVOR_GRACE
                while True:  # the root has exited: its survivors are watched, then killed after a final poll
                    t0 = time.monotonic()
                    survivors = self.watch(proc, tree, log_path, tag)
                    if not survivors:
                        break
                    if t0 >= grace:
                        self.reap(tree, survivors, tag)
                        break
                    time.sleep(max(0.0, 1 - (time.monotonic() - t0)))
            except BaseException:
                self.kill_all(proc, tree)
                raise
        with open(log_path, "rb") as f:
            text = f.read().decode("utf-8", "replace")
        self.guard(f"after {tag}")
        if PACKAGE_FETCH.search(text):
            raise Stop("the client output shows a package fetch or install")
        new = install_marks(home_dir(self.root)) - self.installs
        if new:
            raise Stop("new package-manager or tool content under <root>\\home: " + ", ".join(sorted(new)[:5]))
        code = proc.returncode
        self.runs.append((tag, code))
        self.last_seconds = time.monotonic() - started
        say(f"[{self.client} {tag}] exit code {code}, {self.last_seconds:.1f} s; last lines of its output:")
        for line in redact(text, self.key).splitlines()[-20:]:
            say("    " + line[:300].rstrip())
        return code

    # -- OpenCode --

    def opencode_discovery(self):
        """`--help` and `run --help` under the same environment: logged, and the three switches looked for."""
        write_opencode_config(self.root, self.key, self.smoke is not None)
        texts = []
        for name, extra in (("help", ["--help"]), ("run-help", ["run", "--help"])):
            tag = self.tag(name)
            self.run(tag, [self.exe, *extra], HELP_TIMEOUT)
            with open(os.path.join(self.logs, f"opencode-{self.label}-{tag}.log"), "rb") as f:
                texts.append(f.read().decode("utf-8", "replace"))
        help_path = os.path.join(self.root, "opencode-help.txt")
        if self.smoke is not None:
            rename_aside(help_path)  # attempt 2's help text never overwrites attempt 1's
        with open(help_path, "w", encoding="utf-8", newline="\n") as f:
            for title, text in zip(("opencode-cli.exe --help", "opencode-cli.exe run --help"), texts):
                f.write(f"===== {title}\n{redact(text, self.key)}\n")
        joined = "\n".join(texts)
        for name in SWITCHES:
            say(f"  help text {'names' if name in joined else 'does not name'} {name}")
        flags = {f: bool(re.search(r"(?<![\w-])" + re.escape(f) + r"\b", texts[1])) for f in ("--standalone", "--model", "--continue", "--format")}
        say("  run --help flags: " + ", ".join(f"{k} {'present' if v else 'ABSENT'}" for k, v in flags.items()))
        need = ("--standalone", "--model") if self.canary or self.smoke else ("--standalone", "--model", "--continue")
        missing = [f for f in need if not flags[f]]
        if missing:
            raise Stop(f"opencode run --help does not show {', '.join(missing)}: the invocation this script builds does not fit this OpenCode version")

    def opencode_runs(self):
        self.opencode_discovery()
        # `--standalone`: a private server that exits with the client, instead of the fixed-port background service.
        model = ["--standalone", "--model", f"artemis/{MODEL}"]
        if self.smoke is not None:
            self.bind_version(opencode_version(self.exe))
            self.started_wall = time.time()  # the redirection proof must come from the run itself, not from `--help`
        if self.canary:
            argv = smoke_argv("opencode", self.exe, CONFINE_PROMPT) if self.smoke is not None else [self.exe, "run", *model, "Say ok."]
            self.run("canary", argv, CANARY_TIMEOUT)
            return
        if self.real:
            self.run(self.tag("run"), smoke_argv("opencode", self.exe, self.smoke.task), self.smoke.timeout)
            self.seconds, self.exit = self.last_seconds, self.runs[-1][1]
            return
        self.run("turn1", [self.exe, "run", *model, "Read notes.txt and say how many stages it lists."], FULL_TIMEOUT)
        self.run("turn2", [self.exe, "run", "--standalone", "--continue", "Name the last stage."], FULL_TIMEOUT)

    # -- Claude Code --

    def claude_version(self):
        """`claude --version`, a guarded run like any other (it makes no request), before the run that counts."""
        tag = self.tag("version")
        self.run(tag, [self.exe, "--version"], HELP_TIMEOUT)
        with open(os.path.join(self.logs, f"claude-code-{self.label}-{tag}.log"), "rb") as f:
            found = re.search(r"\b(\d+\.\d+\.\d+)\b", f.read().decode("utf-8", "replace"))
        self.bind_version(found.group(1) if found else "")

    def claude_runs(self):
        fmt = ["--output-format", "json"]
        if self.smoke is not None:
            self.claude_version()
            self.started_wall = time.time()  # the redirection proof must come from the run itself, not from `--version`
        if self.canary:
            argv = (smoke_argv("claude-code", self.exe, CONFINE_PROMPT) if self.smoke is not None
                    else [self.exe, "-p", "Say ok.", *fmt, "--max-turns", "1"])
            self.run("canary", argv, CANARY_TIMEOUT)
            return
        if self.real:
            self.run(self.tag("run"), smoke_argv("claude-code", self.exe, self.smoke.task), self.smoke.timeout)
            self.seconds, self.exit = self.last_seconds, self.runs[-1][1]
            return
        tools = ["--max-turns", "4", "--allowedTools", "Read"]
        self.run("turn1", [self.exe, "-p", "Read notes.txt and say how many stages it lists.", *fmt, *tools], FULL_TIMEOUT)
        self.run("turn2", [self.exe, "-p", "--continue", "Name the last stage.", *fmt, *tools], FULL_TIMEOUT)

    def go(self):
        (self.opencode_runs if self.client == "opencode" else self.claude_runs)()


# ---- results ----


def load_records(out):
    recs = []
    for name in sorted(os.listdir(out)):
        if re.match(r"\d{3,}-.*\.json$", name):
            with open(os.path.join(out, name), encoding="utf-8") as f:
                recs.append(json.load(f))
    return recs


def ends_in_tool_result(body):
    msgs = body.get("messages") if isinstance(body, dict) else None
    if isinstance(msgs, list):  # messages with role "system" after the tool results (Claude Code) do not count
        msgs = [m for m in msgs if not (isinstance(m, dict) and m.get("role") == "system")]
    if not isinstance(msgs, list) or not msgs or not isinstance(msgs[-1], dict):
        return False
    last = msgs[-1]
    if last.get("role") in ("tool", "function"):
        return True
    content = last.get("content")
    return last.get("role") == "user" and isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


def touched_since(path, since):
    """True when `path` (a file, or a directory with anything in it) was created or modified at or after `since`."""
    if since is None:
        return False
    slack = 2.0  # file times on NTFS are coarse

    def fresh(p):
        try:
            st = os.stat(p)
        except OSError:
            return False
        return max(st.st_mtime, st.st_ctime) >= since - slack

    if not os.path.exists(path):
        return False
    if os.path.isfile(path):
        return fresh(path)
    if fresh(path):
        return True
    return any(fresh(os.path.join(d, n)) for d, dirs, files in os.walk(path) for n in dirs + files)


SESSION_ID = re.compile(r'"session_id"\s*:\s*"([0-9a-fA-F-]{36})"')


def opencode_proofs(root):
    """Where OpenCode writes its state when it runs under the redirected home."""
    home = home_dir(root)
    return [os.path.join(home, ".local", "share", "opencode"), os.path.join(home, ".local", "state", "opencode"),
            os.path.join(home, ".cache", "opencode")]


def redirect_ok(root, client, since, log_text):
    """The positive proof that a run used the redirected home: its own state or session transcript is newer than `since`."""
    if client == "claude-code":
        return claude_redirect_ok(root, since, log_text)
    return any(touched_since(p, since) for p in opencode_proofs(root))


def claude_redirect_ok(root, since, log_text):
    """Claude Code wrote under the redirected home in this run: home/claude/.claude.json was touched, or the transcript
    home/claude/projects/**/<session_id>.jsonl was, where <session_id> is the one in the client's own JSON result (only
    that field is read from its output). Nothing else counts."""
    claude = os.path.join(home_dir(root), "claude")
    if touched_since(os.path.join(claude, ".claude.json"), since):
        return True
    for sid in set(SESSION_ID.findall(log_text)):
        for d, _, files in os.walk(os.path.join(claude, "projects")):
            if sid + ".jsonl" in files and touched_since(os.path.join(d, sid + ".jsonl"), since):
                return True
    return False


def evaluate_canary(sess):
    """Step 6's expectations. Returns the problems found (an empty list is a pass)."""
    rep = sess.read_report()
    problems = []
    if rep["requests"] < 1:
        problems.append("the stub saw no request (the client could not reach it, asked for a login or went elsewhere)")
    if rep["classes"]["other"]:
        problems.append("a credential other than the dummy key reached the stub")
    if rep["hosts"] and rep["hosts"] != [STUB_HOST]:
        problems.append("a Host other than " + STUB_HOST + ": " + ", ".join(rep["hosts"]))
    if rep["classes"]["dummy"] < 1:
        problems.append("no request carried the dummy key (a client that sends none has not been configured)")
    seen_dummy = False  # a `none` is allowed only on a first probe, before the first request with the dummy key
    for r in load_records(sess.out):
        if r["credential"] == "dummy":
            seen_dummy = True
        elif r["credential"] == "none" and seen_dummy:
            problems.append("a request without credentials after the first one with the dummy key")
            break
    bad_exits = [f"{tag}={code}" for tag, code in sess.runs if code != 0]
    if bad_exits:
        problems.append("a client exited non-zero: " + ", ".join(bad_exits))
    if sess.polls < 1:
        problems.append("the connection watch never ran")
    if creds_stat() != sess.creds_before:
        problems.append("the credentials file changed")
    # Positive evidence that the redirection worked: the client wrote its state under the root.
    if sess.client == "claude-code":
        try:
            with open(os.path.join(sess.logs, f"{sess.client}-{sess.label}-canary.log"), "rb") as f:
                log_text = f.read().decode("utf-8", "replace")
        except OSError:
            log_text = ""
        if not claude_redirect_ok(sess.root, sess.started_wall, log_text):
            problems.append("no sign Claude Code used the redirected home in this run: neither home/claude/.claude.json "
                            "nor the transcript of this run's own session (the session_id in its JSON result) was written")
    else:
        proofs = opencode_proofs(sess.root)
        if not any(touched_since(p, sess.started_wall) for p in proofs):
            problems.append("no sign the client used the redirected home in this run: none of "
                            + ", ".join(os.path.relpath(p, sess.root) for p in proofs) + " exists")
    return rep, problems


def evaluate_full(sess):
    recs = load_records(sess.out)
    rep = sess.read_report()
    tool_trips = sum(1 for r in recs if ends_in_tool_result(r.get("body")))
    user_ends = sum(1 for r in recs if isinstance(r.get("body"), dict) and r["body"].get("tools") and not ends_in_tool_result(r["body"]))
    say(f"[{sess.client}] {len(recs)} requests recorded; {tool_trips} end in a tool result; {user_ends} carry tools and end in a user message")
    say(f"[{sess.client}] stub report: classes {rep['classes']}, hosts {rep['hosts']}, abort {rep['abort']}, watch polls {sess.polls}")
    problems = []
    if len(recs) < 4:
        problems.append("fewer than four requests")
    if tool_trips < 1:
        problems.append("no tool round trip")
    if user_ends < 2:
        problems.append("no second user turn")
    if rep["hosts"] != [STUB_HOST] or rep["classes"]["other"]:
        problems.append("a Host or credential class outside the expected")
    return problems


def run_client(root, client, canary, key_file, creds_before):
    sess = Session(root, client, canary, key_file, creds_before)
    if canary:
        try:
            os.remove(marker_path(root, client))  # an old pass does not count for this canary
        except FileNotFoundError:
            pass
    version = opencode_version(sess.exe) if client == "opencode" else ""
    say(f"== {client} {sess.label}" + (f" (opencode-cli {version})" if version else ""))
    with sess:
        sess.go()
    rep = sess.read_report()
    say(f"[{client}] stub report: {json.dumps(rep)}")
    if canary:
        rep, problems = evaluate_canary(sess)
        if problems:
            raise Stop(f"{client} canary failed: " + "; ".join(problems))
        with open(marker_path(root, client), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"client": client, "scripts": script_hashes(), "report": rep, "polls": sess.polls, "runs": sess.runs}, f, indent=2)
        say(f"[{client}] canary ok; marker written")
        return True
    problems = evaluate_full(sess)
    for p in problems:
        say(f"[{client}] FAIL: {p}")
    return not problems


# ---- the smoke modes ----


def opencode_version(exe):
    """OpenCode's version, from the file the desktop app keeps beside its bundled CLI (the CLI is not run for it)."""
    try:
        with open(os.path.join(os.path.dirname(exe), "opencode-cli.version"), encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        raise Stop("opencode-cli.version is missing beside the CLI")


def smoke_argv(client, exe, task):
    """The client's command line in a smoke run (with `--standalone`, as the capture uses). The key is never on it."""
    if client == "opencode":
        return [exe, "run", "--standalone", "--model", f"artemis/{MODEL}", task]
    if client == "claude-code":
        return [exe, "-p", task, "--output-format", "json", "--max-turns", "20", "--allowedTools", *CLAUDE_ALLOWED,
                "--disallowedTools", *CLAUDE_DENIED]
    raise ValueError(client)


PROJECT_NAME = re.compile(r"project(-\d+)?")


def smoke_root_ok(root, base):
    """A smoke run root is a direct child of the smoke base (`<base>\\<client>-<time>`, or the canary's)."""
    if not os.path.isabs(root) or norm(os.path.dirname(os.path.abspath(root))) != norm(base):
        raise Stop(f"the smoke root must be a direct child of {base}")


def check_smoke_args(client, root, key_file, server_pid, task, timeout, out, project, base=SMOKE_BASE):
    """Validates a smoke run's arguments and returns the run's key, read from its file (first non-empty line, like the
    server). Nothing here prints the key. Raises Stop for anything the guards cannot vouch for."""
    if client not in CLIENTS:
        raise Stop("an unknown client")
    smoke_root_ok(root, base)
    if not PROJECT_NAME.fullmatch(project):
        raise Stop("the project must be project or project-<n>")
    if not isinstance(server_pid, int) or server_pid <= 0:
        raise Stop("the server pid must be positive")
    if not isinstance(timeout, int) or not 0 < timeout <= SMOKE_MAX_TIMEOUT:
        raise Stop(f"the timeout must be 1 to {SMOKE_MAX_TIMEOUT} s")
    keys = os.path.join(base, "keys")
    if (not os.path.isabs(key_file) or not is_under(key_file, keys) or norm(key_file) == norm(keys) or is_under(key_file, root)
            or not os.path.isfile(key_file)):
        raise Stop(f"the key file must be an existing file under {keys} and outside the run's root")
    if not os.path.isabs(out) or not is_under(out, root) or is_under(out, home_dir(root)) or PROJECT_NAME.fullmatch(
            os.path.relpath(norm(out), norm(root)).split(os.sep)[0]):
        raise Stop("the result file must be under the run's root, outside home and the scratch repos")
    try:
        with open(key_file, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
    except (OSError, ValueError):
        raise Stop("the key file is unreadable")
    key = lines[0] if lines else ""
    if not KEY_SHAPE.fullmatch(key) or key == DUMMY_KEY:
        raise Stop("the key file does not hold a run key (16 to 256 characters of A-Z a-z 0-9 _ = + / . -, not the capture's)")
    if not isinstance(task, str) or not task.strip() or len(task) > 4000 or "\x00" in task or task.lstrip().startswith("-"):
        raise Stop("the task must be 1 to 4000 characters of text that does not begin with a dash")
    if key in task:
        raise Stop("the task holds the run's key")
    return key


def write_json_atomic(path, obj):
    """Writes `obj` to `path` through a temporary file. An earlier file at `path` is renamed aside first, never overwritten."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    rename_aside(path)
    os.replace(tmp, path)


def smoke_verdict(result):
    """Python's judgement of a run that completed under the guards, on the fields in `result`. Sets `hosts_status`
    ("clean": the watch saw the tree connect to exactly 127.0.0.1:8090; "inconclusive": it saw no connection, because the
    requests fell between two polls, or saw anything else) and returns the reason to stop when the run cannot be vouched for:
    no sign the client used the redirected home."""
    result["hosts_status"] = "clean" if result["hosts"] == [STUB_HOST] else "inconclusive"
    if not result["redirect_ok"]:
        return "no sign the client used the redirected home in this run (its state or session transcript under <root>\\home was not written)"
    return None


REFUSAL_TEXT = re.compile(r"denied|reject|blocked|not allowed|permission|refus|haven't granted", re.I)


def tool_results(body):
    """(text, is_error) for every tool result in the last message of a request body, in either API. OpenAI's tool messages
    carry no error flag, so theirs is always False."""
    msgs = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(msgs, list):
        return []
    msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") != "system"]
    if not msgs:
        return []

    def text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
        return json.dumps(content)

    last = msgs[-1]
    if last.get("role") in ("tool", "function"):
        return [(text(last.get("content")), False)]
    content = last.get("content")
    if last.get("role") == "user" and isinstance(content, list):
        return [(text(b.get("content")), b.get("is_error") is True) for b in content
                if isinstance(b, dict) and b.get("type") == "tool_result"]
    return []


def confinement_problems(recs, client, marker):
    """The smoke canary's proof of confinement. The stub made the client's read tool ask for a sentinel file outside the
    run's root; the client must have refused: a request after the call carries a tool result that is an error (Claude Code)
    or says it was refused (OpenCode), and the sentinel's contents (`marker`) are in no request at all. Returns the problems."""
    results = [r for rec in recs for r in tool_results(rec.get("body"))]
    problems = []
    if not results:
        problems.append("the client never made the scripted read, so its refusal was not shown")
    if marker in json.dumps(recs):
        problems.append("the sentinel's contents reached the model: the client read a file outside the run's root")
    if client == "claude-code" and results and not all(err for _, err in results):
        problems.append("the scripted read's tool result is not an error")
    if client != "claude-code" and results and not all(REFUSAL_TEXT.search(t) for t, _ in results):
        problems.append("the scripted read's tool result does not say the read was refused")
    return problems


def smoke_main(argv):
    """`smoke`: one client, one task, against the real server the script started. Exit 0: the run completed under every
    guard (whatever the client's own exit code is, it is in the result file); 2: a guard stopped it, or a check failed."""
    ap = argparse.ArgumentParser(prog="capture_clients.py smoke")
    ap.add_argument("--client", choices=CLIENTS, required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--server-pid", type=int, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", required=True)
    ap.add_argument("--project", default="project", help="the scratch repo's directory name below the root (project, project-2)")
    a = ap.parse_args(argv)
    assert PORT != 8080
    root = os.path.abspath(a.root)
    try:
        key = check_smoke_args(a.client, root, os.path.abspath(a.key_file), a.server_pid, a.task, a.timeout,
                               os.path.abspath(a.out), a.project)
    except Stop as e:
        print(f"STOP: {e}", file=sys.stderr)
        return 2
    out = os.path.abspath(a.out)
    creds_before = creds_stat()
    result = {"client": a.client, "version": "", "exit": None, "seconds": 0.0, "polls": 0, "refused": [], "redirect_ok": False,
              "creds_unchanged": True, "hosts": [], "hosts_status": "inconclusive", "proxy_hosts": [], "timed_out": False,
              "stop": None, "project": a.project}
    sess = None
    code = 0
    try:
        require_creds(creds_before)
        prepare_smoke_root(root, a.project)
        sess = Session(root, a.client, False, None, creds_before,
                       Smoke(key, a.server_pid, a.project, a.task, a.timeout))
        with sess:
            sess.go()
        try:
            with open(os.path.join(sess.logs, f"{a.client}-smoke-{sess.tag('run')}.log"), "rb") as f:
                log_text = f.read().decode("utf-8", "replace")
        except OSError:
            log_text = ""
        result["redirect_ok"] = redirect_ok(root, a.client, sess.started_wall, log_text)
        ids = SESSION_ID.findall(log_text)
        if a.client == "claude-code" and ids:
            result["session_id"] = ids[0]
        if creds_stat() != creds_before:
            raise Stop("the credentials file's size or mtime changed")
    except Stop as e:
        print(f"STOP: {redact(str(e), key)}", file=sys.stderr)
        result["stop"] = redact(str(e), key)
        result["timed_out"] = isinstance(e, TimedOut)
        code = 2
    except Exception as e:  # nothing else may pass as a completed run
        print(f"STOP: internal error {type(e).__name__}\n" + redact(traceback.format_exc(), key), file=sys.stderr)
        result["stop"] = f"internal error: {type(e).__name__}"
        code = 2
    if sess is not None:
        result.update(version=sess.version, exit=sess.exit, seconds=round(sess.seconds, 1), polls=sess.polls,
                      refused=sorted(set(sess.refused)), hosts=sorted(sess.hosts), proxy_hosts=sorted(sess.proxy_hosts))
    result["creds_unchanged"] = creds_stat() == creds_before
    reason = smoke_verdict(result)  # also sets hosts_status, from the hosts the watch saw
    if code == 0 and not result["creds_unchanged"]:
        reason = "the credentials file's size or mtime changed"
    if code == 0 and reason:
        print(f"STOP: {reason}", file=sys.stderr)
        result["stop"], code = reason, 2
    write_json_atomic(out, result)  # an earlier file at that path is renamed aside
    say(f"[{a.client}] smoke result: " + json.dumps({k: v for k, v in result.items() if k != "stop"}))
    return code


def smoke_canary_main(argv):
    """`smoke-canary`: the canary of the smoke's shape (its env, its config with the permission block, Claude's aliases)
    against the recording stub with the canary's dummy key. A pass writes the marker at the fixed path, not under the root."""
    ap = argparse.ArgumentParser(prog="capture_clients.py smoke-canary")
    ap.add_argument("--client", choices=CLIENTS, required=True)
    ap.add_argument("--root", required=True)
    a = ap.parse_args(argv)
    assert PORT != 8080
    root = os.path.abspath(a.root)
    creds_before = creds_stat()
    try:
        require_creds(creds_before)
        smoke_root_ok(root, SMOKE_BASE)
        bound = smoke_hashes()  # fail closed before anything starts: every bound file must exist; the marker binds these
        if is_under(smoke_marker_path(a.client), root):
            raise Stop("the smoke marker would sit under the run's root")
        key_file = prepare_root(root)
        # The sentinel: a file with harmless random contents in the smoke base, outside this root. The stub has the
        # client's read tool ask for it, and the canary passes only if the client refuses.
        marker = "ARTEMIS-SENTINEL-" + secrets.token_hex(12)
        sentinel = os.path.join(SMOKE_BASE, f"sentinel-{a.client}-{secrets.token_hex(4)}.txt")
        if is_under(sentinel, root):
            raise Stop("the sentinel would sit under the run's root")
        os.makedirs(SMOKE_BASE, exist_ok=True)
        with open(sentinel, "w", encoding="utf-8", newline="\n") as f:
            f.write(marker + "\n")
        sess = Session(root, a.client, True, key_file, creds_before, Smoke(DUMMY_KEY, sentinel=sentinel))
        try:
            os.remove(smoke_marker_path(a.client))  # an old pass does not count for this canary
        except FileNotFoundError:
            pass
        say(f"== {a.client} smoke canary")
        with sess:
            sess.go()
        rep, problems = evaluate_canary(sess)
        problems += confinement_problems(load_records(sess.out), a.client, marker)
        if problems:
            raise Stop(f"{a.client} smoke canary failed: " + "; ".join(problems))
        if not sess.version:
            raise Stop("the client's version was not read")
        if creds_stat() != creds_before:
            raise Stop("the credentials file's size or mtime changed")
        check_unchanged(bound)
        write_smoke_marker(a.client, SMOKE_BASE, REPO, {"version": sess.version, "report": rep, "polls": sess.polls,
                                                       "runs": sess.runs, "hosts": sorted(sess.hosts),
                                                       "proxy_hosts": sorted(sess.proxy_hosts), "confined": True},
                           root, hashes=bound)
        say(f"[{a.client}] smoke canary ok (the client refused the read outside the root); marker written")
    except Stop as e:
        print(f"STOP: {redact(str(e))}", file=sys.stderr)
        return 2
    except Exception as e:  # nothing else may pass as a canary
        print(f"STOP: internal error {type(e).__name__}\n" + redact(traceback.format_exc()), file=sys.stderr)
        return 2
    return 0


def selftest_smoke():
    """Checks of the smoke modes' own parts. Starts no client, no stub and no server, reads no
    credential and touches only a temporary directory. Returns the process exit code."""
    import tempfile

    fails = []

    def check(name, ok, detail=""):
        say(f"  {'ok  ' if ok else 'FAIL'}  {name}" + ("" if ok or not detail else f"  ({detail})"))
        if not ok:
            fails.append(name)

    def stops(fn, *a, **kw):
        """The message of the Stop that `fn` raises, or None when it raises none."""
        try:
            fn(*a, **kw)
        except Stop as e:
            return str(e)
        return None

    bs = chr(92)
    tmp = tempfile.mkdtemp(prefix="artemis-selftest-smoke-")
    key = "smoke-run-key-" + "q7Zk" * 6  # gitleaks:allow (invented, for this selftest only)
    base = os.path.join(tmp, "m3c-smoke")
    root = os.path.join(base, "opencode-20261026-120000")
    alias = "qwen3.6-35b-a3b"
    task = "Fix the bugs in calc.py so that the tests in test_calc.py pass, then run the tests with python -m unittest -v."

    home_names = {"USERPROFILE", "HOME", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
                  "XDG_STATE_HOME", "XDG_CACHE_HOME"}
    proxy_names = {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "NO_PROXY", "no_proxy"}
    git_names = {"GIT_CONFIG_NOSYSTEM", "GIT_TERMINAL_PROMPT", "GCM_INTERACTIVE"}
    opencode_names = {"OPENCODE_DISABLE_MODELS_FETCH", "OPENCODE_DISABLE_AUTOUPDATE", "OPENCODE_DISABLE_CLAUDE_CODE", "OPENCODE_CONFIG",
                      "OPENCODE_DISABLE_PROJECT_CONFIG"}
    aliases = {"ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
               "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_SMALL_FAST_MODEL"}
    claude_names = aliases | {"CLAUDE_CONFIG_DIR", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN",
                              "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "DISABLE_TELEMETRY", "DISABLE_ERROR_REPORTING",
                              "DISABLE_AUTOUPDATER", "CLAUDE_CODE_USE_POWERSHELL_TOOL"}

    def t_env():
        parent = {
            "SystemRoot": bs.join(("C:", "Windows")), "Path": "C:" + bs + "bin", "ComSpec": "cmd.exe",
            "ANTHROPIC_API_KEY": "sk-ant-" + "a" * 30, "OPENAI_API_KEY": "sk-" + "b" * 30, "CLAUDE_CODE_OAUTH_TOKEN": "c" * 30,
            "MY_SECRET": "d" * 30, "UV_CACHE_DIR": "e", "OPENCODE_CONFIG": "f", "HTTPS_PROXY": "http://elsewhere:1",
        }
        for client, own in (("opencode", opencode_names), ("claude-code", claude_names)):
            env = child_env(client, root, parent, key=key, smoke=True)
            want = {"SYSTEMROOT", "PATH", "COMSPEC"} | home_names | proxy_names | git_names | own
            check(f"smoke child_env {client}: exactly the allowlist plus its own variables", set(env) == want,
                  f"extra {sorted(set(env) - want)}, missing {sorted(want - set(env))}")
            check(f"smoke child_env {client}: passes verify_env", stops(verify_env, env, root, client, key=key, smoke=True) is None)
            secrets = [v for k, v in parent.items() if FORBIDDEN_NAME.search(k)]
            check(f"smoke child_env {client}: no secret-named parent value is in it", not any(v in secrets for v in env.values()))
            check(f"smoke child_env {client}: the capture's dummy key is not in it", DUMMY_KEY not in env.values())
            bad = dict(env, ANTHROPIC_API_KEY="x")
            check(f"smoke child_env {client}: verify_env refuses a foreign key variable",
                  stops(verify_env, bad, root, client, key=key, smoke=True) is not None)
        cc = child_env("claude-code", root, parent, key=key, smoke=True)
        check("claude-code: every alias is the model", all(cc[n] == alias for n in aliases), str({n: cc.get(n) for n in aliases}))
        check("claude-code: the token is the run's key and the base URL is 8090",
              cc["ANTHROPIC_AUTH_TOKEN"] == key and cc["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8090")
        check("claude-code: the PowerShell tool is off (Bash is the one shell tool)", cc.get("CLAUDE_CODE_USE_POWERSHELL_TOOL") == "0")
        check("claude-code: the capture's own shape has no PowerShell variable",
              "CLAUDE_CODE_USE_POWERSHELL_TOOL" not in child_env("claude-code", root, parent))
        check("claude-code: verify_env refuses a missing alias",
              stops(verify_env, {k: v for k, v in cc.items() if k != "ANTHROPIC_DEFAULT_HAIKU_MODEL"}, root, "claude-code",
                    key=key, smoke=True) is not None)
        check("claude-code: verify_env refuses the capture's key",
              stops(verify_env, cc, root, "claude-code", key=DUMMY_KEY, smoke=True) is not None)
        check("claude-code: the capture's own shape is unchanged (no extra aliases)",
              set(child_env("claude-code", root, parent)) & (aliases - {"ANTHROPIC_MODEL"}) == set())
        oc = child_env("opencode", root, parent, key=key, smoke=True)
        check("opencode: project config is disabled in the smoke", oc.get("OPENCODE_DISABLE_PROJECT_CONFIG") == "1")
        check("opencode: the capture's own shape has no such variable",
              "OPENCODE_DISABLE_PROJECT_CONFIG" not in child_env("opencode", root, parent))
        check("opencode: verify_env refuses a smoke env without it",
              stops(verify_env, {k: v for k, v in oc.items() if k != "OPENCODE_DISABLE_PROJECT_CONFIG"}, root, "opencode",
                    key=key, smoke=True) is not None)

    def t_config():
        rootc = os.path.join(tmp, "cfgroot")
        write_opencode_config(rootc, key, smoke=True)
        env = child_env("opencode", rootc)
        paths = [env["OPENCODE_CONFIG"], os.path.join(env["XDG_CONFIG_HOME"], "opencode", "opencode.json")]
        texts = []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                texts.append(f.read())
        cfg = json.loads(texts[0])
        want = {"*": "deny", "read": "allow", "glob": "allow", "grep": "allow",
                "edit": {"*": "allow", "*:*": "deny", "opencode.json": "deny", "opencode.jsonc": "deny", ".opencode/*": "deny",
                         ".agents/*": "deny", ".claude/*": "deny", ".git/*": "deny"},
                "bash": {"*": "deny", "python -m unittest*": "allow"}, "external_directory": "deny",
                "webfetch": "deny", "websearch": "deny", "task": "deny"}
        check("opencode config: both files are the same", texts[0] == texts[1])
        check("opencode config: the permission block", cfg.get("permission") == want, str(cfg.get("permission")))
        check("opencode config: the order is the rule order (catch-all first; the last matching rule wins)",
              list(cfg["permission"]) == list(want) and list(cfg["permission"]["bash"]) == ["*", "python -m unittest*"]
              and list(cfg["permission"]["edit"]) == list(want["edit"]))
        check("opencode config: no list or todowrite key (neither is a tool in 2.0.30)",
              "list" not in cfg["permission"] and "todowrite" not in cfg["permission"])
        # The 2.0.30 evaluator in miniature: config keys map to actions (bash -> shell, task -> subagent, write/patch -> edit),
        # rules keep their order, the last rule whose action and resource match wins, and no match asks.
        rules = []
        for k, v in cfg["permission"].items():
            action = {"bash": "shell", "task": "subagent", "write": "edit", "patch": "edit"}.get(k, k)
            for res, eff in ({"*": v} if isinstance(v, str) else v).items():
                rules.append((action, res, eff))

        def glob_ok(pattern, text):
            rx = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".")
            return re.fullmatch(rx, text, re.S | re.I) is not None

        def effect(action, resource):
            hit = [e for a, r, e in rules if glob_ok(a, action) and glob_ok(r, resource)]
            return hit[-1] if hit else "ask"

        for action, resource, want_effect in (
            ("read", "calc.py", "allow"), ("glob", "*.py", "allow"), ("grep", "mean", "allow"), ("edit", "calc.py", "allow"),
            ("edit", "pkg/sub/test_calc.py", "allow"), ("edit", "README.md", "allow"),
            ("edit", "opencode.json", "deny"), ("edit", "opencode.jsonc", "deny"), ("edit", "OpenCode.JSON", "deny"),
            ("edit", ".opencode/opencode.json", "deny"), ("edit", ".opencode/agent/deep/x.md", "deny"),
            ("edit", ".agents/a.md", "deny"), ("edit", ".claude/settings.json", "deny"), ("edit", ".git/hooks/pre-commit", "deny"),
            ("edit", ".gitignore", "allow"),
            ("edit", "opencode.json::$DATA", "deny"), ("edit", ".opencode::$INDEX_ALLOCATION/agent/x.md", "deny"),
            ("edit", ".git::$INDEX_ALLOCATION/hooks/pre-commit", "deny"), ("edit", "calc.py:stream", "deny"),
            ("shell", "python -m unittest -v", "allow"), ("shell", "python -m unittest", "allow"), ("shell", "rm -rf x", "deny"),
            ("shell", "curl http://example.invalid", "deny"), ("shell", "python x.py", "deny"),
            ("external_directory", "C:/Users/someone/*", "deny"), ("webfetch", "http://example.invalid", "deny"),
            ("websearch", "q", "deny"), ("subagent", "explore", "deny"), ("skill", "x", "deny"), ("question", "*", "deny"),
            ("doom_loop", "*", "deny"), ("lsp", "*", "deny"), ("mcp", "x", "deny"),
        ):
            got = effect(action, resource)
            check(f"opencode permission: {action} {resource!r} is {want_effect}", got == want_effect, got)

        def listed(rule_list, action):
            """The tool filter: a tool is dropped when the last rule for its action denies every resource."""
            hit = [(r, e) for a, r, e in rule_list if glob_ok(a, action)]
            return not (hit and hit[-1] == ("*", "deny"))

        check("opencode permission: the shell and edit tools stay listed", listed(rules, "shell") and listed(rules, "edit"))
        check("opencode permission: webfetch, websearch, subagent and skill are not listed",
              not any(listed(rules, a) for a in ("webfetch", "websearch", "subagent", "skill")))
        reversed_bash = [("shell", "python -m unittest*", "allow"), ("shell", "*", "deny")]
        check("opencode permission: a reversed bash block would drop the shell tool altogether", not listed(reversed_bash, "shell"))

        def keys_of(node, out):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "apiKey":
                        out.append(v)
                    keys_of(v, out)
            return out

        check("opencode config: the only apiKey is the key file's", keys_of(cfg, []) == [key])
        check("opencode config: the capture's dummy key is not in it", DUMMY_KEY not in texts[0])
        check("opencode config: the base URL is 8090", cfg["provider"]["artemis"]["options"]["baseURL"] == "http://127.0.0.1:8090/v1")
        write_opencode_config(rootc)
        with open(paths[0], encoding="utf-8") as f:
            old = json.load(f)
        check("opencode config: the capture's shape is unchanged (dummy key, no permission block)",
              "permission" not in old and keys_of(old, []) == [DUMMY_KEY])

    def t_marker():
        repo = os.path.join(tmp, "repo")
        for rel in SMOKE_BOUND:
            p = os.path.join(repo, *rel.split("/"))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write("fixture " + rel + "\n")
        os.makedirs(base, exist_ok=True)
        check("the marker's fixed path", smoke_marker_path("claude-code") == "T:" + bs + "artemis-oracle" + bs + "runs" + bs
              + "m3c-smoke" + bs + "canary-ok-smoke-claude-code", smoke_marker_path("claude-code"))
        check("the marker binds the seven files", sorted(smoke_hashes(repo)) == sorted(SMOKE_BOUND) and len(SMOKE_BOUND) == 7)
        check("no marker: refused", stops(require_smoke_canary, "opencode", base, repo) is not None)
        write_smoke_marker("opencode", base, repo, {"report": {}})
        check("a fresh marker is accepted", stops(require_smoke_canary, "opencode", base, repo) is None)
        check("the marker of the other client does not do", stops(require_smoke_canary, "claude-code", base, repo) is not None)
        target = os.path.join(repo, "tools", "smoke", "Smoke.psm1")
        with open(target, "a", encoding="utf-8", newline="\n") as f:
            f.write("changed\n")
        check("a marker whose hashes differ is refused", stops(require_smoke_canary, "opencode", base, repo) is not None)
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write("fixture tools/smoke/Smoke.psm1\n")
        check("restored, it is accepted again", stops(require_smoke_canary, "opencode", base, repo) is None)
        os.remove(target)
        check("a marker whose bound file is missing is void", stops(require_smoke_canary, "opencode", base, repo) is not None)
        check("a marker cannot be written over a missing file", stops(write_smoke_marker, "opencode", base, repo, {}) is not None)
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write("fixture tools/smoke/Smoke.psm1\n")
        with open(smoke_marker_path("opencode", base), encoding="utf-8") as f:
            data = json.load(f)
        data["scripts"].pop("tools/ps/ArtemisServer.psm1")
        with open(smoke_marker_path("opencode", base), "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f)
        check("a marker that binds fewer files is refused", stops(require_smoke_canary, "opencode", base, repo) is not None)
        other = os.path.join(tmp, "elsewhere")
        os.makedirs(other, exist_ok=True)
        write_smoke_marker("claude-code", other, repo, {})
        check("a marker anywhere but the fixed path is not looked at",
              stops(require_smoke_canary, "claude-code", base, repo) is not None)
        with open(os.path.join(base, "canary-ok-claude-code"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"client": "claude-code", "scripts": script_hashes()}, f)
        check("3b's own marker does not do", stops(require_smoke_canary, "claude-code", base, repo) is not None)
        check("a marker cannot be under the run's root", stops(write_smoke_marker, "opencode", base, repo, {}, base) is not None)

    def t_owner():
        listen = [(4242, "127.0.0.1:8090", "0.0.0.0:0", "LISTENING"), (777, "127.0.0.1:5000", "0.0.0.0:0", "LISTENING")]
        check("8090's listener is the server's: accepted", stops(check_8090_owner, listen, 4242) is None)
        check("8090's listener is another pid: refused", stops(check_8090_owner, listen, 4243) is not None)
        check("nothing listens on 8090: refused", stops(check_8090_owner, listen[1:], 4242) is not None)
        check("a second listener on 8090 (any address): refused",
              stops(check_8090_owner, listen + [(999, "0.0.0.0:8090", "0.0.0.0:0", "LISTENING")], 4242) is not None
              and stops(check_8090_owner, listen + [(999, "[::]:8090", "[::]:0", "LISTENING")], 4242) is not None)
        check("the server listening beyond loopback: refused",
              stops(check_8090_owner, [(4242, "0.0.0.0:8090", "0.0.0.0:0", "LISTENING")], 4242) is not None)
        check("connections on 8090 do not count as listeners",
              stops(check_8090_owner, listen + [(999, "127.0.0.1:8090", "127.0.0.1:50000", "ESTABLISHED")], 4242) is None)
        check("the server's image must be artemis", stops(check_server_image, {4242: "artemis.exe"}, 4242) is None
              and stops(check_server_image, {4242: "python.exe"}, 4242) is not None
              and stops(check_server_image, {1: "artemis.exe"}, 4242) is not None)

    def t_argv():
        want_oc = ["x.exe", "run", "--standalone", "--model", "artemis/" + alias, task]
        want_cc = ["x.exe", "-p", task, "--output-format", "json", "--max-turns", "20", "--allowedTools",
                   "Bash(python -m unittest:*)", "Edit(/**)", "Edit(./**)", "--disallowedTools", "WebFetch", "WebSearch", "Task", "Agent",
                   "PowerShell"]
        check("opencode argv", smoke_argv("opencode", "x.exe", task) == want_oc, str(smoke_argv("opencode", "x.exe", task)))
        check("claude-code argv", smoke_argv("claude-code", "x.exe", task) == want_cc, str(smoke_argv("claude-code", "x.exe", task)))
        check("the key on an argv is refused", stops(leak_check, {}, {}, ["x", "--token", key], key) is not None
              and stops(leak_check, {}, {}, ["x", task], key) is None)

    def t_redact():
        line = f"Authorization: Bearer {key} and again {key}, also {DUMMY_KEY} and sk-ant-abcdef"
        out = redact(line, key)
        check("redact removes the run's key", key not in out and DUMMY_KEY not in out and "sk-ant-abcdef" not in out, out)
        check("redact leaves a line without a secret alone", redact("3 stages", key) == "3 stages")

    def t_args():
        os.makedirs(os.path.join(base, "keys"), exist_ok=True)
        kf = os.path.join(base, "keys", "key.txt")
        with open(kf, "w", encoding="utf-8", newline="\n") as f:
            f.write(key + "\n")
        out = os.path.join(root, "verdict-client-1.json")

        def args(**kw):
            a = dict(client="opencode", root=root, key_file=kf, server_pid=4242, task=task, timeout=900, out=out, project="project")
            a.update(kw)
            return check_smoke_args(a["client"], a["root"], a["key_file"], a["server_pid"], a["task"], a["timeout"], a["out"],
                                    a["project"], base)

        for p in (os.path.join(base, "key.txt"), os.path.join(tmp, "elsewhere", "key.txt")):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(key + "\n")  # real files, so the refusal is about where they are
        check("valid arguments give the key from the file", stops(args) is None and args() == key)
        for name, kw in (
            ("a root outside the smoke base", dict(root=os.path.join(tmp, "elsewhere", "r"), out=os.path.join(tmp, "elsewhere", "r", "v.json"))),
            ("the smoke base itself as the root", dict(root=base, out=os.path.join(base, "v.json"))),
            ("a key file under the root", dict(key_file=os.path.join(root, "key.txt"))),
            ("a key file in the smoke base but not in keys", dict(key_file=os.path.join(base, "key.txt"))),
            ("a key file elsewhere", dict(key_file=os.path.join(tmp, "elsewhere", "key.txt"))),
            ("a missing key file", dict(key_file=os.path.join(base, "keys", "none.txt"))),
            ("an out file under home", dict(out=os.path.join(root, "home", "v.json"))),
            ("an out file in the project", dict(out=os.path.join(root, "project", "v.json"))),
            ("an out file in the second project", dict(out=os.path.join(root, "project-2", "v.json"))),
            ("an out file outside the root", dict(out=os.path.join(base, "v.json"))),
            ("an unknown client", dict(client="codex")),
            ("an empty task", dict(task="")),
            ("a task that is an option", dict(task="--auto fix it")),
            ("a task that holds the key", dict(task="use " + key)),
            ("a zero timeout", dict(timeout=0)),
            ("an hour and a second", dict(timeout=3601)),
            ("a server pid of zero", dict(server_pid=0)),
            ("a project that escapes the root", dict(project=".." + bs + "x")),
            ("another project name", dict(project="other")),
        ):
            check("refused: " + name, stops(args, **kw) is not None)
        check("project-2 is allowed", stops(args, project="project-2") is None)
        short = os.path.join(base, "keys", "short.txt")
        for name, text in (("a short key", "abc\n"), ("the capture's dummy key", DUMMY_KEY + "\n"),
                           ("a key with a space", "abcdefghijklmnop qrstuvwxyz\n"), ("an empty key file", "\n")):
            with open(short, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            check("refused: " + name, stops(args, key_file=short) is not None)

    def t_hosts():
        live = {10, 11}
        rows = [(10, "127.0.0.1:50001", "127.0.0.1:8090", "ESTABLISHED"), (10, "127.0.0.1:50002", "127.0.0.1:9", "SYN_SENT"),
                (11, "127.0.0.1:50003", "127.0.0.1:8080", "ESTABLISHED"), (11, "127.0.0.1:50004", "127.0.0.1:11434", "SYN_SENT"),
                (11, "127.0.0.1:50005", "93.184.216.34:443", "ESTABLISHED"), (4242, "127.0.0.1:8090", "0.0.0.0:0", "LISTENING")]
        acc, ref = classify_rows(rows, live)
        hosts, proxies = accepted_endpoints(acc)
        check("the watch accepts 8090 and the proxy port, listed apart", hosts == {"127.0.0.1:8090"} and proxies == {"127.0.0.1:9"},
              f"{hosts} {proxies}")
        check("the watch still refuses 8080, other loopback ports and other hosts", len(ref) == 3, str(ref))

    def t_session():
        """A smoke Session built without a client (find_exe replaced), its gate pointed at a fixture marker, and the ancestor
        names cut to AGENTS.md (the temporary directory sits under the profile, which holds .claude): no process starts."""
        srepo, sbase = os.path.join(tmp, "srepo"), os.path.join(tmp, "sbase")
        for rel in SMOKE_BOUND:
            p = os.path.join(srepo, *rel.split("/"))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write("fixture " + rel + "\n")
        sroot = os.path.join(sbase, "claude-code-20261026-120000")
        os.makedirs(os.path.join(sroot, "project-2"), exist_ok=True)

        class Fake(Session):
            def find_exe(self):
                return os.path.join(tmp, "bin", "client.exe")

        def make(canary, sm):
            return Fake(sroot, "claude-code", canary, None, creds_stat(), sm)

        real = Smoke(key, 4242, "project-2", task, 900)
        gate = globals()["require_smoke_canary"]
        names = globals()["ANCESTOR_NAMES"]
        globals()["require_smoke_canary"] = lambda client: gate(client, sbase, srepo)
        globals()["ANCESTOR_NAMES"] = ("AGENTS.md",)
        try:
            check("a smoke Session without the marker is refused", stops(make, False, real) is not None)
            check("a smoke canary Session needs no marker", stops(make, True, Smoke(DUMMY_KEY)) is None)
            sentinel = os.path.join(sbase, "sentinel-claude-code-0f3a.txt")
            csess = make(True, Smoke(DUMMY_KEY, sentinel=sentinel))
            check("the canary's stub is told to read the sentinel, outside the root",
                  csess.read_path == sentinel and not is_under(sentinel, sroot))
            check("without a sentinel the stub reads notes.txt in the project (the 3b canary)",
                  make(True, Smoke(DUMMY_KEY)).read_path == os.path.join(sroot, "project", "notes.txt"))
            write_smoke_marker("claude-code", sbase, srepo, {"version": "2.1.310"})
            sess = make(False, real)
            check("a smoke Session with the marker: real, no stub, the right repo and log tag",
                  sess.real and sess.stub is None and sess.project == os.path.join(sroot, "project-2")
                  and sess.tag("run") == "run-project-2" and sess.label == "smoke")
            check("a client whose version differs from the marker's is a Stop before the run",
                  stops(sess.bind_version, "2.1.311") is not None)
            check("the Stop tells the operator to rerun smoke-canary",
                  "smoke-canary" in (stops(sess.bind_version, "2.1.311") or ""))
            check("a version that could not be read is a Stop too", stops(sess.bind_version, "") is not None)
            check("a client at the marker's version passes", stops(sess.bind_version, "2.1.310") is None and sess.version == "2.1.310")
            write_smoke_marker("claude-code", sbase, srepo, {})
            check("a marker without a version binds nothing it can match: a Stop",
                  stops(make(False, real).bind_version, "2.1.310") is not None)
            check("a canary Session has no marker version to hold the client to",
                  stops(make(True, Smoke(DUMMY_KEY)).bind_version, "9.9.9") is None)
            write_smoke_marker("claude-code", sbase, srepo, {"version": "2.1.310"})
            check("its guard passes with the run's key in the environment", stops(sess.guard, "selftest") is None)
            sess.env["ANTHROPIC_AUTH_TOKEN"] = DUMMY_KEY
            check("its guard refuses another key in the environment", stops(sess.guard, "selftest") is not None)
            sess.env["ANTHROPIC_AUTH_TOKEN"] = key
            sess.env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = "claude-sonnet"
            check("its guard refuses a changed alias", stops(sess.guard, "selftest") is not None)
            sess.env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = alias
            with open(os.path.join(sroot, "AGENTS.md"), "w", encoding="utf-8", newline="\n") as f:
                f.write("x\n")
            check("its guard refuses an AGENTS.md above the repo", stops(sess.guard, "selftest") is not None)
            os.remove(os.path.join(sroot, "AGENTS.md"))
            check("its guard passes again without it", stops(sess.guard, "selftest") is None)
            with open(os.path.join(srepo, "tools", "ps", "ArtemisServer.psm1"), "a", encoding="utf-8", newline="\n") as f:
                f.write("changed\n")
            check("a bound file changed since the canary: a new smoke Session is refused", stops(make, False, real) is not None)
            # OpenCode's machine-wide config directory: injected, so nothing under the real ProgramData is read or made.
            machine = os.path.join(tmp, "fake-programdata", "opencode")
            dirs = globals()["machine_config_dirs"]
            globals()["machine_config_dirs"] = lambda: [machine]
            try:
                osess = Fake(sroot, "opencode", True, None, creds_stat(), Smoke(DUMMY_KEY))
                check("an opencode smoke Session's guard passes with no machine-wide config", stops(osess.guard, "selftest") is None)
                os.makedirs(machine)
                check("an opencode smoke Session's guard stops when the machine-wide config exists",
                      stops(osess.guard, "selftest") is not None)
                check("a claude-code Session's guard does not look at it", stops(sess.guard, "selftest") is None)
                os.rmdir(machine)
                oproj = os.path.join(sroot, "project")
                os.makedirs(oproj, exist_ok=True)
                with open(os.path.join(oproj, ".env"), "w", encoding="utf-8") as f:
                    f.write("x\n")
                check("an opencode smoke Session's guard stops on a .env in the project", stops(osess.guard, "selftest") is not None)
                os.remove(os.path.join(oproj, ".env"))
                with open(os.path.join(oproj, "bunfig.toml"), "w", encoding="utf-8") as f:
                    f.write("x\n")
                check("an opencode smoke Session's guard stops on a bunfig.toml in the project",
                      stops(osess.guard, "selftest") is not None)
                os.remove(os.path.join(oproj, "bunfig.toml"))
                check("and passes again without them", stops(osess.guard, "selftest") is None)
            finally:
                globals()["machine_config_dirs"] = dirs
        finally:
            globals()["require_smoke_canary"] = gate
            globals()["ANCESTOR_NAMES"] = names

    def t_machine():
        pd = os.path.join(tmp, "pd")
        check("the machine-wide config directories: %ProgramData%\\opencode, then C:\\ProgramData\\opencode",
              machine_config_dirs({"ProgramData": pd}) == [os.path.join(pd, "opencode"), "C:" + bs + "ProgramData" + bs + "opencode"]
              and machine_config_dirs({}) == ["C:" + bs + "ProgramData" + bs + "opencode"]
              and machine_config_dirs({"PROGRAMDATA": pd})[0] == os.path.join(pd, "opencode"))
        scratch = os.path.join(tmp, "scratch-programdata", "opencode")
        check("no machine-wide config directory: passes", stops(check_machine_config, [scratch]) is None)
        os.makedirs(scratch)
        check("a machine-wide config directory: Stop", stops(check_machine_config, [os.path.join(tmp, "none"), scratch]) is not None)
        os.rmdir(scratch)
        # Bun standalone builds load .env* and bunfig.toml from the cwd (a bunfig preload would run code).
        proj = os.path.join(tmp, "bunproj")
        os.makedirs(proj)
        with open(os.path.join(proj, "calc.py"), "w", encoding="utf-8") as f:
            f.write("x = 1\n")
        check("a project without .env* or bunfig.toml passes", stops(check_project_files, proj) is None)
        for name in (".env", ".env.local", ".ENV.production", "bunfig.toml", "BunFig.toml"):
            p = os.path.join(proj, name)
            with open(p, "w", encoding="utf-8") as f:
                f.write("x\n")
            check(f"a project with {name} stops the launch", stops(check_project_files, proj) is not None)
            os.remove(p)
        os.makedirs(os.path.join(proj, ".env.d"))
        check("a directory named .env.d stops it too", stops(check_project_files, proj) is not None)
        os.rmdir(os.path.join(proj, ".env.d"))
        check("a project that is gone does not raise", stops(check_project_files, os.path.join(tmp, "no-such")) is None)
        # the aside name never collides: a second rename in the same second takes a counter
        victim = os.path.join(tmp, "out.json")
        for n in range(3):
            with open(victim, "w", encoding="utf-8") as f:
                f.write(str(n))
            rename_aside(victim)
        asides = sorted(p for p in os.listdir(tmp) if p.startswith("_old-") and "out.json" in p)
        check("rename_aside twice in a second keeps all three", len(asides) == 3 and not os.path.exists(victim), str(asides))
        with open(victim, "w", encoding="utf-8") as f:
            f.write("x")
        check("write_json_atomic over an existing file does not raise", stops(write_json_atomic, victim, {"a": 1}) is None
              and json.load(open(victim, encoding="utf-8")) == {"a": 1})

    def t_confine():
        marker = "ARTEMIS-SENTINEL-" + "0f3a" * 6
        sentinel = os.path.join(base, "sentinel-x.txt")

        def anth(blocks):
            return {"body": {"messages": [{"role": "user", "content": "go"}, {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": sentinel}}]},
                {"role": "user", "content": blocks}]}}

        def oai(text):
            return {"body": {"messages": [{"role": "user", "content": "go"}, {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read", "arguments": json.dumps({"filePath": sentinel})}}]},
                {"role": "tool", "tool_call_id": "c1", "content": text}]}}

        first = {"body": {"messages": [{"role": "user", "content": "go"}]}}
        refused = [{"type": "tool_result", "tool_use_id": "t1", "is_error": True,
                    "content": "Claude requested permissions to read from outside, but you haven't granted it yet."}]
        read_ok = [{"type": "tool_result", "tool_use_id": "t1", "content": marker + "\n"}]
        check("claude-code: a refused read passes", confinement_problems([first, anth(refused)], "claude-code", marker) == [])
        check("claude-code: a read that returned the sentinel fails",
              confinement_problems([first, anth(read_ok)], "claude-code", marker) != [])
        check("claude-code: a tool result that is not an error fails",
              confinement_problems([first, anth([dict(refused[0], is_error=False)])], "claude-code", marker) != [])
        check("claude-code: no tool call at all fails (the refusal was not shown)",
              confinement_problems([first], "claude-code", marker) != [])
        check("opencode: a denied read passes",
              confinement_problems([first, oai("Permission denied: external_directory")], "opencode", marker) == [])
        check("opencode: a read that returned the sentinel fails",
              confinement_problems([first, oai(marker)], "opencode", marker) != [])
        check("opencode: a result that does not say it was refused fails",
              confinement_problems([first, oai("file is empty")], "opencode", marker) != [])
        check("opencode: no tool call at all fails", confinement_problems([first], "opencode", marker) != [])
        check("the sentinel's contents anywhere in a request fail it, whatever the result",
              confinement_problems([dict(first, extra=marker), anth(refused)], "claude-code", marker) != [])

    def t_verdict():
        ok = {"hosts": ["127.0.0.1:8090"], "redirect_ok": True}
        res = dict(ok)
        check("a completed run with the redirect proof and 8090 is clean", smoke_verdict(res) is None and res["hosts_status"] == "clean")
        res = dict(ok, hosts=[])
        check("an empty hosts list is inconclusive, never clean", smoke_verdict(res) is None and res["hosts_status"] == "inconclusive")
        res = dict(ok, redirect_ok=False)
        check("no redirect proof is a Stop", smoke_verdict(res) is not None)
        check("hosts beyond 8090 are never clean",
              (lambda r: (smoke_verdict(r), r["hosts_status"]))(dict(ok, hosts=["127.0.0.1:8090", "1.2.3.4:5"]))[1] == "inconclusive")
        h = {"a": "1"}
        check("unchanged hashes pass", stops(check_unchanged, h, lambda: dict(h)) is None)
        check("hashes that moved during the canary refuse the marker", stops(check_unchanged, h, lambda: {"a": "2"}) is not None)
        check("a creds file that was absent is a Stop in smoke mode", stops(require_creds, None) is not None
              and stops(require_creds, (1, 2)) is None)

    sections = [("environment", t_env), ("opencode config", t_config), ("canary marker", t_marker), ("8090 owner", t_owner),
                ("argv", t_argv), ("redact", t_redact), ("arguments", t_args), ("connection watch", t_hosts),
                ("confinement canary", t_confine), ("verdict", t_verdict), ("machine-wide config", t_machine),
                ("session wiring", t_session)]
    try:
        for title, fn in sections:
            say(f"[{title}]")
            try:
                fn()
            except Exception as e:  # a section that cannot run is a failure, not a crash
                check(f"{title}: raised {type(e).__name__}", False, str(e)[:300])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    say(f"selftest-smoke: {'FAILED, ' + str(len(fails)) + ' check(s)' if fails else 'all checks passed'}")
    return 1 if fails else 0


def blank_opencode_key_main(argv):
    """`blank-opencode-key`: blanks the key in OpenCode's two configs under the root (see `blank_opencode_key`). The
    smoke script runs it after a clean key scan. Exit 0, or 2 when a config path is outside the root."""
    ap = argparse.ArgumentParser(prog="capture_clients.py blank-opencode-key")
    ap.add_argument("--root", required=True)
    a = ap.parse_args(argv)
    try:
        blank_opencode_key(os.path.abspath(a.root))
    except Stop as e:
        print(f"STOP: {e}", file=sys.stderr)
        return 2
    return 0


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    if len(sys.argv) > 1 and sys.argv[1] in ("smoke", "smoke-canary"):  # against the real server, or its canary
        return (smoke_main if sys.argv[1] == "smoke" else smoke_canary_main)(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "blank-opencode-key":
        return blank_opencode_key_main(sys.argv[2:])
    ap = argparse.ArgumentParser(description="Run OpenCode and Claude Code against the recording stub.")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--only", choices=CLIENTS)
    ap.add_argument("--canary", action="store_true")
    ap.add_argument("--print-env-names", action="store_true")
    ap.add_argument("--selftest-smoke", action="store_true")
    args = ap.parse_args()
    if args.selftest_smoke:
        return selftest_smoke()
    if args.print_env_names:
        print_env_names()
        return 0
    assert PORT != 8080
    root = os.path.abspath(args.root)
    creds_before = creds_stat()
    ok = True
    try:
        key_file = prepare_root(root)
        for client in ([args.only] if args.only else CLIENTS):
            ok = run_client(root, client, args.canary, key_file, creds_before) and ok
        if creds_stat() != creds_before:
            raise Stop("the credentials file's size or mtime changed")
    except Stop as e:
        print(f"STOP: {e}", file=sys.stderr)
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
