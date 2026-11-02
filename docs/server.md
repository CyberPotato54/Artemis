# `artemis serve`

`artemis serve` runs Qwen3.6-35B-A3B behind an OpenAI-compatible and an Anthropic-compatible HTTP API. It serves one
model, one request at a time, from one GPU. This page is the settings reference: every key, its flag and default, the
endpoints, what is accepted and ignored, what is not enforced, the logs, the lifecycle, memory use, exit codes and the
known windows where the behaviour is not what you would assume.

## Running it

```
artemis serve --model /path/to/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
artemis serve --config /path/to/artemis.toml
```

- The settings come from one TOML file, and any flag you give replaces the file's value for that key.
- `--config PATH` names the file. A missing file is an error.
- Without `--config`, the default file is `%APPDATA%\artemis\artemis.toml` on Windows, and
  `$XDG_CONFIG_HOME/artemis/artemis.toml` (else `~/.config/artemis/artemis.toml`) elsewhere. If it does not exist, the
  defaults apply. A file saved with a byte-order mark is read normally.
- `model` is the only required setting.
- An unknown key in the file is an error that names the key, so a typo cannot silently drop a setting.
- The server refuses to start (exit 1) on a bad config, an unreadable or empty key file, an address it may not bind, or
  a GGUF that is not the supported model. These checks run before the port opens, so a refused start leaves no `start`
  line in the log. The model is not loaded at this point (see Lifecycle). Its file is opened, and its tensors,
  hyperparameters, chat template and vocabulary are checked against the Qwen3.6-35B-A3B contract; the first
  difference in the tensors and hyperparameters is named. The vocabulary and the chat template must match the ones
  Artemis is tested with (a sha256 pin each); a digest mismatch names the pinned item. A different vocabulary is
  always refused. A different chat template is refused unless `allow_custom_template = true`, which uses it and
  writes a `config_warning`. A render of any template is bounded to 50 million template instructions; one that runs
  out is a 500.
- Stop it with Ctrl-C (see Lifecycle for what that does).

An example file:

```toml
model = "D:/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
port = 8090
api_key_file = "D:/keys/artemis-key.txt"
idle_unload_s = 900

[sampling]
temperature = 0.7
```

The production config is a template, `tools/config/artemis-production.toml.in`. It holds the settings the production
server runs with (`host = "127.0.0.1"`, `health_service = "artemis"`, `lazy_load = true`, `idle_unload_s = 600`,
`queue_timeout_s = 300`, `stall_s = 120` and `stall_load_s = 300`, the last two written out so that a change of default
cannot move them) and four placeholders: `{{MODEL}}`, `{{PORT}}`, `{{API_KEY_FILE}}` and `{{LOG_DIR}}`. Whoever installs
or tests Artemis fills them and writes the result as the config file. The path placeholders sit inside TOML literal
strings (single quotes), so a Windows path needs no escapes. A value may hold no single quote, no line break and no
`{{`; `fill_template` (`crates/server/src/config.rs`) and `New-ArtemisConfig` (`tools/ps/ArtemisServer.psm1`) apply the
same rules and name the placeholder in an error, never its value. The GPU runner fills port 8090 and refuses a filled
config that names 8080 anywhere. The snapshot pool and parking keep their defaults.

## Changes in recent builds

These changes can affect a config or a client that worked with an earlier build:

- The first `serve` that finds no tuned file runs `artemis setup` before it opens the port (about 80 s with the model file in the OS cache, once), and
  later starts read the tuned file it wrote. `first_run_setup = false` skips the measuring, and `ARTEMIS_TUNED=off`
  turns the tuned file off (see The tuned file).
- A relative `model`, `api_key_file` or `log_dir` in a config file is now read from the config file's folder. A flag's
  relative path is still read from the working directory. `~` is not expanded.
- A request with an `Origin` header gets a 403 unless `allowed_origins` lists the origin (Auth and network).
- A POST with a body must carry a JSON content type, or it gets a 415.
- A key under 32 characters refuses the start when `host` is not a loopback address.
- Slow clients are cut off (30 s for a request head, 60 s for a body), at most 128 connections are served at once, and a
  request body of the generation endpoints may hold at most 1,000,000 JSON values.
- The request logs older than `log_keep_days` (14 by default) are removed at start and at each new day.
- New settings: `allowed_origins`, `allow_custom_template`, `special_tokens_in_content`, `anthropic_thinking` and
  `log_keep_days`.
- `health_service` takes any name of 1 to 32 lower-case letters, digits or hyphens, not starting with a hyphen.
- On `/v1/messages`, a prompt over the context now gets `prompt is too long: N tokens > M maximum`.
- On `/v1/messages`, a request that says nothing about thinking and sets `max_tokens` under 1,024 gets half of
  `max_tokens` as its reasoning budget (or the configured budget when that is smaller).
- `/v1/load` and `/v1/unload` read at most 4 KiB of body; one byte more is a 413.
- At most four requests are prepared at once; a fifth waits up to 10 s and then gets a 503.
- A request the door refuses (a foreign origin, a body that is not JSON, a wrong key or Host) closes its connection
  after the error.
- A request leaves one line on stderr, plus a few bounded notes (see Logs), instead of one `info` line for each field; a
  script that reads stderr may notice.

## Settings

Every key has a flag of the same name with underscores turned into hyphens (`idle_unload_s` is `--idle-unload-s`),
except the `[sampling]` table, which has no flags, and the `[speculation]` table, which has two
(`--speculation-enabled` and `--speculation-sources`). A boolean flag takes `true` or `false` (`--lazy-load false`);
given alone it means `true`.

| Key | Flag | Default | Meaning |
|---|---|---|---|
| `model` | `--model` | none (required) | Path to the GGUF. A relative path in the config file is read from the file's folder; one given by the flag is read from the working directory. The same holds for `api_key_file` and `log_dir`. |
| `host` | `--host` | `127.0.0.1` | The address to listen on. Without a key it must be loopback (see Auth and network). |
| `port` | `--port` | `8080` | The port to listen on. 8080 is this binary's own default. |
| `api_key_file` | `--api-key-file` | unset (auth off) | A file whose first non-empty line is the API key. |
| `allowed_origins` | `--allowed-origins` | empty | Browser origins allowed to call the server, each `scheme://host[:port]` with any scheme (`http://localhost:3000`, `vscode-webview://abc`, `app://local`) and nothing after the port. The flag takes a comma-separated list. An entry of any other shape refuses the start, `null` and an empty host included. See Auth and network. |
| `ctx` | `--ctx` | `65536` | The context window in tokens, from 512 to 262,144 (the model's trained window); a value outside that refuses the start. A prompt longer than this is a 400 before it queues. `/health` reports it as `max_context`. |
| `snapshot_pool_gib` | `--snapshot-pool-gib` | `8` | Host RAM for prompt-reuse snapshots, in GiB. `0` turns reuse off. The pool is cut at load to what the machine has (see Memory). |
| `park_snapshots` | `--park-snapshots` | `true` | Keep the snapshots in ordinary RAM across an idle unload or `/v1/unload`, and restore them at the next load. With `false` an unload drops them. A shutdown parks nothing either way. |
| `lazy_load` | `--lazy-load` | `true` | Load the model on the first request that needs it. With `false` the load starts as soon as the port is open. |
| `idle_unload_s` | `--idle-unload-s` | `600` | Unload the model after this many seconds without a request. `0` means never. |
| `queue_timeout_s` | `--queue-timeout-s` | `300` | How long a request may wait in the queue (the wait for a load included) before it gets a 503. At least 1; `0` refuses the start. |
| `stall_s` | `--stall-s` | `120` | The stall watchdog's bound, in seconds: the engine has work and has shown no progress (no token, no prefill segment, no finished command, no change of state) for this long, and the process ends with exit code 4 (see Lifecycle). Work that arrives is not progress. `0` turns the watchdog off, loads included. |
| `stall_load_s` | `--stall-load-s` | `300` | The same bound while the engine's state is `loading` or `recovering` (a load, or the reload after a GPU loss). A load shows progress as it goes (each weight tensor uploaded and each layer of expert copies counts), so this bound judges a load that stops moving, not one that is slow. It replaces `stall_s` for that time; a value below `stall_s` (both nonzero) gets a `config_warning` at start, not a refusal. `0` lifts only this bound: a stuck load is then never judged. With `stall_s = 0` it has no effect. |
| `model_id` | `--model-id` | `qwen3.6-35b-a3b` | The model id the API reports in `/v1/models`, `/health` and responses. |
| `health_service` | `--health-service` | `artemis` | The `service` field of the `/health` body. Any name of 1 to 32 lower-case letters, digits or hyphens, not starting with a hyphen, for a supervisor whose health check looks for a particular name; anything else refuses the start. It changes nothing else. |
| `thinking` | `--thinking` | `true` | Whether a request that says nothing about reasoning gets it. On `/v1/messages` see also `anthropic_thinking`. |
| `thinking_budget` | `--thinking-budget` | `0` | The default reasoning budget in tokens for a request that thinks and names no budget. `0` is no budget. |
| `[sampling]` | none | see below | What a request that sets none of these gets. |
| `[speculation]` | `--speculation-enabled`, `--speculation-sources` | see below | Speculative decoding. The flags set the table's `enabled` and `sources` over the file (`--speculation-sources lookup` or `lookup,mtp`). |
| `log_dir` | `--log-dir` | `%LOCALAPPDATA%\artemis\logs` on Windows, else `~/.local/state/artemis/logs` | Where the JSON-lines logs go. Created if missing. If neither base variable is set, `artemis-logs` under the working directory. |
| `log_level` | `--log-level` | `info` | The level of the human-readable lines on stderr: `error`, `warn`, `info`, `debug` or `trace`. It does not change the JSON-lines logs. |
| `log_keep_days` | `--log-keep-days` | `14` | Days of request logs to keep. When the server starts, and when the log moves to a new day's file, it removes the files `artemis-YYYY-MM-DD.jsonl` in `log_dir` dated that many days before today or earlier (with 14 on 2026-11-14, the files from 2026-11-01 on stay). It touches only names of exactly that form with a real date, so a copy such as `artemis-2026-10-01.jsonl.bak`, `artemis-latest.jsonl` and any other file in a shared folder stay. The unit is calendar days in UTC, so with 1 only today's file stays and the previous day's file goes as the day changes, even though its last lines are minutes old; use 2 or more to keep a full day behind. 0 keeps every file. A file that cannot be removed is left, and nothing fails. |
| `device_uuid` | `--device-uuid` | unset (auto) | Use the GPU with this UUID: 32 hex digits, with or without dashes. By default the engine picks the discrete GPU. |
| `cache_slots` | `--cache-slots` | unset (auto) | The most expert-cache slots per layer the engine will hold, 0 to 256 (a layer has 256 experts; a larger value refuses the start). With `0` every expert is read from host memory. By default the cache is sized to the VRAM left after the model loads. |
| `f32_prefill` | `--f32-prefill` | `false` | Prefill with the f32 kernels instead of the split-f16 cooperative-matrix kernels. Slower; for comparisons and for a prompt that overflows f16. |
| `allow_custom_template` | `--allow-custom-template` | `false` | Start with a GGUF whose chat template differs from the one Artemis is tested with. Without it that start is refused. The server writes a `config_warning` naming the template's sha256. The vocabulary has no such switch. |
| `special_tokens_in_content` | `--special-tokens-in-content` | `true` | Whether the text of a control token in client text (a message, a file an agent read, a tool result, a tool's schema) is read as that token. `true` parses special tokens everywhere, as llama.cpp does. With `false` the text stays plain text and the template's own turn markers stay tokens (see Not enforced). A request that holds the character U+FDD0, which the server uses internally in this mode, is then a 400. |
| `anthropic_thinking` | `--anthropic-thinking` | `true` | Whether a `/v1/messages` request that states no thinking (it sends neither `thinking` nor `output_config.effort`, or only an unknown type or effort) thinks when `thinking` is on. `true` is what llama.cpp's server does. With `false` such a request does not think, which is what Anthropic's API means by an absent field; a request that states thinking is unchanged, and so is the OpenAI endpoint. |
| `first_run_setup` | `--first-run-setup` | `true` | Whether the first `serve` that finds no tuned file runs `artemis setup` to measure this PC before it opens the port (see The tuned file). `false` skips that, so a server whose restarts must stay short never pauses to measure; run `artemis setup` beforehand instead. Given alone the flag means `true`. |

The `[sampling]` table (each value is checked as a request's would be, and a bad one refuses the start):

| Key | Default |
|---|---|
| `temperature` | `0.6` |
| `top_p` | `0.95` |
| `top_k` | `20` |
| `min_p` | `0.0` |
| `presence_penalty` | `0.0` |

The GGUF declares its own sampling defaults (temperature 1.0, top-k 20, top-p 0.95). Artemis uses the values above
because production use of this model settled on them; change them here. A request overrides any of them, and
temperature 0 is greedy.

The `[speculation]` table turns on speculative decoding with two draft sources: prompt lookup, and the model's own MTP
block. Each step drafts a few tokens, from the session's own history or from the MTP block, and checks them in one pass,
keeping the ones the request's own sampler picks; the gain is speed on text that repeats its context. It is on by default, and the production template carries no line for it. A bad
value, or a key not listed here, refuses the start and names `speculation.<key>`; that holds when `enabled = false`
too.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | `false` is the rollback: today's one-token decode, with nothing allocated for speculation. `true` with the default `sources` loads the MTP block, which costs about 58.5 MiB of VRAM and 464 MiB of host memory (see Memory, MTP). Lookup alone costs no VRAM up to a `ctx` of about 170K at the default prefill chunk of 2,048; above that, the verify's attention buffers outgrow prefill's work area, by about 90 MiB at `ctx` 262,144, which can cost the cache a slot a layer. `--speculation-enabled false` sets it for one run; given alone the flag means `true`. |
| `sources` | `["lookup", "mtp"]` | Which sources draft: `"lookup"` (prompt lookup) and `"mtp"` (the model's MTP block). The list names which run; its order does nothing, and `source_choice` says how a step picks between them. An unknown name, a name listed twice and an empty list are each refused (to turn speculation off set `enabled = false`). `--speculation-sources lookup` sets it for one run (comma-separated); the file's list is replaced. `artemis generate` and `spec-check` take the same list as `--sources`. |
| `max_draft` | `4` | The most draft tokens one step checks, from 1 to 15. The default is the faster of 3 and 4 on the speed prompts (4: the edit prompt ran about 1% faster over several rounds, within run noise; the medians of six tied). |
| `ngram_min` | `2` | Prompt lookup's shortest n-gram, at least 1. |
| `ngram_max` | `4` | Its longest, at most 16 and at least `ngram_min`. |
| `min_gain` | `1.03` | The predicted speed-up below which a step runs plain decode. A finite number, 1 or more. |
| `accept_half_life` | `32.0` | Draft tokens per halving of an old outcome's weight in the accept rate. Above 0. Lookup also keeps a rate per kind of match (2-, 3- and 4-token matches and a continuing run), which chooses its drafts and is not shown in `/v1/stats`. |
| `cost_slope_ms` | `4.4` | The controller's cost, in ms, of each extra token in a step, until it has timed its own. 0 or more. The default is the slope measured on the speed prompts at 8,192 positions, rounded to 0.1 ms. |
| `accept_prior` | `0.5` | The accept rate a fresh controller starts from, from 0 to 1. |
| `probe_every` | `64` | MTP drafts one token (`k = 1`) once this many plain steps have run since it last drafted, so its accept rate stays measured. At least 1. |
| `mtp_window` | `2048` | The last positions of the prompt that prefill writes into the MTP's KV. From 1 to `mtp_ctx / 2`. The load cuts it to half the clamped rows (a `ctx` under `mtp_ctx` clamps them). |
| `mtp_ctx` | `8192` | The MTP's KV rows (positions); 16 MiB at 8,192. From 512 to 65,536 and a multiple of 256. The load clamps it to the session's `ctx`, and a session under 512 positions refuses MTP. |
| `mtp_ring` | `256` | The hidden-row ring's length (2 MiB at 256). From 16 to `mtp_ctx / 4` and a multiple of 16. The load clamps it to a quarter of the rows and to the prefill chunk minus 16, rounded down to 16 (the load line reports a cut). |
| `mtp_accept_prior` | `[0.79, 0.81, 0.74, 0.78]` (the pooled depth rates measured on the speed prompts; depth 1 from the edit prompt) | MTP's accept rate at each draft depth until it has measured its own; a depth past the list takes the last value. 1 to 15 values, each from 0 to 1. |
| `mtp_cost_ms` | `1.6` | The prior for the MTP draft pass's time per draft token, in ms, until it is measured. Finite, 0 or more. |
| `source_choice` | `"best"` | Which source drafts when both have a draft: `"priority"` (lookup first) or `"best"` (the source whose draft predicts more tokens per ms; measured faster on the speed prompts). `artemis generate` and `spec-check` take it as `--source-choice`. |
| `mtp_head_rows` | `65536` | `0` drafts MTP tokens with the full output head; N drafts from its first N rows (no VRAM; the draft ids are below N, the output cannot change). 0 is the full head, else 1024 or more; a value above the model's vocabulary loads as the full head. Measured faster at 65,536 on the speed prompts. `--mtp-head-rows` on `generate` and `spec-check`. |

To roll back, set `enabled = false` in `[speculation]` (or start with `--speculation-enabled false`) and restart; no
other setting changes.

Three settings write a warning to the log at start: `snapshot_pool_gib = 0` ("prompt reuse is off"),
`idle_unload_s = 0` ("the model never unloads when idle") and `stall_s = 0` ("the stall watchdog is off"). A fourth
warning is for a `stall_load_s` below `stall_s`, both nonzero ("stall_load_s (N s) is below stall_s (M s): a load is
given less time than a request"); the server still starts. A fifth belongs to the hidden test flag (see For scripts).
Each is a `config_warning` line.

## The tuned file

`artemis setup` measures the PC once and writes what it chose to a per-user file, the tuned file. `serve` reads it at
every start, under `artemis.toml`, so a first run needs no config edit and the config file stays the place for later
changes.

- **Where it is.** `%LOCALAPPDATA%\artemis\tuned.toml` on Windows, and `$XDG_STATE_HOME/artemis/tuned.toml` (else
  `~/.local/state/artemis/tuned.toml`) elsewhere. It lives in the local state folder because its values belong to one
  machine; `artemis.toml` may be copied between machines.
- **`ARTEMIS_TUNED`.** The environment variable `ARTEMIS_TUNED` set to a path moves the file. Set to `off` (in any
  case) it turns the layer off: nothing is read or written, and `artemis setup` refuses because it has nowhere to write.
  Use `off` for a test or a script that must run with the defaults. When the base variable of the default path is unset
  (`LOCALAPPDATA` on Windows, `HOME` elsewhere, with `XDG_STATE_HOME` unset), `serve` treats the layer as off and writes
  one `config_warning` saying so; set `ARTEMIS_TUNED` to a file path to use it.
- **What it may set.** `device_uuid`, and in `[speculation]`: `enabled`, `sources`, `cost_slope_ms` and `mtp_cost_ms`,
  under the same names and rules as in `artemis.toml`, so a line can be copied across. Each chosen value has a comment
  line above it that says why it was chosen. A `[setup]` table records what was measured (the schema, the Artemis
  version, the date, the GPU and driver, the model's size and the scores); `serve` does not apply it.
- **Precedence.** The defaults, then the tuned file, then `artemis.toml`, then the flags. A key that your
  `artemis.toml` or a flag sets is never replaced by a tuned value. `ctx`, `cache_slots` and `f32_prefill` are never
  tuned: setup measures with the values you set for them.
- **The `tuned` log line.** At every start `serve` writes a `tuned` line with the file's `path`, the keys it `applied`
  and the keys it `ignored` because you set them.
- **Refused.** A file with an unknown key, a bad value, a different schema, or text that is not TOML fails the start
  with a message that names the file and the key and ends with `run artemis setup --force`. A tuned value meets the
  same rules as the same key in `artemis.toml`.
- **Out of date.** A file written by another Artemis version, or for a model of another size, still loads, and `serve`
  writes a `config_warning` for each difference. The engine writes another when the GPU or the driver differs from the
  one measured. `artemis setup --force` measures again.
- **First start.** When no tuned file exists, `first_run_setup` is `true` and the model has passed the contract check,
  `serve` runs `artemis setup` (see below), prints what it chose and then opens the port. If that fails, `serve` exits 1
  with the message and writes no file, so the next start tries again.
- **Writes.** Setup writes the file to a temporary name in the same folder, syncs it and renames it over the old one, so
  an interrupted setup leaves no tuned file and no partial one, and the next `serve` runs setup again. A temporary file
  a killed run left behind is removed by the next write.

## `artemis setup`

```
artemis setup [--config PATH] [--model PATH] [--device-uuid UUID] [--force] [--json FILE]
```

Setup measures this PC and writes the tuned file. `artemis serve` runs the same measuring on its first start, when it
finds no tuned file and `first_run_setup` is `true` (the default), and opens the port only after it is done.

- **Flags.** `--config` and `--model` are `serve`'s: setup reads the same config file, so it measures with your `ctx`,
  `cache_slots`, `device_uuid`, `f32_prefill` and `[speculation]` table, and chooses only the settings you left unset.
  `--device-uuid` names the GPU. `--force` measures again and replaces the file; without it, a tuned file that reads is
  reported (when it was measured, on which GPU, the expected speed) and left alone, and the model is not opened.
  `--json FILE` also writes what was measured and chosen, with the reasons, as JSON. Its `chosen` is the tuned file's
  content, and `null` for a key you set yourself.
- **What it does, in order.** It refuses when `ARTEMIS_TUNED` is `off` or the file has no place. It takes the lock
  `<tuned file>.lock` (below), checks the model against the contract and the pins as `serve` does, lists the GPUs and
  picks the discrete one (the one with the most memory when there are several, recorded as `device_uuid`; your
  `device_uuid` or `--device-uuid` picks one instead), and warns when another program holds more than 2 GiB of the GPU's
  memory. It refuses when the host has less free memory than loading the model needs, naming both figures. It measures
  the link to the GPU, then loads the model with MTP and runs short decodes with speculation on, and loads it without
  MTP for prompt lookup alone and plain decode. When the controller's measured costs differ from the built-in ones by
  more than a quarter, it runs one more pass on the MTP load with the measured costs and keeps them if they are at
  least 1.03 times as fast. It keeps MTP when it is at least 1.02 times as fast as lookup alone, and chooses plain
  decode over speculation only when it is at least 1.03 times as fast. Only one engine is open at a time.
- **Time.** The whole run, loads included, has a limit of 600 s; the measuring stops a minute short of it and uses what
  it has. A run takes a few minutes.
- **What it prints.** The start text, the GPUs found, `Loading the model (1 of 2)...` and
  `Measuring <what> (1 of 2)...` lines, and at the end `Done in <m> min <s> s. Settings written to <path>.`, one line
  for each value it chose with the reason, the free host memory with the size the snapshot pool will get, and the
  expected decode speed on short prompts. A card under 15 GiB of memory gets a note that Artemis targets 16 GB cards
  and that its speed on smaller cards is not claimed.
- **The lock.** `<tuned file>.lock` is created exclusively and holds the process id and the time. A second setup that
  finds a lock whose process is still running and that is younger than twice the time limit stops with
  `Another artemis setup is running (pid <p>). If it is not, delete <lock file>.`; a lock whose process is gone (a power
  cut, a closed console window, a killed process) or that is older than the limit belongs to a dead run and is replaced
  at the next start. The lock is released when the run ends, when it fails and when you press Ctrl-C. A Ctrl-C before
  the settings are written stops setup with `Setup was interrupted. No settings were written.`, and a first start
  that was interrupted writes `setup_failed` to the log.
- **Nothing on failure.** An error at any step (a refused model, a missing GPU, too little memory, a load that fails, a
  full disk) writes no tuned file and leaves an existing one as it was.
- **Log lines.** `setup_start` (`reason` is `first run` or `asked`, and `path`), `setup_done` (`seconds`, `limit_hit`,
  `target`, `gpu_name`, `vram_bytes`, `cache_slots`, `scores`, `enabled`, `sources`, `priors`, `measured_slope_ms`,
  `measured_mtp_d1_ms`, `expected_tok_s`, `path`) and `setup_failed` (`message`). `enabled`, `sources` and `priors` are
  the values the server will run with: setup's choice for the settings it chose, and your own value for the settings
  you set. A first `serve` that setup refuses (a lock another setup holds, say) writes `setup_start` and then
  `setup_failed` with the reason, and exits 1. A standalone `artemis setup` that is refused before the measuring starts
  writes none of them and prints the reason.
- **Short restarts.** A server whose restarts must stay short sets `first_run_setup = false` or runs `artemis setup`
  beforehand.
- **A different GPU or driver.** At each engine load the server compares the device's UUID and the driver with the
  tuned file's, and writes a `config_warning` when either differs: `The tuned settings were measured on <gpu> with
  driver <d>; this GPU or driver differs. Run artemis setup --force to measure again.` A `device_uuid` in the tuned file
  that names a GPU the PC no longer has makes the load fail. The log line names the file and `artemis setup --force`;
  the reply to an API client names "the tuned settings file" and the command, and leaves the local path out.
- **Bare `artemis`.** With no command, `artemis` prints how to start the server and exits 0.
- **Tests and scripts.** Set `ARTEMIS_TUNED=off` in any script or test that starts `artemis serve`: without it the
  start runs setup and writes the real tuned file.

## Endpoints

| Endpoint | Auth | Behaviour |
|---|---|---|
| `POST /v1/chat/completions` | key | OpenAI chat: streaming and not, tools, reasoning, logprobs, usage. |
| `POST /v1/messages` | key | Anthropic messages: streaming and not, `tool_use`, `tool_result`, `thinking`. |
| `POST /v1/messages/count_tokens` | key | `{"input_tokens": N}`: the token count of the full prompt, generation prompt included. It renders and tokenizes only, so it works while the model is unloaded and never queues. A prompt over the context is counted, not rejected. |
| `POST /v1/load` | key | Loads the model now if it is idle and answers 200 `{"status":"ready","loaded":true}` once it is loaded, or 503 with the engine's message if the load fails. If the model is already loaded it answers 200 at once. |
| `POST /v1/unload` | key | Ends the request in flight (a non-streaming client gets 503, a streaming one an error event), answers every queued request 503, unloads, and answers 200 `{"status":"idle","loaded":false}` once the memory is free. The next generation request or `/v1/load` loads again. |
| `GET /v1/models`, `GET /v1/models/{id}` | key | One model, with the shape both OpenAI's and Anthropic's list calls accept. `{id}` returns the entry for the id asked for. Each entry carries `context_length` and `max_model_len`, both the configured `ctx` (65,536 by default), for clients that read the window from the model list. |
| `GET /v1/stats` | key | The engine's counters, their age and the watchdog's progress counter (below). Like every `/v1/*` path it needs the key; with no key it is open only to loopback and the Host check applies. |
| `GET /health` | none | The state of the server (below). |

Any other path is a 404 and a wrong method a 405, in the path's error shape.

### `/health`

`/health` takes no auth, accepts any `Authorization` header, and answers within milliseconds even in the middle of a
prefill. The body is:

```json
{"service":"artemis","status":"ready","version":"0.1.0","loaded":true,"model":"qwen3.6-35b-a3b","max_context":65536,"api_key":true}
```

| `status` | HTTP | Meaning |
|---|---|---|
| `ready` | 200 | The model is loaded. |
| `idle` | 200 | The model is not loaded. The next request loads it. |
| `loading` | 503 | A load is running. Requests wait in the queue meanwhile. |
| `recovering` | 503 | The GPU was lost and the engine is reloading. Requests wait. |

The 503 carries the same body. `loaded` is a JSON boolean, `model` is `model_id`, `max_context` is `ctx`, `api_key`
says whether a key is set, and `version` is the binary's version. `service` is `health_service`.

### `/v1/stats`

`GET /v1/stats` answers from a copy of the engine's counters and never waits on the engine, so it answers at once even
while the engine is stuck. The body is:

```json
{"object":"stats","uptime_s":5400,"state":"ready","progress":{"ticks":812345,"pending":0},"age_s":12.4,"log_write_failures":0,"stats":{"snapshots_held":3,"pool_bytes_used":1992294400,"pool_bytes_limit":8589934592,"restores":41,"restore_bytes":2583691264,"reused_tokens":318400,"prefilled_tokens":96210,"evicted_protected":0,"evicted_unprotected":2,"park_errors":0,"chunks_refused":0,"loads":1,"device_losses":0,"after_done_errors":0,"parked_snapshots":0,"cache":{"slots_per_layer":131,"hits":9120344,"misses":1204551,"staged":88123,"cpu_misses":0,"cpu_timeouts":0,"longest_submit_us":151000},"spec":{"steps":1840,"plain_steps":2210,"drafted":7360,"accepted":4210,"replays":1120,"accept_rate":{"lookup":0.71},"by_size":[{"t":1,"steps":2210,"mean_ms":9.1,"mean_miss_mib":38.2,"tokens":2210},{"t":5,"steps":1840,"mean_ms":14.8,"mean_miss_mib":71.5,"tokens":6050}],"mtp_depth_rates":[],"mtp_draft":[]}}}
```

| Field | Meaning |
|---|---|
| `uptime_s` | Whole seconds since the server was built. (The idle-unload `stats` line counts from the backend's start, a fraction of a second earlier.) |
| `state` | The engine's state, as `/health` names it: `idle`, `ready`, `loading` or `recovering`. |
| `progress` | The watchdog's inputs: `ticks`, a counter that moves whenever the engine does something (a token, a prefill segment, a finished command, a load or unload ending, and during a load each weight tensor and each layer of expert copies), and `pending`, the work the engine was given and has not finished. |
| `age_s` | Seconds since the engine last answered a stats question; `null` before its first answer. |
| `log_write_failures` | Log lines that could not be written since the start (see Logs). |
| `stats` | The engine's counters at that answer, `null` before the first. `cache` and `spec` inside it are `null` while no model is loaded (`spec` also with speculation off); their counters start again at each load. |

`stats.spec` counts speculation since the last load, in numbers only (no draft or output text). `mtp_depth_rates` and
`mtp_draft` are always present, and empty until MTP has drafted; once it has, `accept_rate` also lists `"mtp"`, as in
`"accept_rate":{"lookup":0.71,"mtp":0.64}`, `"mtp_depth_rates":[0.64,0.55,0.48]` and
`"mtp_draft":[{"k":1,"passes":310,"mean_ms":2.1},{"k":3,"passes":95,"mean_ms":5.8}]`.

| Field | Meaning |
|---|---|
| `steps`, `plain_steps` | Steps that checked a draft (a verify step), and steps that ran without one. |
| `drafted`, `accepted` | Draft tokens checked, and the ones kept. |
| `replays` | Verify steps that kept fewer than all their drafts. |
| `accept_rate` | The controller's current estimate of the chance a draft token is kept, by source (`{"lookup": 0.71, "mtp": 0.64}`; MTP's is at depth 1). A moving average that starts at `accept_prior`; it is not `accepted` over `drafted`. Lookup's figure pools all its matches; the per-kind rates that choose its drafts are not listed. The request line's `spec_accept_rate` is that ratio, for one request. |
| `mtp_depth_rates` | MTP's accept rate at each draft depth, depth 1 first; empty until MTP has run. |
| `mtp_draft` | The MTP draft pass by `k` (tokens drafted in the pass): one row per `k` that ran, with its `passes` and `mean_ms` per pass. |
| `by_size` | One row per step size that ran: `t` tokens through the model (the last token and `t - 1` drafts, so `t = 1` is a plain step), the `steps` of that size, `mean_ms` and `mean_miss_mib` per step (wall time, and MiB the expert cache read), and the positions kept (`tokens`). |

`after_done_errors` also counts a commit fault that came after a burst had ended: the reply was complete and went out,
the cached session was dropped, and the next request starts fresh.

The copy is refreshed when a request that reached the engine ends and at each heartbeat (see Lifecycle). Its `age_s`
counts from when the engine answered, not from when the request ended: under load the refresh queues behind the next
request, so the copy can be one request stale, and `age_s` says so. A 200 answer writes no request line (a poll every
few seconds would drown the log); a refusal (401, 403) is logged like any other.

### Error shapes

The path decides the shape.

- Under `/v1/messages`: `{"type":"error","error":{"type":T,"message":M}}`, where T is `invalid_request_error`,
  `authentication_error`, `permission_error`, `not_found_error`, `request_too_large`, `overloaded_error` or
  `api_error`.
- Every other path: `{"error":{"message":M,"type":T,"param":P,"code":C}}`, where T is `invalid_request_error`,
  `authentication_error`, `permission_error`, `not_found_error`, `server_error` or `service_unavailable`.
- A prompt over the context is a 400 in each API's words. Under `/v1/messages` the message is
  `prompt is too long: N tokens > M maximum`, where M is the context less one, as Anthropic words it and as Claude Code
  recognises it. Elsewhere it is `the prompt is N tokens; the context holds C` with code `context_length_exceeded`.
- A message is one line of at most 500 bytes. It never holds the key or any prompt text.
- An error after a stream's headers have gone out is sent in the stream: OpenAI gets a `data: {"error":{...}}` line and
  then `data: [DONE]`, Anthropic gets an `event: error`. Because headers are sent only once the request has left the
  queue, queue timeouts, load failures and every validation error still arrive as real HTTP statuses.

The statuses you will see: 400 (a malformed request, a prompt over the context), 401 (key), 403 (Host, Origin,
cross-site fetch), 404, 405, 408 (a request body that stalled), 413 (body over 16 MiB), 415 (a body that is not JSON),
499 in the log only (below), 500 (an engine failure during a request, or the engine's thread gone), 503 (queue full,
queue timeout, load failure, GPU memory in use, busy preparing other requests, unload, shutdown, GPU loss).

## Accepted and ignored

These are accepted so that a client does not fail over a setting it rarely depends on. An ignored field is named in the
request's log line (`ignored`), never its value. A name is cut to its first 40 characters, and the line names the first
32 of them; the rest are counted in `ignored_more`. The request's stderr line carries only the count, as `(n fields
ignored)`. So a request cannot put a long text into the log through a field name, nor make a long line with a million of
them.

- Ignored when not at their default, and then logged: `frequency_penalty`, `repetition_penalty`, `logit_bias`,
  `typical_p`, `mirostat`, `mirostat_tau`, `mirostat_eta`.
- Ignored without a word: `user`, `metadata` (including `metadata.user_id`), `store`, `service_tier`, `stream_options`
  beyond `include_usage` (which is honoured), and the values of the `anthropic-beta` and `anthropic-version` headers.
- Ignored and logged: any field the server does not know, an unknown `thinking.type`, an OpenAI `tool_choice` object of
  an unknown type (taken as `auto`), and `output_config.effort` where it has no effect (beside a `thinking` of
  `enabled` or `disabled`, or an unrecognised value) or has the value `max` (which is also mapped to `high`).

- A `system` or `developer` message after the leading one or two, and a message whose role is not `user`, `assistant`,
  `tool`, `system` or `developer`, are dropped by the model's chat template, as llama.cpp drops them: the model never
  sees them, and the reply is a normal 200. The request's log line counts them in `dropped_messages`, and its stderr
  line says `(n messages dropped by the template)`. The template merges the leading one or two `system` or `developer`
  messages into one system block. Move later instructions into the first system message, or into a user message.

Rejected with a 400, because ignoring them would break the client's contract:

- `n` above 1.
- OpenAI `response_format` of `json_object` or `json_schema`, and Anthropic `output_config.format` (structured
  outputs). There are no grammars.
- `top_logprobs` above 20, or without `logprobs: true`.
- More than 16 stop strings.
- An unknown OpenAI `reasoning_effort` value, a `chat_template_kwargs.enable_thinking` that is not a boolean, an
  unknown `tool_choice` string or Anthropic `tool_choice` type, a missing `max_tokens` on `/v1/messages`, and any field
  of the wrong type (the message names the field).

Fields that are used:

- Sampling: `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `seed` (a negative seed means random).
- `max_tokens` and `max_completion_tokens` are clamped to what is left of the context after the prompt, rather than
  refused: some agents ask for 32,000 or more. If OpenAI's is absent, the room left is used.
- `stop` and `stop_sequences`.
- `logprobs` with `top_logprobs` 0 to 20 on OpenAI chat (the content tokens only; reasoning, tool-call and forced
  tokens carry none). Anthropic has no logprobs.
- `tool_choice` `auto` (the default) and `none`. `none` renders the prompt without the tools, so a `<tool_call>` in the
  reply comes back as content.
- `parallel_tool_calls: false` and Anthropic `disable_parallel_tool_use: true` stop the reply at the end of its first
  tool call, and the finish reason is `tool_calls`.
- Reasoning:

| Request | Thinking | Budget |
|---|---|---|
| OpenAI `reasoning_effort: "none"`, or `chat_template_kwargs.enable_thinking: false` | off | none |
| `reasoning_effort` (or `reasoning.effort`) `low` or `minimal` / `medium` / `high` | on | 1,024 / 4,096 / none |
| Anthropic `thinking: {type: "enabled", budget_tokens: N}` | on | N |
| Anthropic `thinking: {type: "disabled"}` | off | none |
| Anthropic `thinking: {type: "adaptive"}` | on | from `output_config.effort` as below; with no effort, `thinking_budget` |
| Anthropic `output_config.effort` with no `thinking` | on | as `low` / `medium` / `high` above; `max` is `high`; any other value is ignored and the request counts as saying nothing |
| nothing | `thinking` (on `/v1/messages`, also `anthropic_thinking`) | `thinking_budget` |

A `/v1/messages` request that says nothing of thinking (no `thinking`, no `output_config.effort`, or only an unknown
type or effort, which is ignored) gets the default, which is on unless `thinking` or `anthropic_thinking` turns it off.
When it asks for fewer than 1,024 `max_tokens`, its budget is cut to half of `max_tokens` (at least 1), or to
`thinking_budget` when that is smaller. A short request such as a title or a one-word probe then has room left for its
answer. A request that states thinking, and a `count_tokens` call, keep the budget in the table.

If any field says off, thinking is off. A budget is a limit on reasoning tokens: when the model reaches it, the engine
closes the reasoning block and the reply goes on. The request's log line says so (`budget_closed`).

## Not enforced

- **`tool_choice` of `required`, `any` or a named tool** is accepted and rendered as `auto`. The model may answer
  without calling a tool, or call a different one. There are no grammars to force it. The request's `ignored` list
  names `tool_choice`, and its stderr line counts it.
- **No grammars.** Structured outputs, JSON mode and schema-constrained output are refused (above), not approximated.
- **One request at a time.** The engine runs one sequence. Requests queue first in, first out; up to 64 wait, and the
  65th gets a 503 at once.
- **Special tokens in message text are parsed as control tokens, by default.** The whole rendered prompt is tokenized
  with special tokens on, as llama.cpp does. A `<|im_start|>` in a file an agent reads or in a tool result becomes a
  real turn marker, so a tool result can forge a turn. With `special_tokens_in_content = false`, the 27 control tokens
  (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>` and the vision, FIM, audio and TTS markers) are plain text wherever
  they appear in client text: in message content, reasoning, tool-call names and arguments, and tool definitions. The
  template's own turn markers stay tokens, and the six tags `<think>`, `</think>`, `<tool_call>`, `</tool_call>`,
  `<tool_response>` and `</tool_response>` still tokenize as tags, as in llama.cpp with special parsing off. A prompt
  with such text then tokenizes differently from llama.cpp's; a prompt with none tokenizes the same either way. The
  mode is recommended for agents that read untrusted files. This holds for the pinned chat template; with
  `allow_custom_template` it depends on the literals the custom template puts around client text. Without
  `special_tokens_in_content = false`, do not put untrusted text in front of a model whose output you act on.

## Clients

[install.md](install.md#use-it-with-your-coding-agent) has the setup for Claude Code, OpenCode and any OpenAI client:
the base URL, the model id, the key, the 65,536-token context and its error, and `allowed_origins` for a browser or
editor-webview client. An origin with any scheme can be allowed, such as `vscode-webview://abc`.

## Auth and network

- **The key** is the first non-empty line of `api_key_file`, with surrounding whitespace and a CR trimmed. A missing,
  unreadable, empty or non-UTF-8 file, or one over 64 KiB, refuses the start; the message names the path, never the
  key. The key is never logged or printed.
- **Strength:** the server keeps the key as a SHA-256 digest and a character count, not the text, and compares digests
  in constant time. A key under 32 characters refuses the start when the bind address is not loopback (the message
  gives the count), and gets a `config_warning` when it is. To make a key of 32 random bytes in base64 (44 characters),
  in Windows PowerShell 5.1 or PowerShell 7:
  `$b = New-Object byte[] 32; [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b); [Convert]::ToBase64String($b)`.
  Put it on one line in the key file, and keep the file readable only by you.
- **Presenting it:** `Authorization: Bearer <key>` or `x-api-key: <key>`, compared in constant time.
- **Order:** auth runs before the body is read, so an empty or invalid body with a wrong key is a 401, not a 400. Every
  `/v1/*` path needs the key, `/v1/load` and `/v1/unload` included. `/health` never does.
- **With no key:** auth is off, the bind address must be loopback (`127.0.0.0/8`, `::1` or `localhost`) or the server
  refuses to start, and every request's `Host` must be `localhost`, `127.0.0.1`, `[::1]` (any port) or the bind address,
  or it gets 403. That includes `/health`. The check defends against DNS rebinding.
- **With a key:** any bind address is allowed (a hostname is resolved, and its first address used), and the Host check
  is off.
- **Browsers:** a web page in the user's browser can send a request to a loopback server, and with no key nothing else
  stops it. So every request is checked before the Host check and before auth, with or without a key. A request with an
  `Origin` header is allowed only when the origin equals an entry of `allowed_origins` (compared ASCII
  case-insensitively, a trailing `/` ignored); any other gets 403. Without an allowed `Origin`, a `Sec-Fetch-Site` of
  `cross-site` or `same-site` also gets 403. `same-origin`, `none` and an absent header pass, so curl, the SDKs and
  the agents, which send neither header, are unaffected. Add an origin to `allowed_origins` to use a browser-based
  client. A key does not lift this check.
- **Content type:** a POST with a body (a `Content-Length` above 0, or a `Transfer-Encoding` header) must carry a JSON
  media type: `application/json` or `application/*+json`, parameters such as `charset` allowed. Any other type, or
  none, gets 415 before the body is read, and before the API key is checked, so a request with the wrong key and
  the wrong type gets the 415; a form or plain-text post is what a web page can send without asking
  permission first. A POST with no body (`/v1/load`, `/v1/unload`) passes with any type or none. A bare `curl -d` sends
  `application/x-www-form-urlencoded`, so add `-H 'Content-Type: application/json'`.
- No CORS headers are sent.
- **TLS:** the server speaks plain HTTP. Keep the loopback default, or put a TLS reverse proxy or an SSH tunnel in front
  of it when another machine needs to reach it.
- **`/health`** needs no key. It shows the version, the model id, the context size and whether a key is set, and it
  answers any `Authorization` header.
- **One cache:** clients that share one key share the prompt cache, and the cache keeps no separation between them.
- **The port:** while Artemis is down, another program on the same machine can listen on its port and receive what a
  client sends, the key included.
- Request bodies are limited to 16 MiB; a larger one gets 413. `/v1/load` and `/v1/unload` take no body, so their limit
  is 4 KiB (`the request body is over 4 KiB`).
- **Request memory:** a body is parsed into a tree that costs many times its size, so a body of the generation endpoints
  (`/v1/chat/completions`, `/v1/messages`, `/v1/messages/count_tokens`) that holds more than 1,000,000 JSON values
  (every scalar, array, object and object key counts one) gets 400 `the request holds more than 1,000,000 JSON values`.
  A large legitimate request is far below it: 300 messages of 50,000 characters hold about 1,500 values. At most four
  requests are prepared at once (parse, template render, tokenization); a fifth waits up to 10 s and then gets 503 `the
  server is busy preparing other requests; try again`. A request gives up its place once its prompt build ends (a client
  that leaves mid-build does not free it early), so the wait in the queue and the generation hold none, and the
  request's body and parsed form are freed by then.
- **Timeouts and the connection cap:** a client has 30 s to send the head of a request, and a kept-alive connection that
  sends no next request within 30 s is closed (clients reconnect on their own). A request body has 60 s to arrive,
  counted from the handler's first read; a slower one gets 408 `the request body did not arrive within 60 s`. Neither
  limit applies to the response: a stream, or a long prefill with no bytes sent yet, runs as long as it needs. At most
  128 connections are served at once (kept-alive idle ones count, and the queue's 64 places plus the running request fit
  within it); further connections wait in the operating system's backlog until one closes, so a host that opens many
  idle connections cannot exhaust the server's memory. On a network bind, one host can still hold all 128 slots, so put
  a proxy that limits connections per client in front of the server.
- At most 64 requests wait in the queue; the 65th gets a 503 at once.
- Each request is answered over HTTP/1.1, with keep-alive allowed. Once the engine has taken a request, a stream sends a
  keep-alive line every 10 s. While the request waits in the queue or for a load, nothing is sent, because the status is
  decided when the engine takes it.
- A non-streaming client that disconnects is logged as status 499 ("client closed request": not a status the client
  saw). A streaming client that disconnects keeps status 200 in the log, with finish reason `cancelled`. Either way the
  engine stops within a token and keeps the state it reached for reuse.

## Logs

Each request writes one JSON line to `log_dir/artemis-YYYY-MM-DD.jsonl` (the date is UTC). The lifecycle lines go to the
same file. A human-readable line for each lifecycle line also goes to stderr. Each request leaves one stderr line, `{id}
{endpoint} -> {status} {finish}`, followed by ` (n fields ignored)` when it had any and ` (m messages dropped by the
template)` when the chat template dropped any. A few more notes can follow a request on stderr, each bounded in size and
count: a stop string that was empty and ignored, a `chat_template_kwargs` value that is not a boolean, a prompt with no
cut. Content parts and blocks the server skips (a part with no type, or of a type it does not support) are logged only
at level `debug`, since a request can hold thousands. **No prompt or output text and no key appears in any log.** Almost
every field is a count, a time, an id or a name the server chose. The values a client or the model chooses, or that
identify someone, are these: the `model` name the request sent (cut to its first 64 characters), an ignored field's name
(cut to 40 characters), the tool names in `repairs` (cut to 64 characters), the `peer` address on a refusal, and
`log_dir` in the `start` line, which holds the account name when the folder is under the user profile. Check those
before a log is shared. A request's line stays small whatever the request: it holds at most 32 ignored names, 16 repairs
and a 64-character model, and counts the rest. A log line that cannot be written (it cannot be serialized, or its file
cannot be opened or written) is dropped and counted: the first failure writes one warning to stderr, and the count is
`log_write_failures` in every `stats` line and in `/v1/stats`. Logging never fails a request.

Every request is logged, the door's refusals (401, 403, 404, 405, 413, 415) and `/v1/load` and `/v1/unload` included.
The `endpoint` is the route (`unknown` for a 404, never the raw path). A field a request never reached is `null`. The
`peer` field names the client on the refusals that come from the door's checks (401, 403, 415), so a run of wrong keys
shows where it came from.

| Field | Meaning |
|---|---|
| `time` | `YYYY-MM-DDTHH:MM:SSZ`: UTC, to the whole second. It is the request's arrival, not when the line was written; the line is written when the request ends. |
| `id` | The request id. |
| `endpoint` | The route. |
| `peer` | The client's IP address, without a port; set only on a 401, 403 or 415 refusal from the door's checks, `null` on every other line. An address is not prompt text. |
| `model` | The model the request named, cut to its first 64 characters. |
| `status` | The HTTP status (499 for a non-streaming client that left). |
| `prompt_tokens` | The prompt's length in tokens. |
| `cut_at` | The token the generation prompt starts at (where the prompt ends and the model's turn begins). `null` when the prompt has no cut. |
| `user_query` | Whether the turn before the cut is the user's query. |
| `stream` | Whether the reply was asked for as a stream. |
| `tool_calls` | Valid tool calls the reply emitted. |
| `malformed_calls` | `<tool_call>` texts returned as content because they were not valid calls. |
| `think_budget` | The request's reasoning budget in tokens; `null` with thinking off or no budget. |
| `budget_closed` | True when the engine closed the reasoning block at that budget. |
| `reused_tokens` | Prompt tokens restored from a snapshot rather than prefilled. |
| `restored` | Where the restored state came from: `none`, `live`, `interval`, `prompt_end`, `user_query` or `reply_end`. |
| `restore_bytes`, `restore_ms` | What the restore copied and how long it took. |
| `diverged_at` | The token where the prompt parted from the best cached state. |
| `cause` | Why the restore stopped short: `none`, `new_conversation`, `reasoning_dropped`, `reply_retokenized`, `new_input` or `no_snapshot`. |
| `prefilled_tokens` | Prompt tokens the engine prefilled. |
| `completion_tokens`, `reasoning_tokens` | Tokens generated, and how many of them were reasoning. For a reply cut by a stop string, `completion_tokens` (and so the usage the client gets and `decode_tok_s`) includes the tokens the engine generated before the server's cancel reached it: about one without speculation, up to the step's drafts (`max_draft`, 4 by default) more with it. |
| `queue_ms` | Time from the request's arrival to the engine taking it, from the queue and any load or park it waited for. |
| `load_ms` | Model load time charged to this request (a lazy load). |
| `prefill_ms` | Prefill time. |
| `ttft_ms` | Time from arrival to the first token. It includes the queue and the load. |
| `total_ms` | Time from arrival to the line being written: how long the request ran, whatever its end (a refusal, a stream that failed part-way, a client that left). Never `null`; the request ran from `time` to `time` plus this. |
| `prefill_tok_s`, `decode_tok_s` | Token rates, `null` when there were no tokens or no time. `decode_tok_s` is `completion_tokens` over the decode time (the first token included, though prefill produced it); the rate of the decode steps alone is `decode_tokens` over `decode_ms`. |
| `decode_ms` | The decode phase's time in milliseconds. |
| `decode_tokens` | The tokens decoded after the first: `completion_tokens` minus 1 (0 when there was no completion). |
| `decode_misses` | The expert cache's misses during the decode phase. `null` when the engine gave no count. |
| `hit_rate` | The expert cache's hit rate. |
| `miss_mib_per_token` | MiB of expert misses per token. |
| `spec_steps`, `spec_plain_steps` | Speculation: the request's steps that checked a draft, and the ones that ran without. `null` with speculation off, and for a request that ran no decode step. |
| `spec_drafted`, `spec_accepted` | Draft tokens checked, and the ones kept. |
| `spec_accept_rate` | `spec_accepted` over `spec_drafted` for this request; `null` when nothing was drafted. (`/v1/stats` has a different `accept_rate`: the controller's running estimate.) |
| `spec_replays` | Steps that kept fewer than all their drafts. |
| `spec_drafted_lookup`, `spec_accepted_lookup`, `spec_drafted_mtp`, `spec_accepted_mtp` | `spec_drafted` and `spec_accepted` by source; each pair sums to the totals. `null` as they are. |
| `pool_bytes_used` | Bytes of the snapshot pool in use after the request. |
| `finish` | The finish reason. |
| `error_kind` | The error kind for a failed request. |
| `error_message` | The error's message, cut to its first 300 characters; `null` when `error_kind` is. It is the one-line message the client got (it holds no request text), and it makes a failed request explain itself. |
| `ignored` | Names of accepted-and-ignored fields, each cut to its first 40 characters; at most the first 32. |
| `ignored_more` | How many ignored fields came after those 32 and are not named. |
| `repairs` | Repairs made to the reply, at most the first 16. A repair reads `tool "<name>": <what>`, the tool's name cut to 64 characters. A repair about a parameter reads `tool "<name>": parameter of <n> bytes: <what>`: the key the model wrote is shown by its length in bytes and never its text, since a model can write anything it has read into a key. |
| `repairs_more` | How many repairs came after those 16 and are not listed. |
| `dropped_messages` | How many messages the chat template dropped: a `system` or `developer` message after the leading one or two, and a message of a role the template does not know. 0 for most requests. |

The lifecycle lines carry `time` and an `event`:

| `event` | Fields | When |
|---|---|---|
| `start` | `version`, `bind`, `model_id`, `log_dir` | The port is open. The stderr line reads `artemis {version} listening on {bind}, model {model_id}, logs in {log_dir}`, so the folder is on screen when someone is asked for a log. |
| `config_warning` | `message` | At start, once for each setting that switches a safeguard off or sets `stall_load_s` below `stall_s` (above), for the hidden test flag, and for a tuned file that has no place or is out of date (The tuned file). The first load on a GPU other than the tested AMD Radeon RX 9070 XT writes one more, once per process, and a load on a GPU or driver other than the tuned file's writes another each time. |
| `tuned` | `path`, `applied`, `ignored` | At every start with a place for the tuned file. `applied` and `ignored` are the dotted keys the file set and the ones your settings outranked; both are empty when the file is missing. |
| `setup_start` | `reason`, `path` | Setup began measuring. `reason` is `first run` (`serve` found no tuned file) or `asked` (`artemis setup`); `path` is the tuned file it will write. |
| `setup_done` | `seconds`, `limit_hit`, `target`, `gpu_name`, `vram_bytes`, `cache_slots`, `scores`, `enabled`, `sources`, `priors`, `measured_slope_ms`, `measured_mtp_d1_ms`, `expected_tok_s`, `path` | Setup wrote the tuned file. `scores` are the decode speeds in tok/s by candidate; `enabled`, `sources` and `priors` (`cost_slope_ms` and `mtp_cost_ms`, or `null`) are the values the server will run with (setup's choice, or your own value for a key you set); `expected_tok_s` is 0 when nothing was measured. |
| `setup_failed` | `message` | Setup stopped without writing the tuned file. `serve` then exits 1 with the same message. |
| `load` | `seconds`, `detail`, `slots_per_layer`, `vram_used_mib`, `pool_limit_gib`, `pool_cut`, `unparked` | A model loaded. `detail` names the device and whether the load retried with a smaller cache. `slots_per_layer` is the expert cache size, `vram_used_mib` the VRAM the load took, `pool_limit_gib` the snapshot pool's limit at this load, `pool_cut` whether it is below `snapshot_pool_gib`, and `unparked` the number of parked snapshots restored into the pool. |
| `load_failed` | `message` | A load did not finish (a lazy load, `/v1/load` or the load at start). The message is the engine's, never a request's text. |
| `reload_failed` | `message` | The reload after a GPU loss did not finish. The message is the engine's. |
| `unload` | `cause` | `idle`, `request` (`/v1/unload`) or `shutdown`. |
| `device_lost` | `message` | The GPU was lost. The message is the engine's: to debug a loss on an AMD driver, set the environment variable `ARTEMIS_MARKERS=layer` and the message names the first dispatch of the layer in flight, or `ARTEMIS_MARKERS=dispatch` for the exact dispatch. Markers are off by default: they cost about 3% of decode speed or more. |
| `panic` | `thread`, `location`, `message` | A panic on any thread. `thread` is the thread's name, or `unnamed`; `location` is `file:line:col`. `message` is the panic's text only when the payload is a fixed string compiled into the binary (as `panic!("literal")` and `unwrap` on `None` give); any other payload (a formatted message, an error's `Debug`, an index error) can carry request text, so `message` is `null`. |
| `stall` | `doing`, `request`, `quiet_s`, `bound_s`, `ticks` | The watchdog ended the process (see Lifecycle). `doing` is `request`, `load`, `reload` or `command`; `request` is the id of the oldest request still in flight, `null` when none; `quiet_s` is how long the engine showed no progress and `bound_s` the bound it passed (`stall_s` or `stall_load_s`); `ticks` is the progress counter it stopped at. |
| `stats` | `uptime_s`, `age_s`, `log_write_failures`, `stats` | The engine's counters, `spec` included, as `/v1/stats` carries them: the latest answer in `stats` and its age in `age_s`, both `null` before the first answer. Written every 10 minutes (the heartbeat), after each `unload` line for cause `idle` or `request`, and before each `exit` line. |
| `exit` | `reason` | `shutdown`, `device_lost_twice`, `engine_gone` or `stall`. |

`tools/py/request_quality.py` reads the request lines and reports the quality signals they carry, per day and overall:
finish reasons, the share that stopped at the token limit (a request that filled the context window counts here too,
because the server ends it with `length`), malformed tool calls, reasoning budget closes, errors, token totals, the
prefix reuse rate and the slowest requests, with a flag on each share above a simple threshold. A file it cannot read is
named on stderr and skipped; a file given twice, or with its folder, is read once. It uses only the Python standard
library and never changes a log: `python -I tools/py/request_quality.py LOG_DIR [--since DATE]
[--json OUT]`.

## Lifecycle

- **Start.** The server reads the config, opens the GGUF, checks it, builds the tokenizer and the chat template, and
  binds the port. This takes seconds. With `lazy_load = true` (the default) the model stays unloaded until a request
  needs it.
- **Lazy load.** The first generation request on an idle server loads the model on the engine's thread: about 9 s with
  the file in the OS cache, and 95 s once with it cold. On a 32 GB machine the OS cache is unlikely to hold the 21 GB
  file beside the pinned experts, so a reload after an idle unload is expected to be a cold load (not measured). Parked
  snapshots stay in RAM through the unload. `/health` says `loading` (503) meanwhile, and requests wait in the queue;
  they are not answered 503. A client with a deadline shorter than the load will time out first. If the load fails (no
  discrete GPU, a missing feature, weights that do not fit, GPU memory held by another program), the request that
  triggered it and every generation request waiting behind it get 503 with the engine's message, `/health` goes back to
  `idle`, and the next request tries again. A VRAM shortage gets one retry with half the cache before it fails; the
  message then carries the figures of both attempts (see Memory).
- **Load at start.** With `lazy_load = false` a load starts as soon as the port is open. A failure is logged, and the
  server stays up and `idle`; the next request tries again.
- **Idle unload.** After `idle_unload_s` seconds with no request, the engine parks its snapshots in ordinary RAM (if
  `park_snapshots`), unloads the model and frees the GPU. `/health` then says `idle`. The next request loads again and
  restores the parked snapshots into the pool. A request that arrives while this happens waits for it, then lazy-loads;
  it is never failed for it, and its `queue_ms` includes the park. `idle_unload_s = 0` never unloads.
- **On demand.** `POST /v1/unload` and `POST /v1/load` do the same by hand. A guard that needs the VRAM back calls
  `/v1/unload`: it does not wait for the request in flight, it ends it.
- **Queue.** First in, first out, one at a time. A request waits at most `queue_timeout_s` for the engine to start it,
  then gets a 503; the engine later finds it gone and skips it.
- **Shutdown.** On Ctrl-C (or Ctrl-Break on Windows) the server stops taking work (new requests and `/v1/load` get 503
  "the server is shutting down"), ends the queue and the request in flight, unloads the model **without parking**,
  drains open connections, and exits 0. The 30 s grace starts when the stop arrives and covers the unload and the drain
  together; after it the server writes its last lines and waits up to 20 s more for the engine's thread. It writes an
  `unload` line with cause `shutdown` if a model was unloaded, then a `stats` line, then an `exit` line with reason
  `shutdown`, which is the log's last line (the heartbeat and the watchdog end first). A hard kill loses nothing but the
  process: the only state on disk is the logs.
- **Device loss.** If the GPU is lost, the request in flight fails (503 for non-streaming, an error event for
  streaming), `/health` says `recovering` (503) while the engine drops everything, the pool included, and reloads, and
  requests wait. If the GPU is lost twice within 10 minutes, the server logs why and exits with code 2. If the reload
  fails, the requests that were waiting get 503 and the server goes back to `idle`; the next request tries to load
  again.
- **Engine failure.** If the engine's thread ends (a panic), nothing notices at once, though the `panic` line is in the
  log from the moment of the panic. The next request, `/v1/load` or `/v1/unload` that finds the thread gone gets 500
  (not 503), and the server then exits with code 3.
- **Panics.** A panic on any thread writes one `panic` line and the same line on stderr, in place of the default text,
  which would show the panic's message; a backtrace follows on stderr when `RUST_BACKTRACE` is set. The line is written
  through a helper thread that the panic waits on for at most 2 s, so a panic that happens while the log's lock is held
  cannot deadlock; the line is then written when the lock frees.
- **The watchdog: what counts as progress.** Every 5 s (not a setting) the server looks at the engine's progress counter
  (`progress` in `/v1/stats`). The engine has work when it holds a request or a command it has not finished, or while a
  load or a reload runs, even with nothing else pending. Only the engine's own steps count as progress: a token, the
  check before each prefill segment (a segment is at most 4,096 tokens), the end of a command, a load, unload or loss
  ending, a change of state, and, during a load or a reload, each weight tensor uploaded and each layer of expert
  copies. The last of these is why a long load on a slow disk keeps moving; if one tensor or layer still takes longer
  than `stall_load_s`, the `stall` line says to raise it. Work arriving does not, so a stream of new requests behind a
  stuck engine does not hide the stall.
- **The watchdog: the bounds.** The clock starts when work first appears and restarts at each step. When the quiet time
  reaches `stall_s` (or `stall_load_s` while the state is `loading` or `recovering`) the server decides the engine is
  stuck, and from that moment nothing cancels it, a stop included. The check runs every 5 s, so a stall is reported up
  to one tick after the bound. A check 50 ms or less short of the bound counts (the tick's own jitter), so `quiet_s`
  on the `stall` line can read up to 50 ms under the bound.
- **The watchdog: what it writes and how it exits.** It writes three lines within 2 s in all: `stall`, then `stats`
  (from the latest copy; it asks the engine nothing), then `exit` with reason `stall`. A line that cannot be written in
  time ends the writing. It then ends the process with code 4. It does not unload, drain connections or wait for the
  engine's thread, because all of those wait on the stuck engine. `serve`'s own `unload`, `stats` and `exit` lines are
  suppressed from the decision on, so the watchdog's `exit` is the only one. In the `stall` line, `doing` is `load` or
  `reload` in those states; otherwise `request` while a request is in flight and `command` when none is (a `/v1/load`
  or `/v1/unload`, or the snapshot taken after a reply, which runs after the request has ended). `request` is the
  oldest request in flight: the worker is first in, first out, so that is the one it runs or the one waiting behind the
  stall.
- **The heartbeat.** Every 10 minutes (not a setting), counting from the start, never at once, the server asks the
  engine for its counters, waits at most 2 s for the answer, and writes a `stats` line from the latest copy and the
  copy's age, whether or not the engine answered. A gap of more than one period between `stats` lines therefore means
  the server's own runtime stopped. The copy is also refreshed when a request that reached the engine ends.

## Memory

- **The pin.** A loaded model holds about 18.7 GiB of pinned host memory (the experts and the embedding; 19.2 GiB with
  the MTP layer, below), plus the VRAM
  for the dense layers, the KV cache and the expert cache. The figure is this engine's, measured on the RX 9070 XT
  machine it was built on; the VRAM side is sized to what the GPU has free at load.
- **MTP.** With `"mtp"` in `sources` (the default) and speculation on, the load adds the MTP block: about 40 MiB of
  its weights in VRAM, its KV (`mtp_ctx` rows, 16 MiB at the default) and the hidden-row ring (2 MiB), about 58.5 MiB of
  VRAM in all, which comes out of the expert cache (under one slot a layer); and 464 MiB of its experts in host memory,
  on top of the 18.7 GiB pin. `sources = ["lookup"]` or `enabled = false` loads none of it.
- **The pool.** The snapshot pool lives in host RAM beside the pin, up to `snapshot_pool_gib`. At each load the engine
  cuts it to `min(snapshot_pool_gib, whole GiB of (available - 4 GiB))`, where `available` is the lesser of what the
  pinned heap has left and what the OS reports free. A pool cut under 1 GiB is 0: reuse is off, and the server runs
  without it. The cut is logged, never an error: the `load` line records `pool_limit_gib` and `pool_cut`. If the driver
  later refuses a chunk of the pool, reuse shrinks in the same way, and the engine's counters record it.
- **A 32 GB machine.** With 32 GB in total, the OS and other programs leave about 26 GiB free before the load; take
  away the 19.2 GiB pin (MTP on) and the 4 GiB margin, and the pool is about 2 GiB. That still reuses prompts: the
  server tests that exercise reuse need under 1 GiB by design (63 MiB per snapshot plus 20,480 B per token). They ran
  with `--snapshot-pool-gib 2` on the 2026-10-26 build, with the pool uncut at 2 GiB and all passing, and the test
  runner refuses to start them below 26 GiB free. What the pin leaves for other work on a 32 GB machine is not measured.
  Set `snapshot_pool_gib` to what you want to give up rather than relying on the cut: a small value keeps the memory
  use predictable.
- **When a load does not fit.** The `load_failed` line and the 503 carry the engine's own figures. A shortage of host
  memory reads "the embedding, the experts and prefill's host buffers in host memory need N MiB, but only M MiB is
  free. Close other programs or add RAM; the snapshot pool is sized after this check and is not part of these
  figures", and the load is not retried, since the cache is in VRAM. A shortage of GPU memory is retried once with half
  the cache slots, and if that fails too, the message reads "not enough GPU memory: with 128 cache slots per layer,
  <the first attempt's figures>; with 64, <the second's>", where each part is the engine's own account of what did not
  fit and by how much, followed by the advice to close programs that use the GPU, or lower `ctx` or set `cache_slots`.
  Every other load error keeps the engine's text.
- **While unloaded.** An idle unload with `park_snapshots = true` keeps the pool's contents in ordinary RAM, up to its
  size, so a server that has been idle holds that much host memory but no GPU memory and no pin. Set
  `park_snapshots = false` to drop them.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | A requested stop (Ctrl-C, Ctrl-Break, or standard input closing with `--stop-on-stdin-close`). |
| 1 | Refused to start (config, key, bind address, model contract, log directory, or the address could not be listened on), or the server's task panicked. |
| 2 | The GPU was lost twice within 10 minutes. |
| 3 | A request, `/v1/load` or `/v1/unload` found the engine's thread gone (see Engine failure). |
| 4 | The stall watchdog ended the process (see The watchdog): the `stall`, `stats` and `exit` lines are the log's last. |

On every path after the engine starts except 4, the process waits up to 20 s for the engine's thread before it exits.
Code 4 does not wait: the engine is stuck, so the wait would only wait on it. No `unload` line is written.

## For scripts

`--stop-on-stdin-close` is a hidden flag: the server stops as on Ctrl-C when its standard input reaches the end. A
script on Windows cannot send Ctrl-C to a child that shares no console, so it starts the server with standard input
redirected and closes the pipe to stop it. If the script dies, the pipe closes and the server stops with it. Nothing
else reads standard input.

`/v1/load` and `/v1/unload` are the other scriptable controls; a script that needs the VRAM back without stopping the
server calls `/v1/unload`.

`--test-stall-at-token N` (N of 1 or more) is a second hidden flag, for the GPU stall test and nothing else: every
request then stalls the engine's worker at its Nth token, so the watchdog fires. It is a flag of `artemis serve`, not a
config key, so no config file and no environment variable can set it. With it set, `serve` writes a `config_warning`
line at start ("a test hook is set: every request stalls at token N"), so a log always shows it, and it refuses to start
(exit 1, before the log or the model is opened) when the port is 8080.

## Behaviour that differs from the test fake

The server's tests run against a scripted fake engine. The real engine differs in one place that matters here. When a
load fails, every generation request queued behind it fails with the same error and is not retried. A `/v1/load` that
was queued behind it is different: the real worker runs it in its turn, with its own load attempt (and its own
wait, about 9 s or more) rather than answering with the earlier failure. The fake fails those too. The real worker's
behaviour is the one described here.

## Known windows

These are the places where the behaviour is not what the description above would suggest. They are rare and none is a
hang.

- **A `/v1/load` during an idle park answers `ready` while the model unloads.** The engine reports `ready` until the
  park ends, so `/health` and `/v1/load` both read as loaded for the few seconds the park takes. A request in that
  window waits for the unload and then lazy-loads, as above.
- **A stop during a load waits for the load, up to a limit.** The engine cannot interrupt a model load, so the unload
  runs only after it. A stop during a lazy load, a `/v1/load` or the load at start finishes the load, unloads and exits
  0 if the load is warm (about 9 s). The 30 s grace runs from the stop, though, and the server then waits 20 s for the
  engine's thread. A stop during a cold load (about 95 s) therefore ends the process after about 50 s, in the middle of
  the load, with no `unload` line, and exits 0.
- **A dead engine thread is found by the next call.** Nothing watches the thread. The first request, `/v1/load` or
  `/v1/unload` after it ended gets 500, and then the server exits with code 3. Until then `/health` answers from the
  state it last had. The `panic` line shows the thread's end in the log when it happens.
- **A stop after a GPU loss can reload first.** If the GPU is lost while the reply-end snapshot of a turn the stop
  cancelled is taken, the engine reloads (about 9 s) before it shuts down.
- **A loss whose error reached a client that left before reading it is not counted.** The loss is counted when a
  request's error is delivered or when the engine reports a loss no request carried. An error that was sent to a client
  that disconnected before it read it is counted by neither. It fails toward not exiting: the server stays up. The
  `device_lost` line is written for every loss that is counted, so a count missing from the log is the sign.
- **Losses during a lazy load or a park are not counted.** A GPU loss while the model is loading inside a request is a
  failed load: the requests get 503, `/health` goes back to `idle`, and the next request retries. A loss during an idle
  park drops the snapshots and unloads the model anyway. Neither counts toward "two losses within 10 minutes", so that
  rule is narrower than its words: it counts the losses a request or a `/v1/load` carried, and the losses after a reply
  that no request carried (the reply-end snapshot, the reasoning-budget check). The engine's own counters record the
  rest.
