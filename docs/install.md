# Installing Artemis

Artemis runs one model, Qwen3.6-35B-A3B, on your GPU and serves it over the OpenAI and Anthropic HTTP APIs on one port.
It runs on Windows. It is tested on Windows 11 Pro x64 with an AMD Radeon RX 9070 XT.

If you would rather have an AI coding agent do the steps below, the [README](https://github.com/CyberPotato54/Artemis#readme) in the Artemis repository has a prompt for
that. It is under "Set it up with your coding agent".

## What you need

| | Requirement |
|---|---|
| GPU | At least 16 GB of VRAM and a current driver with Vulkan 1.3 support. The driver includes the Vulkan loader (`vulkan-1.dll`). You do not need the Vulkan SDK to run Artemis. Cards under 16 GB may load with a smaller cache; that is untested. |
| Visual C++ runtime | The Microsoft Visual C++ Redistributable for Visual Studio 2015-2022, x64: https://aka.ms/vs/17/release/vc_redist.x64.exe |
| RAM | 32 GB is the target and is untested; 64 GB is tested. The model's experts live in pinned system memory, about 20 GB. On 32 GB close other large programs and set `snapshot_pool_gib` to 1 or 2. |
| Disk | About 22.7 GB (21.1 GiB) for the model file. |

## Install

1. Install the Visual C++ Redistributable from the link above, if it is not installed already.
2. Update your GPU driver to a current release.
3. Check the release archive, `artemis-<version>-<commit>-windows-x64.zip`, before you unzip it. In PowerShell:

   ```powershell
   Get-FileHash <the archive> -Algorithm SHA256
   ```

   The hash must equal the archive's line in `SHA256SUMS.txt` on [the release page](https://github.com/CyberPotato54/Artemis/releases). `Get-FileHash` prints upper case
   and `SHA256SUMS.txt` is lower case, so compare them ignoring letter case. `SHA256SUMS.txt` also has a line for
   `<folder>/artemis.exe`, so you can check the unzipped program the same way. To build `artemis.exe` yourself, see
   [Building from source](#building-from-source).

   The archive holds one folder with the same name as the archive, and that folder holds `artemis.exe`,
   `artemis.toml.example`, this guide as `INSTALL.md`, `server.md` (every setting and endpoint), the two licence files,
   `THIRD-PARTY-LICENSES.txt` and `MODEL.txt`, which names the model file below. Unzip it into `Programs` inside your
   local application data folder and rename the folder to `Artemis`. The profile folder's name often differs from your
   user name, so the commands use `$env:LOCALAPPDATA`, and the first one prints the full path:

   ```powershell
   $zip = '<the full path of the archive>'
   $dir = "$env:LOCALAPPDATA\Programs\Artemis"
   $dir
   if (Test-Path $dir) { throw "$dir exists: move artemis.toml and your key file out, delete the folder, then run this again" }
   Expand-Archive -Path $zip -DestinationPath "$env:LOCALAPPDATA\Programs"
   Rename-Item "$env:LOCALAPPDATA\Programs\$([IO.Path]::GetFileNameWithoutExtension($zip))" Artemis
   ```

   This puts `artemis.exe` at `$dir\artemis.exe`. Use these commands rather than Explorer's "Extract All", which adds a
   second folder of the same name. Neither PowerShell commands nor `artemis.toml` expand `%LOCALAPPDATA%`, so use the
   full path that `$dir` printed wherever a file needs one. Other accounts on the PC can write into a folder made under
   `C:\`, and Windows can load a DLL that sits next to a program, so do not run Artemis from `C:\` or from Downloads.
4. Download the model. This is a one-time download of about 22.7 GB.
   - Repository: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF
   - File: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
   - Size: 22,663,387,424 bytes
   - Save it somewhere with room, for example `C:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
5. Check the download in PowerShell:

   ```powershell
   Get-FileHash C:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -Algorithm SHA256
   ```

   The hash must be the following, ignoring letter case (`Get-FileHash` prints upper case):

   ```
   0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b
   ```

   The loader checks the tensors, the hyperparameters, the chat template and the vocabulary.

After the download Artemis needs no network. It has no telemetry and no update checks. The tokenizer and chat template
come from the GGUF, and the GPU kernels are built into the binary.

### What loads

Artemis is tested with one file, UD-Q4_K_M. It loads Qwen3.6-35B-A3B GGUFs that include the MTP layer (the `-MTP-GGUF`
repository) and whose weight tensors are F32, BF16, Q8_0, Q4_K, Q5_K or Q6_K, with this chat template and vocabulary.
The template and the vocabulary must match the ones Artemis is tested with (a sha256 pin each). A different vocabulary
is always refused. A different chat template is refused unless `allow_custom_template = true` is set. Larger quants
need more system RAM and are untested. IQ, Q2_K, Q3_K and MXFP4 quants and `mmproj` files are refused.

## Run it

1. Start the server with the path of the model file. No config file is needed:

   ```powershell
   $dir = "$env:LOCALAPPDATA\Programs\Artemis"
   & "$dir\artemis.exe" serve --model 'C:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
   ```

   Single quotes keep PowerShell from expanding a `$` in the path. The model's path is the only thing Artemis needs.

   The first start measures your PC for a few minutes. It lists your GPUs, loads the model, times short runs with
   speculation on and off, prints what it chose and why, and writes the choices to a per-user file,
   `%LOCALAPPDATA%\artemis\tuned.toml`. Then it serves. Later starts read that file and begin at once. Nothing is sent
   over the network.

2. Stop it with Ctrl-C.

3. Measure again after a driver update, a new GPU or a different model file:

   ```powershell
   & "$dir\artemis.exe" setup --force
   ```

   `artemis.exe setup` without `--force` reports the file it finds and changes nothing. The server writes a warning to
   its log when its GPU or driver is not the one the file was measured on.

For settings you want to change later, copy `artemis.toml.example` from the archive to `artemis.toml` in the same
folder, edit it, and start the server with `--config "$dir\artemis.toml"`. A setting in `artemis.toml`, or a flag, wins
over the same setting in the tuned file, and setup measures with the values you set.

With no API key set, the server listens on `127.0.0.1` only. To set a key, put it on the first line of a text file
beside `artemis.toml`, add `api_key_file = 'key.txt'` to the config (a relative path in a config file is read from the
config file's folder), and send the key as `Authorization: Bearer <key>` or `x-api-key: <key>`. To listen on another
address, set `host`. On an address that is not loopback the key must be at least 32 characters, or the server refuses to
start; [server.md](server.md) shows how to make one. Every setting, flag and endpoint is in [server.md](server.md).

## Check that it works

The model loads on the first request that needs it. After a first start's measuring, which has just read the file
twice, that load takes about 9 seconds. On a later start with the file not in the OS cache it takes about 95 seconds,
and later loads about 9 seconds, while the file stays in the cache. On a 32 GB machine a reload after an idle unload
is expected to be a cold load, because the OS cache is unlikely to hold the 21 GB file beside the pinned experts; this
is not measured.
`idle_unload_s` (600 s by default) unloads the model after that long without a request and frees the 18.7 GiB the
experts pin (19.2 GiB with the MTP layer). Parked snapshots stay in RAM unless `park_snapshots = false`. `0` keeps the model loaded. While it loads,
`/health` answers 503 with `"status":"loading"`.

1. In a second PowerShell window, check the server:

   ```powershell
   Invoke-RestMethod http://127.0.0.1:8080/health
   ```

   `status` is `idle` before the first request and `ready` once the model is loaded.

2. Send a chat request:

   ```powershell
   $body = @{
     model = "qwen3.6-35b-a3b"
     messages = @(@{ role = "user"; content = "Say hello in five words." })
     max_tokens = 50
     reasoning_effort = "none"
   } | ConvertTo-Json -Depth 5

   Invoke-RestMethod -Uri http://127.0.0.1:8080/v1/chat/completions -Method Post `
     -ContentType "application/json" -Body $body
   ```

   If you set an API key, add `-Headers @{ Authorization = "Bearer <key>" }`.

## Use it with your coding agent

Artemis listens on `http://127.0.0.1:8080`. It serves one model, so any model name reaches it, and it runs one request
at a time.

### Claude Code

Claude Code speaks the Anthropic API. In the PowerShell window where you start it, set:

```powershell
$env:ANTHROPIC_BASE_URL = 'http://127.0.0.1:8080'
$env:ANTHROPIC_AUTH_TOKEN = 'artemis'
$env:ANTHROPIC_MODEL = 'qwen3.6-35b-a3b'
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = 'qwen3.6-35b-a3b'
claude
```

`ANTHROPIC_AUTH_TOKEN` is any text when no key is set, and the key when one is. Setting both model variables to the
served id keeps Claude Code's side calls, such as titles, on the same name.

### OpenCode

Add a provider to `opencode.json`, in the project folder or in `~/.config/opencode/`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "artemis/qwen3.6-35b-a3b",
  "small_model": "artemis/qwen3.6-35b-a3b",
  "provider": {
    "artemis": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Artemis",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "artemis"
      },
      "models": {
        "qwen3.6-35b-a3b": {
          "name": "qwen3.6-35b-a3b",
          "tool_call": true,
          "limit": { "context": 65536, "output": 16384 }
        }
      }
    }
  }
}
```

`apiKey` is any text when no key is set, and the key when one is.

### Any OpenAI client

Set the base URL to `http://127.0.0.1:8080/v1`, the model to `qwen3.6-35b-a3b` and the API key to any text, or to the
key if you set one.

### What to expect

- **The context is 65,536 tokens.** A client that assumes more gets an error once a conversation passes it. On
  `/v1/messages` the error reads `prompt is too long: N tokens > M maximum`, as Anthropic words it. Run `/compact` or
  start a new session. `ctx` can go higher, up to 262,144, at a cost in VRAM.
- **Requests run one at a time.** A client's side calls, such as titles, wait behind its main request. Give them the
  same model id as the main one.
- **A browser or editor-webview client needs its origin allowed.** Such a client sends an `Origin` header and gets a 403
  unless `allowed_origins` lists the origin, for example `allowed_origins = ['http://localhost:3000']`. An entry is
  written `scheme://host[:port]` and the scheme can be any, so a webview, an Electron app or an extension is allowed
  with its own origin, such as `vscode-webview://abc` or `app://local`. Clients that send no `Origin` header, such as
  the two above, curl and the SDKs, are not affected.

## Network and keys

- Artemis has no TLS. Keep the loopback default, or put a TLS reverse proxy or an SSH tunnel in front of it when
  another machine needs to reach it.
- `/health` needs no key. It shows the version, the model id, the context size and whether a key is set.
- Clients that share one key share the prompt cache. The cache has no separation between them.
- While Artemis is down, another program on the same PC can listen on its port and receive what a client sends, the key
  included. Start Artemis before the clients, or use a less common `port`.

## Where the logs are

The request logs are in `%LOCALAPPDATA%\artemis\logs`, one `artemis-YYYY-MM-DD.jsonl` file for each day, and the last 14
days are kept (`log_dir` and `log_keep_days` change both). They hold no prompt or output text and no key. The `start`
line names the log folder, which includes your account name, so look at a log before you share it.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | A requested stop (Ctrl-C). |
| 1 | The server refused to start (config, key, address, model file), or its task panicked. |
| 2 | The GPU was lost twice within 10 minutes. |
| 3 | The engine's thread ended. |
| 4 | The stall watchdog ended the process. |

[server.md](server.md) has the detail.

## Building from source

1. Install Rust with [rustup](https://rustup.rs). The repository pins the version in `rust-toolchain.toml`, and rustup
   installs it the first time you build.
2. Install Visual Studio 2022 Build Tools with the "Desktop development with C++" workload. This provides the MSVC
   linker.
3. The Vulkan SDK is needed to change a shader and for `tools\build-release.ps1`, which first runs
   `cargo run -p kernels --bin update-spirv -- --check` to confirm that the committed compiled shaders are glslc's
   output for the shaders in the tree. The repository holds the compiled shaders, so a plain `cargo build` from an
   unchanged checkout needs no SDK. To change a shader, install the
   [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) and run `cargo run -p kernels --bin update-spirv`.
4. From the repository root:

   ```powershell
   .\tools\build-release.ps1
   ```

   The result is `target\release\artemis.exe`. The script builds with the same flags as the test runner, checks that
   the binary holds no path from your machine and loads its DLLs from System32 only, and prints its sha256. A plain
   `cargo build --release -p artemis-tools` also works; it skips the shader check and the path remap.
5. To make the release archive described in Install, run `.\tools\make-release.ps1` from a clean checkout, in PowerShell
   7.2 or later. It builds
   with `build-release.ps1` and writes `target\dist\artemis-<version>-<commit>-windows-x64.zip`. It also writes
   `target\dist\url-scan.txt`, a report of the web addresses in the binary and the DLLs it imports, and
   `target\dist\SHA256SUMS.txt`, the sha256 of the archive, of `artemis.exe` as it sits in the unzipped folder and, when
   `target\dist\artemis-src-<commit>.zip` exists for the same commit, of the source archive. The checksum file is not
   signed.

## Building from the source archive, offline

The source archive `artemis-src-<commit>.zip` holds the repository at one commit, every Rust crate the build needs, and
the compiled shaders. It builds with no network access and without the Vulkan SDK.

Rust is a one-time prerequisite. rustup downloads the pinned toolchain the first time it is used, so on a machine that is
offline from the start, install the toolchain from the standalone installer instead. Download
https://static.rust-lang.org/dist/rust-1.98.1-x86_64-pc-windows-msvc.msi on a machine with network access and run it
on the build machine. It is Rust 1.98.1, the version in `rust-toolchain.toml`, and it includes rustfmt and clippy. The
build also needs the Visual Studio 2022 Build Tools described in the previous section.

1. Unzip the archive.
2. In PowerShell, go into the `artemis-src-<commit>` folder inside the unzipped archive and run:

   ```powershell
   cargo build --release --offline --frozen -p artemis-tools
   ```

   The result is `target\release\artemis.exe`.

The archive's `.cargo\config.toml` points cargo at the `vendor` folder, so cargo does not contact crates.io.
`tools\make-source-archive.ps1` makes the archive from a clean checkout and writes it to `target\dist`.
`tools\test-offline-build.ps1 -Archive <zip>` unzips it into a temporary folder, removes glslc and the Vulkan SDK from
the environment, gives cargo an empty cargo home, and runs this build. It stops with a message when a
`.cargo\config.toml` in a parent folder of the temporary folder, such as one in your user profile, could change where
cargo finds its crates.

## Troubleshooting

- **The program does not start and names `VCRUNTIME140.dll`.** Install the Visual C++ Redistributable (step 1 of
  Install).
- **A Vulkan loader or "no device" error.** Artemis needs a discrete GPU with Vulkan 1.3. Update the GPU driver. The
  error lists every device it found.
- **The server exits at start naming the model.** The GGUF is not one that loads (see What loads), or the download is
  damaged. The message names the first difference. Check the file against the hash above.
- **The model file will not open and the error mentions os error 32.** Another program still has the file open for
  writing, usually an unfinished download. Artemis opens the model for reading only and does not wait for the writer.
  Let the download finish or close the program, then start again.
- **The port is in use.** The server exits with code 1 and says the address could not be listened on. Change `port` in
  the config, or stop the program that holds it.
- **A load fails and the message gives figures.** "...in host memory need N MiB, but only M MiB is free" means the
  system RAM is short by the difference: close other programs or add RAM, then send the request again. "not enough GPU
  memory: with N cache slots per layer, ...; with N/2, ..." means the GPU was short on both tries, the second with half
  the cache: close programs that use the GPU, lower `ctx` or set `cache_slots`. The server stays up, `/health` goes
  back to `idle` and the next request tries again.
- **The first load is slow.** About 95 seconds is normal while the model file is cold. The stall watchdog counts load
  progress, so a slow load that keeps moving is not stopped. On a slow disk, a load can still stop with a `stall`
  line in the log and exit code 4, with a message that names `stall_load_s`: raise `stall_load_s` in the config.
- **403.** The `Origin` header was not allowed: add the origin to `allowed_origins` (see Use it with your coding agent).
  With no key, a request whose `Host` is not `localhost`, `127.0.0.1` or `[::1]` also gets 403.
- **415.** A POST with a body needs `Content-Type: application/json`. A bare `curl -d` sends a form type, so add
  `-H "Content-Type: application/json"`.
- **400 about a run of whitespace.** A message held more than 262,144 whitespace characters in a row. Remove the run.
