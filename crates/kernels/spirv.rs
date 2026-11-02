//! What `build.rs`, the `update-spirv` binary and `tests/spirv_fresh.rs` share, included in each with `#[path]`: the
//! glslc flags, the source hash that says whether a committed `spirv/<name>.spv` is still the compile of its shader,
//! the manifest format and the rule that picks where a build gets each shader's SPIR-V.
#![allow(dead_code)]

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

/// The glslc flags; a shader is `GLSLC_ARGS`, then `shaders/<name>.comp`, then `-o <out>`, run from the crate directory.
pub const GLSLC_ARGS: [&str; 3] = ["--target-env=vulkan1.3", "-O", "-Ishaders"];

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    fnv1a64_extend(0xcbf2_9ce4_8422_2325, bytes)
}

fn fnv1a64_extend(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The sorted stems of the `*.comp` files in `shaders_dir`.
pub fn shader_names(shaders_dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(shaders_dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", shaders_dir.display()))
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension()? == "comp")
                .then(|| p.file_stem().unwrap().to_string_lossy().into_owned())
        })
        .collect();
    names.sort();
    names
}

/// FNV-1a 64 over `<name>.comp`, then every file in `shaders_dir` that is not a `.comp` (the `.glsl` headers every
/// shader may include) in sorted file-name order, each preceded by its file name and a 0 byte. A change to any header
/// moves every shader's hash, which is safe: it only sends a build to glslc.
pub fn sources_hash(shaders_dir: &Path, name: &str) -> u64 {
    let read =
        |p: &Path| fs::read(p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()));
    let mut h = fnv1a64(&read(&shaders_dir.join(format!("{name}.comp"))));
    let mut others: Vec<String> = fs::read_dir(shaders_dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", shaders_dir.display()))
        .map(|e| e.unwrap())
        .filter(|e| e.path().is_file() && e.path().extension().is_none_or(|x| x != "comp"))
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    others.sort();
    for file in &others {
        h = fnv1a64_extend(h, file.as_bytes());
        h = fnv1a64_extend(h, &[0]);
        h = fnv1a64_extend(h, &read(&shaders_dir.join(file)));
    }
    h
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub sources: u64,
    pub spirv: u64,
}

/// `glslc <version>` on the first line, then `<name> <sources hex16> <spirv hex16>` per shader sorted by name.
pub fn write_manifest(glslc_version: &str, entries: &[Entry]) -> String {
    let mut sorted: Vec<&Entry> = entries.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    let mut text = format!("glslc {glslc_version}\n");
    for e in sorted {
        text.push_str(&format!("{} {:016x} {:016x}\n", e.name, e.sources, e.spirv));
    }
    text
}

pub fn read_manifest(text: &str) -> Result<(String, Vec<Entry>), String> {
    let mut lines = text.lines();
    let version = lines
        .next()
        .and_then(|l| l.strip_prefix("glslc "))
        .ok_or("the manifest's first line is not `glslc <version>`")?
        .to_string();
    let hex = |s: &str, line: &str| {
        (s.len() == 16)
            .then(|| u64::from_str_radix(s, 16).ok())
            .flatten()
            .ok_or_else(|| format!("bad hash `{s}` in manifest line `{line}`"))
    };
    let mut entries = Vec::new();
    for line in lines {
        let parts: Vec<&str> = line.split(' ').collect();
        let [name, sources, spirv] = parts[..] else {
            return Err(format!(
                "manifest line `{line}` is not `<name> <sources> <spirv>`"
            ));
        };
        entries.push(Entry {
            name: name.to_string(),
            sources: hex(sources, line)?,
            spirv: hex(spirv, line)?,
        });
    }
    Ok((version, entries))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// Copy the committed `spirv/<name>.spv`.
    Committed,
    /// Compile the shader with glslc.
    Glslc,
    /// Write an empty file (CI, which has neither GPU nor SDK).
    Stub,
}

/// Where a build gets one shader's SPIR-V. `fresh` means the committed file was compiled from the shader's current
/// sources; `skip` is `ARTEMIS_SKIP_SHADERS=1`.
pub fn choose_source(fresh: bool, glslc_available: bool, skip: bool) -> Result<Source, String> {
    if fresh {
        Ok(Source::Committed)
    } else if skip {
        Ok(Source::Stub)
    } else if glslc_available {
        Ok(Source::Glslc)
    } else {
        Err("the committed SPIR-V is stale and glslc is not available; install the Vulkan SDK and run \
             `cargo run -p kernels --bin update-spirv` (or set ARTEMIS_SKIP_SHADERS=1 for a build that only runs CPU tests). \
             Any file under shaders/ that is not a .comp, an untracked file or an editor backup included, changes every \
             shader's hash and makes all of them stale"
            .to_string())
    }
}

/// `%VULKAN_SDK%\Bin\glslc.exe` when it exists, else `glslc` on `PATH`.
pub fn find_glslc() -> PathBuf {
    std::env::var("VULKAN_SDK")
        .map(|sdk| PathBuf::from(sdk).join("Bin").join("glslc.exe"))
        .ok()
        .filter(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from("glslc"))
}

/// The first line of `glslc --version` (glslc repeats its version number on that line; once is kept), or `None` when
/// glslc does not run or fails.
pub fn glslc_version(glslc: &Path) -> Option<String> {
    let out = Command::new(glslc).arg("--version").output().ok()?;
    out.status.success().then(|| {
        let text = String::from_utf8_lossy(&out.stdout);
        let mut words: Vec<&str> = text
            .lines()
            .next()
            .unwrap_or("")
            .split_whitespace()
            .collect();
        words.dedup();
        words.join(" ")
    })
}

/// Runs glslc on `shaders/<name>.comp` from `crate_dir` (so `-Ishaders` and the source path mean the same wherever the
/// caller runs from), writing `dst`.
pub fn compile(glslc: &Path, crate_dir: &Path, name: &str, dst: &Path) -> Result<(), String> {
    let status = Command::new(glslc)
        .current_dir(crate_dir)
        .args(GLSLC_ARGS)
        .arg(format!("shaders/{name}.comp"))
        .arg("-o")
        .arg(dst)
        .status()
        .map_err(|e| format!("cannot run {} ({e})", glslc.display()))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("glslc failed on shaders/{name}.comp"))
    }
}

/// Whether the committed `<spirv_dir>/<name>.spv` was compiled from the shader's current sources: the manifest has an
/// entry for `name` with the current sources hash, and the `.spv` file exists and hashes to the entry's `spirv` field
/// (an unreadable file is not fresh).
pub fn is_fresh(entries: &[Entry], shaders_dir: &Path, spirv_dir: &Path, name: &str) -> bool {
    entries.iter().find(|e| e.name == name).is_some_and(|e| {
        e.sources == sources_hash(shaders_dir, name)
            && fs::read(spirv_dir.join(format!("{name}.spv")))
                .is_ok_and(|bytes| fnv1a64(&bytes) == e.spirv)
    })
}

/// The `.spv` files in `spirv_dir` whose stem is not in `names`, sorted by file name.
pub fn orphan_spirv(spirv_dir: &Path, names: &[String]) -> Vec<PathBuf> {
    let Ok(dir) = fs::read_dir(spirv_dir) else {
        return Vec::new();
    };
    let mut orphans: Vec<PathBuf> = dir
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "spv"))
        .filter(|p| {
            let stem = p.file_stem().unwrap().to_string_lossy();
            !names.iter().any(|n| *n == stem)
        })
        .collect();
    orphans.sort();
    orphans
}
