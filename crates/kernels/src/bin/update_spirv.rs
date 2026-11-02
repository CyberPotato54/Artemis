//! Compiles every `shaders/*.comp` with glslc into `spirv/<name>.spv` and writes `spirv/MANIFEST`, which build.rs uses
//! to skip glslc for a shader whose sources have not changed. It deletes any `.spv` that no shader produces. `--check`
//! compiles into a temporary directory and exits 1 unless every output is byte-identical to the committed file and
//! `spirv/` holds no other `.spv` (proof that the committed bits are glslc's).
#[path = "../../spirv.rs"]
mod spirv;

use spirv::{
    compile, find_glslc, fnv1a64, glslc_version, orphan_spirv, read_manifest, shader_names,
    sources_hash, write_manifest, Entry,
};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// A directory removed when this drops, so a panic or an early return leaves nothing behind. `process::exit` skips
/// destructors, so `main` drops the guard before it exits.
struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn main() {
    let check = match std::env::args().nth(1).as_deref() {
        None => false,
        Some("--check") => true,
        Some(other) => {
            eprintln!("usage: update-spirv [--check]; got `{other}`");
            std::process::exit(2);
        }
    };
    // Compile from the crate directory, as build.rs does, so `-Ishaders` and the source paths mean the same.
    let krate = Path::new(env!("CARGO_MANIFEST_DIR"));
    let glslc = find_glslc();
    let Some(version) = glslc_version(&glslc) else {
        eprintln!(
            "cannot run {}; install the Vulkan SDK and set VULKAN_SDK",
            glslc.display()
        );
        std::process::exit(1);
    };
    let committed = krate.join("spirv");
    let (out, guard) = if check {
        let dir = std::env::temp_dir().join(format!("artemis-update-spirv-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        (dir.clone(), Some(TempDir(dir)))
    } else {
        (committed.clone(), None)
    };
    fs::create_dir_all(&out).unwrap();

    let shaders = krate.join("shaders");
    let names = shader_names(&shaders);
    let orphans = orphan_spirv(&committed, &names);
    let mut entries = Vec::new();
    let mut mismatches = Vec::new();
    for name in &names {
        let dst = out.join(format!("{name}.spv"));
        compile(&glslc, krate, name, &dst).unwrap_or_else(|e| panic!("{e}"));
        let bytes = fs::read(&dst).unwrap();
        if check
            && fs::read(committed.join(format!("{name}.spv")))
                .ok()
                .as_deref()
                != Some(&bytes)
        {
            mismatches.push(name.clone());
        }
        entries.push(Entry {
            name: name.clone(),
            sources: sources_hash(&shaders, name),
            spirv: fnv1a64(&bytes),
        });
    }

    if check {
        drop(guard);
        // The manifest must say what this glslc output and these sources give (its version line may differ).
        let manifest_ok = fs::read_to_string(committed.join("MANIFEST"))
            .ok()
            .is_some_and(|t| {
                read_manifest(&t).is_ok_and(|(committed_version, _)| {
                    t == write_manifest(&committed_version, &entries)
                })
            });
        if !mismatches.is_empty() || !manifest_ok || !orphans.is_empty() {
            if !mismatches.is_empty() {
                eprintln!(
                    "committed SPIR-V differs from {version}'s output for: {}",
                    mismatches.join(", ")
                );
            }
            if !manifest_ok {
                eprintln!("spirv/MANIFEST does not match the shaders and the committed SPIR-V");
            }
            for orphan in &orphans {
                eprintln!("{} belongs to no shader", orphan.display());
            }
            eprintln!("run `cargo run -p kernels --bin update-spirv` and commit the result");
            std::process::exit(1);
        }
        println!(
            "{} shaders: committed SPIR-V is byte-identical to {version}'s output",
            names.len()
        );
    } else {
        for orphan in &orphans {
            fs::remove_file(orphan).unwrap();
            println!("deleted {}, which no shader produces", orphan.display());
        }
        fs::write(
            committed.join("MANIFEST"),
            write_manifest(&version, &entries),
        )
        .unwrap();
        println!(
            "{} shaders compiled with {version} into {}",
            names.len(),
            committed.display()
        );
    }
}
