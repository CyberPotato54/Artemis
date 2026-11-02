//! The committed SPIR-V under `spirv/` is the compile of the shaders as they stand. This is what lets a build with no
//! Vulkan SDK make real kernels; `cargo run -p kernels --bin update-spirv` regenerates it.
#[path = "../spirv.rs"]
mod spirv;

use spirv::{
    choose_source, fnv1a64, is_fresh, orphan_spirv, read_manifest, shader_names, sources_hash,
    write_manifest, Entry, Source,
};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// A directory removed when this drops, also when an assertion panics.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        let dir = std::env::temp_dir().join(format!("artemis-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn choose_source_rules() {
    // Fresh uses the committed file whatever else is true.
    for (glslc, skip) in [(false, false), (true, false), (false, true), (true, true)] {
        assert_eq!(choose_source(true, glslc, skip), Ok(Source::Committed));
    }
    assert_eq!(choose_source(false, true, false), Ok(Source::Glslc));
    assert_eq!(choose_source(false, true, true), Ok(Source::Stub));
    assert_eq!(choose_source(false, false, true), Ok(Source::Stub));
    let err = choose_source(false, false, false).unwrap_err();
    assert!(err.contains("update-spirv"), "{err}");
}

#[test]
fn every_shader_has_fresh_committed_spirv() {
    let shaders = Path::new("shaders");
    let text = fs::read_to_string("spirv/MANIFEST")
        .expect("spirv/MANIFEST; run `cargo run -p kernels --bin update-spirv`");
    let (_, entries) = read_manifest(&text).unwrap();
    assert_eq!(
        text.lines().next(),
        Some("glslc shaderc v2026.4"),
        "spirv/MANIFEST was written by a different glslc; regenerate it with the Vulkan SDK 1.4.363.0 (glslc shaderc v2026.4)"
    );
    let manifest_names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    let names = shader_names(shaders);
    assert_eq!(
        manifest_names,
        names.iter().map(String::as_str).collect::<Vec<_>>(),
        "spirv/MANIFEST and shaders/*.comp list different shaders; run `cargo run -p kernels --bin update-spirv`"
    );
    let orphans = orphan_spirv(Path::new("spirv"), &names);
    assert!(
        orphans.is_empty(),
        "spirv/ holds .spv files that no shader produces: {orphans:?}; run `cargo run -p kernels --bin update-spirv`"
    );
    for e in &entries {
        assert_eq!(
            sources_hash(shaders, &e.name),
            e.sources,
            "{}: the committed SPIR-V is stale; run `cargo run -p kernels --bin update-spirv`",
            e.name
        );
        let bytes = fs::read(format!("spirv/{}.spv", e.name))
            .unwrap_or_else(|err| panic!("{}: spirv/{}.spv is unreadable ({err}); run `cargo run -p kernels --bin update-spirv`", e.name, e.name));
        assert_eq!(
            fnv1a64(&bytes),
            e.spirv,
            "{}: spirv/{}.spv is not the file the manifest records; run `cargo run -p kernels --bin update-spirv`",
            e.name,
            e.name
        );
    }
}

#[test]
fn an_include_change_moves_every_hash() {
    let guard = TempDir::new("spirv-fresh");
    let dir = guard.0.clone();
    for e in fs::read_dir("shaders").unwrap() {
        let p = e.unwrap().path();
        fs::copy(&p, dir.join(p.file_name().unwrap())).unwrap();
    }
    let names = shader_names(&dir);
    assert!(!names.is_empty());
    let before: Vec<u64> = names.iter().map(|n| sources_hash(&dir, n)).collect();
    let mut common = fs::read(dir.join("common.glsl")).unwrap();
    common.push(b'\n');
    fs::write(dir.join("common.glsl"), common).unwrap();
    for (n, old) in names.iter().zip(&before) {
        assert_ne!(
            sources_hash(&dir, n),
            *old,
            "{n}: its hash ignored a change to common.glsl"
        );
    }
}

#[test]
fn freshness_needs_a_matching_hash_and_the_spv() {
    let guard = TempDir::new("spirv-is-fresh");
    let shaders = guard.0.join("shaders");
    let spirv_dir = guard.0.join("spirv");
    fs::create_dir_all(&shaders).unwrap();
    fs::create_dir_all(&spirv_dir).unwrap();
    fs::write(shaders.join("add.comp"), b"void main() {}\n").unwrap();
    fs::write(shaders.join("common.glsl"), b"// header\n").unwrap();
    let entries = vec![Entry {
        name: "add".into(),
        sources: sources_hash(&shaders, "add"),
        spirv: fnv1a64(b"spv"),
    }];
    // The hash matches but the .spv is missing.
    assert!(!is_fresh(&entries, &shaders, &spirv_dir, "add"));
    fs::write(spirv_dir.join("add.spv"), b"spv").unwrap();
    assert!(is_fresh(&entries, &shaders, &spirv_dir, "add"));
    // The .spv's bytes differ from the ones the manifest recorded.
    fs::write(spirv_dir.join("add.spv"), b"spw").unwrap();
    assert!(!is_fresh(&entries, &shaders, &spirv_dir, "add"));
    fs::write(spirv_dir.join("add.spv"), b"spv").unwrap();
    // A name the manifest does not list.
    assert!(!is_fresh(&entries, &shaders, &spirv_dir, "other"));
    // The source changed after the manifest was written.
    fs::write(shaders.join("add.comp"), b"void main() { }\n").unwrap();
    assert!(!is_fresh(&entries, &shaders, &spirv_dir, "add"));
}

#[test]
fn orphans_are_the_spv_files_no_shader_produces() {
    let guard = TempDir::new("spirv-orphans");
    for f in ["add.spv", "gone.spv", "MANIFEST", "note.txt"] {
        fs::write(guard.0.join(f), b"x").unwrap();
    }
    let orphans = orphan_spirv(&guard.0, &["add".to_string()]);
    assert_eq!(orphans, vec![guard.0.join("gone.spv")]);
}

#[test]
fn manifest_round_trips() {
    let entries = vec![
        Entry {
            name: "add".into(),
            sources: 0x0123_4567_89ab_cdef,
            spirv: 1,
        },
        Entry {
            name: "silu".into(),
            sources: u64::MAX,
            spirv: 0,
        },
    ];
    let text = write_manifest("shaderc v2026.1", &entries);
    assert_eq!(
        read_manifest(&text),
        Ok(("shaderc v2026.1".to_string(), entries))
    );
}
