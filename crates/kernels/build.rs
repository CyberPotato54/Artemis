//! Gets every shaders/*.comp's SPIR-V into OUT_DIR. A shader whose sources match spirv/MANIFEST uses the committed
//! spirv/<name>.spv, so a build needs no Vulkan SDK. A changed shader is compiled with glslc from the Vulkan SDK
//! (shaders/ on the include path) and a warning says to run `cargo run -p kernels --bin update-spirv`. CI has no SDK:
//! with ARTEMIS_SKIP_SHADERS=1 a changed shader gets an empty stub, which is enough to build and run the CPU tests.
use std::{
    env, fs,
    path::{Path, PathBuf},
};

#[path = "spirv.rs"]
mod spirv;
use spirv::{
    choose_source, compile, find_glslc, glslc_version, is_fresh, read_manifest, shader_names,
    Source,
};

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-env-changed=ARTEMIS_SKIP_SHADERS");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    // Directories: cargo rescans everything under them.
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-changed=spirv");
    let skip = env::var("ARTEMIS_SKIP_SHADERS").is_ok_and(|v| v == "1");
    let glslc = find_glslc();
    let glslc_available = glslc_version(&glslc).is_some();
    // No manifest is the bootstrap case (every shader stale). An unreadable or unparseable one is also treated as
    // every shader stale, but says why.
    let manifest = match fs::read_to_string("spirv/MANIFEST") {
        Ok(text) => match read_manifest(&text) {
            Ok((_, entries)) => entries,
            Err(e) => {
                // The parse errors are one line each.
                println!("cargo:warning=spirv/MANIFEST is unparseable ({e}); treating every shader as stale");
                Vec::new()
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => {
            println!(
                "cargo:warning=cannot read spirv/MANIFEST ({e}); treating every shader as stale"
            );
            Vec::new()
        }
    };
    let shaders = Path::new("shaders");
    for name in &shader_names(shaders) {
        let committed = format!("spirv/{name}.spv");
        let dst = out.join(format!("{name}.spv"));
        let fresh = is_fresh(&manifest, shaders, Path::new("spirv"), name);
        match choose_source(fresh, glslc_available, skip).unwrap_or_else(|e| panic!("{name}: {e}"))
        {
            Source::Committed => {
                fs::copy(&committed, &dst)
                    .unwrap_or_else(|e| panic!("cannot copy {committed}: {e}"));
            }
            Source::Stub => fs::write(&dst, []).unwrap(),
            Source::Glslc => {
                println!("cargo:warning={name}: committed SPIR-V is stale; run cargo run -p kernels --bin update-spirv");
                compile(&glslc, Path::new("."), name, &dst).unwrap_or_else(|e| panic!("{e}"));
            }
        }
    }
}
