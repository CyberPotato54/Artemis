//! Compiles shaders/*.comp to SPIR-V with glslc from the Vulkan SDK. CI has no SDK: with
//! ARTEMIS_SKIP_SHADERS=1 it writes empty stubs, which is enough to build and run the CPU tests.
use std::{env, fs, path::PathBuf, process::Command};

const SHADERS: &[&str] = &["read_bw", "spin", "ping", "matvec_q4k", "matvec_q8_0"];

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-env-changed=ARTEMIS_SKIP_SHADERS");
    let skip = env::var("ARTEMIS_SKIP_SHADERS").is_ok_and(|v| v == "1");
    let glslc = env::var("VULKAN_SDK")
        .map(|sdk| PathBuf::from(sdk).join("Bin").join("glslc.exe"))
        .ok()
        .filter(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from("glslc"));
    for name in SHADERS {
        let src = format!("shaders/{name}.comp");
        let dst = out.join(format!("{name}.spv"));
        println!("cargo:rerun-if-changed={src}");
        if skip {
            fs::write(&dst, []).unwrap();
            continue;
        }
        let status = Command::new(&glslc)
            .args(["--target-env=vulkan1.3", "-O", &src, "-o"])
            .arg(&dst)
            .status()
            .unwrap_or_else(|e| {
                panic!(
                    "cannot run {} ({e}); install the Vulkan SDK or set ARTEMIS_SKIP_SHADERS=1",
                    glslc.display()
                )
            });
        assert!(status.success(), "glslc failed on {src}");
    }
}
