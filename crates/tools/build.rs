//! Links artemis.exe with `/DEPENDENTLOADFLAG:0x800` (LOAD_LIBRARY_SEARCH_SYSTEM32) on MSVC. The flag makes the loader
//! resolve the exe's load-time imports (`vcruntime140.dll` among them) from System32 only, so a DLL planted next to
//! `artemis.exe`, in the current directory or on PATH is not loaded. It applies to the `artemis` binary only; tests and
//! other targets are unaffected. `tools\build-release.ps1` reads the flag back out of the built file.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg-bin=artemis=/DEPENDENTLOADFLAG:0x800");
    }
}
