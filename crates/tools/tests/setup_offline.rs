//! Setup reads the model, the GPU and the OS and nothing else: no network code, no test 7 harness.
use std::{fs, path::Path};

const FORBIDDEN: [&str; 8] = [
    "hyper",
    "reqwest",
    "std::net",
    "tokio::net",
    "TcpStream",
    "UdpSocket",
    "http::",
    "test7",
];

fn files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for e in fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let p = e.unwrap().path();
        if p.is_dir() {
            files(&p, out);
        } else {
            out.push(p);
        }
    }
}

#[test]
fn the_setup_module_uses_no_network() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/setup");
    let mut all = Vec::new();
    files(&dir, &mut all);
    assert!(all.len() >= 5, "the setup module's files: {all:?}");
    let mut found = Vec::new();
    for f in &all {
        let text = fs::read_to_string(f).unwrap();
        for word in FORBIDDEN {
            if text.contains(word) {
                found.push(format!("{} names {word}", f.display()));
            }
        }
    }
    assert!(found.is_empty(), "{}", found.join("\n"));
}
