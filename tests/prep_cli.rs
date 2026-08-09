use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use project_willamette::synth::{build_gguf, Preset};

#[test]
fn standalone_and_runtime_prep_interfaces_match() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir();
    let source = directory.join(format!(
        "willamette-prep-source-{}-{nonce}.gguf",
        std::process::id()
    ));
    let standalone = directory.join(format!(
        "willamette-prep-standalone-{}-{nonce}.gguf",
        std::process::id()
    ));
    let compatibility = directory.join(format!(
        "willamette-prep-compat-{}-{nonce}.gguf",
        std::process::id()
    ));
    std::fs::write(&source, build_gguf(Preset::Small, false)).unwrap();

    let prep = Command::new(env!("CARGO_BIN_EXE_willamette-prep"))
        .args(["--model", source.to_str().unwrap(), "--output"])
        .arg(&standalone)
        .output()
        .unwrap();
    assert!(
        prep.status.success(),
        "willamette-prep failed: {}",
        String::from_utf8_lossy(&prep.stderr)
    );

    let runtime = Command::new(env!("CARGO_BIN_EXE_project-willamette"))
        .args([
            "repack-embedding-q6k",
            "--model",
            source.to_str().unwrap(),
            "--output",
        ])
        .arg(&compatibility)
        .output()
        .unwrap();
    assert!(
        runtime.status.success(),
        "runtime compatibility command failed: {}",
        String::from_utf8_lossy(&runtime.stderr)
    );

    assert_eq!(
        std::fs::read(&standalone).unwrap(),
        std::fs::read(&compatibility).unwrap()
    );
    for path in [source, standalone, compatibility] {
        std::fs::remove_file(path).unwrap();
    }
}
