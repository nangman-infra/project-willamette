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
    let dry_run_output = directory.join(format!(
        "willamette-prep-dry-run-{}-{nonce}.gguf",
        std::process::id()
    ));
    std::fs::write(&source, build_gguf(Preset::Small, false)).unwrap();

    let dry_run = Command::new(env!("CARGO_BIN_EXE_willamette-prep"))
        .args(["--model", source.to_str().unwrap(), "--output"])
        .arg(&dry_run_output)
        .args(["--profile", "embedding-q6-k", "--dry-run"])
        .output()
        .unwrap();
    assert!(
        dry_run.status.success(),
        "willamette-prep dry-run failed: {}",
        String::from_utf8_lossy(&dry_run.stderr)
    );
    assert!(!dry_run_output.exists());
    let dry_run_stdout = String::from_utf8_lossy(&dry_run.stdout);
    assert!(dry_run_stdout.contains("Profile:         embedding-q6-k"));
    assert!(dry_run_stdout.contains("Changed tensors: 1"));
    assert!(dry_run_stdout.contains("Tensor layout:"));
    assert!(dry_run_stdout.contains("Dry run: no output written"));

    let prep = Command::new(env!("CARGO_BIN_EXE_willamette-prep"))
        .args(["--model", source.to_str().unwrap(), "--output"])
        .arg(&standalone)
        .args(["--profile", "embedding-q6-k"])
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
