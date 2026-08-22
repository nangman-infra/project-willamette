use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use project_willamette::synth::{build_gguf, Preset};
use serde_json::Value;

fn model_path() -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "willamette-bench-json-{}-{nonce}.gguf",
        std::process::id()
    ))
}

#[test]
fn bench_json_emits_one_versioned_object() {
    let model = model_path();
    std::fs::write(&model, build_gguf(Preset::Tiny, false)).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_project-willamette"))
        .args(["bench", "--model"])
        .arg(&model)
        .args(["--decode-steps", "1", "--format", "json"])
        .output()
        .unwrap();
    std::fs::remove_file(&model).unwrap();

    assert!(
        output.status.success(),
        "bench failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["runtime"]["name"], "willamette");
    assert_eq!(value["model"]["architecture"], "bitnet-b1.58");
    assert_eq!(value["config"]["decode_steps"], 1);
    for key in [
        "matvec_ms",
        "forward_no_cache_ms",
        "cached_forward_ms",
        "lm_head_ms",
        "argmax_ms",
        "complete_token_ms",
        "complete_tokens_per_second",
    ] {
        assert!(value["metrics"][key].as_f64().unwrap().is_finite());
    }
}

#[test]
fn bench_rejects_zero_decode_steps() {
    let model = model_path();
    std::fs::write(&model, build_gguf(Preset::Tiny, false)).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_project-willamette"))
        .args(["bench", "--model"])
        .arg(&model)
        .args(["--decode-steps", "0", "--format", "json"])
        .output()
        .unwrap();
    std::fs::remove_file(&model).unwrap();

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("--decode-steps must be greater than zero"));
}

#[test]
#[ignore = "requires the pinned 144810912-byte SmolLM-135M-Instruct-Q8_0.gguf"]
fn bench_json_accepts_pinned_q8_0_model() {
    let model = std::path::Path::new("models/SmolLM-135M-Instruct-Q8_0.gguf");

    let output = Command::new(env!("CARGO_BIN_EXE_project-willamette"))
        .args(["bench", "--model"])
        .arg(model)
        .args(["--decode-steps", "1", "--format", "json"])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "bench failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["model"]["architecture"], "llama");
    assert!(value["runtime"]["kernel"]
        .as_str()
        .unwrap()
        .starts_with("Q8_0 "));
    assert!(value["metrics"]["complete_token_ms"]
        .as_f64()
        .unwrap()
        .is_finite());
}
