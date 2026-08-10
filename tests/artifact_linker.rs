use std::io::Read;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use project_willamette::gguf::linker::{link_artifact, ArtifactProfile};
use project_willamette::memory::mmap::ModelMmap;
use sha2::{Digest, Sha256};

const MODEL_PATH: &str = "./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf";
const ARTIFACT_BYTES: u64 = 800_468_160;
const ARTIFACT_SHA256: &str = "492e4d2a8db2eefc5f8c86acd08eea6707294de67ce871b5d732e9bfcb468376";

#[test]
#[ignore = "requires the external BitNet GGUF and writes a 0.745 GiB artifact"]
fn pinned_q6k_artifact_checksum_is_stable() {
    if !Path::new(MODEL_PATH).exists() {
        panic!("real GGUF not found at {MODEL_PATH}; see REPRODUCIBILITY.md");
    }
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let output_path = std::env::temp_dir().join(format!(
        "willamette-pinned-artifact-{}-{nonce}.gguf",
        std::process::id()
    ));
    let mmap = ModelMmap::open(MODEL_PATH).unwrap();
    let report =
        link_artifact(mmap.as_bytes(), &output_path, ArtifactProfile::EmbeddingQ6K).unwrap();
    assert_eq!(report.plan.output_bytes, ARTIFACT_BYTES);

    let mut file = std::fs::File::open(&output_path).unwrap();
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).unwrap();
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    drop(file);
    std::fs::remove_file(output_path).unwrap();
    assert_eq!(format!("{:x}", hasher.finalize()), ARTIFACT_SHA256);
}
