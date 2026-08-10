//! Compatibility wrapper for the original narrow Q6_K repacker API.

use crate::error::WillametteError;
use crate::gguf::linker::{link_artifact, ArtifactProfile};
use std::path::Path;

/// Create a derived GGUF whose tied `token_embd.weight` is standard Q6_K.
pub fn repack_embedding_q6k(source: &[u8], output_path: &Path) -> Result<u64, WillametteError> {
    let report = link_artifact(source, output_path, ArtifactProfile::EmbeddingQ6K)?;
    Ok(report.plan.output_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::reader::GgufFile;
    use crate::gguf::types::GgmlType;
    use crate::synth::{build_gguf, Preset};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn repacks_synthetic_embedding_and_preserves_graph() {
        let source = build_gguf(Preset::Small, false);
        let source_gguf = GgufFile::parse(&source).unwrap();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "willamette-q6k-repack-{}-{nonce}.gguf",
            std::process::id()
        ));
        let expected_size = repack_embedding_q6k(&source, &path).unwrap();
        let repacked = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(repacked.len() as u64, expected_size);

        let gguf = GgufFile::parse(&repacked).unwrap();
        let embedding = gguf
            .tensors
            .iter()
            .find(|tensor| tensor.name == "token_embd.weight")
            .unwrap();
        assert_eq!(embedding.ggml_type, GgmlType::Q6K);
        embedding.verify_byte_len().unwrap();
        for source_tensor in source_gguf
            .tensors
            .iter()
            .filter(|tensor| tensor.name != "token_embd.weight")
        {
            let repacked_tensor = gguf
                .tensors
                .iter()
                .find(|tensor| tensor.name == source_tensor.name)
                .unwrap();
            assert_eq!(
                repacked_tensor.data, source_tensor.data,
                "{}",
                source_tensor.name
            );
            assert_eq!(
                repacked_tensor.scale_data, source_tensor.scale_data,
                "{} scale block",
                source_tensor.name
            );
        }
        crate::model::graph::ModelGraph::from_gguf(&gguf).unwrap();
    }

    #[test]
    fn does_not_overwrite_existing_output() {
        let source = build_gguf(Preset::Small, false);
        let path = std::env::temp_dir().join(format!(
            "willamette-q6k-existing-{}-{}.gguf",
            std::process::id(),
            TEST_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, b"keep").unwrap();
        let error = repack_embedding_q6k(&source, &path).unwrap_err();
        assert!(
            matches!(error, WillametteError::Io(ref io) if io.kind() == std::io::ErrorKind::AlreadyExists)
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"keep");
        std::fs::remove_file(path).unwrap();
    }
}
