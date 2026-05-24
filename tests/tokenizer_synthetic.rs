//! Synthetic-GGUF tests for the tokenizer factory's error paths.
//!
//! These tests construct minimal in-memory GGUF byte buffers (no model file
//! required) to exercise each branch of `Tokenizer::from_gguf_metadata` that
//! must reject malformed or unsupported input. They guard the rule that the
//! tokenizer never silently falls back to a fake or partial vocabulary.

use std::collections::HashMap;
use std::io::Write;

use byteorder::{LittleEndian, WriteBytesExt};

use project_willamette::error::WillametteError;
use project_willamette::gguf::reader::{GgufFile, GgufValue, GGUF_MAGIC};
use project_willamette::tokenizer::Tokenizer;

/// Construct a minimal valid GGUF byte buffer that has zero tensors and the
/// provided string-typed metadata KVs.
fn build_synthetic_gguf_strings(kvs: &[(&str, &str)]) -> Vec<u8> {
    let mut buf = Vec::new();
    // Magic "GGUF" — single source of truth
    buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
    // Version 3
    buf.write_u32::<LittleEndian>(3).unwrap();
    // tensor_count = 0
    buf.write_u64::<LittleEndian>(0).unwrap();
    // metadata_kv_count
    buf.write_u64::<LittleEndian>(kvs.len() as u64).unwrap();

    for (k, v) in kvs {
        // Key
        buf.write_u64::<LittleEndian>(k.len() as u64).unwrap();
        buf.write_all(k.as_bytes()).unwrap();
        // Value type tag (8 = String)
        buf.write_u32::<LittleEndian>(8).unwrap();
        // Value
        buf.write_u64::<LittleEndian>(v.len() as u64).unwrap();
        buf.write_all(v.as_bytes()).unwrap();
    }
    buf
}

#[test]
fn empty_metadata_rejected() {
    let meta: HashMap<String, GgufValue> = HashMap::new();
    let result = Tokenizer::from_gguf_metadata(&meta);
    assert!(
        matches!(result, Err(WillametteError::UnsupportedTokenizer(_))),
        "empty metadata must produce UnsupportedTokenizer (got Ok or other Err variant)"
    );
}

#[test]
fn missing_tokenizer_model_key_rejected() {
    // GGUF with one metadata KV that is NOT tokenizer.ggml.model
    let buf = build_synthetic_gguf_strings(&[("general.architecture", "test-arch")]);
    let gguf = GgufFile::parse(&buf).expect("synthetic GGUF should parse");
    let result = Tokenizer::from_gguf_metadata(&gguf.metadata);
    match result {
        Err(WillametteError::UnsupportedTokenizer(msg)) => {
            assert!(
                msg.contains("tokenizer.ggml.model"),
                "error message should mention the missing key, got: {}",
                msg
            );
        }
        Err(other) => panic!(
            "expected UnsupportedTokenizer with key message, got different error: {}",
            other
        ),
        Ok(_) => panic!("expected UnsupportedTokenizer, got Ok(_)"),
    }
}

#[test]
fn non_gpt2_tokenizer_model_rejected() {
    // tokenizer.ggml.model = "llama" (SentencePiece — out of Stage 2 scope)
    let buf = build_synthetic_gguf_strings(&[("tokenizer.ggml.model", "llama")]);
    let gguf = GgufFile::parse(&buf).expect("synthetic GGUF should parse");
    let result = Tokenizer::from_gguf_metadata(&gguf.metadata);
    match result {
        Err(WillametteError::UnsupportedTokenizer(msg)) => {
            assert!(
                msg.contains("gpt2") || msg.contains("llama"),
                "error message should explain the rejection, got: {}",
                msg
            );
        }
        Err(other) => panic!(
            "expected UnsupportedTokenizer for non-gpt2, got different error: {}",
            other
        ),
        Ok(_) => panic!("expected UnsupportedTokenizer for non-gpt2, got Ok(_)"),
    }
}

#[test]
fn gpt2_without_tokens_rejected() {
    // tokenizer.ggml.model = "gpt2" but no tokens array
    let buf = build_synthetic_gguf_strings(&[("tokenizer.ggml.model", "gpt2")]);
    let gguf = GgufFile::parse(&buf).expect("synthetic GGUF should parse");
    let result = Tokenizer::from_gguf_metadata(&gguf.metadata);
    match result {
        Err(WillametteError::UnsupportedTokenizer(msg)) => {
            assert!(
                msg.contains("tokenizer.ggml.tokens"),
                "error message should mention the missing tokens key, got: {}",
                msg
            );
        }
        Err(other) => panic!(
            "expected UnsupportedTokenizer for missing tokens, got different error: {}",
            other
        ),
        Ok(_) => panic!("expected UnsupportedTokenizer for missing tokens, got Ok(_)"),
    }
}

/// Meta-test: scan `src/` for any pattern that would indicate a fake
/// tokenizer or fake-weight code has re-entered the codebase. This guards
/// the project-wide rule that real inference paths must use only real
/// GGUF-backed data.
#[test]
fn source_tree_contains_no_fake_tokenizer_or_pseudo_weights() {
    use std::fs;
    use std::path::Path;

    // Patterns that indicated forbidden code in earlier project versions.
    //
    // Each pattern is split so this very file does NOT match itself.
    let forbidden: &[(&str, &str)] = &[
        ("Simple", "Tokenizer"),
        ("Llm", "Generator"),
        ("positional_", "noise"),
        ("init_buf", "(seed:"),
    ];

    fn walk(dir: &Path, out: &mut Vec<String>, forbidden: &[(&str, &str)]) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out, forbidden);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                let Ok(content) = fs::read_to_string(&path) else {
                    continue;
                };
                for (a, b) in forbidden {
                    let needle = format!("{}{}", a, b);
                    if content.contains(&needle) {
                        out.push(format!("{}: \"{}\"", path.display(), needle));
                    }
                }
            }
        }
    }

    let mut found = Vec::new();
    walk(Path::new("src"), &mut found, forbidden);
    assert!(
        found.is_empty(),
        "Forbidden patterns found in src/ — re-introducing fake code is not allowed:\n{:#?}",
        found
    );
}
