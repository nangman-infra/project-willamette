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
use project_willamette::tokenizer::{EncodeOptions, PromptPart, Tokenizer};

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

// ────────────────────────────────────────────────────────────────────
// Valid-path synthetic GGUF — exercises the success branches of
// `Tokenizer::from_gguf_metadata` and `Tokenizer::encode_with_specials`
// without needing the 1.1 GiB BitNet model file. Together with the
// rejection tests above this gives in-CI coverage for both halves of
// the tokenizer load path.
// ────────────────────────────────────────────────────────────────────

fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
    buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
    buf.write_all(s.as_bytes()).unwrap();
}

fn write_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
    write_gguf_string(buf, key);
    buf.write_u32::<LittleEndian>(8).unwrap(); // type tag: String
    write_gguf_string(buf, value);
}

fn write_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
    write_gguf_string(buf, key);
    buf.write_u32::<LittleEndian>(4).unwrap(); // type tag: Uint32
    buf.write_u32::<LittleEndian>(value).unwrap();
}

fn write_kv_bool(buf: &mut Vec<u8>, key: &str, value: bool) {
    write_gguf_string(buf, key);
    buf.write_u32::<LittleEndian>(7).unwrap(); // type tag: Bool
    buf.write_u8(u8::from(value)).unwrap();
}

fn write_kv_string_array(buf: &mut Vec<u8>, key: &str, values: &[String]) {
    write_gguf_string(buf, key);
    buf.write_u32::<LittleEndian>(9).unwrap(); // type tag: Array
    buf.write_u32::<LittleEndian>(8).unwrap(); // inner type: String
    buf.write_u64::<LittleEndian>(values.len() as u64).unwrap();
    for v in values {
        write_gguf_string(buf, v);
    }
}

/// Reproduce the GPT-2 byte → Unicode mapping used by
/// `src/tokenizer/byte_unicode.rs`. Returns 256 single-char strings
/// indexed by raw byte value. Every byte-level BPE encoder must include
/// all of these in its vocab so that any UTF-8 input round-trips.
///
/// The explicit index loops mirror the OpenAI `bytes_to_unicode()`
/// algorithm 1:1 — the same allow is on
/// `src/tokenizer/byte_unicode.rs` for the production version.
#[allow(clippy::needless_range_loop)]
fn gpt2_byte_unicode_vocab() -> Vec<String> {
    let mut printable = [false; 256];
    for b in (b'!' as usize)..=(b'~' as usize) {
        printable[b] = true;
    }
    for b in 0xA1usize..=0xACusize {
        printable[b] = true;
    }
    for b in 0xAEusize..=0xFFusize {
        printable[b] = true;
    }
    let mut byte_to_char: [char; 256] = ['\0'; 256];
    for b in 0usize..256 {
        if printable[b] {
            byte_to_char[b] = char::from_u32(b as u32).expect("printable code point");
        }
    }
    let mut next_code: u32 = 256;
    for b in 0usize..256 {
        if !printable[b] {
            byte_to_char[b] = char::from_u32(next_code).expect("U+0100.. valid");
            next_code += 1;
        }
    }
    byte_to_char.iter().map(|c| c.to_string()).collect()
}

/// Build a minimal valid byte-level BPE tokenizer GGUF (no tensors,
/// metadata only). 258-token vocab: 256 byte-unicode glyphs + BOS + EOS.
/// No merges, so encoding falls back to one byte = one id.
fn build_valid_synthetic_tokenizer_gguf() -> Vec<u8> {
    let mut tokens: Vec<String> = gpt2_byte_unicode_vocab();
    tokens.push("<|begin_of_text|>".to_string()); // id 256 — BOS
    tokens.push("<|end_of_text|>".to_string()); // id 257 — EOS
    let merges: Vec<String> = Vec::new();

    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap(); // version
    buf.write_u64::<LittleEndian>(0).unwrap(); // tensor_count = 0
    buf.write_u64::<LittleEndian>(6).unwrap(); // metadata_kv_count

    write_kv_string(&mut buf, "tokenizer.ggml.model", "gpt2");
    write_kv_string_array(&mut buf, "tokenizer.ggml.tokens", &tokens);
    write_kv_string_array(&mut buf, "tokenizer.ggml.merges", &merges);
    write_kv_u32(&mut buf, "tokenizer.ggml.bos_token_id", 256);
    write_kv_u32(&mut buf, "tokenizer.ggml.eos_token_id", 257);
    write_kv_bool(&mut buf, "tokenizer.ggml.add_bos_token", true);

    buf
}

fn synthetic_tokenizer() -> Tokenizer {
    let buf = build_valid_synthetic_tokenizer_gguf();
    let gguf = GgufFile::parse(&buf).expect("synthetic GGUF should parse");
    Tokenizer::from_gguf_metadata(&gguf.metadata).expect("synthetic tokenizer should build")
}

#[test]
fn synthetic_tokenizer_loads_with_byte_unicode_vocab() {
    let tok = synthetic_tokenizer();
    assert_eq!(tok.model_type, "gpt2");
    assert_eq!(tok.vocab_size(), 258);
    assert_eq!(tok.bos_id, Some(256));
    assert_eq!(tok.eos_id, Some(257));
    // Spot-check that the byte_unicode glyph for the ASCII letter
    // 'a' (byte 0x61) is in the vocab at id 0x61.
    assert_eq!(tok.token_str(0x61), Some("a"));
}

#[test]
fn synthetic_tokenizer_encodes_ascii_via_byte_fallback() {
    let tok = synthetic_tokenizer();
    let ids = tok
        .encode("ab", EncodeOptions::none())
        .expect("encode plain ASCII");
    // With no merges, each byte maps 1:1: 'a' = 0x61, 'b' = 0x62.
    assert_eq!(ids, vec![0x61, 0x62]);
}

#[test]
fn synthetic_encode_with_specials_text_only_equals_plain_encode() {
    let tok = synthetic_tokenizer();
    let text = "abc";
    let plain = tok.encode(text, EncodeOptions::none()).expect("plain");
    let specials = tok
        .encode_with_specials(&[PromptPart::Text(text)])
        .expect("specials");
    assert_eq!(plain, specials);
}

#[test]
fn synthetic_encode_with_specials_inserts_special_id_verbatim() {
    let tok = synthetic_tokenizer();
    let bos = tok.bos_id.unwrap();
    let eos = tok.eos_id.unwrap();
    let ids = tok
        .encode_with_specials(&[
            PromptPart::Special(bos),
            PromptPart::Text(""),
            PromptPart::Special(eos),
        ])
        .expect("specials");
    // Empty Text contributes zero ids; only the two special tokens land.
    assert_eq!(ids, vec![bos, eos]);
}

#[test]
fn synthetic_encode_with_specials_mixes_text_and_special() {
    let tok = synthetic_tokenizer();
    let bos = tok.bos_id.unwrap();
    let eos = tok.eos_id.unwrap();
    let ids = tok
        .encode_with_specials(&[
            PromptPart::Special(bos),
            PromptPart::Text("ab"),
            PromptPart::Special(eos),
        ])
        .expect("mixed");
    assert_eq!(ids.first().copied(), Some(bos));
    assert_eq!(ids.last().copied(), Some(eos));
    // Middle is the 2 bytes of "ab" each as one id.
    assert_eq!(ids.len(), 4);
    assert_eq!(&ids[1..3], &[0x61, 0x62]);
}

#[test]
fn synthetic_encode_with_specials_rejects_out_of_range_id() {
    let tok = synthetic_tokenizer();
    let bogus = tok.vocab_size() as u32 + 9999;
    let err = tok
        .encode_with_specials(&[PromptPart::Text("a"), PromptPart::Special(bogus)])
        .expect_err("out-of-range special id should be rejected");
    let msg = format!("{}", err);
    assert!(
        msg.contains("out of vocab"),
        "error should mention vocab range; got: {}",
        msg
    );
}

#[test]
fn synthetic_decode_round_trips_ascii_via_specials() {
    let tok = synthetic_tokenizer();
    let ids = tok
        .encode_with_specials(&[PromptPart::Text("abc")])
        .expect("encode_with_specials");
    let decoded = tok.decode(&ids).expect("decode");
    assert_eq!(decoded, "abc");
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
