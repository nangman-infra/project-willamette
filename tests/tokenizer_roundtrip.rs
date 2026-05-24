//! Roundtrip tests for the GGUF byte-level BPE tokenizer.
//!
//! These tests require the real `microsoft/bitnet-b1.58-2B-4T-gguf` file at
//! `./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf`. If the file is
//! absent, every test prints a clear SKIP message and passes — Stage 2's
//! invariants are checked exhaustively only when the real model is present.

use std::path::Path;

use project_willamette::gguf::reader::GgufFile;
use project_willamette::memory::mmap::ModelMmap;
use project_willamette::tokenizer::{EncodeOptions, Tokenizer};

const MODEL_PATH: &str = "./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf";

/// Open the real model, parse GGUF, build the tokenizer, and pass it to `f`.
/// Skips the test (returns without asserting) if the file is missing.
fn with_real_tokenizer<F: FnOnce(&Tokenizer)>(f: F) {
    if !Path::new(MODEL_PATH).exists() {
        eprintln!(
            "SKIP: real GGUF not found at {} — Stage 2 roundtrip tests require it",
            MODEL_PATH
        );
        return;
    }
    let mmap = ModelMmap::open(MODEL_PATH).expect("open real GGUF");
    let bytes = mmap.as_bytes();
    let gguf = GgufFile::parse(bytes).expect("parse real GGUF");
    let tokenizer =
        Tokenizer::from_gguf_metadata(&gguf.metadata).expect("build tokenizer from real GGUF");
    f(&tokenizer);
}

fn assert_roundtrip(tok: &Tokenizer, text: &str) {
    let opts = EncodeOptions::none();
    let ids = tok.encode(text, opts).expect("encode");
    let decoded = tok.decode(&ids).expect("decode");
    assert_eq!(
        decoded, text,
        "byte-level BPE roundtrip failed for {:?}: got {:?}",
        text, decoded
    );
}

#[test]
fn roundtrip_hello_ascii() {
    with_real_tokenizer(|tok| assert_roundtrip(tok, "hello"));
}

#[test]
fn roundtrip_hello_with_punctuation() {
    with_real_tokenizer(|tok| assert_roundtrip(tok, "Hello, world!"));
}

#[test]
fn roundtrip_korean_annyeong() {
    with_real_tokenizer(|tok| assert_roundtrip(tok, "안녕하세요"));
}

#[test]
fn roundtrip_korean_phrase() {
    with_real_tokenizer(|tok| {
        assert_roundtrip(
            tok,
            "프로젝트 윌라메트는 진짜 BitNet GGUF 추론 런타임입니다.",
        );
    });
}

#[test]
fn roundtrip_emoji_simple() {
    with_real_tokenizer(|tok| assert_roundtrip(tok, "hello 🎉 world"));
}

#[test]
fn roundtrip_emoji_dense_and_korean() {
    with_real_tokenizer(|tok| {
        assert_roundtrip(tok, "Hi 🚀 안녕 🌟 emoji ✨ 한글 + 123! 🎉");
    });
}

#[test]
fn roundtrip_empty_string() {
    with_real_tokenizer(|tok| {
        let opts = EncodeOptions::none();
        let ids = tok.encode("", opts).expect("encode empty");
        assert!(ids.is_empty(), "empty input should produce no tokens");
        let decoded = tok.decode(&ids).expect("decode empty");
        assert_eq!(decoded, "");
    });
}

#[test]
fn roundtrip_whitespace_and_newlines() {
    with_real_tokenizer(|tok| {
        assert_roundtrip(tok, "  leading\n\ttabs and newlines  ");
    });
}

#[test]
fn add_bos_true_prepends_bos_id() {
    with_real_tokenizer(|tok| {
        let text = "hello";
        let ids_no = tok
            .encode(text, EncodeOptions::none())
            .expect("no-bos encode");
        let ids_yes = tok
            .encode(
                text,
                EncodeOptions {
                    add_bos: true,
                    add_eos: false,
                },
            )
            .expect("bos encode");

        let bos = tok.bos_id.expect("model must declare bos_token_id");
        assert_eq!(
            ids_yes.len(),
            ids_no.len() + 1,
            "add_bos should add exactly one token"
        );
        assert_eq!(ids_yes[0], bos, "first token must be the BOS id");
        assert_eq!(
            &ids_yes[1..],
            &ids_no[..],
            "remaining ids must match the no-BOS encoding"
        );
    });
}

#[test]
fn add_bos_false_does_not_prepend_bos_id() {
    with_real_tokenizer(|tok| {
        let ids = tok
            .encode(
                "hello world",
                EncodeOptions {
                    add_bos: false,
                    add_eos: false,
                },
            )
            .expect("encode");
        if let Some(bos) = tok.bos_id {
            assert!(
                !ids.contains(&bos),
                "BOS id {} unexpectedly present in {:?}",
                bos,
                ids
            );
        }
    });
}

#[test]
fn metadata_special_tokens_match_inspect_log() {
    // Values verified in Stage 1 against the official GGUF (inspect.log
    // lines 20, 21, 24).
    with_real_tokenizer(|tok| {
        assert_eq!(tok.bos_id, Some(128000));
        assert_eq!(tok.eos_id, Some(128001));
        assert_eq!(tok.pad_id, Some(128001));
        assert!(tok.default_add_bos);
        assert_eq!(tok.vocab_size(), 128256);
        assert_eq!(tok.model_type, "gpt2");
    });
}

#[test]
fn roundtrip_does_not_depend_on_token_id_hardcoding() {
    // This test guards an explicit project rule: token-id sequences may
    // differ from bitnet.cpp until Stage 5 cross-validation. Roundtrip
    // correctness is the only contract Stage 2 commits to. Document that
    // by asserting roundtrip without hardcoding any specific id list.
    with_real_tokenizer(|tok| {
        for text in [
            "a",
            "ab",
            "abc",
            "hello",
            "안",
            "안녕",
            "🎉",
            "hello 안녕 🎉",
        ] {
            let ids = tok.encode(text, EncodeOptions::none()).expect("encode");
            assert!(
                !ids.is_empty() || text.is_empty(),
                "non-empty input must produce ids"
            );
            let decoded = tok.decode(&ids).expect("decode");
            assert_eq!(decoded, text);
        }
    });
}

#[test]
fn encode_with_specials_text_only_matches_plain_encode() {
    use project_willamette::tokenizer::PromptPart;

    with_real_tokenizer(|tok| {
        let text = "Human: hello\n\nBITNETAssistant: ";
        let want = tok.encode(text, EncodeOptions::none()).expect("encode");
        let got = tok
            .encode_with_specials(&[PromptPart::Text(text)])
            .expect("encode_with_specials");
        assert_eq!(
            got, want,
            "text-only encode_with_specials must equal plain encode"
        );
    });
}

#[test]
fn encode_with_specials_inserts_special_verbatim() {
    use project_willamette::tokenizer::PromptPart;

    with_real_tokenizer(|tok| {
        let eot_id: u32 = 128009; // <|eot_id|> for LLaMA-3 family
        let ids = tok
            .encode_with_specials(&[
                PromptPart::Text("Hello"),
                PromptPart::Special(eot_id),
                PromptPart::Text(" world"),
            ])
            .expect("encode_with_specials");
        let hello = tok.encode("Hello", EncodeOptions::none()).expect("hello");
        let world = tok.encode(" world", EncodeOptions::none()).expect("world");
        let mut want = Vec::new();
        want.extend(&hello);
        want.push(eot_id);
        want.extend(&world);
        assert_eq!(ids, want);
        assert!(
            ids.iter().any(|&id| id == eot_id),
            "eot_id must appear in the output verbatim"
        );
    });
}

#[test]
fn encode_with_specials_rejects_out_of_range_special_id() {
    use project_willamette::tokenizer::PromptPart;

    with_real_tokenizer(|tok| {
        let bogus = tok.vocab_size() as u32 + 1;
        let err = tok
            .encode_with_specials(&[PromptPart::Text("hi"), PromptPart::Special(bogus)])
            .expect_err("must reject out-of-vocab id");
        let msg = format!("{}", err);
        assert!(
            msg.contains("out of vocab"),
            "error should mention vocab range; got: {}",
            msg
        );
    });
}

#[test]
fn encode_with_specials_with_leading_bos_special() {
    // Demonstrates the contract: BOS isn't auto-prepended. The caller
    // does it explicitly with a Special part if they want it.
    use project_willamette::tokenizer::PromptPart;

    with_real_tokenizer(|tok| {
        let bos = tok.bos_id.expect("bos id present in real model");
        let ids = tok
            .encode_with_specials(&[PromptPart::Special(bos), PromptPart::Text("hello")])
            .expect("encode_with_specials");
        assert_eq!(ids.first().copied(), Some(bos), "first id should be BOS");
        let body = tok.encode("hello", EncodeOptions::none()).expect("hello");
        assert_eq!(&ids[1..], &body[..]);
    });
}
