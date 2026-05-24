//! Stage 5-A integration test — exercise the same pipeline the `run`
//! CLI uses: encode → single-token forward → logits → argmax → decode.

use std::path::Path;

use project_willamette::gguf::reader::GgufFile;
use project_willamette::memory::mmap::ModelMmap;
use project_willamette::model::generate::greedy_next_token_from_single_position_zero;
use project_willamette::model::ModelGraph;
use project_willamette::tokenizer::Tokenizer;

const MODEL_PATH: &str = "./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf";

#[test]
fn run_pipeline_predicts_in_range_token_id() {
    if !Path::new(MODEL_PATH).exists() {
        eprintln!("SKIP: real GGUF not found at {}", MODEL_PATH);
        return;
    }
    let mmap = ModelMmap::open(MODEL_PATH).expect("open");
    let bytes = mmap.as_bytes();
    let gguf = GgufFile::parse(bytes).expect("parse");
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata).expect("tokenizer");
    let graph = ModelGraph::from_gguf(&gguf).expect("graph");

    let opts = tokenizer.default_encode_options();
    let ids = tokenizer.encode("hello", opts).expect("encode");
    assert!(!ids.is_empty());
    let last = *ids.last().unwrap();

    let next = greedy_next_token_from_single_position_zero(&graph, last).expect("greedy");
    assert!(next < graph.config.vocab_size);

    // Decoding must not panic.
    let _ = tokenizer.decode(&[next]).expect("decode");
}

#[test]
fn run_pipeline_is_deterministic() {
    if !Path::new(MODEL_PATH).exists() {
        eprintln!("SKIP: real GGUF not found at {}", MODEL_PATH);
        return;
    }
    let mmap = ModelMmap::open(MODEL_PATH).expect("open");
    let bytes = mmap.as_bytes();
    let gguf = GgufFile::parse(bytes).expect("parse");
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata).expect("tokenizer");
    let graph = ModelGraph::from_gguf(&gguf).expect("graph");
    let opts = tokenizer.default_encode_options();
    let ids = tokenizer.encode("hello", opts).expect("encode");
    let last = *ids.last().unwrap();

    let a = greedy_next_token_from_single_position_zero(&graph, last).expect("a");
    let b = greedy_next_token_from_single_position_zero(&graph, last).expect("b");
    assert_eq!(a, b);
}
