//! Qwen2 architecture acceptance tests.

use std::path::Path;

use project_willamette::gguf::reader::GgufFile;
use project_willamette::gguf::types::GgmlType;
use project_willamette::memory::mmap::ModelMmap;
use project_willamette::model::architecture::ForwardVariant;
use project_willamette::model::generate::generate_with_cache_and_sampler;
use project_willamette::model::sampler::{Sampler, SamplingParams};
use project_willamette::model::ModelGraph;
use project_willamette::tokenizer::{EncodeOptions, Tokenizer};
use sha2::{Digest, Sha256};

const MODEL_PATH: &str = "./models/Qwen2.5-3B-Instruct-Q4_K_M.gguf";
const MODEL_SHA256: &str = "626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d";

#[test]
fn official_qwen2_5_3b_graph_loads_when_available() {
    if !Path::new(MODEL_PATH).is_file() {
        return;
    }

    let mmap = ModelMmap::open(MODEL_PATH).expect("open Qwen2.5 model");
    let gguf = GgufFile::parse(mmap.as_bytes()).expect("parse Qwen2.5 model");
    let graph = ModelGraph::from_gguf(&gguf).expect("build Qwen2.5 graph");

    assert_eq!(graph.config.architecture, "qwen2");
    assert_eq!(graph.forward_variant, ForwardVariant::Qwen2);
    assert_eq!(graph.config.block_count, 36);
    assert_eq!(graph.config.embedding_length, 2048);
    assert_eq!(graph.config.head_count, 16);
    assert_eq!(graph.config.head_count_kv, 2);
    assert_eq!(graph.config.kv_dim, 256);
    assert_eq!(graph.config.vocab_size, 151_936);
    assert_eq!(graph.layers.len(), 36);
    assert!(graph.layers.iter().all(|layer| {
        layer.attn_q_bias.is_some()
            && layer.attn_k_bias.is_some()
            && layer.attn_v_bias.is_some()
            && layer.attn_q_bias.unwrap().ggml_type == GgmlType::F32
            && layer.attn_k_bias.unwrap().ggml_type == GgmlType::F32
            && layer.attn_v_bias.unwrap().ggml_type == GgmlType::F32
            && layer
                .attn_q_bias_f32
                .as_ref()
                .is_some_and(|bias| bias.len() == 2048)
            && layer
                .attn_k_bias_f32
                .as_ref()
                .is_some_and(|bias| bias.len() == 256)
            && layer
                .attn_v_bias_f32
                .as_ref()
                .is_some_and(|bias| bias.len() == 256)
    }));

    let layer = &graph.layers[0];
    let input = vec![0.0; graph.config.embedding_length as usize];
    let mut q = vec![0.0; graph.config.embedding_length as usize];
    let mut k = vec![0.0; graph.config.kv_dim as usize];
    let mut v = vec![0.0; graph.config.kv_dim as usize];
    layer
        .project_qkv(&input, &mut q, &mut k, &mut v)
        .expect("project biased QKV");
    assert_eq!(q.as_slice(), layer.attn_q_bias_f32.as_deref().unwrap());
    assert_eq!(k.as_slice(), layer.attn_k_bias_f32.as_deref().unwrap());
    assert_eq!(v.as_slice(), layer.attn_v_bias_f32.as_deref().unwrap());
}

#[test]
fn official_qwen2_5_tokenizer_matches_llama_cpp_when_available() {
    if !Path::new(MODEL_PATH).is_file() {
        return;
    }

    let mmap = ModelMmap::open(MODEL_PATH).expect("open Qwen2.5 model");
    let gguf = GgufFile::parse(mmap.as_bytes()).expect("parse Qwen2.5 model");
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata).expect("load Qwen2 tokenizer");
    let ids = tokenizer
        .encode(
            "안녕하세요 123, WORLD'S!\n펌프 P-204",
            EncodeOptions::none(),
        )
        .expect("tokenize mixed Qwen2 text");

    assert_eq!(
        ids,
        [
            126246, 144370, 91145, 220, 16, 17, 18, 11, 50891, 13272, 4894, 144732, 126445, 393,
            12, 17, 15, 19,
        ]
    );
}

#[test]
#[ignore = "requires the pinned 2104932768-byte Qwen2.5-3B-Instruct-Q4_K_M.gguf"]
fn pinned_qwen2_5_korean_report_matches_llama_cpp() {
    let mmap = ModelMmap::open(MODEL_PATH).expect("open Qwen2.5 model");
    assert_eq!(
        format!("{:x}", Sha256::digest(mmap.as_bytes())),
        MODEL_SHA256
    );
    let gguf = GgufFile::parse(mmap.as_bytes()).expect("parse Qwen2.5 model");
    let graph = ModelGraph::from_gguf(&gguf).expect("build Qwen2.5 graph");
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata).expect("load Qwen2 tokenizer");
    let prompt_text = "다음 정비 메모를 정확히 6줄의 보고서로 변환하세요. 각 줄은 지정된 필드명과 콜론으로 시작하고, 메모에 없는 내용은 추가하지 마세요. 필드 순서: 설비, 시각, 증상, 조치, 작업시간, 결과. 메모: 펌프 P-204. 14:20에 베어링 소음 증가 확인. 전원을 차단하고 체결 볼트를 조였다. 작업시간은 20분. 시험 운전 후 소음이 사라졌다.";
    let (prompt, stop_id) = tokenizer
        .encode_chatml_turn(
            Some("You are a concise and accurate local assistant."),
            prompt_text,
        )
        .expect("tokenize ChatML prompt");
    let mut sampler = Sampler::new(SamplingParams::greedy());
    let generated = generate_with_cache_and_sampler(
        &graph,
        &prompt,
        100,
        tokenizer.eos_id,
        &[stop_id],
        graph.config.context_length as usize,
        &mut sampler,
        |_, _, _| {},
    )
    .expect("generate Korean report");

    assert_eq!(
        tokenizer.decode_lossy(&generated).unwrap(),
        "설비: 펌프 P-204\n시각: 14:20\n증상: 베어링 소음 증가\n조치: 전원 차단 및 체결 볼트 조임\n작업시간: 20분\n결과: 시험 운전 후 소음 사라짐"
    );
}
