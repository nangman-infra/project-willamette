//! Pinned classic-Llama F16 acceptance artifact.

use std::path::Path;

use project_willamette::gguf::reader::GgufFile;
use project_willamette::memory::mmap::ModelMmap;
use project_willamette::model::cached_forward::forward_with_cache;
use project_willamette::model::forward::forward_single_token_position_zero;
use project_willamette::model::generate::generate_with_cache_and_sampler;
use project_willamette::model::kv_cache::KVCache;
use project_willamette::model::multi_forward::multi_token_forward;
use project_willamette::model::sampler::{Sampler, SamplingParams};
use project_willamette::model::ModelGraph;
use project_willamette::tokenizer::Tokenizer;
use sha2::{Digest, Sha256};

const MODEL_PATH: &str = "./models/stories260K.F16.gguf";
const MODEL_SHA256: &str = "57a81ed1c8b032ba29319eae80c3e568dbb5a16ce665a09da1a0efe2e4eb69e3";
const MODEL_15M_PATH: &str = "./models/stories15M.F16.gguf";
const MODEL_15M_SHA256: &str = "35111216f325b8feb5b895095cfc7df1b6652368cb4893e9004a25825f517f54";
const SMOLLM_PATH: &str = "./models/SmolLM-135M-Instruct-f16.gguf";
const SMOLLM_SHA256: &str = "1fc02c21fba7874b15955d21dc59182aeae382abea412419ffd2fbaa84861790";
const SMOLLM_Q4_PATH: &str = "./models/SmolLM-135M-Instruct-Q4_0.gguf";
const SMOLLM_Q4_SHA256: &str = "a637fd6dcfd1b1333779ce2db780996cf4ed2a64aa0f9f6be0bb46689eb232a1";
const SMOLLM_Q8_PATH: &str = "./models/SmolLM-135M-Instruct-Q8_0.gguf";
const SMOLLM_Q8_SHA256: &str = "76520babb0daebccb6e17d2f38504ece61356a0ca958d8e8795ef4d23c23c1f0";
const SMOLLM2_360M_Q8_PATH: &str = "./models/SmolLM2-360M-Instruct-Q8_0.gguf";
const SMOLLM2_360M_Q8_SHA256: &str =
    "48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201";

#[test]
#[ignore = "requires the pinned 601248-byte stories260K.F16.gguf; see REPRODUCIBILITY.md"]
fn pinned_llama_f16_matches_llama_cpp_golden() {
    assert!(Path::new(MODEL_PATH).is_file(), "missing {MODEL_PATH}");
    let mmap = ModelMmap::open(MODEL_PATH).expect("open model");
    assert_eq!(
        format!("{:x}", Sha256::digest(mmap.as_bytes())),
        MODEL_SHA256
    );

    let gguf = GgufFile::parse(mmap.as_bytes()).expect("parse model");
    let graph = ModelGraph::from_gguf(&gguf).expect("build graph");
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata).expect("load tokenizer");

    assert_eq!(graph.config.architecture, "llama");
    assert_eq!(graph.config.block_count, 5);
    assert_eq!(graph.config.embedding_length, 64);
    assert_eq!(graph.config.feed_forward_length, 172);
    assert_eq!(graph.config.head_count, 8);
    assert_eq!(graph.config.head_count_kv, 4);
    assert_eq!(graph.config.rope_freq_base, 10_000.0);
    assert_eq!(graph.config.vocab_size, 512);
    assert!(!graph.lm_head_is_tied());
    assert!(graph
        .layers
        .iter()
        .all(|layer| layer.attn_sub_norm.is_none() && layer.ffn_sub_norm.is_none()));

    let prompt = tokenizer
        .encode("One day", tokenizer.default_encode_options())
        .expect("tokenize");
    assert_eq!(prompt, [1, 385, 328]);
    assert_eq!(tokenizer.decode(&prompt).unwrap(), "One day");

    let position_zero = forward_single_token_position_zero(&graph, 1).expect("single forward");
    let no_cache = multi_token_forward(&graph, &[1]).expect("multi forward");
    assert!(cosine(&position_zero, &no_cache) > 0.999_999);

    let mut cache = KVCache::new(graph.layers.len(), graph.config.kv_dim as usize, 4);
    let cached = forward_with_cache(&graph, &mut cache, 1, 0).expect("cached forward");
    assert!(cosine(&position_zero, &cached) > 0.999);

    let mut sampler = Sampler::new(SamplingParams::greedy());
    let generated = generate_with_cache_and_sampler(
        &graph,
        &prompt,
        5,
        tokenizer.eos_id,
        &[],
        graph.config.context_length as usize,
        &mut sampler,
        |_, _, _| {},
    )
    .expect("generate");
    assert_eq!(generated, [432, 261, 376, 298, 315]);
    assert_eq!(
        tokenizer.decode_lossy(&generated).unwrap(),
        ", a little gir"
    );
}

#[test]
#[ignore = "requires the pinned 49550112-byte stories15M.F16.gguf; see REPRODUCIBILITY.md"]
fn pinned_llama_15m_matches_llama_cpp_golden() {
    let mmap = ModelMmap::open(MODEL_15M_PATH).expect("open model");
    assert_eq!(
        format!("{:x}", Sha256::digest(mmap.as_bytes())),
        MODEL_15M_SHA256
    );
    let gguf = GgufFile::parse(mmap.as_bytes()).expect("parse model");
    let graph = ModelGraph::from_gguf(&gguf).expect("build graph");
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata).expect("load tokenizer");
    let prompt = tokenizer
        .encode("One day, Timmy went to", tokenizer.default_encode_options())
        .expect("tokenize");
    assert_eq!(prompt, [1, 3118, 2462, 29892, 7870, 1357, 3512, 304]);
    assert_eq!(tokenizer.decode(&prompt).unwrap(), "One day, Timmy went to");

    let mut sampler = Sampler::new(SamplingParams::greedy());
    let generated = generate_with_cache_and_sampler(
        &graph,
        &prompt,
        10,
        tokenizer.eos_id,
        &[],
        graph.config.context_length as usize,
        &mut sampler,
        |_, _, _| {},
    )
    .expect("generate");
    assert_eq!(
        generated,
        [278, 14089, 411, 670, 16823, 29889, 940, 4446, 263, 4802]
    );
    assert_eq!(
        tokenizer.decode_lossy(&generated).unwrap(),
        " the park with his mom. He saw a big"
    );
}

#[test]
#[ignore = "requires the pinned 270885792-byte SmolLM-135M-Instruct-f16.gguf; see REPRODUCIBILITY.md"]
fn pinned_smollm_135m_instruct_matches_llama_cpp_golden() {
    let mmap = ModelMmap::open(SMOLLM_PATH).expect("open model");
    assert_eq!(
        format!("{:x}", Sha256::digest(mmap.as_bytes())),
        SMOLLM_SHA256
    );
    let gguf = GgufFile::parse(mmap.as_bytes()).expect("parse model");
    let graph = ModelGraph::from_gguf(&gguf).expect("build graph");
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata).expect("load tokenizer");

    assert_eq!(graph.config.architecture, "llama");
    assert_eq!(graph.config.block_count, 30);
    assert_eq!(graph.config.embedding_length, 576);
    assert_eq!(graph.config.feed_forward_length, 1536);
    assert_eq!(graph.config.head_count, 9);
    assert_eq!(graph.config.head_count_kv, 3);
    assert_eq!(graph.config.vocab_size, 49_152);
    assert!(graph.lm_head_is_tied());

    let tokenizer_golden = tokenizer
        .encode(
            "Question: What is 84 * 3 / 2?",
            tokenizer.default_encode_options(),
        )
        .expect("tokenize");
    assert_eq!(
        tokenizer_golden,
        [17872, 42, 1812, 314, 216, 40, 36, 1672, 216, 35, 2272, 216, 34, 47]
    );

    let prompt = tokenizer
        .encode(
            "Question: What is 2 + 2? Answer:",
            tokenizer.default_encode_options(),
        )
        .expect("tokenize prompt");
    assert_eq!(
        prompt,
        [17872, 42, 1812, 314, 216, 34, 1232, 216, 34, 47, 19842, 42]
    );
    let mut sampler = Sampler::new(SamplingParams::greedy());
    let generated = generate_with_cache_and_sampler(
        &graph,
        &prompt,
        20,
        tokenizer.eos_id,
        &[],
        graph.config.context_length as usize,
        &mut sampler,
        |_, _, _| {},
    )
    .expect("generate");
    assert_eq!(generated, [216, 36]);
    assert_eq!(tokenizer.decode_lossy(&generated).unwrap(), " 4");
}

#[test]
#[ignore = "requires the pinned 144810912-byte SmolLM-135M-Instruct-Q8_0.gguf; see REPRODUCIBILITY.md"]
fn pinned_smollm_q8_0_matches_llama_cpp_golden() {
    let mmap = ModelMmap::open(SMOLLM_Q8_PATH).expect("open model");
    assert_eq!(
        format!("{:x}", Sha256::digest(mmap.as_bytes())),
        SMOLLM_Q8_SHA256
    );
    let gguf = GgufFile::parse(mmap.as_bytes()).expect("parse model");
    let graph = ModelGraph::from_gguf(&gguf).expect("build graph");
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata).expect("load tokenizer");
    assert_eq!(
        graph.token_embd.ggml_type,
        project_willamette::gguf::types::GgmlType::Q8_0
    );
    assert!(graph
        .layers
        .iter()
        .all(|layer| layer.attn_q.ggml_type == project_willamette::gguf::types::GgmlType::Q8_0));

    let prompt = tokenizer
        .encode(
            "Question: What is 2 + 2? Answer:",
            tokenizer.default_encode_options(),
        )
        .expect("tokenize prompt");
    let mut sampler = Sampler::new(SamplingParams::greedy());
    let generated = generate_with_cache_and_sampler(
        &graph,
        &prompt,
        20,
        tokenizer.eos_id,
        &[],
        graph.config.context_length as usize,
        &mut sampler,
        |_, _, _| {},
    )
    .expect("generate");
    assert_eq!(generated, [216, 36]);
    assert_eq!(tokenizer.decode_lossy(&generated).unwrap(), " 4");
}

#[test]
#[ignore = "requires the pinned 386404992-byte SmolLM2-360M-Instruct-Q8_0.gguf; see REPRODUCIBILITY.md"]
fn pinned_smollm2_360m_q8_0_matches_llama_cpp_golden() {
    let mmap = ModelMmap::open(SMOLLM2_360M_Q8_PATH).expect("open model");
    assert_eq!(
        format!("{:x}", Sha256::digest(mmap.as_bytes())),
        SMOLLM2_360M_Q8_SHA256
    );
    let gguf = GgufFile::parse(mmap.as_bytes()).expect("parse model");
    let graph = ModelGraph::from_gguf(&gguf).expect("build graph");
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata).expect("load tokenizer");
    assert_eq!(graph.config.block_count, 32);
    assert_eq!(graph.config.embedding_length, 960);
    assert_eq!(graph.config.context_length, 8192);

    let system = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
    let prompt_text = "What is the capital of France? Answer in one sentence.";
    let (prompt, stop_id) = tokenizer
        .encode_chatml_turn(Some(system), prompt_text)
        .expect("tokenize ChatML prompt");
    assert_eq!(
        prompt,
        [
            1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018, 411,
            407, 19712, 8182, 2, 198, 1, 4093, 198, 1780, 314, 260, 3575, 282, 4649, 47, 19842,
            281, 582, 6330, 30, 2, 198, 1, 520, 9531, 198,
        ]
    );
    let mut sampler = Sampler::new(SamplingParams::greedy());
    let generated = generate_with_cache_and_sampler(
        &graph,
        &prompt,
        30,
        tokenizer.eos_id,
        &[stop_id],
        graph.config.context_length as usize,
        &mut sampler,
        |_, _, _| {},
    )
    .expect("generate");
    assert_eq!(generated, [504, 3575, 282, 4649, 314, 7042, 30]);
    assert_eq!(
        tokenizer.decode_lossy(&generated).unwrap(),
        "The capital of France is Paris."
    );
}

#[test]
#[ignore = "requires the pinned 91726752-byte SmolLM-135M-Instruct-Q4_0.gguf; see REPRODUCIBILITY.md"]
fn pinned_smollm_q4_0_matches_llama_cpp_golden() {
    let mmap = ModelMmap::open(SMOLLM_Q4_PATH).expect("open model");
    assert_eq!(
        format!("{:x}", Sha256::digest(mmap.as_bytes())),
        SMOLLM_Q4_SHA256
    );
    let gguf = GgufFile::parse(mmap.as_bytes()).expect("parse model");
    let graph = ModelGraph::from_gguf(&gguf).expect("build graph");
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata).expect("load tokenizer");

    // This upstream quant keeps the tied embedding/lm-head at Q8_0 while all
    // transformer linears use Q4_0.
    assert_eq!(
        graph.token_embd.ggml_type,
        project_willamette::gguf::types::GgmlType::Q8_0
    );
    assert!(graph.layers.iter().all(|layer| {
        [
            &layer.attn_q,
            &layer.attn_k,
            &layer.attn_v,
            &layer.attn_output,
            &layer.ffn_gate,
            &layer.ffn_up,
            &layer.ffn_down,
        ]
        .into_iter()
        .all(|tensor| tensor.ggml_type == project_willamette::gguf::types::GgmlType::Q4_0)
    }));

    let prompt = tokenizer
        .encode(
            "Question: What is 2 + 2? Answer:",
            tokenizer.default_encode_options(),
        )
        .expect("tokenize prompt");
    let mut sampler = Sampler::new(SamplingParams::greedy());
    let generated = generate_with_cache_and_sampler(
        &graph,
        &prompt,
        20,
        tokenizer.eos_id,
        &[],
        graph.config.context_length as usize,
        &mut sampler,
        |_, _, _| {},
    )
    .expect("generate");
    assert_eq!(generated, [216, 36]);
    assert_eq!(tokenizer.decode_lossy(&generated).unwrap(), " 4");
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let aa = a.iter().map(|x| x * x).sum::<f32>();
    let bb = b.iter().map(|x| x * x).sum::<f32>();
    dot / (aa.sqrt() * bb.sqrt())
}
