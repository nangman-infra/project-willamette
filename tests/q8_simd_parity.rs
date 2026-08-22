//! External-model diagnostic for Q8_0 scalar/SIMD greedy stability.

use project_willamette::gguf::reader::GgufFile;
use project_willamette::memory::mmap::ModelMmap;
use project_willamette::model::cached_forward::{forward_with_cache_into, ForwardWorkspace};
use project_willamette::model::kv_cache::KVCache;
use project_willamette::model::lm_head::compute_logits_from_graph;
use project_willamette::model::ModelGraph;
use project_willamette::tokenizer::Tokenizer;
use sha2::{Digest, Sha256};

const MODEL_PATH: &str = "./models/SmolLM2-360M-Instruct-Q8_0.gguf";
const MODEL_SHA256: &str = "48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201";
const SYSTEM: &str = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
const PROMPT: &str = "Write one accurate paragraph explaining why the sky looks blue during the day and red near sunset. Mention Rayleigh scattering, wavelength, and the longer path through the atmosphere. Finish with a concise conclusion.";

#[derive(Debug)]
struct TraceStep {
    selected_id: u32,
    selected_logit: f32,
    runner_up_id: u32,
    runner_up_logit: f32,
}

impl TraceStep {
    fn margin(&self) -> f32 {
        self.selected_logit - self.runner_up_logit
    }
}

#[test]
#[ignore = "requires the pinned 386404992-byte SmolLM2-360M-Instruct-Q8_0.gguf"]
fn trace_smollm2_360m_q8_greedy_margins() {
    let mmap = ModelMmap::open(MODEL_PATH).expect("open model");
    assert_eq!(
        format!("{:x}", Sha256::digest(mmap.as_bytes())),
        MODEL_SHA256
    );
    let gguf = GgufFile::parse(mmap.as_bytes()).expect("parse model");
    let graph = ModelGraph::from_gguf(&gguf).expect("build graph");
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata).expect("load tokenizer");
    let (prompt, _) = tokenizer
        .encode_chatml_turn(Some(SYSTEM), PROMPT)
        .expect("tokenize ChatML prompt");

    let trace = greedy_trace(&graph, &prompt, 120);
    assert_first_step_envelope(&trace[0]);
    for (step, sample) in trace.iter().enumerate() {
        println!(
            "step={step} selected={} selected_logit={:.9} runner_up={} runner_up_logit={:.9} margin={:.9}",
            sample.selected_id,
            sample.selected_logit,
            sample.runner_up_id,
            sample.runner_up_logit,
            sample.margin()
        );
    }
    println!(
        "generated={:?}",
        trace
            .iter()
            .map(|step| step.selected_id)
            .collect::<Vec<_>>()
    );
}

fn assert_first_step_envelope(step: &TraceStep) {
    let mut candidate_ids = [step.selected_id, step.runner_up_id];
    candidate_ids.sort_unstable();
    assert_eq!(candidate_ids, [504, 30300]);
    assert!(
        (17.5..=19.0).contains(&step.selected_logit),
        "unexpected top logit: {step:?}"
    );
    assert!(
        (17.5..=19.0).contains(&step.runner_up_logit),
        "unexpected runner-up logit: {step:?}"
    );
    assert!(
        (0.0..=0.25).contains(&step.margin()),
        "unexpected first-step margin: {step:?}"
    );
}

fn greedy_trace(graph: &ModelGraph<'_>, prompt: &[u32], steps: usize) -> Vec<TraceStep> {
    let mut cache = KVCache::new(
        graph.layers.len(),
        graph.config.kv_dim as usize,
        prompt.len() + steps,
    );
    let mut workspace = ForwardWorkspace::new(graph);
    let mut hidden = Vec::new();
    for (position, &token) in prompt.iter().enumerate() {
        forward_with_cache_into(
            graph,
            &mut cache,
            &mut workspace,
            token,
            position as u32,
            &mut hidden,
        )
        .expect("prefill");
    }

    let mut trace = Vec::with_capacity(steps);
    for step in 0..steps {
        let logits = compute_logits_from_graph(&hidden, graph).expect("lm-head");
        let sample = top_two(&logits);
        let selected = sample.selected_id;
        trace.push(sample);
        if step + 1 < steps {
            let position = prompt.len() + step;
            forward_with_cache_into(
                graph,
                &mut cache,
                &mut workspace,
                selected,
                position as u32,
                &mut hidden,
            )
            .expect("decode");
        }
    }
    trace
}

fn top_two(logits: &[f32]) -> TraceStep {
    let mut best = (u32::MAX, f32::NEG_INFINITY);
    let mut second = (u32::MAX, f32::NEG_INFINITY);
    for (index, &logit) in logits.iter().enumerate() {
        assert!(logit.is_finite(), "non-finite logit at {index}");
        if logit > best.1 {
            second = best;
            best = (index as u32, logit);
        } else if logit > second.1 {
            second = (index as u32, logit);
        }
    }
    TraceStep {
        selected_id: best.0,
        selected_logit: best.1,
        runner_up_id: second.0,
        runner_up_logit: second.1,
    }
}
