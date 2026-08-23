#![allow(clippy::needless_range_loop)]
// Token-position + head-index indexing reads more naturally than
// iterator chains over multiple parallel arrays.

//! Stage 5-C — single-token forward with a `KVCache`.
//!
//! Designed to be called once per token, in position order
//! (`0, 1, 2, …`). For each layer it:
//!
//!   1. RMSNorms the hidden state with `attn_norm`.
//!   2. Computes Q/K/V via BitLinear matvecs.
//!   3. Applies NEOX RoPE at the supplied position.
//!   4. Appends the new (K, V) to the cache for this layer.
//!   5. Runs scaled-dot-product attention from this Q against the
//!      cached `[K, V]` window (positions `0..=position`).
//!   6. Applies `attn_sub_norm`, the output BitLinear, and the first
//!      residual.
//!   7. Runs the FFN half (Stage 4-D2) and the second residual.
//!
//! After all layers, applies `output_norm` and returns the final
//! hidden — the same shape Stage 4-D5 hands to `compute_logits`.
//!
//! Numerical equivalence with the no-cache path (`multi_token_forward`)
//! is verified by `tests/kv_cache_forward.rs`.

use crate::error::WillametteError;
use crate::model::architecture::ForwardVariant;
use crate::model::attention::{apply_rope_multi_head, softmax_inplace};
use crate::model::ffn::{elementwise_mul, relu_square, silu};
use crate::model::graph::{LayerWeights, ModelGraph};
use crate::model::kv_cache::KVCache;
use crate::model::linear::linear_batched_f32;
use crate::model::primitives::{
    attention_scale, embedding_gather, kv_head_for_q_head, rms_norm_f32, AttentionShape, RopeType,
};
use crate::model::q4_k::Q8KActivation;
use crate::model::stage_timing::time_stage;

/// Per-token constants pulled from `ModelConfig` — packaged so the
/// inner per-layer helper doesn't take a dozen scalar arguments.
struct LayerCtx {
    variant: ForwardVariant,
    n_embd: usize,
    kv_dim: usize,
    n_ff: usize,
    head_dim: usize,
    n_rot: usize,
    freq_base: f32,
    eps: f32,
    n_heads: usize,
    n_heads_kv: u32,
    shape: AttentionShape,
    scale: f32,
}

/// Reusable buffers for cached single-token forward passes.
///
/// Construct one workspace per generation/chat session and pass it to the
/// `_into` APIs to avoid rebuilding layer and attention scratch vectors for
/// every token. The compatibility wrappers below still allocate a temporary
/// workspace for callers that prefer the original `Result<Vec<f32>>` API.
pub struct ForwardWorkspace {
    hidden: Vec<f32>,
    scratch_k: Vec<f32>,
    scratch_v: Vec<f32>,
    x_norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    scores: Vec<f32>,
    sub_normed: Vec<f32>,
    wo_out: Vec<f32>,
    x_norm_ffn: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    fused: Vec<f32>,
    fused_norm: Vec<f32>,
    down: Vec<f32>,
    q8_k_activation: Q8KActivation,
}

/// Reusable buffers for layer-major cached prompt prefill.
///
/// Projection inputs and outputs are token-major across the complete chunk.
/// Dequantized layer-cache and attention-score scratch are shared through the
/// embedded single-token workspace.
pub struct PrefillWorkspace {
    token: ForwardWorkspace,
    hidden: Vec<f32>,
    x_norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    fused: Vec<f32>,
}

impl PrefillWorkspace {
    pub fn new(graph: &ModelGraph<'_>) -> Self {
        Self {
            token: ForwardWorkspace::new(graph),
            hidden: Vec::new(),
            x_norm: Vec::new(),
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            attn_out: Vec::new(),
            gate: Vec::new(),
            up: Vec::new(),
            fused: Vec::new(),
        }
    }

    fn prepare(&mut self, ctx: &LayerCtx, token_count: usize) -> Result<(), WillametteError> {
        let embd_len = checked_token_buffer_len(token_count, ctx.n_embd, "embedding")?;
        let kv_len = checked_token_buffer_len(token_count, ctx.kv_dim, "KV")?;
        let ffn_len = checked_token_buffer_len(token_count, ctx.n_ff, "FFN")?;
        self.hidden.resize(embd_len, 0.0);
        self.x_norm.resize(embd_len, 0.0);
        self.q.resize(embd_len, 0.0);
        self.k.resize(kv_len, 0.0);
        self.v.resize(kv_len, 0.0);
        self.attn_out.resize(embd_len, 0.0);
        self.gate.resize(ffn_len, 0.0);
        self.up.resize(ffn_len, 0.0);
        self.fused.resize(ffn_len, 0.0);
        self.token.prepare(ctx);
        Ok(())
    }
}

fn checked_token_buffer_len(
    token_count: usize,
    width: usize,
    name: &str,
) -> Result<usize, WillametteError> {
    token_count.checked_mul(width).ok_or_else(|| {
        WillametteError::GgufParse(format!(
            "prefill_with_cache: token-major {name} buffer size overflow"
        ))
    })
}

impl ForwardWorkspace {
    pub fn new(graph: &ModelGraph<'_>) -> Self {
        let n_embd = graph.config.embedding_length as usize;
        let kv_dim = graph.config.kv_dim as usize;
        let n_ff = graph.config.feed_forward_length as usize;
        Self {
            hidden: vec![0.0; n_embd],
            scratch_k: Vec::new(),
            scratch_v: Vec::new(),
            x_norm: vec![0.0; n_embd],
            q: vec![0.0; n_embd],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            attn_out: vec![0.0; n_embd],
            scores: Vec::new(),
            sub_normed: vec![0.0; n_embd],
            wo_out: vec![0.0; n_embd],
            x_norm_ffn: vec![0.0; n_embd],
            gate: vec![0.0; n_ff],
            up: vec![0.0; n_ff],
            fused: vec![0.0; n_ff],
            fused_norm: vec![0.0; n_ff],
            down: vec![0.0; n_embd],
            q8_k_activation: Q8KActivation::new(),
        }
    }

    fn prepare(&mut self, ctx: &LayerCtx) {
        self.hidden.resize(ctx.n_embd, 0.0);
        self.x_norm.resize(ctx.n_embd, 0.0);
        self.q.resize(ctx.n_embd, 0.0);
        self.k.resize(ctx.kv_dim, 0.0);
        self.v.resize(ctx.kv_dim, 0.0);
        self.attn_out.resize(ctx.n_embd, 0.0);
        self.sub_normed.resize(ctx.n_embd, 0.0);
        self.wo_out.resize(ctx.n_embd, 0.0);
        self.x_norm_ffn.resize(ctx.n_embd, 0.0);
        self.gate.resize(ctx.n_ff, 0.0);
        self.up.resize(ctx.n_ff, 0.0);
        self.fused.resize(ctx.n_ff, 0.0);
        self.fused_norm.resize(ctx.n_ff, 0.0);
        self.down.resize(ctx.n_embd, 0.0);
    }
}

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0_f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Single-token forward at the given position, reading/writing `cache`.
///
/// Returns the post-`output_norm` hidden state (length `n_embd`). The
/// `cache.position()` must equal `position` on entry — i.e. tokens must
/// be processed strictly in order.
pub fn forward_with_cache(
    graph: &ModelGraph<'_>,
    cache: &mut KVCache,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, WillametteError> {
    ensure_supported_variant(graph.forward_variant)?;
    let mut workspace = ForwardWorkspace::new(graph);
    let mut output = Vec::new();
    forward_with_cache_into(
        graph,
        cache,
        &mut workspace,
        token_id,
        position,
        &mut output,
    )?;
    Ok(output)
}

/// Allocation-reusing variant of [`forward_with_cache`].
///
/// On error, cache entries appended by this call are rolled back and `output`
/// is cleared. Workspace contents are unspecified but safe to reuse.
pub fn forward_with_cache_into(
    graph: &ModelGraph<'_>,
    cache: &mut KVCache,
    workspace: &mut ForwardWorkspace,
    token_id: u32,
    position: u32,
    output: &mut Vec<f32>,
) -> Result<(), WillametteError> {
    forward_with_cache_progress_into(graph, cache, workspace, token_id, position, output, |_| {})
}

/// Layer-major cached forward for a non-empty token chunk.
///
/// Unlike repeatedly calling [`forward_with_cache`], each transformer layer's
/// cache is dequantized once after all K/V entries for the chunk are appended.
/// The returned hidden state is for the final token in `token_ids`.
pub fn prefill_with_cache(
    graph: &ModelGraph<'_>,
    cache: &mut KVCache,
    token_ids: &[u32],
    start_position: u32,
) -> Result<Vec<f32>, WillametteError> {
    let mut workspace = PrefillWorkspace::new(graph);
    let mut output = Vec::new();
    prefill_with_cache_into(
        graph,
        cache,
        &mut workspace,
        token_ids,
        start_position,
        &mut output,
    )?;
    Ok(output)
}

/// Allocation-reusing variant of [`prefill_with_cache`].
///
/// All inputs and cache capacity are validated before mutation. Any later
/// failure rolls every layer back to its entry position and clears `output`.
pub fn prefill_with_cache_into(
    graph: &ModelGraph<'_>,
    cache: &mut KVCache,
    workspace: &mut PrefillWorkspace,
    token_ids: &[u32],
    start_position: u32,
    output: &mut Vec<f32>,
) -> Result<(), WillametteError> {
    prefill_with_cache_progress_into(
        graph,
        cache,
        workspace,
        token_ids,
        start_position,
        output,
        |_| {},
    )
}

/// Layer-progress variant of [`prefill_with_cache_into`].
///
/// `on_layer` runs after every transformer block for the complete chunk.
pub fn prefill_with_cache_progress_into<F: FnMut(u32)>(
    graph: &ModelGraph<'_>,
    cache: &mut KVCache,
    workspace: &mut PrefillWorkspace,
    token_ids: &[u32],
    start_position: u32,
    output: &mut Vec<f32>,
    mut on_layer: F,
) -> Result<(), WillametteError> {
    output.clear();
    let ctx = build_layer_ctx(graph)?;
    let prefix = validate_prefill_inputs(graph, cache, token_ids, start_position, &ctx)?;
    let checkpoint = cache.checkpoint();

    let result = (|| {
        workspace.prepare(&ctx, token_ids.len())?;
        for (token_idx, &token_id) in token_ids.iter().enumerate() {
            let hidden = token_slice_mut(&mut workspace.hidden, token_idx, ctx.n_embd);
            time_stage!("embedding", {
                embedding_gather(graph.token_embd, token_id, hidden)?;
            });
        }

        for layer in &graph.layers {
            for token_idx in 0..token_ids.len() {
                time_stage!("attn_norm", {
                    rms_norm_f32(
                        token_slice(&workspace.hidden, token_idx, ctx.n_embd),
                        &layer.attn_norm_f32,
                        ctx.eps,
                        token_slice_mut(&mut workspace.x_norm, token_idx, ctx.n_embd),
                    )?;
                });
            }
            time_stage!("matvec_qkv", {
                layer.project_qkv_batched(
                    &workspace.x_norm,
                    &mut workspace.q,
                    &mut workspace.k,
                    &mut workspace.v,
                )?;
            });
            for token_idx in 0..token_ids.len() {
                let position = start_position + token_idx as u32;
                let q = token_slice_mut(&mut workspace.q, token_idx, ctx.n_embd);
                let k = token_slice_mut(&mut workspace.k, token_idx, ctx.kv_dim);
                time_stage!("rope", {
                    apply_rope_multi_head(
                        q,
                        ctx.n_heads as u32,
                        ctx.head_dim,
                        ctx.n_rot,
                        position,
                        ctx.freq_base,
                        rope_type(ctx.variant),
                    )?;
                    apply_rope_multi_head(
                        k,
                        ctx.n_heads_kv,
                        ctx.head_dim,
                        ctx.n_rot,
                        position,
                        ctx.freq_base,
                        rope_type(ctx.variant),
                    )?;
                });
                time_stage!("kv_append", {
                    cache.append(
                        layer.index as usize,
                        k,
                        token_slice(&workspace.v, token_idx, ctx.kv_dim),
                    )?;
                });
            }

            time_stage!("kv_read_into", {
                cache.read_into(
                    layer.index as usize,
                    &mut workspace.token.scratch_k,
                    &mut workspace.token.scratch_v,
                )?;
            });
            for token_idx in 0..token_ids.len() {
                time_stage!("attn_softmax_v", {
                    scaled_dot_product_attention_into(
                        token_slice(&workspace.q, token_idx, ctx.n_embd),
                        &workspace.token.scratch_k,
                        &workspace.token.scratch_v,
                        prefix + token_idx + 1,
                        &ctx,
                        token_slice_mut(&mut workspace.attn_out, token_idx, ctx.n_embd),
                        &mut workspace.token.scores,
                    );
                });
                if ctx.variant == ForwardVariant::BitNetSubNorm {
                    time_stage!("attn_sub_norm", {
                        rms_norm_f32(
                            token_slice(&workspace.attn_out, token_idx, ctx.n_embd),
                            layer.attn_sub_norm_f32.as_deref().ok_or_else(|| {
                                WillametteError::GgufParse(
                                    "BitNet layer is missing attn_sub_norm".to_string(),
                                )
                            })?,
                            ctx.eps,
                            token_slice_mut(&mut workspace.x_norm, token_idx, ctx.n_embd),
                        )?;
                    });
                }
            }
            let attn_projection_input = match ctx.variant {
                ForwardVariant::BitNetSubNorm => &workspace.x_norm,
                ForwardVariant::VanillaLlama | ForwardVariant::Qwen2 => &workspace.attn_out,
            };
            time_stage!("matvec_attn_output", {
                linear_batched_f32(layer.attn_output, attn_projection_input, &mut workspace.q)?;
            });
            for token_idx in 0..token_ids.len() {
                time_stage!("residual_attn", {
                    let hidden = token_slice_mut(&mut workspace.hidden, token_idx, ctx.n_embd);
                    let projected = token_slice(&workspace.q, token_idx, ctx.n_embd);
                    for d in 0..ctx.n_embd {
                        hidden[d] += projected[d];
                    }
                });
                time_stage!("ffn_norm", {
                    rms_norm_f32(
                        token_slice(&workspace.hidden, token_idx, ctx.n_embd),
                        &layer.ffn_norm_f32,
                        ctx.eps,
                        token_slice_mut(&mut workspace.x_norm, token_idx, ctx.n_embd),
                    )?;
                });
            }
            time_stage!("matvec_ffn_gate_up", {
                linear_batched_f32(layer.ffn_gate, &workspace.x_norm, &mut workspace.gate)?;
                linear_batched_f32(layer.ffn_up, &workspace.x_norm, &mut workspace.up)?;
            });
            for token_idx in 0..token_ids.len() {
                time_stage!("ffn_relu2_emul", {
                    let gate = token_slice_mut(&mut workspace.gate, token_idx, ctx.n_ff);
                    match ctx.variant {
                        ForwardVariant::BitNetSubNorm => relu_square(gate),
                        ForwardVariant::VanillaLlama | ForwardVariant::Qwen2 => silu(gate),
                    }
                    elementwise_mul(
                        gate,
                        token_slice(&workspace.up, token_idx, ctx.n_ff),
                        token_slice_mut(&mut workspace.fused, token_idx, ctx.n_ff),
                    )?;
                });
                if ctx.variant == ForwardVariant::BitNetSubNorm {
                    time_stage!("ffn_sub_norm", {
                        rms_norm_f32(
                            token_slice(&workspace.fused, token_idx, ctx.n_ff),
                            layer.ffn_sub_norm_f32.as_deref().ok_or_else(|| {
                                WillametteError::GgufParse(
                                    "BitNet layer is missing ffn_sub_norm".to_string(),
                                )
                            })?,
                            ctx.eps,
                            token_slice_mut(&mut workspace.gate, token_idx, ctx.n_ff),
                        )?;
                    });
                }
            }
            let down_input = match ctx.variant {
                ForwardVariant::BitNetSubNorm => &workspace.gate,
                ForwardVariant::VanillaLlama | ForwardVariant::Qwen2 => &workspace.fused,
            };
            time_stage!("matvec_ffn_down", {
                linear_batched_f32(layer.ffn_down, down_input, &mut workspace.q)?;
            });
            for token_idx in 0..token_ids.len() {
                time_stage!("residual_ffn", {
                    let hidden = token_slice_mut(&mut workspace.hidden, token_idx, ctx.n_embd);
                    let down = token_slice(&workspace.q, token_idx, ctx.n_embd);
                    for d in 0..ctx.n_embd {
                        hidden[d] += down[d];
                    }
                });
                time_stage!("check_finite", {
                    check_finite_hidden(
                        token_slice(&workspace.hidden, token_idx, ctx.n_embd),
                        layer.index,
                    )?;
                });
            }
            on_layer(layer.index);
        }

        output.resize(ctx.n_embd, 0.0);
        let final_hidden = token_slice(&workspace.hidden, token_ids.len() - 1, ctx.n_embd);
        time_stage!("output_norm", {
            rms_norm_f32(final_hidden, &graph.output_norm_f32, ctx.eps, output)?;
        });
        check_finite_output(output)?;
        Ok(())
    })();
    if result.is_err() {
        cache.rollback(checkpoint);
        output.clear();
    }
    result
}

/// Same as [`forward_with_cache`] but calls `on_layer(layer_idx)`
/// after each transformer block finishes. Used by the TUI to update
/// the layer-progress indicator in the dashboard. The overhead is
/// one closure call per layer (≤ 30 calls/token for BitNet b1.58 2B)
/// — well below the matvec cost.
pub fn forward_with_cache_progress<F: FnMut(u32)>(
    graph: &ModelGraph<'_>,
    cache: &mut KVCache,
    token_id: u32,
    position: u32,
    on_layer: F,
) -> Result<Vec<f32>, WillametteError> {
    ensure_supported_variant(graph.forward_variant)?;
    let mut workspace = ForwardWorkspace::new(graph);
    let mut output = Vec::new();
    forward_with_cache_progress_into(
        graph,
        cache,
        &mut workspace,
        token_id,
        position,
        &mut output,
        on_layer,
    )?;
    Ok(output)
}

/// Allocation-reusing variant of [`forward_with_cache_progress`].
///
/// On error, cache entries appended by this call are rolled back and `output`
/// is cleared. Layer progress callbacks that already fired cannot be undone.
pub fn forward_with_cache_progress_into<F: FnMut(u32)>(
    graph: &ModelGraph<'_>,
    cache: &mut KVCache,
    workspace: &mut ForwardWorkspace,
    token_id: u32,
    position: u32,
    output: &mut Vec<f32>,
    mut on_layer: F,
) -> Result<(), WillametteError> {
    if let Err(error) = ensure_supported_variant(graph.forward_variant) {
        output.clear();
        return Err(error);
    }
    let ctx = build_layer_ctx(graph)?;

    validate_cache_inputs(graph, cache, token_id, position, ctx.kv_dim)?;
    let checkpoint = cache.checkpoint();
    output.clear();
    let result = (|| {
        workspace.prepare(&ctx);

        time_stage!("embedding", {
            embedding_gather(graph.token_embd, token_id, &mut workspace.hidden)?;
        });

        for layer in &graph.layers {
            forward_one_layer(layer, cache, workspace, &ctx, position)?;
            on_layer(layer.index);
        }

        output.resize(ctx.n_embd, 0.0);
        time_stage!("output_norm", {
            rms_norm_f32(&workspace.hidden, &graph.output_norm_f32, ctx.eps, output)?;
        });
        Ok(())
    })();
    if result.is_err() {
        cache.rollback(checkpoint);
        output.clear();
    }
    result
}

fn build_layer_ctx(graph: &ModelGraph<'_>) -> Result<LayerCtx, WillametteError> {
    ensure_supported_variant(graph.forward_variant)?;
    let cfg = &graph.config;
    Ok(LayerCtx {
        n_embd: cfg.embedding_length as usize,
        variant: graph.forward_variant,
        kv_dim: cfg.kv_dim as usize,
        n_ff: cfg.feed_forward_length as usize,
        head_dim: cfg.head_dim as usize,
        n_rot: cfg.rope_dimension_count as usize,
        freq_base: cfg.rope_freq_base,
        eps: cfg.layer_norm_rms_epsilon,
        n_heads: cfg.head_count as usize,
        n_heads_kv: cfg.head_count_kv,
        shape: AttentionShape::from_config(cfg.head_count, cfg.head_count_kv, cfg.head_dim)?,
        scale: attention_scale(cfg.head_dim as usize),
    })
}

fn ensure_supported_variant(variant: ForwardVariant) -> Result<(), WillametteError> {
    match variant {
        ForwardVariant::BitNetSubNorm | ForwardVariant::VanillaLlama | ForwardVariant::Qwen2 => {
            Ok(())
        }
    }
}

fn validate_cache_inputs(
    graph: &ModelGraph<'_>,
    cache: &KVCache,
    token_id: u32,
    position: u32,
    kv_dim: usize,
) -> Result<(), WillametteError> {
    if cache.kv_dim != kv_dim {
        return Err(WillametteError::GgufParse(format!(
            "forward_with_cache: cache.kv_dim={} != model kv_dim={}",
            cache.kv_dim, kv_dim
        )));
    }
    if cache.n_layers != graph.layers.len() {
        return Err(WillametteError::GgufParse(format!(
            "forward_with_cache: cache.n_layers={} != model layers={}",
            cache.n_layers,
            graph.layers.len()
        )));
    }
    if cache.position() as u32 != position {
        return Err(WillametteError::GgufParse(format!(
            "forward_with_cache: cache.position()={} != position={}; tokens must be processed in order",
            cache.position(),
            position
        )));
    }
    if token_id >= graph.config.vocab_size {
        return Err(WillametteError::GgufParse(format!(
            "forward_with_cache: token_id {} out of vocab range {}",
            token_id, graph.config.vocab_size
        )));
    }
    Ok(())
}

fn validate_prefill_inputs(
    graph: &ModelGraph<'_>,
    cache: &KVCache,
    token_ids: &[u32],
    start_position: u32,
    ctx: &LayerCtx,
) -> Result<usize, WillametteError> {
    if token_ids.is_empty() {
        return Err(WillametteError::GgufParse(
            "prefill_with_cache: token chunk must not be empty".to_string(),
        ));
    }
    let prefix = cache.validate_append(graph.layers.len(), ctx.kv_dim, token_ids.len())?;
    if prefix != start_position as usize {
        return Err(WillametteError::GgufParse(format!(
            "prefill_with_cache: cache.position()={} != start_position={}",
            prefix, start_position
        )));
    }
    let last_offset = u32::try_from(token_ids.len() - 1).map_err(|_| {
        WillametteError::GgufParse("prefill_with_cache: token chunk is too large".to_string())
    })?;
    start_position.checked_add(last_offset).ok_or_else(|| {
        WillametteError::GgufParse("prefill_with_cache: position overflows u32".to_string())
    })?;
    for &token_id in token_ids {
        if token_id >= graph.config.vocab_size {
            return Err(WillametteError::GgufParse(format!(
                "prefill_with_cache: token_id {} out of vocab range {}",
                token_id, graph.config.vocab_size
            )));
        }
    }
    if graph.output_norm_f32.len() != ctx.n_embd {
        return Err(WillametteError::GgufParse(format!(
            "prefill_with_cache: output norm length {} != embedding length {}",
            graph.output_norm_f32.len(),
            ctx.n_embd
        )));
    }
    for (expected_idx, layer) in graph.layers.iter().enumerate() {
        if layer.index as usize != expected_idx {
            return Err(WillametteError::GgufParse(format!(
                "prefill_with_cache: layer index {} != expected {}",
                layer.index, expected_idx
            )));
        }
        if layer.attn_norm_f32.len() != ctx.n_embd || layer.ffn_norm_f32.len() != ctx.n_embd {
            return Err(WillametteError::GgufParse(format!(
                "prefill_with_cache: layer {} norm dimensions are invalid",
                layer.index
            )));
        }
        if ctx.variant == ForwardVariant::BitNetSubNorm
            && (layer.attn_sub_norm_f32.as_deref().map_or(0, <[f32]>::len) != ctx.n_embd
                || layer.ffn_sub_norm_f32.as_deref().map_or(0, <[f32]>::len) != ctx.n_ff)
        {
            return Err(WillametteError::GgufParse(format!(
                "prefill_with_cache: BitNet layer {} sub-norm dimensions are invalid",
                layer.index
            )));
        }
    }
    Ok(prefix)
}

fn token_slice(values: &[f32], token_idx: usize, width: usize) -> &[f32] {
    &values[token_idx * width..(token_idx + 1) * width]
}

fn token_slice_mut(values: &mut [f32], token_idx: usize, width: usize) -> &mut [f32] {
    &mut values[token_idx * width..(token_idx + 1) * width]
}

#[allow(clippy::too_many_arguments)]
fn forward_one_layer(
    layer: &LayerWeights<'_>,
    cache: &mut KVCache,
    workspace: &mut ForwardWorkspace,
    ctx: &LayerCtx,
    position: u32,
) -> Result<(), WillametteError> {
    project_qkv_and_append(layer, cache, workspace, ctx, position)?;
    let layer_idx = layer.index as usize;
    time_stage!("kv_read_into", {
        cache.read_into(
            layer_idx,
            &mut workspace.scratch_k,
            &mut workspace.scratch_v,
        )?;
    });
    let n_past = workspace.scratch_k.len() / ctx.kv_dim;
    finish_one_layer(layer, workspace, ctx, n_past)
}

fn project_qkv_and_append(
    layer: &LayerWeights<'_>,
    cache: &mut KVCache,
    workspace: &mut ForwardWorkspace,
    ctx: &LayerCtx,
    position: u32,
) -> Result<(), WillametteError> {
    time_stage!("attn_norm", {
        rms_norm_f32(
            &workspace.hidden,
            &layer.attn_norm_f32,
            ctx.eps,
            &mut workspace.x_norm,
        )?;
    });

    time_stage!("matvec_qkv", {
        layer.project_qkv_graph_validated(
            &workspace.x_norm,
            &mut workspace.q,
            &mut workspace.k,
            &mut workspace.v,
            &mut workspace.q8_k_activation,
        )?;
    });

    time_stage!("rope", {
        apply_rope_multi_head(
            &mut workspace.q,
            ctx.n_heads as u32,
            ctx.head_dim,
            ctx.n_rot,
            position,
            ctx.freq_base,
            rope_type(ctx.variant),
        )?;
        apply_rope_multi_head(
            &mut workspace.k,
            ctx.n_heads_kv,
            ctx.head_dim,
            ctx.n_rot,
            position,
            ctx.freq_base,
            rope_type(ctx.variant),
        )?;
    });

    let layer_idx = layer.index as usize;
    time_stage!("kv_append", {
        cache.append(layer_idx, &workspace.k, &workspace.v)?;
    });
    Ok(())
}

fn finish_one_layer(
    layer: &LayerWeights<'_>,
    workspace: &mut ForwardWorkspace,
    ctx: &LayerCtx,
    n_past: usize,
) -> Result<(), WillametteError> {
    time_stage!("attn_softmax_v", {
        scaled_dot_product_attention_into(
            &workspace.q,
            &workspace.scratch_k,
            &workspace.scratch_v,
            n_past,
            ctx,
            &mut workspace.attn_out,
            &mut workspace.scores,
        )
    });

    let attn_projection_input = match ctx.variant {
        ForwardVariant::BitNetSubNorm => {
            time_stage!("attn_sub_norm", {
                rms_norm_f32(
                    &workspace.attn_out,
                    layer.attn_sub_norm_f32.as_deref().ok_or_else(|| {
                        WillametteError::GgufParse(
                            "BitNet layer is missing attn_sub_norm".to_string(),
                        )
                    })?,
                    ctx.eps,
                    &mut workspace.sub_normed,
                )?;
            });
            &workspace.sub_normed
        }
        ForwardVariant::VanillaLlama | ForwardVariant::Qwen2 => &workspace.attn_out,
    };
    time_stage!("matvec_attn_output", {
        layer.project_attn_output(
            attn_projection_input,
            &mut workspace.wo_out,
            &mut workspace.q8_k_activation,
        )?;
    });
    time_stage!("residual_attn", {
        for d in 0..ctx.n_embd {
            workspace.hidden[d] += workspace.wo_out[d];
        }
    });

    apply_ffn_block(layer, workspace, ctx)?;
    time_stage!("check_finite", {
        check_finite_hidden(&workspace.hidden, layer.index)?;
    });
    Ok(())
}

fn scaled_dot_product_attention_into(
    q: &[f32],
    cached_k: &[f32],
    cached_v: &[f32],
    n_past: usize,
    ctx: &LayerCtx,
    attn_out: &mut [f32],
    scores: &mut Vec<f32>,
) {
    attn_out.fill(0.0);
    for h in 0..ctx.n_heads {
        let kv_h = kv_head_for_q_head(h as u32, ctx.shape.group_size) as usize;
        let q_h = &q[h * ctx.head_dim..(h + 1) * ctx.head_dim];
        scores.clear();
        scores.reserve(n_past);
        for p in 0..n_past {
            let base = p * ctx.kv_dim + kv_h * ctx.head_dim;
            let k_h = &cached_k[base..base + ctx.head_dim];
            scores.push(dot_f32(q_h, k_h) * ctx.scale);
        }
        softmax_inplace(scores);

        let out_h = &mut attn_out[h * ctx.head_dim..(h + 1) * ctx.head_dim];
        for p in 0..n_past {
            let base = p * ctx.kv_dim + kv_h * ctx.head_dim;
            let v_h = &cached_v[base..base + ctx.head_dim];
            let w = scores[p];
            for d in 0..ctx.head_dim {
                out_h[d] += w * v_h[d];
            }
        }
    }
}

fn apply_ffn_block(
    layer: &LayerWeights<'_>,
    workspace: &mut ForwardWorkspace,
    ctx: &LayerCtx,
) -> Result<(), WillametteError> {
    time_stage!("ffn_norm", {
        rms_norm_f32(
            &workspace.hidden,
            &layer.ffn_norm_f32,
            ctx.eps,
            &mut workspace.x_norm_ffn,
        )?;
    });
    time_stage!("matvec_ffn_gate_up", {
        layer.project_ffn_gate_up(
            &workspace.x_norm_ffn,
            &mut workspace.gate,
            &mut workspace.up,
            &mut workspace.q8_k_activation,
        )?;
    });
    time_stage!("ffn_relu2_emul", {
        match ctx.variant {
            ForwardVariant::BitNetSubNorm => relu_square(&mut workspace.gate),
            ForwardVariant::VanillaLlama | ForwardVariant::Qwen2 => silu(&mut workspace.gate),
        }
        elementwise_mul(&workspace.gate, &workspace.up, &mut workspace.fused)?;
    });
    let down_input = match ctx.variant {
        ForwardVariant::BitNetSubNorm => {
            time_stage!("ffn_sub_norm", {
                rms_norm_f32(
                    &workspace.fused,
                    layer.ffn_sub_norm_f32.as_deref().ok_or_else(|| {
                        WillametteError::GgufParse(
                            "BitNet layer is missing ffn_sub_norm".to_string(),
                        )
                    })?,
                    ctx.eps,
                    &mut workspace.fused_norm,
                )?;
            });
            &workspace.fused_norm
        }
        ForwardVariant::VanillaLlama | ForwardVariant::Qwen2 => &workspace.fused,
    };
    time_stage!("matvec_ffn_down", {
        layer.project_ffn_down(
            down_input,
            &mut workspace.down,
            &mut workspace.q8_k_activation,
        )?;
    });
    time_stage!("residual_ffn", {
        for d in 0..ctx.n_embd {
            workspace.hidden[d] += workspace.down[d];
        }
    });
    Ok(())
}

fn rope_type(variant: ForwardVariant) -> RopeType {
    match variant {
        ForwardVariant::BitNetSubNorm | ForwardVariant::Qwen2 => RopeType::Neox,
        ForwardVariant::VanillaLlama => RopeType::Norm,
    }
}

fn check_finite_hidden(hidden: &[f32], layer_idx: u32) -> Result<(), WillametteError> {
    for v in hidden {
        if !v.is_finite() {
            return Err(WillametteError::GgufParse(format!(
                "forward_with_cache: non-finite hidden after layer {}",
                layer_idx
            )));
        }
    }
    Ok(())
}

fn check_finite_output(output: &[f32]) -> Result<(), WillametteError> {
    if output.iter().any(|value| !value.is_finite()) {
        return Err(WillametteError::GgufParse(
            "prefill_with_cache: non-finite final hidden".to_string(),
        ));
    }
    Ok(())
}
