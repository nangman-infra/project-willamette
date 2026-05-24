//! Source-pinned tensor registry for a BitNet b1.58 model.
//!
//! Stage 4-A only — every field is a borrow into a parsed
//! [`crate::gguf::reader::GgufFile`]. No tensor data is copied, no dequant
//! happens, no forward kernels run.
//!
//! The shape and dtype rules enforced here are cited in
//! [`docs/BITNET_FORWARD_PLAN.md`](../../docs/BITNET_FORWARD_PLAN.md) §4 and
//! §5 against `src/llama.cpp:8717..8760` of the pinned commit.

use std::collections::HashMap;

use crate::error::WillametteError;
use crate::gguf::reader::GgufFile;
use crate::gguf::tensor::TensorView;
use crate::gguf::types::GgmlType;
use crate::model::config::BitNetConfig;
use crate::model::primitives::f32_tensor_to_vec;

#[derive(Debug)]
pub struct LayerWeights<'a> {
    pub index: u32,

    pub attn_norm: &'a TensorView<'a>,
    /// Pre-decoded `attn_norm` weights (Stage 10-A). Forward paths read
    /// this directly instead of decoding the F32 view on every token.
    pub attn_norm_f32: Vec<f32>,
    pub attn_q: &'a TensorView<'a>,
    pub attn_k: &'a TensorView<'a>,
    pub attn_v: &'a TensorView<'a>,
    pub attn_output: &'a TensorView<'a>,
    pub attn_sub_norm: &'a TensorView<'a>,
    /// Pre-decoded `attn_sub_norm` weights (Stage 10-A).
    pub attn_sub_norm_f32: Vec<f32>,

    pub ffn_norm: &'a TensorView<'a>,
    /// Pre-decoded `ffn_norm` weights (Stage 10-A).
    pub ffn_norm_f32: Vec<f32>,
    pub ffn_gate: &'a TensorView<'a>,
    pub ffn_up: &'a TensorView<'a>,
    pub ffn_down: &'a TensorView<'a>,
    pub ffn_sub_norm: &'a TensorView<'a>,
    /// Pre-decoded `ffn_sub_norm` weights (Stage 10-A).
    pub ffn_sub_norm_f32: Vec<f32>,
}

#[derive(Debug)]
pub struct ModelGraph<'a> {
    pub config: BitNetConfig,

    pub token_embd: &'a TensorView<'a>,
    pub output_norm: &'a TensorView<'a>,
    /// Pre-decoded `output_norm` weights (Stage 10-A). Forward paths
    /// read this directly so we don't re-decode 4 bytes/element on
    /// every token.
    pub output_norm_f32: Vec<f32>,

    /// Final projection. For BitNet b1.58 this always references
    /// `token_embd` — the weight-tying rule is unconditional in
    /// `build_bitnet_158` (`src/llama.cpp:15527`). See
    /// `docs/BITNET_FORWARD_PLAN.md` §6 for the citation.
    pub lm_head: &'a TensorView<'a>,
    /// True iff the file contained a separate `output.weight` tensor.
    /// Currently `false` for `microsoft/bitnet-b1.58-2B-4T-gguf`. Even
    /// when true, the forward graph still uses `token_embd`, so this is
    /// informational only.
    pub has_output_weight_tensor: bool,

    pub layers: Vec<LayerWeights<'a>>,
}

impl<'a> ModelGraph<'a> {
    pub fn from_gguf(gguf: &'a GgufFile<'a>) -> Result<Self, WillametteError> {
        let config = BitNetConfig::from_gguf_metadata(&gguf.metadata)?;

        let by_name: HashMap<&str, &TensorView<'a>> =
            gguf.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
        if by_name.len() != gguf.tensors.len() {
            // Indicates duplicate tensor names in the file — we shouldn't see
            // this in valid GGUFs but the GGUF spec doesn't forbid it.
            return Err(WillametteError::GgufParse(
                "duplicate tensor names in GGUF tensor directory".to_string(),
            ));
        }

        // ── top-level tensors ──
        let token_embd = require_tensor(&by_name, "token_embd.weight")?;
        check_dtype(token_embd, GgmlType::F16)?;
        check_shape(
            token_embd,
            &[config.embedding_length as u64, config.vocab_size as u64],
        )?;

        let output_norm = require_tensor(&by_name, "output_norm.weight")?;
        check_dtype(output_norm, GgmlType::F32)?;
        check_shape(output_norm, &[config.embedding_length as u64])?;

        let (lm_head, has_output_weight_tensor) = if let Some(out) = by_name.get("output.weight") {
            check_dtype(out, GgmlType::F16)?;
            check_shape(
                out,
                &[config.embedding_length as u64, config.vocab_size as u64],
            )?;
            // Even if the file ships a separate output.weight, BitNet
            // b1.58 forward uses tok_embd. Use it.
            (token_embd, true)
        } else {
            (token_embd, false)
        };

        // ── per-layer tensors ──
        let mut layers: Vec<LayerWeights<'a>> = Vec::with_capacity(config.block_count as usize);
        for il in 0..config.block_count {
            let attn_norm = require_layer_tensor(&by_name, il, "attn_norm")?;
            check_dtype(attn_norm, GgmlType::F32)?;
            check_shape(attn_norm, &[config.embedding_length as u64])?;

            let attn_sub_norm = require_layer_tensor(&by_name, il, "attn_sub_norm")?;
            check_dtype(attn_sub_norm, GgmlType::F32)?;
            check_shape(attn_sub_norm, &[config.embedding_length as u64])?;

            let attn_q = require_layer_tensor(&by_name, il, "attn_q")?;
            check_dtype(attn_q, GgmlType::BitNetI2S)?;
            check_shape(
                attn_q,
                &[
                    config.embedding_length as u64,
                    (config.head_dim * config.head_count) as u64,
                ],
            )?;

            let attn_k = require_layer_tensor(&by_name, il, "attn_k")?;
            check_dtype(attn_k, GgmlType::BitNetI2S)?;
            check_shape(
                attn_k,
                &[config.embedding_length as u64, config.kv_dim as u64],
            )?;

            let attn_v = require_layer_tensor(&by_name, il, "attn_v")?;
            check_dtype(attn_v, GgmlType::BitNetI2S)?;
            check_shape(
                attn_v,
                &[config.embedding_length as u64, config.kv_dim as u64],
            )?;

            let attn_output = require_layer_tensor(&by_name, il, "attn_output")?;
            check_dtype(attn_output, GgmlType::BitNetI2S)?;
            check_shape(
                attn_output,
                &[
                    (config.head_dim * config.head_count) as u64,
                    config.embedding_length as u64,
                ],
            )?;

            let ffn_norm = require_layer_tensor(&by_name, il, "ffn_norm")?;
            check_dtype(ffn_norm, GgmlType::F32)?;
            check_shape(ffn_norm, &[config.embedding_length as u64])?;

            let ffn_sub_norm = require_layer_tensor(&by_name, il, "ffn_sub_norm")?;
            check_dtype(ffn_sub_norm, GgmlType::F32)?;
            check_shape(ffn_sub_norm, &[config.feed_forward_length as u64])?;

            let ffn_gate = require_layer_tensor(&by_name, il, "ffn_gate")?;
            check_dtype(ffn_gate, GgmlType::BitNetI2S)?;
            check_shape(
                ffn_gate,
                &[
                    config.embedding_length as u64,
                    config.feed_forward_length as u64,
                ],
            )?;

            let ffn_up = require_layer_tensor(&by_name, il, "ffn_up")?;
            check_dtype(ffn_up, GgmlType::BitNetI2S)?;
            check_shape(
                ffn_up,
                &[
                    config.embedding_length as u64,
                    config.feed_forward_length as u64,
                ],
            )?;

            let ffn_down = require_layer_tensor(&by_name, il, "ffn_down")?;
            check_dtype(ffn_down, GgmlType::BitNetI2S)?;
            check_shape(
                ffn_down,
                &[
                    config.feed_forward_length as u64,
                    config.embedding_length as u64,
                ],
            )?;

            let attn_norm_f32 = f32_tensor_to_vec(attn_norm)?;
            let attn_sub_norm_f32 = f32_tensor_to_vec(attn_sub_norm)?;
            let ffn_norm_f32 = f32_tensor_to_vec(ffn_norm)?;
            let ffn_sub_norm_f32 = f32_tensor_to_vec(ffn_sub_norm)?;

            layers.push(LayerWeights {
                index: il,
                attn_norm,
                attn_norm_f32,
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                attn_sub_norm,
                attn_sub_norm_f32,
                ffn_norm,
                ffn_norm_f32,
                ffn_gate,
                ffn_up,
                ffn_down,
                ffn_sub_norm,
                ffn_sub_norm_f32,
            });
        }

        let output_norm_f32 = f32_tensor_to_vec(output_norm)?;

        Ok(Self {
            config,
            token_embd,
            output_norm,
            output_norm_f32,
            lm_head,
            has_output_weight_tensor,
            layers,
        })
    }

    /// True iff the lm_head reference is the same tensor as `token_embd`
    /// (i.e. weight-tied, which is always true for this architecture).
    pub fn lm_head_is_tied(&self) -> bool {
        // Pointer equality between the borrowed tensors. Both come from
        // gguf.tensors, so identical address means identical tensor.
        std::ptr::eq(self.lm_head as *const _, self.token_embd as *const _)
    }
}

// ── helpers ──

fn require_tensor<'a>(
    by_name: &HashMap<&str, &'a TensorView<'a>>,
    name: &str,
) -> Result<&'a TensorView<'a>, WillametteError> {
    by_name
        .get(name)
        .copied()
        .ok_or_else(|| WillametteError::MissingMetadata(vec![format!("tensor {}", name)]))
}

fn require_layer_tensor<'a>(
    by_name: &HashMap<&str, &'a TensorView<'a>>,
    layer: u32,
    suffix: &str,
) -> Result<&'a TensorView<'a>, WillametteError> {
    let name = format!("blk.{}.{}.weight", layer, suffix);
    by_name
        .get(name.as_str())
        .copied()
        .ok_or_else(|| WillametteError::MissingMetadata(vec![format!("tensor {}", name)]))
}

fn check_dtype(t: &TensorView<'_>, expected: GgmlType) -> Result<(), WillametteError> {
    if t.ggml_type != expected {
        return Err(WillametteError::GgufParse(format!(
            "tensor {:?}: expected dtype {} ({}), got {} ({})",
            t.name,
            expected.name(),
            expected.to_raw(),
            t.ggml_type.name(),
            t.ggml_type.to_raw(),
        )));
    }
    Ok(())
}

fn check_shape(t: &TensorView<'_>, expected: &[u64]) -> Result<(), WillametteError> {
    if t.shape != expected {
        return Err(WillametteError::GgufParse(format!(
            "tensor {:?}: expected shape {:?}, got {:?}",
            t.name, expected, t.shape
        )));
    }
    Ok(())
}
