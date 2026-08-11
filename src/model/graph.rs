//! Source-pinned tensor registry for supported BitNet and classic Llama models.
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
use crate::model::architecture::{resolve, ForwardVariant, LayerTensorRole};
use crate::model::config::ModelConfig;
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
    pub attn_sub_norm: Option<&'a TensorView<'a>>,
    /// Pre-decoded `attn_sub_norm` weights (Stage 10-A).
    pub attn_sub_norm_f32: Option<Vec<f32>>,

    pub ffn_norm: &'a TensorView<'a>,
    /// Pre-decoded `ffn_norm` weights (Stage 10-A).
    pub ffn_norm_f32: Vec<f32>,
    pub ffn_gate: &'a TensorView<'a>,
    pub ffn_up: &'a TensorView<'a>,
    pub ffn_down: &'a TensorView<'a>,
    pub ffn_sub_norm: Option<&'a TensorView<'a>>,
    /// Pre-decoded `ffn_sub_norm` weights (Stage 10-A).
    pub ffn_sub_norm_f32: Option<Vec<f32>>,
}

#[derive(Debug)]
pub struct ModelGraph<'a> {
    pub config: ModelConfig,
    pub forward_variant: ForwardVariant,

    pub token_embd: &'a TensorView<'a>,
    pub output_norm: &'a TensorView<'a>,
    /// Pre-decoded `output_norm` weights (Stage 10-A). Forward paths
    /// read this directly so we don't re-decode 4 bytes/element on
    /// every token.
    pub output_norm_f32: Vec<f32>,

    /// Final projection. BitNet always references `token_embd`; classic Llama
    /// uses `output.weight` when present and otherwise ties the embedding.
    pub lm_head: &'a TensorView<'a>,
    /// True iff the file contained a separate `output.weight` tensor.
    /// Currently `false` for `microsoft/bitnet-b1.58-2B-4T-gguf` and may be
    /// either value for classic Llama.
    pub has_output_weight_tensor: bool,

    pub layers: Vec<LayerWeights<'a>>,
}

impl<'a> ModelGraph<'a> {
    pub fn from_gguf(gguf: &'a GgufFile<'a>) -> Result<Self, WillametteError> {
        let config = ModelConfig::from_gguf_metadata(&gguf.metadata)?;
        let architecture = resolve(&config.architecture)
            .ok_or_else(|| WillametteError::UnsupportedArchitecture(config.architecture.clone()))?;
        let forward_variant = architecture.forward_variant();
        validate_role_contract(
            &config.architecture,
            architecture.layer_tensor_roles(),
            forward_variant,
        )?;

        let by_name: HashMap<&str, &TensorView<'a>> =
            gguf.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
        if by_name.len() != gguf.tensors.len() {
            // Indicates duplicate tensor names in the file — we shouldn't see
            // this in valid GGUFs but the GGUF spec doesn't forbid it.
            return Err(WillametteError::GgufParse(
                "duplicate tensor names in GGUF tensor directory".to_string(),
            ));
        }
        if forward_variant == ForwardVariant::VanillaLlama {
            reject_llama_bias_tensors(&gguf.tensors)?;
        }

        // ── top-level tensors ──
        let token_embd = require_tensor(&by_name, "token_embd.weight")?;
        match forward_variant {
            ForwardVariant::BitNetSubNorm => {
                check_dtype_one_of(token_embd, &[GgmlType::F16, GgmlType::Q6K])?;
            }
            ForwardVariant::VanillaLlama => {
                check_dtype_one_of(token_embd, &[GgmlType::F16, GgmlType::Q4_0, GgmlType::Q8_0])?;
            }
        }
        check_shape(
            token_embd,
            &[config.embedding_length as u64, config.vocab_size as u64],
        )?;

        let output_norm = require_tensor(&by_name, "output_norm.weight")?;
        check_dtype(output_norm, GgmlType::F32)?;
        check_shape(output_norm, &[config.embedding_length as u64])?;

        let (lm_head, has_output_weight_tensor) = if let Some(out) = by_name.get("output.weight") {
            match forward_variant {
                ForwardVariant::BitNetSubNorm => {
                    check_dtype_one_of(out, &[GgmlType::F16, GgmlType::Q6K])?;
                }
                ForwardVariant::VanillaLlama => {
                    check_dtype_one_of(out, &[GgmlType::F16, GgmlType::Q4_0, GgmlType::Q8_0])?;
                }
            }
            check_shape(
                out,
                &[config.embedding_length as u64, config.vocab_size as u64],
            )?;
            let selected = match forward_variant {
                ForwardVariant::BitNetSubNorm => token_embd,
                ForwardVariant::VanillaLlama => *out,
            };
            (selected, true)
        } else {
            (token_embd, false)
        };

        // ── per-layer tensors ──
        let mut layers: Vec<LayerWeights<'a>> = Vec::with_capacity(config.block_count as usize);
        for il in 0..config.block_count {
            let mut tensors = HashMap::with_capacity(architecture.layer_tensor_roles().len());
            for &role in architecture.layer_tensor_roles() {
                let tensor = require_layer_tensor(&by_name, il, role.suffix())?;
                validate_layer_tensor(tensor, role, &config, forward_variant)?;
                tensors.insert(role, tensor);
            }

            let required = |role| require_role(&tensors, il, role);
            let attn_norm = required(LayerTensorRole::AttnNorm)?;
            let attn_sub_norm = tensors.get(&LayerTensorRole::AttnSubNorm).copied();
            let attn_q = required(LayerTensorRole::AttnQ)?;
            let attn_k = required(LayerTensorRole::AttnK)?;
            let attn_v = required(LayerTensorRole::AttnV)?;
            let attn_output = required(LayerTensorRole::AttnOutput)?;
            let ffn_norm = required(LayerTensorRole::FfnNorm)?;
            let ffn_sub_norm = tensors.get(&LayerTensorRole::FfnSubNorm).copied();
            let ffn_gate = required(LayerTensorRole::FfnGate)?;
            let ffn_up = required(LayerTensorRole::FfnUp)?;
            let ffn_down = required(LayerTensorRole::FfnDown)?;

            let attn_norm_f32 = f32_tensor_to_vec(attn_norm)?;
            let attn_sub_norm_f32 = attn_sub_norm.map(f32_tensor_to_vec).transpose()?;
            let ffn_norm_f32 = f32_tensor_to_vec(ffn_norm)?;
            let ffn_sub_norm_f32 = ffn_sub_norm.map(f32_tensor_to_vec).transpose()?;

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
            forward_variant,
            token_embd,
            output_norm,
            output_norm_f32,
            lm_head,
            has_output_weight_tensor,
            layers,
        })
    }

    /// True iff the lm_head reference is the same tensor as `token_embd`.
    pub fn lm_head_is_tied(&self) -> bool {
        // Pointer equality between the borrowed tensors. Both come from
        // gguf.tensors, so identical address means identical tensor.
        std::ptr::eq(self.lm_head as *const _, self.token_embd as *const _)
    }
}

fn require_role<'a>(
    tensors: &HashMap<LayerTensorRole, &'a TensorView<'a>>,
    layer: u32,
    role: LayerTensorRole,
) -> Result<&'a TensorView<'a>, WillametteError> {
    tensors.get(&role).copied().ok_or_else(|| {
        WillametteError::NotImplemented(format!(
            "layer {layer} tensor role {:?} is not provided by this architecture",
            role
        ))
    })
}

fn validate_role_contract(
    architecture: &str,
    roles: &[LayerTensorRole],
    variant: ForwardVariant,
) -> Result<(), WillametteError> {
    use LayerTensorRole::{
        AttnK, AttnNorm, AttnOutput, AttnQ, AttnSubNorm, AttnV, FfnDown, FfnGate, FfnNorm,
        FfnSubNorm, FfnUp,
    };
    for (index, role) in roles.iter().enumerate() {
        if roles[..index].contains(role) {
            return Err(WillametteError::NotImplemented(format!(
                "architecture {architecture:?} declares duplicate tensor role {role:?}"
            )));
        }
    }
    for required in [
        AttnNorm, AttnQ, AttnK, AttnV, AttnOutput, FfnNorm, FfnGate, FfnUp, FfnDown,
    ] {
        if !roles.contains(&required) {
            return Err(WillametteError::NotImplemented(format!(
                "architecture {architecture:?} does not declare required tensor role {required:?}"
            )));
        }
    }
    if variant == ForwardVariant::BitNetSubNorm
        && (!roles.contains(&AttnSubNorm) || !roles.contains(&FfnSubNorm))
    {
        return Err(WillametteError::NotImplemented(format!(
            "architecture {architecture:?} uses BitNetSubNorm without both sub-norm tensor roles"
        )));
    }
    Ok(())
}

fn reject_llama_bias_tensors(tensors: &[TensorView<'_>]) -> Result<(), WillametteError> {
    if let Some(tensor) = tensors.iter().find(|tensor| tensor.name.ends_with(".bias")) {
        return Err(WillametteError::NotImplemented(format!(
            "Llama bias tensor {:?} is not supported",
            tensor.name
        )));
    }
    Ok(())
}

fn validate_layer_tensor(
    tensor: &TensorView<'_>,
    role: LayerTensorRole,
    config: &ModelConfig,
    variant: ForwardVariant,
) -> Result<(), WillametteError> {
    use LayerTensorRole::{
        AttnK, AttnNorm, AttnOutput, AttnQ, AttnSubNorm, AttnV, FfnDown, FfnGate, FfnNorm,
        FfnSubNorm, FfnUp,
    };
    let n_embd = config.embedding_length as u64;
    let n_ff = config.feed_forward_length as u64;
    let kv_dim = config.kv_dim as u64;
    match role {
        AttnNorm | AttnSubNorm | FfnNorm => {
            check_dtype(tensor, GgmlType::F32)?;
            check_shape(tensor, &[n_embd])
        }
        FfnSubNorm => {
            check_dtype(tensor, GgmlType::F32)?;
            check_shape(tensor, &[n_ff])
        }
        AttnQ => {
            check_linear_dtype(tensor, variant)?;
            check_shape(tensor, &[n_embd, n_embd])
        }
        AttnK | AttnV => {
            check_linear_dtype(tensor, variant)?;
            check_shape(tensor, &[n_embd, kv_dim])
        }
        AttnOutput => {
            check_linear_dtype(tensor, variant)?;
            check_shape(tensor, &[n_embd, n_embd])
        }
        FfnGate | FfnUp => {
            check_linear_dtype(tensor, variant)?;
            check_shape(tensor, &[n_embd, n_ff])
        }
        FfnDown => {
            check_linear_dtype(tensor, variant)?;
            check_shape(tensor, &[n_ff, n_embd])
        }
    }
}

fn check_linear_dtype(
    tensor: &TensorView<'_>,
    variant: ForwardVariant,
) -> Result<(), WillametteError> {
    match variant {
        ForwardVariant::BitNetSubNorm => check_dtype(tensor, GgmlType::BitNetI2S),
        ForwardVariant::VanillaLlama => {
            check_dtype_one_of(tensor, &[GgmlType::F16, GgmlType::Q4_0, GgmlType::Q8_0])
        }
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

fn check_dtype_one_of(t: &TensorView<'_>, expected: &[GgmlType]) -> Result<(), WillametteError> {
    if !expected.contains(&t.ggml_type) {
        let names = expected
            .iter()
            .map(|dtype| dtype.name())
            .collect::<Vec<_>>()
            .join(" or ");
        return Err(WillametteError::GgufParse(format!(
            "tensor {:?}: expected dtype {}, got {} ({})",
            t.name,
            names,
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

#[cfg(test)]
mod tests {
    use super::*;
    use LayerTensorRole::{
        AttnK, AttnNorm, AttnOutput, AttnQ, AttnSubNorm, AttnV, FfnDown, FfnGate, FfnNorm,
        FfnSubNorm, FfnUp,
    };

    const BITNET_ROLES: &[LayerTensorRole] = &[
        AttnNorm,
        AttnSubNorm,
        AttnQ,
        AttnK,
        AttnV,
        AttnOutput,
        FfnNorm,
        FfnSubNorm,
        FfnGate,
        FfnUp,
        FfnDown,
    ];

    #[test]
    fn role_contract_rejects_duplicates() {
        let mut roles = BITNET_ROLES.to_vec();
        roles.push(AttnQ);
        assert!(matches!(
            validate_role_contract("test", &roles, ForwardVariant::BitNetSubNorm),
            Err(WillametteError::NotImplemented(_))
        ));
    }

    #[test]
    fn role_contract_rejects_missing_core_role() {
        let roles = BITNET_ROLES
            .iter()
            .copied()
            .filter(|role| *role != FfnDown)
            .collect::<Vec<_>>();
        assert!(matches!(
            validate_role_contract("test", &roles, ForwardVariant::BitNetSubNorm),
            Err(WillametteError::NotImplemented(_))
        ));
    }

    #[test]
    fn bitnet_contract_requires_both_sub_norms() {
        let roles = BITNET_ROLES
            .iter()
            .copied()
            .filter(|role| *role != FfnSubNorm)
            .collect::<Vec<_>>();
        assert!(matches!(
            validate_role_contract("test", &roles, ForwardVariant::BitNetSubNorm),
            Err(WillametteError::NotImplemented(_))
        ));
    }

    #[test]
    fn llama_bias_tensors_are_rejected() {
        let tensor = TensorView {
            name: "blk.0.attn_q.bias".to_string(),
            shape: vec![4],
            ggml_type: GgmlType::F32,
            offset: 0,
            byte_len: 0,
            data: &[],
            scale_data: None,
        };
        assert!(matches!(
            reject_llama_bias_tensors(&[tensor]),
            Err(WillametteError::NotImplemented(_))
        ));
    }

    #[test]
    fn llama_accepts_q4_0_linears_but_bitnet_does_not() {
        let tensor = TensorView {
            name: "blk.0.attn_q.weight".to_string(),
            shape: vec![32, 32],
            ggml_type: GgmlType::Q4_0,
            offset: 0,
            byte_len: 576,
            data: &[],
            scale_data: None,
        };
        assert!(check_linear_dtype(&tensor, ForwardVariant::VanillaLlama).is_ok());
        assert!(check_linear_dtype(&tensor, ForwardVariant::BitNetSubNorm).is_err());
    }
}
