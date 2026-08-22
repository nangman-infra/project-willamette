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
use crate::model::linear::linear_matvec_f32;
use crate::model::primitives::f32_tensor_to_vec;

#[derive(Debug)]
pub struct LayerWeights<'a> {
    pub index: u32,

    pub attn_norm: &'a TensorView<'a>,
    /// Pre-decoded `attn_norm` weights (Stage 10-A). Forward paths read
    /// this directly instead of decoding the F32 view on every token.
    pub attn_norm_f32: Vec<f32>,
    pub attn_q: &'a TensorView<'a>,
    pub attn_q_bias: Option<&'a TensorView<'a>>,
    pub attn_q_bias_f32: Option<Vec<f32>>,
    pub attn_k: &'a TensorView<'a>,
    pub attn_k_bias: Option<&'a TensorView<'a>>,
    pub attn_k_bias_f32: Option<Vec<f32>>,
    pub attn_v: &'a TensorView<'a>,
    pub attn_v_bias: Option<&'a TensorView<'a>>,
    pub attn_v_bias_f32: Option<Vec<f32>>,
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
        reject_unsupported_bias_tensors(
            &config.architecture,
            &gguf.tensors,
            forward_variant,
            config.block_count,
        )?;

        // ── top-level tensors ──
        let token_embd = require_tensor(&by_name, "token_embd.weight")?;
        match forward_variant {
            ForwardVariant::BitNetSubNorm => {
                check_dtype_one_of(token_embd, &[GgmlType::F16, GgmlType::Q6K])?;
            }
            ForwardVariant::VanillaLlama | ForwardVariant::Qwen2 => {
                check_dtype_one_of(
                    token_embd,
                    &[
                        GgmlType::F16,
                        GgmlType::Q4_0,
                        GgmlType::Q4K,
                        GgmlType::Q6K,
                        GgmlType::Q8_0,
                    ],
                )?;
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
                ForwardVariant::VanillaLlama | ForwardVariant::Qwen2 => {
                    check_dtype_one_of(
                        out,
                        &[
                            GgmlType::F16,
                            GgmlType::Q4_0,
                            GgmlType::Q4K,
                            GgmlType::Q6K,
                            GgmlType::Q8_0,
                        ],
                    )?;
                }
            }
            check_shape(
                out,
                &[config.embedding_length as u64, config.vocab_size as u64],
            )?;
            let selected = match forward_variant {
                ForwardVariant::BitNetSubNorm => token_embd,
                ForwardVariant::VanillaLlama | ForwardVariant::Qwen2 => *out,
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
                let tensor = require_layer_tensor(&by_name, il, role)?;
                validate_layer_tensor(tensor, role, &config, forward_variant)?;
                tensors.insert(role, tensor);
            }

            let required = |role| require_role(&tensors, il, role);
            let attn_norm = required(LayerTensorRole::AttnNorm)?;
            let attn_sub_norm = tensors.get(&LayerTensorRole::AttnSubNorm).copied();
            let attn_q = required(LayerTensorRole::AttnQ)?;
            let attn_q_bias = tensors.get(&LayerTensorRole::AttnQBias).copied();
            let attn_k = required(LayerTensorRole::AttnK)?;
            let attn_k_bias = tensors.get(&LayerTensorRole::AttnKBias).copied();
            let attn_v = required(LayerTensorRole::AttnV)?;
            let attn_v_bias = tensors.get(&LayerTensorRole::AttnVBias).copied();
            let attn_output = required(LayerTensorRole::AttnOutput)?;
            let ffn_norm = required(LayerTensorRole::FfnNorm)?;
            let ffn_sub_norm = tensors.get(&LayerTensorRole::FfnSubNorm).copied();
            let ffn_gate = required(LayerTensorRole::FfnGate)?;
            let ffn_up = required(LayerTensorRole::FfnUp)?;
            let ffn_down = required(LayerTensorRole::FfnDown)?;

            let attn_norm_f32 = f32_tensor_to_vec(attn_norm)?;
            let attn_q_bias_f32 = attn_q_bias.map(f32_tensor_to_vec).transpose()?;
            let attn_k_bias_f32 = attn_k_bias.map(f32_tensor_to_vec).transpose()?;
            let attn_v_bias_f32 = attn_v_bias.map(f32_tensor_to_vec).transpose()?;
            let attn_sub_norm_f32 = attn_sub_norm.map(f32_tensor_to_vec).transpose()?;
            let ffn_norm_f32 = f32_tensor_to_vec(ffn_norm)?;
            let ffn_sub_norm_f32 = ffn_sub_norm.map(f32_tensor_to_vec).transpose()?;

            layers.push(LayerWeights {
                index: il,
                attn_norm,
                attn_norm_f32,
                attn_q,
                attn_q_bias,
                attn_q_bias_f32,
                attn_k,
                attn_k_bias,
                attn_k_bias_f32,
                attn_v,
                attn_v_bias,
                attn_v_bias_f32,
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

impl LayerWeights<'_> {
    /// Compute Q/K/V and apply architecture-provided projection biases.
    pub fn project_qkv(
        &self,
        input: &[f32],
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
    ) -> Result<(), WillametteError> {
        linear_matvec_f32(self.attn_q, input, q)?;
        add_projection_bias(q, self.attn_q_bias_f32.as_deref())?;
        linear_matvec_f32(self.attn_k, input, k)?;
        add_projection_bias(k, self.attn_k_bias_f32.as_deref())?;
        linear_matvec_f32(self.attn_v, input, v)?;
        add_projection_bias(v, self.attn_v_bias_f32.as_deref())
    }
}

fn add_projection_bias(
    projection: &mut [f32],
    bias: Option<&[f32]>,
) -> Result<(), WillametteError> {
    let Some(bias) = bias else {
        return Ok(());
    };
    if projection.len() != bias.len() {
        return Err(WillametteError::GgufParse(format!(
            "projection length {} does not match bias length {}",
            projection.len(),
            bias.len()
        )));
    }
    for (value, bias) in projection.iter_mut().zip(bias) {
        *value += bias;
    }
    Ok(())
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
        AttnK, AttnKBias, AttnNorm, AttnOutput, AttnQ, AttnQBias, AttnSubNorm, AttnV, AttnVBias,
        FfnDown, FfnGate, FfnNorm, FfnSubNorm, FfnUp,
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
    if variant == ForwardVariant::Qwen2
        && [AttnQBias, AttnKBias, AttnVBias]
            .iter()
            .any(|role| !roles.contains(role))
    {
        return Err(WillametteError::NotImplemented(format!(
            "architecture {architecture:?} uses Qwen2 without all Q/K/V bias tensor roles"
        )));
    }
    Ok(())
}

fn reject_unsupported_bias_tensors(
    architecture: &str,
    tensors: &[TensorView<'_>],
    variant: ForwardVariant,
    block_count: u32,
) -> Result<(), WillametteError> {
    for tensor in tensors
        .iter()
        .filter(|tensor| tensor.name.ends_with(".bias"))
    {
        let supported = variant == ForwardVariant::Qwen2
            && (0..block_count).any(|layer| {
                ["attn_q", "attn_k", "attn_v"]
                    .iter()
                    .any(|projection| tensor.name == format!("blk.{layer}.{projection}.bias"))
            });
        if !supported {
            return Err(WillametteError::NotImplemented(format!(
                "architecture {architecture:?} bias tensor {:?} is not supported",
                tensor.name
            )));
        }
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
        AttnK, AttnKBias, AttnNorm, AttnOutput, AttnQ, AttnQBias, AttnSubNorm, AttnV, AttnVBias,
        FfnDown, FfnGate, FfnNorm, FfnSubNorm, FfnUp,
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
        AttnQBias => {
            check_dtype(tensor, GgmlType::F32)?;
            check_shape(tensor, &[n_embd])
        }
        AttnKBias | AttnVBias => {
            check_dtype(tensor, GgmlType::F32)?;
            check_shape(tensor, &[kv_dim])
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
        ForwardVariant::VanillaLlama | ForwardVariant::Qwen2 => check_dtype_one_of(
            tensor,
            &[
                GgmlType::F16,
                GgmlType::Q4_0,
                GgmlType::Q4K,
                GgmlType::Q6K,
                GgmlType::Q8_0,
            ],
        ),
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
    role: LayerTensorRole,
) -> Result<&'a TensorView<'a>, WillametteError> {
    let name = format!("blk.{}.{}.{}", layer, role.suffix(), role.tensor_kind());
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
    fn qwen2_contract_requires_all_qkv_biases() {
        let roles = resolve("qwen2")
            .unwrap()
            .layer_tensor_roles()
            .iter()
            .copied()
            .filter(|role| *role != LayerTensorRole::AttnVBias)
            .collect::<Vec<_>>();
        assert!(matches!(
            validate_role_contract("qwen2", &roles, ForwardVariant::Qwen2),
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
            reject_unsupported_bias_tensors("llama", &[tensor], ForwardVariant::VanillaLlama, 1),
            Err(WillametteError::NotImplemented(_))
        ));
    }

    #[test]
    fn qwen2_rejects_non_qkv_biases() {
        let tensor = TensorView {
            name: "blk.0.attn_output.bias".to_string(),
            shape: vec![4],
            ggml_type: GgmlType::F32,
            offset: 0,
            byte_len: 0,
            data: &[],
            scale_data: None,
        };
        assert!(matches!(
            reject_unsupported_bias_tensors("qwen2", &[tensor], ForwardVariant::Qwen2, 1),
            Err(WillametteError::NotImplemented(_))
        ));
    }

    #[test]
    fn projection_bias_is_added_and_length_checked() {
        let mut projection = [1.0, 2.0];
        add_projection_bias(&mut projection, Some(&[0.25, -0.5])).unwrap();
        assert_eq!(projection, [1.25, 1.5]);
        assert!(add_projection_bias(&mut projection, Some(&[1.0])).is_err());
    }

    #[test]
    fn qwen2_bias_requires_f32_and_projection_shape() {
        let config = ModelConfig {
            architecture: "qwen2".to_string(),
            block_count: 1,
            embedding_length: 4,
            feed_forward_length: 8,
            context_length: 16,
            head_count: 2,
            head_count_kv: 1,
            head_dim: 2,
            kv_dim: 2,
            layer_norm_rms_epsilon: 1e-6,
            rope_dimension_count: 2,
            rope_freq_base: 10_000.0,
            vocab_size: 4,
        };
        let wrong_dtype = TensorView {
            name: "blk.0.attn_q.bias".to_string(),
            shape: vec![4],
            ggml_type: GgmlType::F16,
            offset: 0,
            byte_len: 0,
            data: &[],
            scale_data: None,
        };
        let wrong_shape = TensorView {
            name: "blk.0.attn_k.bias".to_string(),
            shape: vec![4],
            ggml_type: GgmlType::F32,
            offset: 0,
            byte_len: 0,
            data: &[],
            scale_data: None,
        };

        assert!(validate_layer_tensor(
            &wrong_dtype,
            LayerTensorRole::AttnQBias,
            &config,
            ForwardVariant::Qwen2,
        )
        .is_err());
        assert!(validate_layer_tensor(
            &wrong_shape,
            LayerTensorRole::AttnKBias,
            &config,
            ForwardVariant::Qwen2,
        )
        .is_err());
    }

    #[test]
    fn llama_accepts_quantized_linears_but_bitnet_does_not() {
        for ggml_type in [GgmlType::Q4_0, GgmlType::Q4K, GgmlType::Q6K] {
            let tensor = TensorView {
                name: "blk.0.attn_q.weight".to_string(),
                shape: vec![256, 256],
                ggml_type,
                offset: 0,
                byte_len: 0,
                data: &[],
                scale_data: None,
            };
            assert!(check_linear_dtype(&tensor, ForwardVariant::VanillaLlama).is_ok());
            assert!(check_linear_dtype(&tensor, ForwardVariant::BitNetSubNorm).is_err());
        }
    }
}
