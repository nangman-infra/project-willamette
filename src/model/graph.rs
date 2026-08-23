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
use crate::model::linear::{
    graph_validated_linear_matvec_f32, graph_validated_linear_matvec_prequantized,
    linear_batched_f32, linear_matvec_f32, GraphLinearBackend,
};
use crate::model::primitives::f32_tensor_to_vec;
use crate::model::q4_k::Q8KActivation;

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

    attn_q_backend: GraphLinearBackend,
    attn_k_backend: GraphLinearBackend,
    attn_v_backend: GraphLinearBackend,
    attn_output_backend: GraphLinearBackend,
    ffn_gate_backend: GraphLinearBackend,
    ffn_up_backend: GraphLinearBackend,
    ffn_down_backend: GraphLinearBackend,
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

    lm_head_backend: GraphLinearBackend,

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
        validate_q4_k_super_scales(&gguf.tensors)?;
        validate_q6_k_scales(&gguf.tensors)?;
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
                attn_q_backend: GraphLinearBackend::resolve(attn_q),
                attn_k_backend: GraphLinearBackend::resolve(attn_k),
                attn_v_backend: GraphLinearBackend::resolve(attn_v),
                attn_output_backend: GraphLinearBackend::resolve(attn_output),
                ffn_gate_backend: GraphLinearBackend::resolve(ffn_gate),
                ffn_up_backend: GraphLinearBackend::resolve(ffn_up),
                ffn_down_backend: GraphLinearBackend::resolve(ffn_down),
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
            lm_head_backend: GraphLinearBackend::resolve(lm_head),
            layers,
        })
    }

    /// True iff the lm_head reference is the same tensor as `token_embd`.
    pub fn lm_head_is_tied(&self) -> bool {
        // Pointer equality between the borrowed tensors. Both come from
        // gguf.tensors, so identical address means identical tensor.
        std::ptr::eq(self.lm_head as *const _, self.token_embd as *const _)
    }

    pub(crate) fn project_lm_head(
        &self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), WillametteError> {
        if !self.lm_head_backend.uses_q8_k() {
            return graph_validated_linear_matvec_f32(
                self.lm_head,
                self.lm_head_backend,
                input,
                output,
            );
        }
        let activation = Q8KActivation::from_f32(input)?;
        graph_validated_linear_matvec_prequantized(
            self.lm_head,
            self.lm_head_backend,
            input,
            &activation,
            output,
        )
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

    pub(crate) fn project_qkv_graph_validated(
        &self,
        input: &[f32],
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
        activation: &mut Q8KActivation,
    ) -> Result<(), WillametteError> {
        if self.attn_q_backend.uses_q8_k()
            || self.attn_k_backend.uses_q8_k()
            || self.attn_v_backend.uses_q8_k()
        {
            activation.quantize(input)?;
        }
        graph_validated_linear_matvec_prequantized(
            self.attn_q,
            self.attn_q_backend,
            input,
            activation,
            q,
        )?;
        add_projection_bias(q, self.attn_q_bias_f32.as_deref())?;
        graph_validated_linear_matvec_prequantized(
            self.attn_k,
            self.attn_k_backend,
            input,
            activation,
            k,
        )?;
        add_projection_bias(k, self.attn_k_bias_f32.as_deref())?;
        graph_validated_linear_matvec_prequantized(
            self.attn_v,
            self.attn_v_backend,
            input,
            activation,
            v,
        )?;
        add_projection_bias(v, self.attn_v_bias_f32.as_deref())
    }

    /// Compute token-major Q/K/V and apply each projection bias once per token.
    pub fn project_qkv_batched(
        &self,
        input: &[f32],
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
    ) -> Result<(), WillametteError> {
        linear_batched_f32(self.attn_q, input, q)?;
        add_projection_bias_batched(q, self.attn_q_bias_f32.as_deref())?;
        linear_batched_f32(self.attn_k, input, k)?;
        add_projection_bias_batched(k, self.attn_k_bias_f32.as_deref())?;
        linear_batched_f32(self.attn_v, input, v)?;
        add_projection_bias_batched(v, self.attn_v_bias_f32.as_deref())
    }

    pub(crate) fn project_attn_output(
        &self,
        input: &[f32],
        output: &mut [f32],
        activation: &mut Q8KActivation,
    ) -> Result<(), WillametteError> {
        self.project_single(
            self.attn_output,
            self.attn_output_backend,
            input,
            output,
            activation,
        )
    }

    pub(crate) fn project_ffn_gate_up(
        &self,
        input: &[f32],
        gate: &mut [f32],
        up: &mut [f32],
        activation: &mut Q8KActivation,
    ) -> Result<(), WillametteError> {
        if self.ffn_gate_backend.uses_q8_k() || self.ffn_up_backend.uses_q8_k() {
            activation.quantize(input)?;
        }
        graph_validated_linear_matvec_prequantized(
            self.ffn_gate,
            self.ffn_gate_backend,
            input,
            activation,
            gate,
        )?;
        graph_validated_linear_matvec_prequantized(
            self.ffn_up,
            self.ffn_up_backend,
            input,
            activation,
            up,
        )
    }

    pub(crate) fn project_ffn_down(
        &self,
        input: &[f32],
        output: &mut [f32],
        activation: &mut Q8KActivation,
    ) -> Result<(), WillametteError> {
        self.project_single(
            self.ffn_down,
            self.ffn_down_backend,
            input,
            output,
            activation,
        )
    }

    fn project_single(
        &self,
        weight: &TensorView<'_>,
        backend: GraphLinearBackend,
        input: &[f32],
        output: &mut [f32],
        activation: &mut Q8KActivation,
    ) -> Result<(), WillametteError> {
        if backend.uses_q8_k() {
            activation.quantize(input)?;
        }
        graph_validated_linear_matvec_prequantized(weight, backend, input, activation, output)
    }
}

fn validate_q4_k_super_scales(tensors: &[TensorView<'_>]) -> Result<(), WillametteError> {
    let block_bytes = TensorView::Q4K_BYTES_PER_BLOCK as usize;
    for tensor in tensors
        .iter()
        .filter(|tensor| tensor.ggml_type == GgmlType::Q4K)
    {
        tensor.verify_byte_len()?;
        let expected = usize::try_from(TensorView::q4k_expected_byte_len(&tensor.shape)?)
            .map_err(|_| WillametteError::GgufParse("Q4_K tensor size overflow".to_string()))?;
        if tensor.data.len() != expected {
            return Err(WillametteError::GgufParse(format!(
                "tensor {:?}: Q4_K data length {} != expected {expected}",
                tensor.name,
                tensor.data.len()
            )));
        }
        if !tensor.data.len().is_multiple_of(block_bytes) {
            return Err(WillametteError::GgufParse(format!(
                "tensor {:?}: Q4_K data length {} is not a multiple of {block_bytes}",
                tensor.name,
                tensor.data.len()
            )));
        }
        for (block_index, block) in tensor.data.chunks_exact(block_bytes).enumerate() {
            let d = crate::model::primitives::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin =
                crate::model::primitives::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            if !d.is_finite() || !dmin.is_finite() {
                return Err(WillametteError::GgufParse(format!(
                    "tensor {:?}: Q4_K block {block_index} has a non-finite d or dmin scale",
                    tensor.name
                )));
            }
        }
    }
    Ok(())
}

fn validate_q6_k_scales(tensors: &[TensorView<'_>]) -> Result<(), WillametteError> {
    let block_bytes = TensorView::Q6K_BYTES_PER_BLOCK as usize;
    for tensor in tensors
        .iter()
        .filter(|tensor| tensor.ggml_type == GgmlType::Q6K)
    {
        tensor.verify_byte_len()?;
        let expected = usize::try_from(TensorView::q6k_expected_byte_len(&tensor.shape)?)
            .map_err(|_| WillametteError::GgufParse("Q6_K tensor size overflow".to_string()))?;
        if tensor.data.len() != expected {
            return Err(WillametteError::GgufParse(format!(
                "tensor {:?}: Q6_K data length {} != expected {expected}",
                tensor.name,
                tensor.data.len()
            )));
        }
        if !tensor.data.len().is_multiple_of(block_bytes) {
            return Err(WillametteError::GgufParse(format!(
                "tensor {:?}: Q6_K data length {} is not a multiple of {block_bytes}",
                tensor.name,
                tensor.data.len()
            )));
        }
        crate::model::q6_k::validate_d_scales(tensor.data).map_err(|error| {
            WillametteError::GgufParse(format!("tensor {:?}: {error}", tensor.name))
        })?;
    }
    Ok(())
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

fn add_projection_bias_batched(
    projection: &mut [f32],
    bias: Option<&[f32]>,
) -> Result<(), WillametteError> {
    let Some(bias) = bias else {
        return Ok(());
    };
    if bias.is_empty() || !projection.len().is_multiple_of(bias.len()) {
        return Err(WillametteError::GgufParse(format!(
            "batched projection length {} is not a positive multiple of bias length {}",
            projection.len(),
            bias.len()
        )));
    }
    for row in projection.chunks_exact_mut(bias.len()) {
        add_projection_bias(row, Some(bias))?;
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
    use crate::gguf::reader::GgufValue;
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
    fn graph_load_rejects_non_finite_q4_k_super_scale() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::Str("bitnet-b1.58".to_string()),
        );
        for (key, value) in [
            ("block_count", 1),
            ("embedding_length", 256),
            ("feed_forward_length", 256),
            ("context_length", 16),
            ("attention.head_count", 1),
            ("attention.head_count_kv", 1),
            ("rope.dimension_count", 256),
            ("vocab_size", 1),
        ] {
            metadata.insert(format!("bitnet-b1.58.{key}"), GgufValue::Uint32(value));
        }
        metadata.insert(
            "bitnet-b1.58.attention.layer_norm_rms_epsilon".to_string(),
            GgufValue::Float32(1e-5),
        );
        metadata.insert(
            "bitnet-b1.58.rope.freq_base".to_string(),
            GgufValue::Float32(10_000.0),
        );

        let mut data = vec![0u8; TensorView::Q4K_BYTES_PER_BLOCK as usize];
        data[..2].copy_from_slice(&0x7e00_u16.to_le_bytes());
        let gguf = GgufFile {
            version: 3,
            tensor_count: 1,
            metadata,
            tensors: vec![TensorView {
                name: "malformed.weight".to_string(),
                shape: vec![256, 1],
                ggml_type: GgmlType::Q4K,
                offset: 0,
                byte_len: data.len() as u64,
                data: &data,
                scale_data: None,
            }],
            alignment: 32,
            data_section_start: 0,
            tensor_descriptors: Vec::new(),
        };

        let error = ModelGraph::from_gguf(&gguf).unwrap_err();
        assert!(error.to_string().contains("non-finite d or dmin scale"));
    }

    #[test]
    fn graph_load_rejects_non_finite_q6_k_scale() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::Str("bitnet-b1.58".to_string()),
        );
        for (key, value) in [
            ("block_count", 1),
            ("embedding_length", 256),
            ("feed_forward_length", 256),
            ("context_length", 16),
            ("attention.head_count", 1),
            ("attention.head_count_kv", 1),
            ("rope.dimension_count", 256),
            ("vocab_size", 1),
        ] {
            metadata.insert(format!("bitnet-b1.58.{key}"), GgufValue::Uint32(value));
        }
        metadata.insert(
            "bitnet-b1.58.attention.layer_norm_rms_epsilon".to_string(),
            GgufValue::Float32(1e-5),
        );
        metadata.insert(
            "bitnet-b1.58.rope.freq_base".to_string(),
            GgufValue::Float32(10_000.0),
        );

        let mut data = vec![0_u8; TensorView::Q6K_BYTES_PER_BLOCK as usize];
        data[208..210].copy_from_slice(&0x7e00_u16.to_le_bytes());
        let gguf = GgufFile {
            version: 3,
            tensor_count: 1,
            metadata,
            tensors: vec![TensorView {
                name: "malformed.weight".to_string(),
                shape: vec![256, 1],
                ggml_type: GgmlType::Q6K,
                offset: 0,
                byte_len: data.len() as u64,
                data: &data,
                scale_data: None,
            }],
            alignment: 32,
            data_section_start: 0,
            tensor_descriptors: Vec::new(),
        };

        let error = ModelGraph::from_gguf(&gguf).unwrap_err();
        assert!(error.to_string().contains("non-finite d scale"));
    }

    #[test]
    fn quantized_scale_validation_rejects_shape_storage_mismatch() {
        let q4_data = vec![0_u8; TensorView::Q4K_BYTES_PER_BLOCK as usize];
        let q4 = TensorView {
            name: "short-q4.weight".to_string(),
            shape: vec![256, 2],
            ggml_type: GgmlType::Q4K,
            offset: 0,
            byte_len: (q4_data.len() * 2) as u64,
            data: &q4_data,
            scale_data: None,
        };
        assert!(validate_q4_k_super_scales(&[q4])
            .unwrap_err()
            .to_string()
            .contains("data length"));

        let q6_data = vec![0_u8; TensorView::Q6K_BYTES_PER_BLOCK as usize];
        let q6 = TensorView {
            name: "short-q6.weight".to_string(),
            shape: vec![256, 2],
            ggml_type: GgmlType::Q6K,
            offset: 0,
            byte_len: (q6_data.len() * 2) as u64,
            data: &q6_data,
            scale_data: None,
        };
        assert!(validate_q6_k_scales(&[q6])
            .unwrap_err()
            .to_string()
            .contains("data length"));
    }

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
    fn batched_qkv_bias_matches_repeated_projection_exactly() {
        use half::f16;

        let bytes = [1.0_f32, -2.0, 0.5, 3.0]
            .into_iter()
            .flat_map(|value| f16::from_f32(value).to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let weight = TensorView {
            name: "test.weight".to_string(),
            shape: vec![2, 2],
            ggml_type: GgmlType::F16,
            offset: 0,
            byte_len: bytes.len() as u64,
            data: &bytes,
            scale_data: None,
        };
        let layer = LayerWeights {
            index: 0,
            attn_norm: &weight,
            attn_norm_f32: vec![],
            attn_q: &weight,
            attn_q_bias: None,
            attn_q_bias_f32: Some(vec![0.25, -0.5]),
            attn_k: &weight,
            attn_k_bias: None,
            attn_k_bias_f32: Some(vec![-1.0, 2.0]),
            attn_v: &weight,
            attn_v_bias: None,
            attn_v_bias_f32: Some(vec![4.0, -3.0]),
            attn_output: &weight,
            attn_sub_norm: None,
            attn_sub_norm_f32: None,
            ffn_norm: &weight,
            ffn_norm_f32: vec![],
            ffn_gate: &weight,
            ffn_up: &weight,
            ffn_down: &weight,
            ffn_sub_norm: None,
            ffn_sub_norm_f32: None,
            attn_q_backend: GraphLinearBackend::Checked,
            attn_k_backend: GraphLinearBackend::Checked,
            attn_v_backend: GraphLinearBackend::Checked,
            attn_output_backend: GraphLinearBackend::Checked,
            ffn_gate_backend: GraphLinearBackend::Checked,
            ffn_up_backend: GraphLinearBackend::Checked,
            ffn_down_backend: GraphLinearBackend::Checked,
        };
        let input = [2.0, -1.0, -0.5, 4.0, 3.0, 0.25];
        let mut expected_q = vec![0.0; 6];
        let mut expected_k = vec![0.0; 6];
        let mut expected_v = vec![0.0; 6];
        for token in 0..3 {
            layer
                .project_qkv(
                    &input[token * 2..token * 2 + 2],
                    &mut expected_q[token * 2..token * 2 + 2],
                    &mut expected_k[token * 2..token * 2 + 2],
                    &mut expected_v[token * 2..token * 2 + 2],
                )
                .unwrap();
        }

        let mut actual_q = vec![0.0; 6];
        let mut actual_k = vec![0.0; 6];
        let mut actual_v = vec![0.0; 6];
        layer
            .project_qkv_batched(&input, &mut actual_q, &mut actual_k, &mut actual_v)
            .unwrap();

        assert_eq!(actual_q, expected_q);
        assert_eq!(actual_k, expected_k);
        assert_eq!(actual_v, expected_v);
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
