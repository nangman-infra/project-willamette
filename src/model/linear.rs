//! Architecture-neutral matrix-vector dispatch for supported GGUF weights.

use rayon::prelude::*;

use crate::error::WillametteError;
use crate::gguf::tensor::TensorView;
use crate::gguf::types::GgmlType;
use crate::model::bitlinear::bitlinear_i2s_matvec_f32;
use crate::model::primitives::f16_to_f32;
use crate::model::q4_k::Q8KActivation;

#[derive(Clone, Copy, Debug)]
pub(crate) enum GraphLinearBackend {
    Checked,
    Q4K(Q4KBackend),
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Q6K(Q6KBackend),
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Q4KBackend {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2,
    Scalar,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Clone, Copy, Debug)]
pub(crate) enum Q6KBackend {
    Avx2,
}

impl GraphLinearBackend {
    pub(crate) fn resolve(weight: &TensorView<'_>) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if weight.ggml_type == GgmlType::Q6K && std::arch::is_x86_feature_detected!("avx2") {
            return Self::Q6K(Q6KBackend::Avx2);
        }
        if weight.ggml_type != GgmlType::Q4K {
            return Self::Checked;
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            return Self::Q4K(Q4KBackend::Avx2);
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::arch::is_x86_feature_detected!("sse2") {
            return Self::Q4K(Q4KBackend::Sse2);
        }
        Self::Q4K(Q4KBackend::Scalar)
    }

    pub(crate) fn uses_q8_k(self) -> bool {
        match self {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Q4K(Q4KBackend::Avx2) => true,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Q6K(_) => true,
            Self::Q4K(_) | Self::Checked => false,
        }
    }
}

pub(crate) fn graph_validated_linear_matvec_f32(
    weight: &TensorView<'_>,
    backend: GraphLinearBackend,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), WillametteError> {
    let GraphLinearBackend::Q4K(backend) = backend else {
        return linear_matvec_f32(weight, input, output);
    };

    debug_assert_eq!(weight.ggml_type, GgmlType::Q4K);
    debug_assert_eq!(weight.shape.len(), 2);
    let in_dim = weight.shape[0] as usize;
    let out_dim = weight.shape[1] as usize;
    debug_assert_eq!(input.len(), in_dim);
    debug_assert_eq!(output.len(), out_dim);
    let row_bytes = in_dim / TensorView::Q4K_ELEMENTS_PER_BLOCK as usize
        * TensorView::Q4K_BYTES_PER_BLOCK as usize;
    debug_assert_eq!(weight.data.len(), row_bytes * out_dim);

    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let bytes = &weight.data[row * row_bytes..(row + 1) * row_bytes];
        *out = q4_k_dot_row_graph_validated(bytes, input, backend);
    });
    Ok(())
}

/// Uses a prepared Q8_K activation for validated Q4_K/Q6_K weights and preserves
/// the checked f32 dispatch for every other graph-supported dtype.
pub(crate) fn graph_validated_linear_matvec_prequantized(
    weight: &TensorView<'_>,
    backend: GraphLinearBackend,
    input: &[f32],
    activation: &Q8KActivation,
    output: &mut [f32],
) -> Result<(), WillametteError> {
    if !backend.uses_q8_k() {
        return graph_validated_linear_matvec_f32(weight, backend, input, output);
    }
    let (type_name, elements_per_block, bytes_per_block) = match backend {
        GraphLinearBackend::Q4K(_) => (
            "Q4_K",
            TensorView::Q4K_ELEMENTS_PER_BLOCK as usize,
            TensorView::Q4K_BYTES_PER_BLOCK as usize,
        ),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        GraphLinearBackend::Q6K(_) => (
            "Q6_K",
            TensorView::Q6K_ELEMENTS_PER_BLOCK as usize,
            TensorView::Q6K_BYTES_PER_BLOCK as usize,
        ),
        GraphLinearBackend::Checked => unreachable!("checked backend returned above"),
    };

    let in_dim = usize::try_from(*weight.shape.first().ok_or_else(|| {
        WillametteError::GgufParse(format!("graph {type_name} linear has no input dimension"))
    })?)
    .map_err(|_| {
        WillametteError::GgufParse(format!("graph {type_name} input dimension overflow"))
    })?;
    let out_dim = usize::try_from(*weight.shape.get(1).ok_or_else(|| {
        WillametteError::GgufParse(format!("graph {type_name} linear has no output dimension"))
    })?)
    .map_err(|_| {
        WillametteError::GgufParse(format!("graph {type_name} output dimension overflow"))
    })?;
    if input.len() != in_dim || activation.len() != in_dim || output.len() != out_dim {
        return Err(WillametteError::GgufParse(format!(
            "graph {type_name} linear dimensions input={} activation={} output={} != [{in_dim}, {out_dim}]",
            input.len(),
            activation.len(),
            output.len()
        )));
    }
    let row_bytes = in_dim / elements_per_block * bytes_per_block;

    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let bytes = &weight.data[row * row_bytes..(row + 1) * row_bytes];
        *out = match backend {
            GraphLinearBackend::Q4K(backend) => {
                q4_k_dot_row_q8_graph_validated(bytes, activation, backend)
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            GraphLinearBackend::Q6K(Q6KBackend::Avx2) => {
                // SAFETY: backend resolution checked AVX2 once at graph load,
                // and graph validation established finite, complete Q6_K blocks.
                unsafe {
                    crate::model::q6_k_simd::dot_row_q8_avx2_validated(bytes, &activation.blocks)
                }
            }
            GraphLinearBackend::Checked => unreachable!("checked backend returned above"),
        };
    });
    Ok(())
}

#[inline]
fn q4_k_dot_row_q8_graph_validated(row: &[u8], input: &Q8KActivation, backend: Q4KBackend) -> f32 {
    match backend {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Q4KBackend::Avx2 => {
            // SAFETY: graph backend resolution established AVX2 support.
            unsafe { crate::model::q4_k_simd::dot_row_q8_avx2_validated(row, &input.blocks) }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Q4KBackend::Sse2 => crate::model::q4_k_simd::dot_row_q8_sse2_validated(row, &input.blocks),
        Q4KBackend::Scalar => crate::model::q4_k::dot_row_q8_scalar_validated(row, &input.blocks),
    }
}

#[inline]
fn q4_k_dot_row_graph_validated(row: &[u8], input: &[f32], backend: Q4KBackend) -> f32 {
    match backend {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Q4KBackend::Avx2 => crate::model::q4_k_simd::dot_row_avx2_validated(row, input),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Q4KBackend::Sse2 => crate::model::q4_k_simd::dot_row_sse2_validated(row, input),
        Q4KBackend::Scalar => q4_k_dot_row_scalar_graph_validated(row, input),
    }
}

fn q4_k_dot_row_scalar_graph_validated(row: &[u8], input: &[f32]) -> f32 {
    const BLOCK_BYTES: usize = TensorView::Q4K_BYTES_PER_BLOCK as usize;
    const BLOCK_VALUES: usize = TensorView::Q4K_ELEMENTS_PER_BLOCK as usize;

    let mut sum = 0.0_f32;
    for (block_index, block) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let quants = &block[16..144];
        let input_offset = block_index * BLOCK_VALUES;
        for band in 0..4 {
            let low_group = 2 * band;
            let high_group = low_group + 1;
            let (low_scale, low_min) = crate::model::q4_k::scale_min(scales, low_group);
            let (high_scale, high_min) = crate::model::q4_k::scale_min(scales, high_group);
            for index in 0..32 {
                let packed = quants[band * 32 + index];
                let low_weight =
                    d * f32::from(low_scale) * f32::from(packed & 0x0f) - dmin * f32::from(low_min);
                let high_weight =
                    d * f32::from(high_scale) * f32::from(packed >> 4) - dmin * f32::from(high_min);
                sum += low_weight * input[input_offset + low_group * 32 + index];
                sum += high_weight * input[input_offset + high_group * 32 + index];
            }
        }
    }
    sum
}

pub fn linear_matvec_f32(
    weight: &TensorView<'_>,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), WillametteError> {
    match weight.ggml_type {
        GgmlType::BitNetI2S => bitlinear_i2s_matvec_f32(weight, input, output),
        GgmlType::F16 => f16_matvec_f32(weight, input, output),
        GgmlType::Q4_0 => q4_0_matvec_f32(weight, input, output),
        GgmlType::Q4K => q4_k_matvec_f32(weight, input, output),
        GgmlType::Q6K => q6_k_matvec_f32(weight, input, output),
        GgmlType::Q8_0 => q8_0_matvec_f32(weight, input, output),
        other => Err(WillametteError::UnsupportedTensorType(other.to_raw())),
    }
}

/// Applies a linear weight to token-major inputs `[token][in_dim]` and writes
/// token-major outputs `[token][out_dim]`.
pub fn linear_batched_f32(
    weight: &TensorView<'_>,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), WillametteError> {
    match weight.ggml_type {
        GgmlType::Q4K => q4_k_batched_f32(weight, input, output),
        GgmlType::BitNetI2S | GgmlType::F16 | GgmlType::Q4_0 | GgmlType::Q6K | GgmlType::Q8_0 => {
            let (in_dim, out_dim, _) = batched_dimensions(weight, input, output)?;
            for (input_token, output_token) in input
                .chunks_exact(in_dim)
                .zip(output.chunks_exact_mut(out_dim))
            {
                linear_matvec_f32(weight, input_token, output_token)?;
            }
            Ok(())
        }
        other => Err(WillametteError::UnsupportedTensorType(other.to_raw())),
    }
}

fn batched_dimensions(
    weight: &TensorView<'_>,
    input: &[f32],
    output: &[f32],
) -> Result<(usize, usize, usize), WillametteError> {
    if weight.shape.len() != 2 {
        return Err(WillametteError::GgufParse(format!(
            "batched linear {:?}: expected 2 dimensions, got {:?}",
            weight.name, weight.shape
        )));
    }
    let in_dim = usize::try_from(weight.shape[0])
        .map_err(|_| WillametteError::GgufParse("batched input dimension overflow".to_string()))?;
    let out_dim = usize::try_from(weight.shape[1])
        .map_err(|_| WillametteError::GgufParse("batched output dimension overflow".to_string()))?;
    if in_dim == 0 || out_dim == 0 {
        return Err(WillametteError::GgufParse(format!(
            "batched linear {:?}: dimensions must be nonzero",
            weight.name
        )));
    }
    if input.is_empty() || !input.len().is_multiple_of(in_dim) {
        return Err(WillametteError::GgufParse(format!(
            "batched linear {:?}: input length {} is not a positive multiple of {in_dim}",
            weight.name,
            input.len()
        )));
    }
    let tokens = input.len() / in_dim;
    let expected_output = tokens.checked_mul(out_dim).ok_or_else(|| {
        WillametteError::GgufParse("batched linear output size overflow".to_string())
    })?;
    if output.len() != expected_output {
        return Err(WillametteError::GgufParse(format!(
            "batched linear {:?}: output length {} != {tokens} * {out_dim}",
            weight.name,
            output.len()
        )));
    }
    Ok((in_dim, out_dim, tokens))
}

fn q4_k_batched_f32(
    weight: &TensorView<'_>,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), WillametteError> {
    let (in_dim, out_dim, tokens) = batched_dimensions(weight, input, output)?;
    let row_bytes = usize::try_from(TensorView::q4k_expected_byte_len(&[weight.shape[0]])?)
        .map_err(|_| WillametteError::GgufParse("Q4_K row size overflow".to_string()))?;
    let expected = row_bytes
        .checked_mul(out_dim)
        .ok_or_else(|| WillametteError::GgufParse("Q4_K tensor size overflow".to_string()))?;
    if weight.data.len() != expected {
        return Err(WillametteError::GgufParse(format!(
            "Q4_K linear {:?}: data length {} != expected {}",
            weight.name,
            weight.data.len(),
            expected
        )));
    }

    let mut rows = vec![0.0; output.len()];
    rows.par_chunks_mut(tokens)
        .enumerate()
        .try_for_each(|(row, values)| {
            crate::model::q4_k::dot_rows(
                &weight.data[row * row_bytes..(row + 1) * row_bytes],
                input,
                in_dim,
                values,
            )
        })?;

    for (row, values) in rows.chunks_exact(tokens).enumerate() {
        for (token, &value) in values.iter().enumerate() {
            output[token * out_dim + row] = value;
        }
    }
    Ok(())
}

pub fn q4_k_matvec_f32(
    weight: &TensorView<'_>,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), WillametteError> {
    quantized_matvec_f32(
        weight,
        input,
        output,
        GgmlType::Q4K,
        "Q4_K",
        TensorView::q4k_expected_byte_len,
        crate::model::q4_k::dot_row,
    )
}

pub fn q6_k_matvec_f32(
    weight: &TensorView<'_>,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), WillametteError> {
    quantized_matvec_f32(
        weight,
        input,
        output,
        GgmlType::Q6K,
        "Q6_K",
        TensorView::q6k_expected_byte_len,
        crate::model::q6_k::dot_row,
    )
}

fn quantized_matvec_f32(
    weight: &TensorView<'_>,
    input: &[f32],
    output: &mut [f32],
    expected_type: GgmlType,
    type_name: &str,
    expected_byte_len: fn(&[u64]) -> Result<u64, WillametteError>,
    dot_row: fn(&[u8], &[f32]) -> Result<f32, WillametteError>,
) -> Result<(), WillametteError> {
    if weight.ggml_type != expected_type {
        return Err(WillametteError::UnsupportedTensorType(
            weight.ggml_type.to_raw(),
        ));
    }
    if weight.shape.len() != 2 {
        return Err(WillametteError::GgufParse(format!(
            "{type_name} linear {:?}: expected 2 dimensions, got {:?}",
            weight.name, weight.shape
        )));
    }
    let in_dim = usize::try_from(weight.shape[0])
        .map_err(|_| WillametteError::GgufParse(format!("{type_name} input dimension overflow")))?;
    let out_dim = usize::try_from(weight.shape[1]).map_err(|_| {
        WillametteError::GgufParse(format!("{type_name} output dimension overflow"))
    })?;
    if input.len() != in_dim || output.len() != out_dim {
        return Err(WillametteError::GgufParse(format!(
            "{type_name} linear {:?}: input/output lengths {}/{} != {}/{}",
            weight.name,
            input.len(),
            output.len(),
            in_dim,
            out_dim
        )));
    }
    let row_bytes = usize::try_from(expected_byte_len(&[weight.shape[0]])?)
        .map_err(|_| WillametteError::GgufParse(format!("{type_name} row size overflow")))?;
    let expected = row_bytes
        .checked_mul(out_dim)
        .ok_or_else(|| WillametteError::GgufParse(format!("{type_name} tensor size overflow")))?;
    if weight.data.len() != expected {
        return Err(WillametteError::GgufParse(format!(
            "{type_name} linear {:?}: data length {} != expected {}",
            weight.name,
            weight.data.len(),
            expected
        )));
    }
    output
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(row, out)| {
            *out = dot_row(&weight.data[row * row_bytes..(row + 1) * row_bytes], input)?;
            Ok(())
        })
}

pub fn q4_0_matvec_f32(
    weight: &TensorView<'_>,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), WillametteError> {
    if weight.ggml_type != GgmlType::Q4_0 {
        return Err(WillametteError::UnsupportedTensorType(
            weight.ggml_type.to_raw(),
        ));
    }
    if weight.shape.len() != 2 {
        return Err(WillametteError::GgufParse(format!(
            "Q4_0 linear {:?}: expected 2 dimensions, got {:?}",
            weight.name, weight.shape
        )));
    }
    let in_dim = usize::try_from(weight.shape[0])
        .map_err(|_| WillametteError::GgufParse("Q4_0 input dimension overflow".to_string()))?;
    let out_dim = usize::try_from(weight.shape[1])
        .map_err(|_| WillametteError::GgufParse("Q4_0 output dimension overflow".to_string()))?;
    if input.len() != in_dim || output.len() != out_dim {
        return Err(WillametteError::GgufParse(format!(
            "Q4_0 linear {:?}: input/output lengths {}/{} != {}/{}",
            weight.name,
            input.len(),
            output.len(),
            in_dim,
            out_dim
        )));
    }
    let row_bytes = usize::try_from(TensorView::q4_0_expected_byte_len(&[weight.shape[0]])?)
        .map_err(|_| WillametteError::GgufParse("Q4_0 row size overflow".to_string()))?;
    let expected = row_bytes
        .checked_mul(out_dim)
        .ok_or_else(|| WillametteError::GgufParse("Q4_0 tensor size overflow".to_string()))?;
    if weight.data.len() != expected {
        return Err(WillametteError::GgufParse(format!(
            "Q4_0 linear {:?}: data length {} != expected {}",
            weight.name,
            weight.data.len(),
            expected
        )));
    }
    output
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(row, out)| {
            *out = crate::model::q4_0::dot_row(
                &weight.data[row * row_bytes..(row + 1) * row_bytes],
                input,
            )?;
            Ok(())
        })
}

pub fn q8_0_matvec_f32(
    weight: &TensorView<'_>,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), WillametteError> {
    if weight.ggml_type != GgmlType::Q8_0 {
        return Err(WillametteError::UnsupportedTensorType(
            weight.ggml_type.to_raw(),
        ));
    }
    if weight.shape.len() != 2 {
        return Err(WillametteError::GgufParse(format!(
            "Q8_0 linear {:?}: expected 2 dimensions, got {:?}",
            weight.name, weight.shape
        )));
    }
    let in_dim = usize::try_from(weight.shape[0])
        .map_err(|_| WillametteError::GgufParse("Q8_0 input dimension overflow".to_string()))?;
    let out_dim = usize::try_from(weight.shape[1])
        .map_err(|_| WillametteError::GgufParse("Q8_0 output dimension overflow".to_string()))?;
    if input.len() != in_dim || output.len() != out_dim {
        return Err(WillametteError::GgufParse(format!(
            "Q8_0 linear {:?}: input/output lengths {}/{} != {}/{}",
            weight.name,
            input.len(),
            output.len(),
            in_dim,
            out_dim
        )));
    }
    let row_bytes = usize::try_from(TensorView::q8_0_expected_byte_len(&[weight.shape[0]])?)
        .map_err(|_| WillametteError::GgufParse("Q8_0 row size overflow".to_string()))?;
    let expected = row_bytes
        .checked_mul(out_dim)
        .ok_or_else(|| WillametteError::GgufParse("Q8_0 tensor size overflow".to_string()))?;
    if weight.data.len() != expected {
        return Err(WillametteError::GgufParse(format!(
            "Q8_0 linear {:?}: data length {} != expected {}",
            weight.name,
            weight.data.len(),
            expected
        )));
    }
    output
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(row, out)| {
            *out = crate::model::q8_0::dot_row(
                &weight.data[row * row_bytes..(row + 1) * row_bytes],
                input,
            )?;
            Ok(())
        })
}

pub fn f16_matvec_f32(
    weight: &TensorView<'_>,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), WillametteError> {
    if weight.ggml_type != GgmlType::F16 {
        return Err(WillametteError::UnsupportedTensorType(
            weight.ggml_type.to_raw(),
        ));
    }
    if weight.shape.len() != 2 {
        return Err(WillametteError::GgufParse(format!(
            "F16 linear {:?}: expected 2 dimensions, got {:?}",
            weight.name, weight.shape
        )));
    }
    let in_dim = usize::try_from(weight.shape[0]).map_err(|_| {
        WillametteError::GgufParse(format!(
            "F16 linear {:?}: input dimension overflow",
            weight.name
        ))
    })?;
    let out_dim = usize::try_from(weight.shape[1]).map_err(|_| {
        WillametteError::GgufParse(format!(
            "F16 linear {:?}: output dimension overflow",
            weight.name
        ))
    })?;
    if input.len() != in_dim || output.len() != out_dim {
        return Err(WillametteError::GgufParse(format!(
            "F16 linear {:?}: input/output lengths {}/{} != {}/{}",
            weight.name,
            input.len(),
            output.len(),
            in_dim,
            out_dim
        )));
    }
    let row_bytes = in_dim
        .checked_mul(2)
        .ok_or_else(|| WillametteError::GgufParse("F16 linear row size overflow".to_string()))?;
    let expected = row_bytes
        .checked_mul(out_dim)
        .ok_or_else(|| WillametteError::GgufParse("F16 linear tensor size overflow".to_string()))?;
    if weight.data.len() != expected {
        return Err(WillametteError::GgufParse(format!(
            "F16 linear {:?}: data length {} != expected {}",
            weight.name,
            weight.data.len(),
            expected
        )));
    }

    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let bytes = &weight.data[row * row_bytes..(row + 1) * row_bytes];
        let mut sum = 0.0_f32;
        for (column, &value) in input.iter().enumerate() {
            let offset = column * 2;
            let bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            sum += f16_to_f32(bits) * value;
        }
        *out = sum;
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn tensor(values: &[f32], shape: Vec<u64>) -> TensorView<'_> {
        let bytes = values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect::<Vec<_>>()
            .leak();
        TensorView {
            name: "test.weight".to_string(),
            shape,
            ggml_type: GgmlType::F16,
            offset: 0,
            byte_len: bytes.len() as u64,
            data: bytes,
            scale_data: None,
        }
    }

    #[test]
    fn f16_rectangular_matvec_matches_hand_result() {
        let weight = tensor(&[1.0, 2.0, 3.0, -1.0, 0.5, 4.0], vec![3, 2]);
        let mut output = [0.0; 2];
        f16_matvec_f32(&weight, &[2.0, -1.0, 0.5], &mut output).unwrap();
        assert_eq!(output, [1.5, -0.5]);
    }

    #[test]
    fn f16_matvec_rejects_length_mismatch() {
        let weight = tensor(&[1.0, 2.0], vec![2, 1]);
        assert!(f16_matvec_f32(&weight, &[1.0], &mut [0.0]).is_err());
    }

    #[test]
    fn f16_batched_fallback_is_token_major() {
        let weight = tensor(&[1.0, 2.0, 3.0, -1.0, 0.5, 4.0], vec![3, 2]);
        let input = [2.0, -1.0, 0.5, -3.0, 2.0, 1.0, 0.25, 4.0, -2.0];
        let mut output = [0.0; 6];
        linear_batched_f32(&weight, &input, &mut output).unwrap();

        assert_eq!(output, [1.5, -0.5, 4.0, 8.0, 2.25, -6.25]);
    }

    #[test]
    fn q8_0_rectangular_matvec_matches_hand_result() {
        let mut bytes = Vec::new();
        for (scale, quant) in [(0.5, 2i8), (0.25, -4i8)] {
            bytes.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            bytes.extend(std::iter::repeat_n(quant as u8, 32));
        }
        let tensor = TensorView {
            name: "q8.weight".to_string(),
            shape: vec![32, 2],
            ggml_type: GgmlType::Q8_0,
            offset: 0,
            byte_len: bytes.len() as u64,
            data: &bytes,
            scale_data: None,
        };
        let mut output = [0.0; 2];
        q8_0_matvec_f32(&tensor, &[1.0; 32], &mut output).unwrap();
        assert_eq!(output, [32.0, -32.0]);
    }

    #[test]
    fn q4_0_rectangular_matvec_matches_hand_result() {
        let mut bytes = Vec::new();
        for (scale, nibble) in [(0.5, 10u8), (0.25, 4u8)] {
            bytes.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            bytes.extend(std::iter::repeat_n(nibble | (nibble << 4), 16));
        }
        let tensor = TensorView {
            name: "q4.weight".to_string(),
            shape: vec![32, 2],
            ggml_type: GgmlType::Q4_0,
            offset: 0,
            byte_len: bytes.len() as u64,
            data: &bytes,
            scale_data: None,
        };
        let mut output = [0.0; 2];
        q4_0_matvec_f32(&tensor, &[1.0; 32], &mut output).unwrap();
        assert_eq!(output, [32.0, -32.0]);
    }

    fn q4_k_constant_block(quant: u8) -> Vec<u8> {
        let mut block = vec![0u8; 144];
        block[..2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        block[4..8].fill(1);
        block[12..16].fill(1);
        block[16..].fill(quant | (quant << 4));
        block
    }

    #[test]
    fn q4_k_rectangular_matvec_matches_hand_result() {
        let mut bytes = q4_k_constant_block(2);
        bytes.extend(q4_k_constant_block(4));
        let tensor = TensorView {
            name: "q4_k.weight".to_string(),
            shape: vec![256, 2],
            ggml_type: GgmlType::Q4K,
            offset: 0,
            byte_len: bytes.len() as u64,
            data: &bytes,
            scale_data: None,
        };
        let mut output = [0.0; 2];
        q4_k_matvec_f32(&tensor, &[1.0; 256], &mut output).unwrap();
        assert_eq!(output, [512.0, 1_024.0]);
    }

    #[test]
    fn q4_k_batched_matches_repeated_matvec_exactly() {
        let mut bytes = q4_k_constant_block(2);
        bytes.extend(q4_k_constant_block(4));
        bytes.extend(q4_k_constant_block(7));
        let tensor = TensorView {
            name: "q4_k.batch.weight".to_string(),
            shape: vec![256, 3],
            ggml_type: GgmlType::Q4K,
            offset: 0,
            byte_len: bytes.len() as u64,
            data: &bytes,
            scale_data: None,
        };

        for tokens in [1, 2, 5, 8, 9, 32] {
            let input = (0..tokens * 256)
                .map(|index| {
                    let token = index / 256;
                    let column = index % 256;
                    (column as f32 * 0.03125).sin() + token as f32 * 0.125 - 0.5
                })
                .collect::<Vec<_>>();
            let mut expected = vec![0.0; tokens * 3];
            for (input_token, output_token) in
                input.chunks_exact(256).zip(expected.chunks_exact_mut(3))
            {
                linear_matvec_f32(&tensor, input_token, output_token).unwrap();
            }
            let mut actual = vec![f32::NAN; tokens * 3];
            linear_batched_f32(&tensor, &input, &mut actual).unwrap();

            assert_eq!(actual, expected, "token count {tokens}");
        }
    }

    #[test]
    fn q4_k_graph_fixed_backend_matches_checked_matvec_bitwise() {
        let mut bytes = q4_k_constant_block(2);
        bytes.extend(q4_k_constant_block(7));
        let tensor = TensorView {
            name: "q4_k.graph.weight".to_string(),
            shape: vec![256, 2],
            ggml_type: GgmlType::Q4K,
            offset: 0,
            byte_len: bytes.len() as u64,
            data: &bytes,
            scale_data: None,
        };
        let input = (0..256)
            .map(|index| (index as f32 * 0.03125).sin() - 0.25)
            .collect::<Vec<_>>();
        let mut expected = [0.0; 2];
        q4_k_matvec_f32(&tensor, &input, &mut expected).unwrap();

        let backend = GraphLinearBackend::resolve(&tensor);
        let mut actual = [0.0; 2];
        graph_validated_linear_matvec_f32(&tensor, backend, &input, &mut actual).unwrap();

        assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
    }

    #[test]
    fn q4_k_graph_prequantized_dispatch_matches_q8_kernel() {
        let mut bytes = q4_k_constant_block(2);
        bytes.extend(q4_k_constant_block(7));
        let tensor = TensorView {
            name: "q4_k.graph.q8.weight".to_string(),
            shape: vec![256, 2],
            ggml_type: GgmlType::Q4K,
            offset: 0,
            byte_len: bytes.len() as u64,
            data: &bytes,
            scale_data: None,
        };
        let input = (0..256)
            .map(|index| (index as f32 * 0.071).cos() * 1.7 - 0.2)
            .collect::<Vec<_>>();
        let activation = Q8KActivation::from_f32(&input).unwrap();
        let backend = GraphLinearBackend::resolve(&tensor);
        let mut expected = [0.0; 2];
        if backend.uses_q8_k() {
            expected = [
                crate::model::q4_k::dot_row_q8_k(&bytes[..144], &activation).unwrap(),
                crate::model::q4_k::dot_row_q8_k(&bytes[144..], &activation).unwrap(),
            ];
        } else {
            graph_validated_linear_matvec_f32(&tensor, backend, &input, &mut expected).unwrap();
        }

        let mut actual = [0.0; 2];
        graph_validated_linear_matvec_prequantized(
            &tensor,
            backend,
            &input,
            &activation,
            &mut actual,
        )
        .unwrap();

        assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
    }

    #[test]
    fn q4_k_prequantized_scalar_backend_preserves_f32_result() {
        let bytes = q4_k_constant_block(7);
        let tensor = TensorView {
            name: "q4_k.graph.scalar.weight".to_string(),
            shape: vec![256, 1],
            ggml_type: GgmlType::Q4K,
            offset: 0,
            byte_len: bytes.len() as u64,
            data: &bytes,
            scale_data: None,
        };
        let input = (0..256)
            .map(|index| (index as f32 * 0.071).cos() * 1.7 - 0.2)
            .collect::<Vec<_>>();
        let activation = Q8KActivation::from_f32(&input).unwrap();
        let backend = GraphLinearBackend::Q4K(Q4KBackend::Scalar);
        let mut expected = [0.0];
        graph_validated_linear_matvec_f32(&tensor, backend, &input, &mut expected).unwrap();
        let mut actual = [0.0];
        graph_validated_linear_matvec_prequantized(
            &tensor,
            backend,
            &input,
            &activation,
            &mut actual,
        )
        .unwrap();

        assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
    }

    #[test]
    fn graph_prequantized_dispatch_preserves_non_q4_k_fallback() {
        let weight = tensor(&[1.0, 2.0, 3.0, -1.0, 0.5, 4.0], vec![3, 2]);
        let input = [2.0, -1.0, 0.5];
        let mut expected = [0.0; 2];
        linear_matvec_f32(&weight, &input, &mut expected).unwrap();

        let mut actual = [0.0; 2];
        graph_validated_linear_matvec_prequantized(
            &weight,
            GraphLinearBackend::resolve(&weight),
            &input,
            &Q8KActivation::new(),
            &mut actual,
        )
        .unwrap();

        assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn q4_k_graph_prequantized_dispatch_rejects_bad_dimensions_without_writing() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let bytes = q4_k_constant_block(2);
        let tensor = TensorView {
            name: "q4_k.graph.invalid.weight".to_string(),
            shape: vec![256, 1],
            ggml_type: GgmlType::Q4K,
            offset: 0,
            byte_len: bytes.len() as u64,
            data: &bytes,
            scale_data: None,
        };
        let activation = Q8KActivation::from_f32(&[0.0; 512]).unwrap();
        let mut output = [17.0];

        assert!(graph_validated_linear_matvec_prequantized(
            &tensor,
            GraphLinearBackend::Q4K(Q4KBackend::Avx2),
            &[0.0; 256],
            &activation,
            &mut output,
        )
        .is_err());
        assert_eq!(output, [17.0]);
    }

    #[test]
    fn batched_linear_rejects_malformed_inputs_without_writing_output() {
        let bytes = q4_k_constant_block(2);
        let valid = TensorView {
            name: "q4_k.invalid.weight".to_string(),
            shape: vec![256, 1],
            ggml_type: GgmlType::Q4K,
            offset: 0,
            byte_len: bytes.len() as u64,
            data: &bytes,
            scale_data: None,
        };

        for (input, output_len) in [(vec![], 0), (vec![0.0; 255], 1), (vec![0.0; 256], 2)] {
            let mut output = vec![17.0; output_len];
            assert!(linear_batched_f32(&valid, &input, &mut output).is_err());
            assert!(output.iter().all(|&value| value == 17.0));
        }

        let bad_shape = TensorView {
            name: valid.name.clone(),
            shape: vec![256],
            ggml_type: valid.ggml_type,
            offset: valid.offset,
            byte_len: valid.byte_len,
            data: valid.data,
            scale_data: valid.scale_data,
        };
        let mut output = [17.0];
        assert!(linear_batched_f32(&bad_shape, &[0.0; 256], &mut output).is_err());
        assert_eq!(output, [17.0]);

        let bad_data = TensorView {
            data: &bytes[..143],
            byte_len: 143,
            ..valid
        };
        assert!(linear_batched_f32(&bad_data, &[0.0; 256], &mut output).is_err());
        assert_eq!(output, [17.0]);
    }

    fn q6_k_constant_block(scale: u8) -> Vec<u8> {
        let mut block = vec![0x11; 128];
        block.extend_from_slice(&[0xaa; 64]);
        block.extend_from_slice(&[scale; 16]);
        block.extend_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        block
    }

    #[test]
    fn q6_k_rectangular_matvec_matches_hand_result() {
        let mut bytes = q6_k_constant_block(1);
        bytes.extend(q6_k_constant_block(2));
        let tensor = TensorView {
            name: "q6_k.weight".to_string(),
            shape: vec![256, 2],
            ggml_type: GgmlType::Q6K,
            offset: 0,
            byte_len: bytes.len() as u64,
            data: &bytes,
            scale_data: None,
        };
        let mut output = [0.0; 2];
        q6_k_matvec_f32(&tensor, &[1.0; 256], &mut output).unwrap();
        assert_eq!(output, [256.0, 512.0]);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn q6_k_graph_prequantized_avx2_dispatch_matches_checked_kernel() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut bytes = q6_k_constant_block(1);
        bytes.extend(q6_k_constant_block(2));
        let tensor = TensorView {
            name: "q6_k.graph.weight".to_string(),
            shape: vec![256, 2],
            ggml_type: GgmlType::Q6K,
            offset: 0,
            byte_len: bytes.len() as u64,
            data: &bytes,
            scale_data: None,
        };
        let input = (0..256)
            .map(|index| (index as f32 * 0.071).cos() * 1.7 - 0.2)
            .collect::<Vec<_>>();
        let activation = Q8KActivation::from_f32(&input).unwrap();
        let expected = [
            crate::model::q6_k::dot_row_q8_k(&bytes[..210], &activation).unwrap(),
            crate::model::q6_k::dot_row_q8_k(&bytes[210..], &activation).unwrap(),
        ];
        let backend = GraphLinearBackend::resolve(&tensor);
        assert!(matches!(backend, GraphLinearBackend::Q6K(Q6KBackend::Avx2)));

        let mut actual = [0.0; 2];
        graph_validated_linear_matvec_prequantized(
            &tensor,
            backend,
            &input,
            &activation,
            &mut actual,
        )
        .unwrap();
        assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
    }
}
