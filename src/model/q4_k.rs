//! Canonical GGML Q4_K row decoding with x86 SIMD dot dispatch.

use crate::error::WillametteError;
use crate::gguf::tensor::TensorView;
use half::f16;

const BLOCK_VALUES: usize = 256;

#[repr(C)]
#[derive(Clone, Debug)]
pub(super) struct Q8KBlock {
    pub(super) d: f32,
    pub(super) qs: [i8; BLOCK_VALUES],
    pub(super) bsums: [i16; 16],
}

impl Default for Q8KBlock {
    fn default() -> Self {
        Self {
            d: 0.0,
            qs: [0; BLOCK_VALUES],
            bsums: [0; 16],
        }
    }
}

/// Owned Q8_K activation blocks whose allocation can be reused across rows.
#[derive(Clone, Debug, Default)]
pub struct Q8KActivation {
    pub(super) blocks: Vec<Q8KBlock>,
    values: usize,
}

impl Q8KActivation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_f32(input: &[f32]) -> Result<Self, WillametteError> {
        let mut activation = Self::new();
        activation.quantize(input)?;
        Ok(activation)
    }

    /// Replaces this activation while retaining storage when its capacity is sufficient.
    pub fn quantize(&mut self, input: &[f32]) -> Result<(), WillametteError> {
        if input.is_empty() || !input.len().is_multiple_of(BLOCK_VALUES) {
            return Err(WillametteError::GgufParse(format!(
                "Q8_K activation length {} is not a positive multiple of {BLOCK_VALUES}",
                input.len()
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(WillametteError::GgufParse(
                "Q8_K activation contains a non-finite value".to_string(),
            ));
        }

        let block_count = input.len() / BLOCK_VALUES;
        self.blocks.resize_with(block_count, Q8KBlock::default);
        for (source, target) in input.chunks_exact(BLOCK_VALUES).zip(&mut self.blocks) {
            quantize_q8_k_block(source, target);
        }
        self.values = input.len();
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.values
    }

    pub fn is_empty(&self) -> bool {
        self.values == 0
    }
}

fn quantize_q8_k_block(input: &[f32], output: &mut Q8KBlock) {
    let mut amax = 0.0_f32;
    let mut max = 0.0_f32;
    for &value in input {
        let absolute = value.abs();
        if absolute > amax {
            amax = absolute;
            max = value;
        }
    }

    if amax == 0.0 {
        *output = Q8KBlock::default();
        return;
    }

    let iscale = -127.0 / max;
    output.d = 1.0 / iscale;
    for (group, values) in input.chunks_exact(16).enumerate() {
        let mut sum = 0_i16;
        for (index, &value) in values.iter().enumerate() {
            let quant = (value * iscale).round_ties_even().clamp(-127.0, 127.0) as i8;
            output.qs[group * 16 + index] = quant;
            sum += i16::from(quant);
        }
        output.bsums[group] = sum;
    }
}

fn validate_row(row: &[u8], values: usize) -> Result<(), WillametteError> {
    if values == 0 || !values.is_multiple_of(TensorView::Q4K_ELEMENTS_PER_BLOCK as usize) {
        return Err(WillametteError::GgufParse(format!(
            "Q4_K value count {values} is not a positive multiple of {}",
            TensorView::Q4K_ELEMENTS_PER_BLOCK
        )));
    }
    let expected = values / TensorView::Q4K_ELEMENTS_PER_BLOCK as usize
        * TensorView::Q4K_BYTES_PER_BLOCK as usize;
    if row.len() != expected {
        return Err(WillametteError::GgufParse(format!(
            "Q4_K row has {} bytes, expected {expected}",
            row.len()
        )));
    }
    Ok(())
}

fn block_super_scales(block: &[u8]) -> Result<(f32, f32), WillametteError> {
    let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    let dmin = f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
    if !d.is_finite() || !dmin.is_finite() {
        return Err(WillametteError::GgufParse(
            "Q4_K block has a non-finite d or dmin scale".to_string(),
        ));
    }
    Ok((d, dmin))
}

// This is upstream GGML's get_scale_min_k4 packing for eight 6-bit pairs.
pub(super) fn scale_min(scales: &[u8], group: usize) -> (u8, u8) {
    if group < 4 {
        (scales[group] & 0x3f, scales[group + 4] & 0x3f)
    } else {
        (
            (scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4),
            (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
        )
    }
}

fn for_each_weight(block: &[u8], mut visit: impl FnMut(usize, f32)) -> Result<(), WillametteError> {
    let (d, dmin) = block_super_scales(block)?;
    let scales = &block[4..16];
    let qs = &block[16..144];

    for band in 0..4 {
        let low_group = 2 * band;
        let high_group = low_group + 1;
        let (low_scale, low_min) = scale_min(scales, low_group);
        let (high_scale, high_min) = scale_min(scales, high_group);
        for index in 0..32 {
            let packed = qs[band * 32 + index];
            let low_q = packed & 0x0f;
            let high_q = packed >> 4;
            visit(
                low_group * 32 + index,
                d * f32::from(low_scale) * f32::from(low_q) - dmin * f32::from(low_min),
            );
            visit(
                high_group * 32 + index,
                d * f32::from(high_scale) * f32::from(high_q) - dmin * f32::from(high_min),
            );
        }
    }
    Ok(())
}

pub fn dequantize_row(row: &[u8], output: &mut [f32]) -> Result<(), WillametteError> {
    validate_row(row, output.len())?;
    for (block_index, block) in row
        .chunks_exact(TensorView::Q4K_BYTES_PER_BLOCK as usize)
        .enumerate()
    {
        let output_offset = block_index * TensorView::Q4K_ELEMENTS_PER_BLOCK as usize;
        for_each_weight(block, |index, weight| {
            output[output_offset + index] = weight
        })?;
    }
    Ok(())
}

pub fn dot_row(row: &[u8], input: &[f32]) -> Result<f32, WillametteError> {
    validate_row(row, input.len())?;
    validate_super_scales(row)?;

    dot_row_validated(row, input)
}

/// Computes a Q4_K row dot product against a previously quantized Q8_K activation.
pub fn dot_row_q8_k(row: &[u8], input: &Q8KActivation) -> Result<f32, WillametteError> {
    if input.values == 0 || input.blocks.len().checked_mul(BLOCK_VALUES) != Some(input.values) {
        return Err(WillametteError::GgufParse(
            "Q8_K activation has invalid dimensions".to_string(),
        ));
    }
    validate_row(row, input.values)?;
    validate_super_scales(row)?;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime detection established AVX2 support and both rows were validated.
        return Ok(unsafe { super::q4_k_simd::dot_row_q8_avx2_validated(row, &input.blocks) });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("sse2") {
        return Ok(super::q4_k_simd::dot_row_q8_sse2_validated(
            row,
            &input.blocks,
        ));
    }

    Ok(dot_row_q8_scalar_validated(row, &input.blocks))
}

pub(crate) fn dot_rows(
    row: &[u8],
    inputs: &[f32],
    input_dim: usize,
    outputs: &mut [f32],
) -> Result<(), WillametteError> {
    validate_row(row, input_dim)?;
    let expected = input_dim.checked_mul(outputs.len()).ok_or_else(|| {
        WillametteError::GgufParse("Q4_K batched input size overflow".to_string())
    })?;
    if inputs.len() != expected {
        return Err(WillametteError::GgufParse(format!(
            "Q4_K batched input length {} != expected {expected}",
            inputs.len()
        )));
    }
    if outputs.is_empty() {
        return Err(WillametteError::GgufParse(
            "Q4_K batched token count must be positive".to_string(),
        ));
    }
    validate_super_scales(row)?;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        super::q4_k_simd::dot_rows_avx2_validated(row, inputs, input_dim, outputs);
        return Ok(());
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("sse2") {
        super::q4_k_simd::dot_rows_sse2_validated(row, inputs, input_dim, outputs);
        return Ok(());
    }

    for (input, output) in inputs.chunks_exact(input_dim).zip(outputs) {
        *output = dot_row_scalar(row, input)?;
    }
    Ok(())
}

fn validate_super_scales(row: &[u8]) -> Result<(), WillametteError> {
    for block in row.chunks_exact(TensorView::Q4K_BYTES_PER_BLOCK as usize) {
        block_super_scales(block)?;
    }
    Ok(())
}

fn dot_row_validated(row: &[u8], input: &[f32]) -> Result<f32, WillametteError> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        return Ok(super::q4_k_simd::dot_row_avx2_validated(row, input));
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("sse2") {
        return Ok(super::q4_k_simd::dot_row_sse2_validated(row, input));
    }

    dot_row_scalar(row, input)
}

fn dot_row_scalar(row: &[u8], input: &[f32]) -> Result<f32, WillametteError> {
    let mut sum = 0.0_f32;
    for (block_index, block) in row
        .chunks_exact(TensorView::Q4K_BYTES_PER_BLOCK as usize)
        .enumerate()
    {
        let input_offset = block_index * TensorView::Q4K_ELEMENTS_PER_BLOCK as usize;
        for_each_weight(block, |index, weight| {
            sum += weight * input[input_offset + index]
        })?;
    }
    Ok(sum)
}

pub(super) fn dot_row_q8_scalar_validated(row: &[u8], input: &[Q8KBlock]) -> f32 {
    let mut sum = 0.0_f32;
    for (block, activation) in row
        .chunks_exact(TensorView::Q4K_BYTES_PER_BLOCK as usize)
        .zip(input)
    {
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let dmin = f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
        let scales = &block[4..16];
        let qs = &block[16..144];
        let mut scaled_sum = 0_i32;
        let mut minimum_sum = 0_i32;

        for group in 0..8 {
            let (scale, minimum) = scale_min(scales, group);
            let band = group / 2;
            let high_nibble = group % 2 != 0;
            let mut dot = 0_i32;
            for index in 0..32 {
                let packed = qs[band * 32 + index];
                let quant = if high_nibble {
                    packed >> 4
                } else {
                    packed & 0x0f
                };
                dot += i32::from(quant) * i32::from(activation.qs[group * 32 + index]);
            }
            scaled_sum += i32::from(scale) * dot;
            minimum_sum += i32::from(minimum)
                * (i32::from(activation.bsums[group * 2])
                    + i32::from(activation.bsums[group * 2 + 1]));
        }

        sum += d * activation.d * scaled_sum as f32 - dmin * activation.d * minimum_sum as f32;
    }
    sum
}

pub fn active_kernel_label() -> &'static str {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        return "Q4_K AVX2";
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("sse2") {
        return "Q4_K SSE2";
    }
    "Q4_K scalar"
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCALES: [u8; 8] = [1, 2, 3, 4, 17, 34, 51, 63];
    const MINS: [u8; 8] = [5, 6, 7, 8, 9, 10, 11, 12];
    const PACKED_SCALES_MINS: [u8; 12] = [
        0x41, 0x82, 0xc3, 0xc4, 0x05, 0x06, 0x07, 0x08, 0x91, 0xa2, 0xb3, 0xcf,
    ];

    fn pinned_block(d: f32, dmin: f32) -> Vec<u8> {
        let mut block = vec![0u8; 144];
        block[..2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        block[4..16].copy_from_slice(&PACKED_SCALES_MINS);
        for band in 0..4 {
            let low_q = (2 * band + 1) as u8;
            let high_q = (2 * band + 2) as u8;
            block[16 + band * 32..16 + (band + 1) * 32].fill(low_q | (high_q << 4));
        }
        block
    }

    #[test]
    fn decodes_upstream_scale_min_mapping_for_all_groups() {
        let block = pinned_block(0.5, 0.25);
        let mut output = [0.0; 256];
        dequantize_row(&block, &mut output).unwrap();

        for group in 0..8 {
            let q = (group + 1) as f32;
            let expected = 0.5 * f32::from(SCALES[group]) * q - 0.25 * f32::from(MINS[group]);
            assert!(output[group * 32..(group + 1) * 32]
                .iter()
                .all(|&value| value == expected));
        }
    }

    #[test]
    fn decodes_each_band_low_nibble_then_high_nibble() {
        let mut block = pinned_block(1.0, 0.0);
        block[16..].fill(0);
        for band in 0..4 {
            block[16 + band * 32 + band] = 0x0f;
            block[16 + band * 32 + band + 4] = 0xf0;
        }
        let mut output = [0.0; 256];
        dequantize_row(&block, &mut output).unwrap();

        for band in 0..4 {
            assert_eq!(
                output[2 * band * 32 + band],
                f32::from(SCALES[2 * band]) * 15.0
            );
            assert_eq!(
                output[(2 * band + 1) * 32 + band + 4],
                f32::from(SCALES[2 * band + 1]) * 15.0
            );
        }
    }

    #[test]
    fn direct_dot_matches_dequantized_dot_across_blocks() {
        let mut row = pinned_block(0.5, 0.25);
        row.extend(pinned_block(-0.125, 0.375));
        let input: Vec<f32> = (0..512)
            .map(|index| (index as f32 * 0.071).sin() + (index % 13) as f32 * 0.03)
            .collect();
        let mut output = vec![0.0; 512];
        dequantize_row(&row, &mut output).unwrap();
        let expected: f32 = output.iter().zip(&input).map(|(a, b)| a * b).sum();
        let sum_abs: f32 = output.iter().zip(&input).map(|(a, b)| (a * b).abs()).sum();
        let actual = dot_row(&row, &input).unwrap();
        assert!((actual - expected).abs() <= 1e-6 * sum_abs.max(1.0));
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn parity_fixture(block_count: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let mut state = 0x6d2b_79f5_u32;
        let mut next_byte = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        let mut row = Vec::with_capacity(block_count * 144);
        for block_index in 0..block_count {
            let mut block = vec![0u8; 144];
            let d = if block_index % 2 == 0 {
                0.03125 * (block_index + 1) as f32
            } else {
                -0.0234375 * (block_index + 1) as f32
            };
            let dmin = 0.015625 * (block_index + 2) as f32;
            block[..2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
            block[2..4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
            for byte in &mut block[4..] {
                *byte = next_byte();
            }
            row.extend(block);
        }

        let input = (0..block_count * 256)
            .map(|index| {
                let magnitude = 0.25 + ((index * 37 + 11) % 97) as f32 / 19.0;
                let oscillation = (index as f32 * 0.137).sin() * 0.125;
                if index % 2 == 0 {
                    magnitude + oscillation
                } else {
                    -magnitude + oscillation
                }
            })
            .collect::<Vec<_>>();
        let mut dequantized = vec![0.0; input.len()];
        dequantize_row(&row, &mut dequantized).unwrap();
        (row, input, dequantized)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn assert_simd_parity(label: &str, simd: f32, scalar: f32, dequantized: f32, sum_abs: f32) {
        let tolerance = 1e-5 * sum_abs.max(1.0);
        assert!(
            (simd - scalar).abs() <= tolerance,
            "{label}={simd}, scalar={scalar}, tolerance={tolerance}"
        );
        assert!(
            (simd - dequantized).abs() <= tolerance,
            "{label}={simd}, dequantized={dequantized}, tolerance={tolerance}"
        );
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn parity_values() -> (Vec<u8>, Vec<f32>, f32, f32, f32) {
        let (row, input, weights) = parity_fixture(5);
        let scalar = dot_row_scalar(&row, &input).unwrap();
        let dequantized = weights
            .iter()
            .zip(&input)
            .map(|(weight, value)| weight * value)
            .sum::<f32>();
        let sum_abs = weights
            .iter()
            .zip(&input)
            .map(|(weight, value)| (weight * value).abs())
            .sum::<f32>();

        (row, input, scalar, dequantized, sum_abs)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn assert_batched_simd_exact(
        label: &str,
        single: fn(&[u8], &[f32]) -> f32,
        kernel: fn(&[u8], &[f32], usize, &mut [f32]),
    ) {
        let (row, base_input, _) = parity_fixture(3);
        let input_dim = base_input.len();

        for tokens in [1, 2, 5, 8, 9, 32] {
            let inputs = (0..tokens)
                .flat_map(|token| {
                    base_input.iter().enumerate().map(move |(index, &value)| {
                        value * (1.0 + token as f32 * 0.03125)
                            + ((index + token * 7) % 11) as f32 * 0.0078125
                    })
                })
                .collect::<Vec<_>>();
            let expected = inputs
                .chunks_exact(input_dim)
                .map(|input| single(&row, input))
                .collect::<Vec<_>>();
            let mut actual = vec![f32::NAN; tokens];
            kernel(&row, &inputs, input_dim, &mut actual);

            for (token, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{label} token count {tokens}, token {token}: {actual} != {expected}"
                );
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn sse2_dot_matches_scalar_and_dequantized_with_cancellation() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        let (row, input, scalar, dequantized, sum_abs) = parity_values();
        let simd = super::super::q4_k_simd::dot_row_sse2_validated(&row, &input);
        assert_simd_parity("SSE2", simd, scalar, dequantized, sum_abs);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn avx2_dot_matches_scalar_and_dequantized_with_cancellation() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let (row, input, scalar, dequantized, sum_abs) = parity_values();
        let simd = super::super::q4_k_simd::dot_row_avx2_validated(&row, &input);
        assert_simd_parity("AVX2", simd, scalar, dequantized, sum_abs);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn sse2_batched_dot_matches_repeated_single_row_exactly() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        assert_batched_simd_exact(
            "SSE2",
            super::super::q4_k_simd::dot_row_sse2_validated,
            super::super::q4_k_simd::dot_rows_sse2_validated,
        );
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn avx2_batched_dot_matches_repeated_single_row_exactly() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        assert_batched_simd_exact(
            "AVX2",
            super::super::q4_k_simd::dot_row_avx2_validated,
            super::super::q4_k_simd::dot_rows_avx2_validated,
        );
    }

    #[test]
    fn rejects_malformed_rows_and_non_finite_super_scales() {
        assert!(dot_row(&[0; 144], &[0.0; 128]).is_err());
        assert!(dot_row(&[0; 143], &[0.0; 256]).is_err());
        assert!(dequantize_row(&[0; 144], &mut [0.0; 128]).is_err());

        for offset in [0, 2] {
            for scale in [f16::NAN, f16::INFINITY, f16::NEG_INFINITY] {
                let mut block = pinned_block(1.0, 1.0);
                block[offset..offset + 2].copy_from_slice(&scale.to_bits().to_le_bytes());
                assert!(dot_row(&block, &[0.0; 256]).is_err());
                assert!(dequantize_row(&block, &mut [0.0; 256]).is_err());
            }
        }
    }

    #[test]
    fn batched_dot_rejects_inconsistent_token_storage() {
        let row = pinned_block(1.0, 0.5);
        let mut output = [0.0; 2];
        assert!(dot_rows(&row, &[0.0; 256], 256, &mut output).is_err());
        assert!(dot_rows(&row[..143], &[0.0; 512], 256, &mut output).is_err());
        assert!(dot_rows(&[], &[], 0, &mut output).is_err());
        assert!(dot_rows(&row, &[], 256, &mut []).is_err());
        assert_eq!(output, [0.0; 2]);
    }

    #[test]
    fn q8_k_quantization_has_pinned_layout_sums_and_ties_to_even() {
        assert_eq!(std::mem::size_of::<Q8KBlock>(), 292);
        let mut input = vec![1.0; 256];
        input[0] = -127.0;
        input[1] = 0.5;
        input[2] = 1.5;
        input[3] = 2.5;

        let activation = Q8KActivation::from_f32(&input).unwrap();
        let block = &activation.blocks[0];
        assert_eq!(block.d, 1.0);
        assert_eq!(&block.qs[..4], &[-127, 0, 2, 2]);
        assert_eq!(block.bsums[0], -111);
        assert!(block.bsums[1..].iter().all(|&sum| sum == 16));
        assert_eq!(activation.len(), 256);
        assert!(!activation.is_empty());
    }

    #[test]
    fn q8_k_quantization_uses_first_signed_extremum_and_handles_zero() {
        let mut tied = vec![0.0; 256];
        tied[0] = 4.0;
        tied[1] = -4.0;
        let activation = Q8KActivation::from_f32(&tied).unwrap();
        assert!(activation.blocks[0].d.is_sign_negative());
        assert_eq!(activation.blocks[0].qs[0], -127);
        assert_eq!(activation.blocks[0].qs[1], 127);

        let zero = Q8KActivation::from_f32(&[0.0; 256]).unwrap();
        assert_eq!(zero.blocks[0].d.to_bits(), 0.0_f32.to_bits());
        assert_eq!(zero.blocks[0].qs, [0; 256]);
        assert_eq!(zero.blocks[0].bsums, [0; 16]);
    }

    #[test]
    fn q8_k_quantization_reuses_storage_and_rejects_bad_input_without_mutation() {
        let mut activation = Q8KActivation::new();
        activation.quantize(&[1.0; 512]).unwrap();
        let capacity = activation.blocks.capacity();
        activation.quantize(&[-2.0; 256]).unwrap();
        assert_eq!(activation.blocks.capacity(), capacity);
        let retained_d = activation.blocks[0].d;

        assert!(activation.quantize(&[]).is_err());
        assert!(activation.quantize(&[0.0; 255]).is_err());
        let mut nonfinite = [0.0; 256];
        nonfinite[17] = f32::NAN;
        assert!(activation.quantize(&nonfinite).is_err());
        nonfinite[17] = f32::INFINITY;
        assert!(activation.quantize(&nonfinite).is_err());
        assert_eq!(activation.len(), 256);
        assert_eq!(activation.blocks[0].d, retained_d);
    }

    fn reconstructed_q8_reference(row: &[u8], activation: &Q8KActivation) -> (f32, f32) {
        let mut weights = vec![0.0; activation.len()];
        dequantize_row(row, &mut weights).unwrap();
        let mut sum = 0.0_f32;
        let mut sum_abs = 0.0_f32;
        for (block_index, block) in activation.blocks.iter().enumerate() {
            for index in 0..256 {
                let product =
                    weights[block_index * 256 + index] * block.d * f32::from(block.qs[index]);
                sum += product;
                sum_abs += product.abs();
            }
        }
        (sum, sum_abs)
    }

    #[test]
    fn q4_k_q8_k_scalar_matches_reconstructed_reference_across_blocks() {
        let (row, input, _) = parity_fixture_portable(5);
        let activation = Q8KActivation::from_f32(&input).unwrap();
        let scalar = dot_row_q8_scalar_validated(&row, &activation.blocks);
        let (reference, sum_abs) = reconstructed_q8_reference(&row, &activation);
        let tolerance = 2e-6 * sum_abs.max(1.0);
        assert!(
            (scalar - reference).abs() <= tolerance,
            "scalar={scalar}, reference={reference}, tolerance={tolerance}"
        );
        let dispatched = dot_row_q8_k(&row, &activation).unwrap();
        assert!((dispatched - scalar).abs() <= 1e-6 * scalar.abs().max(1.0));
    }

    #[test]
    fn q4_k_q8_k_rejects_malformed_rows_and_dimension_mismatches() {
        let row = pinned_block(1.0, 0.5);
        let one = Q8KActivation::from_f32(&[0.0; 256]).unwrap();
        let two = Q8KActivation::from_f32(&[0.0; 512]).unwrap();
        assert!(dot_row_q8_k(&row[..143], &one).is_err());
        assert!(dot_row_q8_k(&row, &two).is_err());
        assert!(dot_row_q8_k(&row, &Q8KActivation::new()).is_err());

        let malformed = Q8KActivation {
            blocks: Vec::new(),
            values: 256,
        };
        assert!(dot_row_q8_k(&row, &malformed).is_err());
    }

    fn parity_fixture_portable(block_count: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let mut state = 0x6d2b_79f5_u32;
        let mut next_byte = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        let mut row = Vec::with_capacity(block_count * 144);
        for block_index in 0..block_count {
            let mut block = vec![0u8; 144];
            let d = if block_index % 2 == 0 {
                0.03125 * (block_index + 1) as f32
            } else {
                -0.0234375 * (block_index + 1) as f32
            };
            let dmin = 0.015625 * (block_index + 2) as f32;
            block[..2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
            block[2..4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
            for byte in &mut block[4..] {
                *byte = next_byte();
            }
            row.extend(block);
        }
        let input = (0..block_count * 256)
            .map(|index| {
                let magnitude = 0.25 + ((index * 37 + 11) % 97) as f32 / 19.0;
                let oscillation = (index as f32 * 0.137).sin() * 0.125;
                if index % 2 == 0 {
                    magnitude + oscillation
                } else {
                    -magnitude + oscillation
                }
            })
            .collect::<Vec<_>>();
        let mut dequantized = vec![0.0; input.len()];
        dequantize_row(&row, &mut dequantized).unwrap();
        (row, input, dequantized)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn q4_k_q8_k_simd_matches_scalar_on_cancellation_heavy_blocks() {
        let (row, input, _) = parity_fixture_portable(7);
        let activation = Q8KActivation::from_f32(&input).unwrap();
        let scalar = dot_row_q8_scalar_validated(&row, &activation.blocks);
        let (_, sum_abs) = reconstructed_q8_reference(&row, &activation);
        let tolerance = 2e-6 * sum_abs.max(1.0);

        if std::arch::is_x86_feature_detected!("sse2") {
            let sse = super::super::q4_k_simd::dot_row_q8_sse2_validated(&row, &activation.blocks);
            assert_eq!(sse.to_bits(), scalar.to_bits());
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            let avx = unsafe {
                super::super::q4_k_simd::dot_row_q8_avx2_validated(&row, &activation.blocks)
            };
            assert!(
                (avx - scalar).abs() <= tolerance,
                "AVX2={avx}, scalar={scalar}, tolerance={tolerance}"
            );
        }
    }
}
