//! Canonical GGML Q4_K row decoding with x86 SIMD dot dispatch.

use crate::error::WillametteError;
use crate::gguf::tensor::TensorView;
use half::f16;

fn validate_row(row: &[u8], values: usize) -> Result<(), WillametteError> {
    if !values.is_multiple_of(TensorView::Q4K_ELEMENTS_PER_BLOCK as usize) {
        return Err(WillametteError::GgufParse(format!(
            "Q4_K value count {values} is not a multiple of {}",
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
    for block in row.chunks_exact(TensorView::Q4K_BYTES_PER_BLOCK as usize) {
        block_super_scales(block)?;
    }

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
}
