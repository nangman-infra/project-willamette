//! Scalar standard-GGML Q6_K decoding used by embedding gather and lm-head.

use crate::error::WillametteError;
use crate::gguf::tensor::TensorView;
use crate::model::primitives::f16_to_f32;
use half::f16;

fn validate_row(row: &[u8], values: usize) -> Result<(), WillametteError> {
    if !values.is_multiple_of(TensorView::Q6K_ELEMENTS_PER_BLOCK as usize) {
        return Err(WillametteError::GgufParse(format!(
            "Q6_K value count {values} is not a multiple of 256"
        )));
    }
    let expected = values / TensorView::Q6K_ELEMENTS_PER_BLOCK as usize
        * TensorView::Q6K_BYTES_PER_BLOCK as usize;
    if row.len() != expected {
        return Err(WillametteError::GgufParse(format!(
            "Q6_K row has {} bytes, expected {expected}",
            row.len()
        )));
    }
    Ok(())
}

fn for_each_weight(block: &[u8], mut visit: impl FnMut(usize, f32)) {
    let ql = &block[..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

    for half in 0..2 {
        let ql = &ql[half * 64..];
        let qh = &qh[half * 32..];
        let scales = &scales[half * 8..];
        let output = half * 128;
        for l in 0..32 {
            let scale = l / 16;
            let q1 = ((ql[l] & 0x0f) | ((qh[l] & 3) << 4)) as i8 - 32;
            let q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
            visit(output + l, d * scales[scale] as i8 as f32 * q1 as f32);
            visit(
                output + l + 32,
                d * scales[scale + 2] as i8 as f32 * q2 as f32,
            );
            visit(
                output + l + 64,
                d * scales[scale + 4] as i8 as f32 * q3 as f32,
            );
            visit(
                output + l + 96,
                d * scales[scale + 6] as i8 as f32 * q4 as f32,
            );
        }
    }
}

pub fn dequantize_row(row: &[u8], out: &mut [f32]) -> Result<(), WillametteError> {
    validate_row(row, out.len())?;
    for (block_index, block) in row
        .chunks_exact(TensorView::Q6K_BYTES_PER_BLOCK as usize)
        .enumerate()
    {
        let output = block_index * TensorView::Q6K_ELEMENTS_PER_BLOCK as usize;
        for_each_weight(block, |index, weight| out[output + index] = weight);
    }
    Ok(())
}

pub fn dot_row(row: &[u8], input: &[f32]) -> Result<f32, WillametteError> {
    validate_row(row, input.len())?;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("sse2") {
        // SAFETY: runtime detection establishes SSE2 support. validate_row
        // established complete 210-byte blocks and matching input length.
        return Ok(unsafe { super::q6_k_sse2::dot_row_sse2_validated(row, input) });
    }

    Ok(dot_row_scalar_validated(row, input))
}

fn dot_row_scalar_validated(row: &[u8], input: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for (block_index, block) in row
        .chunks_exact(TensorView::Q6K_BYTES_PER_BLOCK as usize)
        .enumerate()
    {
        let input_offset = block_index * TensorView::Q6K_ELEMENTS_PER_BLOCK as usize;
        for_each_weight(block, |index, weight| {
            sum += weight * input[input_offset + index]
        });
    }
    sum
}

fn nearest_int(value: f32) -> i32 {
    debug_assert!(value.abs() <= 4_194_303.0);
    (((value + 12_582_912.0).to_bits() & 0x007f_ffff) as i32) - 0x0040_0000
}

fn signed_abs_max(input: &[f32]) -> f32 {
    let mut max = 0.0_f32;
    for &value in input {
        if value.abs() > max.abs() {
            max = value;
        }
    }
    max
}

fn store_levels(input: &[f32], levels: &mut [i8], inverse_scale: f32) {
    for (level, &value) in levels.iter_mut().zip(input) {
        *level = (nearest_int(inverse_scale * value).clamp(-32, 31) + 32) as i8;
    }
}

fn weighted_sums(input: &[f32], inverse_scale: f32) -> (f32, f32) {
    let mut sum_lx = 0.0_f32;
    let mut sum_l2 = 0.0_f32;
    for &value in input {
        let level = nearest_int(inverse_scale * value).clamp(-32, 31);
        let weight = value * value;
        sum_lx += weight * value * level as f32;
        sum_l2 += weight * (level * level) as f32;
    }
    (sum_lx, sum_l2)
}

fn make_qx_quants(input: &[f32], levels: &mut [i8]) -> f32 {
    let max = signed_abs_max(input);
    if max.abs() < 1e-15 {
        levels.fill(0);
        return 0.0;
    }

    let mut inverse_scale = -32.0 / max;
    store_levels(input, levels, inverse_scale);
    let (mut sum_lx, mut sum_l2) = weighted_sums(input, inverse_scale);
    let mut scale = if sum_l2 != 0.0 { sum_lx / sum_l2 } else { 0.0 };
    let mut best = scale * sum_lx;
    for adjustment in -9..=9 {
        if adjustment == 0 {
            continue;
        }
        inverse_scale = -(32.0 + 0.1 * adjustment as f32) / max;
        (sum_lx, sum_l2) = weighted_sums(input, inverse_scale);
        if sum_l2 > 0.0 && sum_lx * sum_lx > best * sum_l2 {
            store_levels(input, levels, inverse_scale);
            scale = sum_lx / sum_l2;
            best = scale * sum_lx;
        }
    }
    scale
}

fn quantize_group_scales(input: &[f32], levels: &mut [i8; 256]) -> ([f32; 16], f32) {
    let mut scales = [0.0_f32; 16];
    let mut max_scale = 0.0_f32;
    for (group, scale_slot) in scales.iter_mut().enumerate() {
        let scale = make_qx_quants(
            &input[group * 16..(group + 1) * 16],
            &mut levels[group * 16..(group + 1) * 16],
        );
        *scale_slot = scale;
        if scale.abs() > max_scale.abs() {
            max_scale = scale;
        }
    }
    (scales, max_scale)
}

fn requantize_groups(
    input: &[f32],
    levels: &mut [i8; 256],
    scales: &[f32; 16],
    inverse_scale: f32,
    d: f32,
) -> [i8; 16] {
    let mut quant_scales = [0i8; 16];
    for (group, (&scale, quant_scale)) in scales.iter().zip(&mut quant_scales).enumerate() {
        *quant_scale = nearest_int(inverse_scale * scale).min(127) as i8;
        let group_scale = d * *quant_scale as f32;
        if group_scale != 0.0 {
            for index in 0..16 {
                let level = nearest_int(input[group * 16 + index] / group_scale).clamp(-32, 31);
                levels[group * 16 + index] = (level + 32) as i8;
            }
        }
    }
    quant_scales
}

fn pack_block(levels: &[i8; 256], quant_scales: &[i8; 16], d_bits: u16, output: &mut [u8]) {
    for half in 0..2 {
        let source = half * 128;
        let ql = half * 64;
        let qh = 128 + half * 32;
        for index in 0..32 {
            let q1 = levels[source + index] as u8;
            let q2 = levels[source + index + 32] as u8;
            let q3 = levels[source + index + 64] as u8;
            let q4 = levels[source + index + 96] as u8;
            output[ql + index] = (q1 & 0x0f) | ((q3 & 0x0f) << 4);
            output[ql + index + 32] = (q2 & 0x0f) | ((q4 & 0x0f) << 4);
            output[qh + index] = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
        }
    }
    for (destination, &scale) in output[192..208].iter_mut().zip(quant_scales) {
        *destination = scale as u8;
    }
    output[208..210].copy_from_slice(&d_bits.to_le_bytes());
}

fn quantize_block(input: &[f32], output: &mut [u8]) {
    let mut levels = [0i8; 256];
    let (scales, max_scale) = quantize_group_scales(input, &mut levels);
    if max_scale.abs() < 1e-15 {
        output.fill(0);
        return;
    }

    let inverse_scale = -128.0 / max_scale;
    let d_bits = f16::from_f32(1.0 / inverse_scale).to_bits();
    let d = f16::from_bits(d_bits).to_f32();
    let quant_scales = requantize_groups(input, &mut levels, &scales, inverse_scale, d);
    pack_block(&levels, &quant_scales, d_bits, output);
}

/// Quantize one f32 row to standard GGML Q6_K blocks.
pub fn quantize_row(input: &[f32], output: &mut [u8]) -> Result<(), WillametteError> {
    validate_row(output, input.len())?;
    for (input_block, output_block) in input.chunks_exact(256).zip(output.chunks_exact_mut(210)) {
        quantize_block(input_block, output_block);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ones_block() -> Vec<u8> {
        let mut block = vec![0x11; 128];
        block.extend_from_slice(&[0xaa; 64]);
        block.extend_from_slice(&[1; 16]);
        block.extend_from_slice(&0x3c00u16.to_le_bytes());
        block
    }

    #[test]
    fn decodes_exact_ones_block() {
        let mut output = vec![0.0; 256];
        dequantize_row(&ones_block(), &mut output).unwrap();
        assert!(output.iter().all(|&value| value == 1.0));
    }

    #[test]
    fn decodes_pinned_upstream_bit_and_scale_mapping() {
        let mut block = vec![0u8; 210];
        block[0] = 0xf0;
        block[32] = 0xe1;
        block[128] = 0xe4;
        for (index, scale) in block[192..208].iter_mut().enumerate() {
            *scale = (index + 1) as u8;
        }
        block[200] = (-2i8) as u8;
        block[64] = 0x0f;
        block[160] = 0x03;
        block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());

        let mut output = vec![0.0; 256];
        dequantize_row(&block, &mut output).unwrap();
        assert_eq!(output[0], -32.0);
        assert_eq!(output[32], -45.0);
        assert_eq!(output[64], 75.0);
        assert_eq!(output[96], 210.0);
        assert_eq!(output[128], -62.0);
    }

    #[test]
    fn direct_dot_matches_dequantized_dot() {
        let block = ones_block();
        let input: Vec<f32> = (0..256).map(|index| index as f32 / 32.0 - 4.0).collect();
        let mut output = vec![0.0; 256];
        dequantize_row(&block, &mut output).unwrap();
        let expected: f32 = output.iter().zip(&input).map(|(a, b)| a * b).sum();
        assert_eq!(dot_row_scalar_validated(&block, &input), expected);
        let dispatched = dot_row(&block, &input).unwrap();
        assert!((dispatched - expected).abs() < 1e-3);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn sse2_dot_matches_scalar_with_cancellation() {
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        let input: Vec<f32> = (0..512)
            .map(|index| ((index as f32 * 0.173).sin() * 5.0) - (index % 11) as f32 * 0.2)
            .collect();
        let mut quantized = vec![0u8; 420];
        quantize_row(&input, &mut quantized).unwrap();
        let probe: Vec<f32> = (0..512)
            .map(|index| ((index as f32 * 0.311).cos() * 2.0) + (index % 5) as f32)
            .collect();
        let scalar = dot_row_scalar_validated(&quantized, &probe);
        // SAFETY: the feature check above established SSE2 support and the
        // quantized row has two complete blocks matching probe.len().
        let simd = unsafe { super::super::q6_k_sse2::dot_row_sse2_validated(&quantized, &probe) };
        let mut decoded = vec![0.0; probe.len()];
        dequantize_row(&quantized, &mut decoded).unwrap();
        let sum_abs: f32 = decoded
            .iter()
            .zip(&probe)
            .map(|(weight, value)| (weight * value).abs())
            .sum();
        let tolerance = 1e-5 * sum_abs.max(1.0);
        assert!(
            (simd - scalar).abs() <= tolerance,
            "SSE2={simd}, scalar={scalar}, tolerance={tolerance}"
        );
    }

    #[test]
    fn quantize_roundtrip_has_small_error() {
        let input: Vec<f32> = (0..512)
            .map(|index| ((index as f32 * 0.37).sin() * 3.0) + (index % 7) as f32 * 0.1)
            .collect();
        let mut quantized = vec![0u8; 420];
        quantize_row(&input, &mut quantized).unwrap();
        let mut decoded = vec![0.0; input.len()];
        dequantize_row(&quantized, &mut decoded).unwrap();
        let max_error = input
            .iter()
            .zip(decoded)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_error < 0.15, "max error {max_error}");
    }
}
