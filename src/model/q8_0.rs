//! GGML Q8_0 row decoding with SIMD dot dispatch and scalar fallback.

use crate::error::WillametteError;
use crate::gguf::tensor::TensorView;
use crate::model::primitives::f16_to_f32;

fn validate_lengths(data: &[u8], values: usize) -> Result<(), WillametteError> {
    if !values.is_multiple_of(TensorView::Q8_0_ELEMENTS_PER_BLOCK as usize) {
        return Err(WillametteError::GgufParse(format!(
            "Q8_0 row length {values} is not a multiple of {}",
            TensorView::Q8_0_ELEMENTS_PER_BLOCK
        )));
    }
    let expected = values / TensorView::Q8_0_ELEMENTS_PER_BLOCK as usize
        * TensorView::Q8_0_BYTES_PER_BLOCK as usize;
    if data.len() != expected {
        return Err(WillametteError::GgufParse(format!(
            "Q8_0 row data length {} != expected {expected}",
            data.len()
        )));
    }
    Ok(())
}

fn block_scale(block: &[u8]) -> Result<f32, WillametteError> {
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    if !scale.is_finite() {
        return Err(WillametteError::GgufParse(
            "Q8_0 block has a non-finite scale".to_string(),
        ));
    }
    Ok(scale)
}

pub fn dequantize_row(data: &[u8], output: &mut [f32]) -> Result<(), WillametteError> {
    validate_lengths(data, output.len())?;
    for (block_index, block) in data
        .chunks_exact(TensorView::Q8_0_BYTES_PER_BLOCK as usize)
        .enumerate()
    {
        let scale = block_scale(block)?;
        let output_offset = block_index * TensorView::Q8_0_ELEMENTS_PER_BLOCK as usize;
        for (index, &quant) in block[2..].iter().enumerate() {
            output[output_offset + index] = scale * f32::from(quant as i8);
        }
    }
    Ok(())
}

pub fn dot_row(data: &[u8], input: &[f32]) -> Result<f32, WillametteError> {
    validate_lengths(data, input.len())?;

    #[cfg(not(willamette_q8_scalar))]
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { super::q8_0_simd::dot_row_neon(data, input) };
    }
    #[cfg(not(willamette_q8_scalar))]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { super::q8_0_simd::dot_row_avx2(data, input) };
    }
    #[cfg(not(willamette_q8_scalar))]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("sse2") {
        return unsafe { super::q8_0_simd::dot_row_sse2(data, input) };
    }

    dot_row_scalar(data, input)
}

fn dot_row_scalar(data: &[u8], input: &[f32]) -> Result<f32, WillametteError> {
    let mut sum = 0.0_f32;
    for (block_index, block) in data
        .chunks_exact(TensorView::Q8_0_BYTES_PER_BLOCK as usize)
        .enumerate()
    {
        let scale = block_scale(block)?;
        let input_offset = block_index * TensorView::Q8_0_ELEMENTS_PER_BLOCK as usize;
        for (index, &quant) in block[2..].iter().enumerate() {
            sum += scale * f32::from(quant as i8) * input[input_offset + index];
        }
    }
    Ok(sum)
}

pub fn active_kernel_label() -> &'static str {
    if cfg!(willamette_q8_scalar) {
        return "Q8_0 scalar";
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return "Q8_0 NEON";
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        return "Q8_0 AVX2";
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("sse2") {
        return "Q8_0 SSE2";
    }
    "Q8_0 scalar"
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn block(scale: f32, values: impl Iterator<Item = i8>) -> Vec<u8> {
        let mut bytes = f16::from_f32(scale).to_bits().to_le_bytes().to_vec();
        bytes.extend(values.map(|value| value as u8));
        bytes
    }

    #[test]
    fn dequantizes_signed_values_with_per_block_scale() {
        let data = block(0.5, -16i8..16);
        let mut output = [0.0; 32];
        dequantize_row(&data, &mut output).unwrap();
        assert_eq!(output[0], -8.0);
        assert_eq!(output[16], 0.0);
        assert_eq!(output[31], 7.5);
    }

    #[test]
    fn direct_dot_matches_dequantized_dot() {
        let data = block(0.25, (0..32).map(|value| value as i8 - 16));
        let input = (0..32).map(|value| value as f32 / 7.0).collect::<Vec<_>>();
        let mut dequantized = [0.0; 32];
        dequantize_row(&data, &mut dequantized).unwrap();
        let expected = dequantized
            .iter()
            .zip(&input)
            .map(|(weight, value)| weight * value)
            .sum::<f32>();
        assert!((dot_row(&data, &input).unwrap() - expected).abs() < 1e-5);
    }

    #[test]
    fn dispatched_dot_matches_scalar_across_blocks() {
        let mut data = Vec::new();
        for block_index in 0..4 {
            data.extend(block(
                0.125 * (block_index + 1) as f32,
                (0..32).map(move |value| ((value * 17 + block_index * 11) % 127) as i8 - 63),
            ));
        }
        let input = (0..128)
            .map(|value| ((value * 29 % 101) as f32 - 50.0) / 13.0)
            .collect::<Vec<_>>();
        let expected = dot_row_scalar(&data, &input).unwrap();
        let actual = dot_row(&data, &input).unwrap();
        assert!((actual - expected).abs() < 1e-3, "{actual} != {expected}");
    }

    #[test]
    fn rejects_non_finite_scale() {
        let mut data = vec![0; 34];
        data[..2].copy_from_slice(&f16::NAN.to_bits().to_le_bytes());
        assert!(dot_row(&data, &[0.0; 32]).is_err());
    }
}
