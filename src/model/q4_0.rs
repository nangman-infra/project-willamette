//! Scalar GGML Q4_0 row decoding shared by embeddings, Linear, and lm-head.

use crate::error::WillametteError;
use crate::gguf::tensor::TensorView;
use crate::model::primitives::f16_to_f32;

fn validate_lengths(data: &[u8], values: usize) -> Result<(), WillametteError> {
    if !values.is_multiple_of(TensorView::Q4_0_ELEMENTS_PER_BLOCK as usize) {
        return Err(WillametteError::GgufParse(format!(
            "Q4_0 row length {values} is not a multiple of {}",
            TensorView::Q4_0_ELEMENTS_PER_BLOCK
        )));
    }
    let expected = values / TensorView::Q4_0_ELEMENTS_PER_BLOCK as usize
        * TensorView::Q4_0_BYTES_PER_BLOCK as usize;
    if data.len() != expected {
        return Err(WillametteError::GgufParse(format!(
            "Q4_0 row data length {} != expected {expected}",
            data.len()
        )));
    }
    Ok(())
}

fn block_scale(block: &[u8]) -> Result<f32, WillametteError> {
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    if !scale.is_finite() {
        return Err(WillametteError::GgufParse(
            "Q4_0 block has a non-finite scale".to_string(),
        ));
    }
    Ok(scale)
}

pub fn dequantize_row(data: &[u8], output: &mut [f32]) -> Result<(), WillametteError> {
    validate_lengths(data, output.len())?;
    for (block_index, block) in data
        .chunks_exact(TensorView::Q4_0_BYTES_PER_BLOCK as usize)
        .enumerate()
    {
        let scale = block_scale(block)?;
        let output_offset = block_index * TensorView::Q4_0_ELEMENTS_PER_BLOCK as usize;
        for (index, &packed) in block[2..].iter().enumerate() {
            output[output_offset + index] = scale * f32::from((packed & 0x0f) as i8 - 8);
            output[output_offset + index + 16] = scale * f32::from((packed >> 4) as i8 - 8);
        }
    }
    Ok(())
}

pub fn dot_row(data: &[u8], input: &[f32]) -> Result<f32, WillametteError> {
    validate_lengths(data, input.len())?;
    let mut sum = 0.0_f32;
    for (block_index, block) in data
        .chunks_exact(TensorView::Q4_0_BYTES_PER_BLOCK as usize)
        .enumerate()
    {
        let scale = block_scale(block)?;
        let input_offset = block_index * TensorView::Q4_0_ELEMENTS_PER_BLOCK as usize;
        for (index, &packed) in block[2..].iter().enumerate() {
            let low = f32::from((packed & 0x0f) as i8 - 8);
            let high = f32::from((packed >> 4) as i8 - 8);
            sum += scale * low * input[input_offset + index];
            sum += scale * high * input[input_offset + index + 16];
        }
    }
    Ok(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn block(scale: f32, values: [u8; 32]) -> Vec<u8> {
        let mut bytes = f16::from_f32(scale).to_bits().to_le_bytes().to_vec();
        bytes.extend((0..16).map(|index| values[index] | (values[index + 16] << 4)));
        bytes
    }

    #[test]
    fn dequantizes_nibbles_in_ggml_half_block_order() {
        let mut values = [8; 32];
        for index in 0..16 {
            values[index] = index as u8;
            values[index + 16] = 15 - index as u8;
        }
        let data = block(0.5, values);
        let mut output = [0.0; 32];
        dequantize_row(&data, &mut output).unwrap();
        assert_eq!(output[0], -4.0);
        assert_eq!(output[15], 3.5);
        assert_eq!(output[16], 3.5);
        assert_eq!(output[31], -4.0);
    }

    #[test]
    fn direct_dot_matches_dequantized_dot_across_blocks() {
        let mut data = block(0.25, std::array::from_fn(|index| (index % 16) as u8));
        data.extend(block(
            -0.5,
            std::array::from_fn(|index| 15 - (index % 16) as u8),
        ));
        let input = (0..64).map(|value| value as f32 / 11.0).collect::<Vec<_>>();
        let mut dequantized = [0.0; 64];
        dequantize_row(&data, &mut dequantized).unwrap();
        let expected = dequantized
            .iter()
            .zip(&input)
            .map(|(weight, value)| weight * value)
            .sum::<f32>();
        assert!((dot_row(&data, &input).unwrap() - expected).abs() < 1e-4);
    }

    #[test]
    fn rejects_bad_lengths_and_non_finite_scale() {
        assert!(dot_row(&[0; 18], &[0.0; 16]).is_err());
        assert!(dot_row(&[0; 17], &[0.0; 32]).is_err());

        let mut data = vec![0; 18];
        data[..2].copy_from_slice(&f16::NAN.to_bits().to_le_bytes());
        assert!(dot_row(&data, &[0.0; 32]).is_err());
    }
}
