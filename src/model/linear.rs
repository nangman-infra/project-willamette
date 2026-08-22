//! Architecture-neutral matrix-vector dispatch for supported GGUF weights.

use rayon::prelude::*;

use crate::error::WillametteError;
use crate::gguf::tensor::TensorView;
use crate::gguf::types::GgmlType;
use crate::model::bitlinear::bitlinear_i2s_matvec_f32;
use crate::model::primitives::f16_to_f32;

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
}
