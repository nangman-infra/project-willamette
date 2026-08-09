//! Narrow GGUF repacker for converting the tied F16 embedding to Q6_K.

use crate::error::WillametteError;
use crate::gguf::reader::GgufFile;
use crate::gguf::tensor::TensorView;
use crate::gguf::types::GgmlType;
use crate::model::primitives::f16_to_f32;
use crate::model::q6_k;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn read_u32(data: &[u8], offset: usize) -> Result<u32, WillametteError> {
    let bytes: [u8; 4] = data
        .get(offset..offset + 4)
        .ok_or_else(|| {
            WillametteError::GgufParse("tensor descriptor u32 out of bounds".to_string())
        })?
        .try_into()
        .expect("slice length checked");
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(data: &[u8], offset: usize) -> Result<u64, WillametteError> {
    let bytes: [u8; 8] = data
        .get(offset..offset + 8)
        .ok_or_else(|| {
            WillametteError::GgufParse("tensor descriptor u64 out of bounds".to_string())
        })?
        .try_into()
        .expect("slice length checked");
    Ok(u64::from_le_bytes(bytes))
}

fn descriptor_fields(
    header: &[u8],
    mut cursor: usize,
    tensor: &TensorView<'_>,
) -> Result<(usize, usize, usize), WillametteError> {
    let name_len = usize::try_from(read_u64(header, cursor)?)
        .map_err(|_| WillametteError::GgufParse("tensor name length overflow".to_string()))?;
    cursor += 8;
    if header.get(cursor..cursor + name_len) != Some(tensor.name.as_bytes()) {
        return Err(WillametteError::GgufParse(format!(
            "tensor descriptor order mismatch at {:?}",
            tensor.name
        )));
    }
    cursor += name_len;
    let dimensions = read_u32(header, cursor)? as usize;
    cursor += 4;
    if dimensions != tensor.shape.len() {
        return Err(WillametteError::GgufParse(format!(
            "tensor descriptor dimension mismatch at {:?}",
            tensor.name
        )));
    }
    for &dimension in &tensor.shape {
        if read_u64(header, cursor)? != dimension {
            return Err(WillametteError::GgufParse(format!(
                "tensor descriptor shape mismatch at {:?}",
                tensor.name
            )));
        }
        cursor += 8;
    }
    let dtype_offset = cursor;
    if read_u32(header, dtype_offset)? != tensor.ggml_type.to_raw() {
        return Err(WillametteError::GgufParse(format!(
            "tensor descriptor dtype mismatch at {:?}",
            tensor.name
        )));
    }
    let tensor_offset = dtype_offset + 4;
    Ok((dtype_offset, tensor_offset, tensor_offset + 8))
}

/// Create a derived GGUF whose tied `token_embd.weight` is standard Q6_K.
pub fn repack_embedding_q6k(source: &[u8], output_path: &Path) -> Result<u64, WillametteError> {
    let gguf = GgufFile::parse(source)?;
    let embedding = gguf
        .tensors
        .iter()
        .find(|tensor| tensor.name == "token_embd.weight")
        .ok_or_else(|| WillametteError::GgufParse("missing token_embd.weight".to_string()))?;
    if embedding.ggml_type != GgmlType::F16 || embedding.shape.len() != 2 {
        return Err(WillametteError::GgufParse(
            "token_embd.weight must be a 2-D F16 tensor".to_string(),
        ));
    }
    let data_start = gguf
        .tensors
        .iter()
        .map(|tensor| tensor.offset)
        .min()
        .ok_or_else(|| WillametteError::GgufParse("GGUF has no tensors".to_string()))?;
    if embedding.offset != data_start {
        return Err(WillametteError::GgufParse(
            "token_embd.weight must be the first data tensor for narrow repacking".to_string(),
        ));
    }

    let q6_bytes = TensorView::q6k_expected_byte_len(&embedding.shape)?;
    let removed_bytes = embedding.byte_len.checked_sub(q6_bytes).ok_or_else(|| {
        WillametteError::GgufParse("Q6_K embedding is not smaller than F16".to_string())
    })?;
    if !removed_bytes.is_multiple_of(gguf.alignment) {
        return Err(WillametteError::GgufParse(format!(
            "embedding size reduction {removed_bytes} breaks alignment {}",
            gguf.alignment
        )));
    }
    let old_embedding_end = embedding.offset + embedding.byte_len;
    if gguf
        .tensors
        .iter()
        .any(|tensor| tensor.offset > embedding.offset && tensor.offset < old_embedding_end)
    {
        return Err(WillametteError::GgufParse(
            "another tensor overlaps token_embd.weight".to_string(),
        ));
    }

    let data_start_usize = usize::try_from(data_start)
        .map_err(|_| WillametteError::GgufParse("data start does not fit usize".to_string()))?;
    let mut header = source[..data_start_usize].to_vec();
    let mut descriptor = usize::try_from(gguf.tensor_info_offset).map_err(|_| {
        WillametteError::GgufParse("tensor directory offset does not fit usize".to_string())
    })?;
    for tensor in &gguf.tensors {
        let (dtype_offset, offset_offset, next) = descriptor_fields(&header, descriptor, tensor)?;
        let relative_offset = read_u64(&header, offset_offset)?;
        if tensor.name == embedding.name {
            header[dtype_offset..dtype_offset + 4]
                .copy_from_slice(&GgmlType::Q6K.to_raw().to_le_bytes());
        } else if tensor.offset >= old_embedding_end {
            let shifted = relative_offset.checked_sub(removed_bytes).ok_or_else(|| {
                WillametteError::GgufParse(format!("offset underflow at {:?}", tensor.name))
            })?;
            header[offset_offset..offset_offset + 8].copy_from_slice(&shifted.to_le_bytes());
        }
        descriptor = next;
    }

    if output_path.try_exists()? {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("output already exists: {}", output_path.display()),
        )
        .into());
    }
    let file_name = output_path
        .file_name()
        .ok_or_else(|| WillametteError::GgufParse("output path has no file name".to_string()))?;
    let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temp_name = format!(
        ".{}.{}.{}.tmp",
        file_name.to_string_lossy(),
        std::process::id(),
        counter
    );
    let temp_path = output_path.with_file_name(temp_name);

    let mut temp_created = false;
    let write_result = (|| -> Result<(), WillametteError> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)?;
        temp_created = true;
        let mut output = BufWriter::with_capacity(1024 * 1024, file);
        output.write_all(&header)?;
        let row_elements = embedding.shape[0] as usize;
        let f16_row_bytes = row_elements * 2;
        let q6_row_bytes = row_elements / 256 * 210;
        let mut row = vec![0.0_f32; row_elements];
        let mut quantized = vec![0u8; q6_row_bytes];
        for f16_bytes in embedding.data.chunks_exact(f16_row_bytes) {
            for (index, bytes) in f16_bytes.chunks_exact(2).enumerate() {
                row[index] = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
            }
            if let Some((index, value)) =
                row.iter().enumerate().find(|(_, value)| !value.is_finite())
            {
                return Err(WillametteError::GgufParse(format!(
                    "token_embd.weight contains non-finite value {value} at row element {index}"
                )));
            }
            q6_k::quantize_row(&row, &mut quantized)?;
            output.write_all(&quantized)?;
        }
        output.write_all(&source[old_embedding_end as usize..])?;
        output.flush()?;
        output.get_ref().sync_all()?;
        drop(output);

        // Publishing via a hard link is atomic and never replaces an existing path.
        std::fs::hard_link(&temp_path, output_path)?;
        Ok(())
    })();
    let cleanup_result = temp_created.then(|| std::fs::remove_file(&temp_path));
    match (write_result, cleanup_result) {
        (Ok(()), _) => {}
        (Err(write_error), Some(Err(cleanup_error))) => {
            return Err(WillametteError::GgufParse(format!(
                "{write_error}; also failed to remove temporary file {}: {cleanup_error}",
                temp_path.display()
            )));
        }
        (Err(write_error), _) => return Err(write_error),
    }
    Ok(source.len() as u64 - removed_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{build_gguf, Preset};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn repacks_synthetic_embedding_and_preserves_graph() {
        let source = build_gguf(Preset::Small, false);
        let source_gguf = GgufFile::parse(&source).unwrap();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "willamette-q6k-repack-{}-{nonce}.gguf",
            std::process::id()
        ));
        let expected_size = repack_embedding_q6k(&source, &path).unwrap();
        let repacked = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(repacked.len() as u64, expected_size);

        let gguf = GgufFile::parse(&repacked).unwrap();
        let embedding = gguf
            .tensors
            .iter()
            .find(|tensor| tensor.name == "token_embd.weight")
            .unwrap();
        assert_eq!(embedding.ggml_type, GgmlType::Q6K);
        embedding.verify_byte_len().unwrap();
        let removed_bytes = source_gguf
            .tensors
            .iter()
            .find(|tensor| tensor.name == "token_embd.weight")
            .unwrap()
            .byte_len
            - embedding.byte_len;
        for source_tensor in source_gguf
            .tensors
            .iter()
            .filter(|tensor| tensor.name != "token_embd.weight")
        {
            let repacked_tensor = gguf
                .tensors
                .iter()
                .find(|tensor| tensor.name == source_tensor.name)
                .unwrap();
            assert_eq!(
                repacked_tensor.data, source_tensor.data,
                "{}",
                source_tensor.name
            );
            assert_eq!(
                repacked_tensor.scale_data, source_tensor.scale_data,
                "{} scale block",
                source_tensor.name
            );
            assert_eq!(repacked_tensor.offset, source_tensor.offset - removed_bytes);
        }
        crate::model::graph::ModelGraph::from_gguf(&gguf).unwrap();
    }

    #[test]
    fn does_not_overwrite_existing_output() {
        let source = build_gguf(Preset::Small, false);
        let path = std::env::temp_dir().join(format!(
            "willamette-q6k-existing-{}-{}.gguf",
            std::process::id(),
            TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, b"keep").unwrap();
        let error = repack_embedding_q6k(&source, &path).unwrap_err();
        assert!(
            matches!(error, WillametteError::Io(ref io) if io.kind() == std::io::ErrorKind::AlreadyExists)
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"keep");
        std::fs::remove_file(path).unwrap();
    }
}
