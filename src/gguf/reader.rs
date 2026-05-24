use std::collections::HashMap;
use std::io::{Cursor, Read};

use byteorder::{LittleEndian, ReadBytesExt};

use crate::error::WillametteError;
use crate::gguf::tensor::TensorView;
use crate::gguf::types::{GgmlType, GgufMetadataValueType};

// ── GGUF constants ──

/// Magic bytes: "GGUF" in little-endian u32 = 0x4655_4747.
///
/// Bytes on disk are `[b'G', b'G', b'U', b'F']` = `[0x47, 0x47, 0x55, 0x46]`,
/// which decoded little-endian gives `0x4655_4747`.
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// Default alignment for tensor data (GGUF v3 spec).
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

// ── Public metadata value representation ──

/// A single value stored in the GGUF metadata key-value section.
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    Str(String),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Try to extract a u64 (also accepting u32 / u16 / u8).
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::Uint8(v) => Some(*v as u64),
            GgufValue::Uint16(v) => Some(*v as u64),
            GgufValue::Uint32(v) => Some(*v as u64),
            GgufValue::Uint64(v) => Some(*v),
            GgufValue::Int32(v) if *v >= 0 => Some(*v as u64),
            GgufValue::Int64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    /// Try to extract a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Try to extract a f32.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::Float32(v) => Some(*v),
            GgufValue::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// Try to extract an array of strings.
    pub fn as_string_array(&self) -> Option<Vec<&str>> {
        match self {
            GgufValue::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr {
                    out.push(v.as_str()?);
                }
                Some(out)
            }
            _ => None,
        }
    }
}

// ── GGUF Reader ──

/// Result of parsing a GGUF file. Holds metadata and tensor views.
///
/// All tensor data is zero-copy — the `TensorView.data` slices point directly
/// into the source byte buffer (which is backed by mmap).
pub struct GgufFile<'a> {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorView<'a>>,
    pub alignment: u64,
}

impl<'a> GgufFile<'a> {
    /// Parse a GGUF file from a byte buffer (typically from `ModelMmap::as_bytes()`).
    ///
    /// This function never panics on malformed input — it returns a
    /// `WillametteError` instead.
    pub fn parse(data: &'a [u8]) -> Result<Self, WillametteError> {
        let file_len = data.len() as u64;
        let mut cur = Cursor::new(data);

        // ── 1. Magic ──
        let magic = cur
            .read_u32::<LittleEndian>()
            .map_err(|e| WillametteError::GgufParse(format!("reading magic: {}", e)))?;
        if magic != GGUF_MAGIC {
            return Err(WillametteError::InvalidMagic(magic));
        }

        // ── 2. Version ──
        let version = cur
            .read_u32::<LittleEndian>()
            .map_err(|e| WillametteError::GgufParse(format!("reading version: {}", e)))?;
        if version != 2 && version != 3 {
            return Err(WillametteError::UnsupportedVersion(version));
        }

        // ── 3. Counts ──
        let tensor_count = cur
            .read_u64::<LittleEndian>()
            .map_err(|e| WillametteError::GgufParse(format!("reading tensor_count: {}", e)))?;
        let metadata_kv_count = cur
            .read_u64::<LittleEndian>()
            .map_err(|e| WillametteError::GgufParse(format!("reading metadata_kv_count: {}", e)))?;

        // ── 4. Metadata key-values ──
        let mut metadata = HashMap::new();
        for i in 0..metadata_kv_count {
            let key = read_gguf_string(&mut cur)
                .map_err(|e| WillametteError::GgufParse(format!("metadata[{}] key: {}", i, e)))?;
            let value = read_gguf_value(&mut cur).map_err(|e| {
                WillametteError::GgufParse(format!(
                    "metadata[{}] (key=\"{}\") value: {}",
                    i, key, e
                ))
            })?;
            metadata.insert(key, value);
        }

        // ── 5. Read alignment from metadata (default 32) ──
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(WillametteError::GgufParse(format!(
                "invalid alignment value: {}",
                alignment
            )));
        }

        // ── 6. Tensor info entries ──
        // Each entry describes name, ndims, shape, ggml_type, and the
        // *relative* offset of the tensor data from the start of the data
        // section (NOT from start of file).
        struct RawTensorInfo {
            name: String,
            shape: Vec<u64>,
            ggml_type: GgmlType,
            relative_offset: u64,
        }

        let mut raw_infos: Vec<RawTensorInfo> = Vec::with_capacity(tensor_count as usize);
        for i in 0..tensor_count {
            let name = read_gguf_string(&mut cur)
                .map_err(|e| WillametteError::GgufParse(format!("tensor[{}] name: {}", i, e)))?;
            let n_dims = cur
                .read_u32::<LittleEndian>()
                .map_err(|e| WillametteError::GgufParse(format!("tensor[{}] n_dims: {}", i, e)))?;
            let mut shape = Vec::with_capacity(n_dims as usize);
            for d in 0..n_dims {
                let dim = cur.read_u64::<LittleEndian>().map_err(|e| {
                    WillametteError::GgufParse(format!("tensor[{}] shape[{}]: {}", i, d, e))
                })?;
                shape.push(dim);
            }
            let raw_type = cur.read_u32::<LittleEndian>().map_err(|e| {
                WillametteError::GgufParse(format!("tensor[{}] ggml_type: {}", i, e))
            })?;
            let ggml_type = GgmlType::from_raw(raw_type);

            let relative_offset = cur
                .read_u64::<LittleEndian>()
                .map_err(|e| WillametteError::GgufParse(format!("tensor[{}] offset: {}", i, e)))?;

            raw_infos.push(RawTensorInfo {
                name,
                shape,
                ggml_type,
                relative_offset,
            });
        }

        // ── 7. Compute the absolute start of the tensor data section ──
        // After all header + metadata + tensor info entries, the data section
        // begins at the next alignment boundary.
        let header_end = cur.position();
        let data_section_start = align_offset(header_end, alignment);

        // ── 8. Build TensorViews ──
        let mut tensors: Vec<TensorView<'a>> = Vec::with_capacity(raw_infos.len());
        for info in &raw_infos {
            let abs_offset = data_section_start
                .checked_add(info.relative_offset)
                .ok_or_else(|| {
                    WillametteError::GgufParse(format!("tensor \"{}\" offset overflow", info.name))
                })?;

            // We need to know byte_len. For known types we can compute it from
            // shape; for unknown types we compute it from the distance to the
            // next tensor or end of file.
            let byte_len = compute_tensor_byte_len(&info.shape, info.ggml_type);

            let end = abs_offset.checked_add(byte_len).ok_or_else(|| {
                WillametteError::GgufParse(format!("tensor \"{}\" end offset overflow", info.name))
            })?;

            if end > file_len {
                return Err(WillametteError::TensorOutOfBounds {
                    name: info.name.clone(),
                    offset: abs_offset,
                    end,
                    file_len,
                });
            }

            let tensor_data = &data[abs_offset as usize..end as usize];

            // I2_S tensors have a 32-byte trailing block (4-byte f32 scale +
            // 28-byte alignment padding). See docs/I2_S_LAYOUT.md and
            // docs/BITLINEAR_I2S_MATVEC.md for the source citations.
            let scale_data = if info.ggml_type == GgmlType::BitNetI2S {
                let scale_end = end
                    .checked_add(TensorView::I2S_TRAILING_SCALE_BLOCK_BYTES)
                    .ok_or_else(|| {
                        WillametteError::GgufParse(format!(
                            "tensor \"{}\" scale-block offset overflow",
                            info.name
                        ))
                    })?;
                if scale_end > file_len {
                    return Err(WillametteError::TensorOutOfBounds {
                        name: format!("{} (scale block)", info.name),
                        offset: end,
                        end: scale_end,
                        file_len,
                    });
                }
                Some(&data[end as usize..scale_end as usize])
            } else {
                None
            };

            tensors.push(TensorView {
                name: info.name.clone(),
                shape: info.shape.clone(),
                ggml_type: info.ggml_type,
                offset: abs_offset,
                byte_len,
                data: tensor_data,
                scale_data,
            });
        }

        Ok(GgufFile {
            version,
            tensor_count,
            metadata,
            tensors,
            alignment,
        })
    }
}

// ── helpers ──

fn align_offset(offset: u64, alignment: u64) -> u64 {
    let remainder = offset % alignment;
    if remainder == 0 {
        offset
    } else {
        offset + (alignment - remainder)
    }
}

/// Read a GGUF-encoded string: u64 length followed by that many UTF-8 bytes.
fn read_gguf_string(cur: &mut Cursor<&[u8]>) -> Result<String, String> {
    let len = cur
        .read_u64::<LittleEndian>()
        .map_err(|e| format!("string length: {}", e))?;
    if len > 1_048_576 {
        return Err(format!("string length {} exceeds 1 MiB safety limit", len));
    }
    let mut buf = vec![0u8; len as usize];
    cur.read_exact(&mut buf)
        .map_err(|e| format!("string body ({} bytes): {}", len, e))?;
    String::from_utf8(buf).map_err(|e| format!("invalid UTF-8: {}", e))
}

/// Read a single GGUF metadata value (type-tag + payload).
fn read_gguf_value(cur: &mut Cursor<&[u8]>) -> Result<GgufValue, String> {
    let raw_type = cur
        .read_u32::<LittleEndian>()
        .map_err(|e| format!("value type tag: {}", e))?;
    let vtype = GgufMetadataValueType::from_raw(raw_type);

    match vtype {
        GgufMetadataValueType::Uint8 => {
            let v = cur.read_u8().map_err(|e| format!("uint8: {}", e))?;
            Ok(GgufValue::Uint8(v))
        }
        GgufMetadataValueType::Int8 => {
            let v = cur.read_i8().map_err(|e| format!("int8: {}", e))?;
            Ok(GgufValue::Int8(v))
        }
        GgufMetadataValueType::Uint16 => {
            let v = cur
                .read_u16::<LittleEndian>()
                .map_err(|e| format!("uint16: {}", e))?;
            Ok(GgufValue::Uint16(v))
        }
        GgufMetadataValueType::Int16 => {
            let v = cur
                .read_i16::<LittleEndian>()
                .map_err(|e| format!("int16: {}", e))?;
            Ok(GgufValue::Int16(v))
        }
        GgufMetadataValueType::Uint32 => {
            let v = cur
                .read_u32::<LittleEndian>()
                .map_err(|e| format!("uint32: {}", e))?;
            Ok(GgufValue::Uint32(v))
        }
        GgufMetadataValueType::Int32 => {
            let v = cur
                .read_i32::<LittleEndian>()
                .map_err(|e| format!("int32: {}", e))?;
            Ok(GgufValue::Int32(v))
        }
        GgufMetadataValueType::Float32 => {
            let v = cur
                .read_f32::<LittleEndian>()
                .map_err(|e| format!("float32: {}", e))?;
            Ok(GgufValue::Float32(v))
        }
        GgufMetadataValueType::Bool => {
            let v = cur.read_u8().map_err(|e| format!("bool: {}", e))?;
            Ok(GgufValue::Bool(v != 0))
        }
        GgufMetadataValueType::String => {
            let s = read_gguf_string(cur)?;
            Ok(GgufValue::Str(s))
        }
        GgufMetadataValueType::Array => {
            // Array: element type (u32) + count (u64) + count × value
            let elem_type_raw = cur
                .read_u32::<LittleEndian>()
                .map_err(|e| format!("array element type: {}", e))?;
            let count = cur
                .read_u64::<LittleEndian>()
                .map_err(|e| format!("array count: {}", e))?;
            if count > 10_000_000 {
                return Err(format!("array count {} exceeds 10M safety limit", count));
            }
            let elem_type = GgufMetadataValueType::from_raw(elem_type_raw);
            let mut arr = Vec::with_capacity(count as usize);
            for i in 0..count {
                let v = read_gguf_typed_value(cur, elem_type)
                    .map_err(|e| format!("array[{}]: {}", i, e))?;
                arr.push(v);
            }
            Ok(GgufValue::Array(arr))
        }
        GgufMetadataValueType::Uint64 => {
            let v = cur
                .read_u64::<LittleEndian>()
                .map_err(|e| format!("uint64: {}", e))?;
            Ok(GgufValue::Uint64(v))
        }
        GgufMetadataValueType::Int64 => {
            let v = cur
                .read_i64::<LittleEndian>()
                .map_err(|e| format!("int64: {}", e))?;
            Ok(GgufValue::Int64(v))
        }
        GgufMetadataValueType::Float64 => {
            let v = cur
                .read_f64::<LittleEndian>()
                .map_err(|e| format!("float64: {}", e))?;
            Ok(GgufValue::Float64(v))
        }
        GgufMetadataValueType::Unknown(t) => Err(format!("unknown metadata value type: {}", t)),
    }
}

/// Read a value whose type tag is already known (used for array elements).
fn read_gguf_typed_value(
    cur: &mut Cursor<&[u8]>,
    vtype: GgufMetadataValueType,
) -> Result<GgufValue, String> {
    match vtype {
        GgufMetadataValueType::Uint8 => {
            Ok(GgufValue::Uint8(cur.read_u8().map_err(|e| e.to_string())?))
        }
        GgufMetadataValueType::Int8 => {
            Ok(GgufValue::Int8(cur.read_i8().map_err(|e| e.to_string())?))
        }
        GgufMetadataValueType::Uint16 => Ok(GgufValue::Uint16(
            cur.read_u16::<LittleEndian>().map_err(|e| e.to_string())?,
        )),
        GgufMetadataValueType::Int16 => Ok(GgufValue::Int16(
            cur.read_i16::<LittleEndian>().map_err(|e| e.to_string())?,
        )),
        GgufMetadataValueType::Uint32 => Ok(GgufValue::Uint32(
            cur.read_u32::<LittleEndian>().map_err(|e| e.to_string())?,
        )),
        GgufMetadataValueType::Int32 => Ok(GgufValue::Int32(
            cur.read_i32::<LittleEndian>().map_err(|e| e.to_string())?,
        )),
        GgufMetadataValueType::Float32 => Ok(GgufValue::Float32(
            cur.read_f32::<LittleEndian>().map_err(|e| e.to_string())?,
        )),
        GgufMetadataValueType::Bool => Ok(GgufValue::Bool(
            cur.read_u8().map_err(|e| e.to_string())? != 0,
        )),
        GgufMetadataValueType::String => {
            let s = read_gguf_string(cur)?;
            Ok(GgufValue::Str(s))
        }
        GgufMetadataValueType::Uint64 => Ok(GgufValue::Uint64(
            cur.read_u64::<LittleEndian>().map_err(|e| e.to_string())?,
        )),
        GgufMetadataValueType::Int64 => Ok(GgufValue::Int64(
            cur.read_i64::<LittleEndian>().map_err(|e| e.to_string())?,
        )),
        GgufMetadataValueType::Float64 => Ok(GgufValue::Float64(
            cur.read_f64::<LittleEndian>().map_err(|e| e.to_string())?,
        )),
        GgufMetadataValueType::Array => {
            // Nested arrays
            let elem_type_raw = cur.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
            let count = cur.read_u64::<LittleEndian>().map_err(|e| e.to_string())?;
            if count > 10_000_000 {
                return Err(format!("nested array count {} too large", count));
            }
            let elem_type = GgufMetadataValueType::from_raw(elem_type_raw);
            let mut arr = Vec::with_capacity(count as usize);
            for _ in 0..count {
                arr.push(read_gguf_typed_value(cur, elem_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        GgufMetadataValueType::Unknown(t) => {
            Err(format!("unknown type tag {} in array element", t))
        }
    }
}

/// Compute byte length for a tensor given its shape and ggml type.
///
/// For block-quantised types (Q4_0, Q8_0, BitNet I2_S, etc.) the size is
/// computed as `n_elements / block_size * bytes_per_block`.
///
/// For types whose block layout is not yet implemented, we return 0 and the
/// caller should use inter-tensor offsets instead (or return an error).
fn compute_tensor_byte_len(shape: &[u64], ggml_type: GgmlType) -> u64 {
    let n_elements: u64 = shape.iter().product();
    if n_elements == 0 {
        return 0;
    }

    match ggml_type {
        // ── Scalar types ──
        GgmlType::F32 => n_elements * 4,
        GgmlType::F16 => n_elements * 2,
        GgmlType::BF16 => n_elements * 2,
        GgmlType::F64 => n_elements * 8,
        GgmlType::I8 => n_elements,
        GgmlType::I16 => n_elements * 2,
        GgmlType::I32 => n_elements * 4,
        GgmlType::I64 => n_elements * 8,

        // ── Standard quantised types (block_size, bytes_per_block) ──
        GgmlType::Q4_0 => n_elements / 32 * 18, // 32 elem, 2 + 16 bytes
        GgmlType::Q4_1 => n_elements / 32 * 20, // 32 elem, 2+2+16 bytes
        GgmlType::Q5_0 => n_elements / 32 * 22,
        GgmlType::Q5_1 => n_elements / 32 * 24,
        GgmlType::Q8_0 => n_elements / 32 * 34, // 32 elem, 2 + 32 bytes
        GgmlType::Q8_1 => n_elements / 32 * 40,
        GgmlType::Q2K => n_elements / 256 * 84,
        GgmlType::Q3K => n_elements / 256 * 110,
        GgmlType::Q4K => n_elements / 256 * 144,
        GgmlType::Q5K => n_elements / 256 * 176,
        GgmlType::Q6K => n_elements / 256 * 210,
        GgmlType::Q8K => n_elements / 256 * 292,

        // ── BitNet I2_S: 128 ternary elements per 32-byte block ──
        // Each element uses 2 bits → 128 * 2 bits = 256 bits = 32 bytes per block.
        GgmlType::BitNetI2S => n_elements / 128 * 32,

        // ── BitNet I8_S: 1 byte per element (int8 activations) ──
        GgmlType::BitNetI8S => n_elements,

        // ── BitNet TL1/TL2: layout sizes not confirmed yet.
        // Return a best-effort estimate; will be validated on load.
        GgmlType::BitNetTL1 => n_elements / 128 * 32,
        GgmlType::BitNetTL2 => n_elements / 128 * 32,

        // ── Everything else: we can't compute, return 0 ──
        _ => 0,
    }
}
