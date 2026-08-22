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

const MAX_TENSOR_COUNT: u64 = 1_000_000;
const MAX_METADATA_KV_COUNT: u64 = 1_000_000;
const MAX_TENSOR_DIMS: u32 = 4;
const MAX_METADATA_ARRAY_ELEMENTS: u64 = 1_000_000;
const MAX_METADATA_ARRAY_DEPTH: usize = 16;

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
#[derive(Debug, Clone, Copy)]
pub(crate) struct TensorDescriptorLocation {
    pub dtype_offset: u64,
    pub relative_offset_offset: u64,
}

pub struct GgufFile<'a> {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorView<'a>>,
    pub alignment: u64,
    pub(crate) data_section_start: u64,
    pub(crate) tensor_descriptors: Vec<TensorDescriptorLocation>,
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

        validate_count(
            "tensor_count",
            tensor_count,
            MAX_TENSOR_COUNT,
            data.len().saturating_sub(cur.position() as usize),
            24,
        )?;
        validate_count(
            "metadata_kv_count",
            metadata_kv_count,
            MAX_METADATA_KV_COUNT,
            data.len().saturating_sub(cur.position() as usize),
            13,
        )?;

        // ── 4. Metadata key-values ──
        let mut metadata = HashMap::new();
        metadata
            .try_reserve(metadata_kv_count as usize)
            .map_err(|e| WillametteError::GgufParse(format!("allocating metadata map: {}", e)))?;
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

        validate_count(
            "tensor_count",
            tensor_count,
            MAX_TENSOR_COUNT,
            data.len().saturating_sub(cur.position() as usize),
            24,
        )?;
        let mut raw_infos: Vec<RawTensorInfo> = Vec::new();
        raw_infos
            .try_reserve_exact(tensor_count as usize)
            .map_err(|e| {
                WillametteError::GgufParse(format!("allocating tensor directory: {}", e))
            })?;
        let mut tensor_descriptors = Vec::new();
        tensor_descriptors
            .try_reserve_exact(tensor_count as usize)
            .map_err(|e| {
                WillametteError::GgufParse(format!("allocating descriptor locations: {}", e))
            })?;
        for i in 0..tensor_count {
            let name = read_gguf_string(&mut cur)
                .map_err(|e| WillametteError::GgufParse(format!("tensor[{}] name: {}", i, e)))?;
            let n_dims = cur
                .read_u32::<LittleEndian>()
                .map_err(|e| WillametteError::GgufParse(format!("tensor[{}] n_dims: {}", i, e)))?;
            if n_dims > MAX_TENSOR_DIMS {
                return Err(WillametteError::GgufParse(format!(
                    "tensor[{}] n_dims {} exceeds {}",
                    i, n_dims, MAX_TENSOR_DIMS
                )));
            }
            let mut shape = Vec::new();
            shape.try_reserve_exact(n_dims as usize).map_err(|e| {
                WillametteError::GgufParse(format!("allocating tensor[{}] shape: {}", i, e))
            })?;
            for d in 0..n_dims {
                let dim = cur.read_u64::<LittleEndian>().map_err(|e| {
                    WillametteError::GgufParse(format!("tensor[{}] shape[{}]: {}", i, d, e))
                })?;
                shape.push(dim);
            }
            let dtype_offset = cur.position();
            let raw_type = cur.read_u32::<LittleEndian>().map_err(|e| {
                WillametteError::GgufParse(format!("tensor[{}] ggml_type: {}", i, e))
            })?;
            let ggml_type = GgmlType::from_raw(raw_type);

            let relative_offset_offset = cur.position();
            let relative_offset = cur
                .read_u64::<LittleEndian>()
                .map_err(|e| WillametteError::GgufParse(format!("tensor[{}] offset: {}", i, e)))?;

            raw_infos.push(RawTensorInfo {
                name,
                shape,
                ggml_type,
                relative_offset,
            });
            tensor_descriptors.push(TensorDescriptorLocation {
                dtype_offset,
                relative_offset_offset,
            });
        }

        // ── 7. Compute the absolute start of the tensor data section ──
        // After all header + metadata + tensor info entries, the data section
        // begins at the next alignment boundary.
        let header_end = cur.position();
        let data_section_start = align_offset(header_end, alignment).ok_or_else(|| {
            WillametteError::GgufParse("tensor data section offset overflow".into())
        })?;
        if !raw_infos.is_empty() && data_section_start > file_len {
            return Err(WillametteError::GgufParse(format!(
                "tensor data section starts beyond EOF: {} > {}",
                data_section_start, file_len
            )));
        }

        // ── 8. Build TensorViews ──
        let mut relative_offsets = Vec::new();
        relative_offsets
            .try_reserve_exact(raw_infos.len())
            .map_err(|e| WillametteError::GgufParse(format!("allocating tensor offsets: {}", e)))?;
        relative_offsets.extend(raw_infos.iter().map(|info| info.relative_offset));

        let mut tensors: Vec<TensorView<'a>> = Vec::new();
        tensors
            .try_reserve_exact(raw_infos.len())
            .map_err(|e| WillametteError::GgufParse(format!("allocating tensor views: {}", e)))?;
        for (info_index, info) in raw_infos.into_iter().enumerate() {
            let abs_offset = data_section_start
                .checked_add(info.relative_offset)
                .ok_or_else(|| {
                    WillametteError::GgufParse(format!("tensor \"{}\" offset overflow", info.name))
                })?;

            if relative_offsets
                .iter()
                .enumerate()
                .any(|(other_index, offset)| {
                    other_index != info_index && *offset == info.relative_offset
                })
            {
                return Err(WillametteError::GgufParse(format!(
                    "tensor \"{}\" shares data offset {} with another tensor",
                    info.name, info.relative_offset
                )));
            }
            let next_abs_offset = relative_offsets
                .iter()
                .copied()
                .filter(|offset| *offset > info.relative_offset)
                .min()
                .map(|offset| {
                    data_section_start.checked_add(offset).ok_or_else(|| {
                        WillametteError::GgufParse(format!(
                            "tensor \"{}\" next offset overflow",
                            info.name
                        ))
                    })
                })
                .transpose()?;

            let byte_len = compute_tensor_byte_len(&info.shape, info.ggml_type)
                .map_err(|e| {
                    WillametteError::GgufParse(format!("tensor \"{}\": {}", info.name, e))
                })?
                .ok_or_else(|| {
                    WillametteError::GgufParse(format!(
                        "tensor \"{}\" has unsupported type {}; byte length is unknown",
                        info.name, info.ggml_type
                    ))
                })?;

            let end = abs_offset.checked_add(byte_len).ok_or_else(|| {
                WillametteError::GgufParse(format!("tensor \"{}\" end offset overflow", info.name))
            })?;

            if end > file_len {
                return Err(WillametteError::TensorOutOfBounds {
                    name: info.name,
                    offset: abs_offset,
                    end,
                    file_len,
                });
            }
            if next_abs_offset.is_some_and(|next| end > next) {
                return Err(WillametteError::GgufParse(format!(
                    "tensor \"{}\" range {}..{} overlaps next tensor at {}",
                    info.name,
                    abs_offset,
                    end,
                    next_abs_offset.unwrap()
                )));
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
                if next_abs_offset.is_some_and(|next| scale_end > next) {
                    return Err(WillametteError::GgufParse(format!(
                        "I2_S tensor \"{}\" scale block ends at {}, overlapping next tensor at {}",
                        info.name,
                        scale_end,
                        next_abs_offset.unwrap()
                    )));
                }
                Some(&data[end as usize..scale_end as usize])
            } else {
                None
            };

            tensors.push(TensorView {
                name: info.name,
                shape: info.shape,
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
            data_section_start,
            tensor_descriptors,
        })
    }
}

// ── helpers ──

fn align_offset(offset: u64, alignment: u64) -> Option<u64> {
    let remainder = offset % alignment;
    if remainder == 0 {
        Some(offset)
    } else {
        offset.checked_add(alignment - remainder)
    }
}

fn validate_count(
    name: &str,
    count: u64,
    limit: u64,
    remaining_bytes: usize,
    minimum_entry_bytes: u64,
) -> Result<(), WillametteError> {
    if count > limit {
        return Err(WillametteError::GgufParse(format!(
            "{} {} exceeds safety limit {}",
            name, count, limit
        )));
    }
    let minimum_bytes = count.checked_mul(minimum_entry_bytes).ok_or_else(|| {
        WillametteError::GgufParse(format!("{} minimum byte count overflow", name))
    })?;
    if minimum_bytes > remaining_bytes as u64 {
        return Err(WillametteError::GgufParse(format!(
            "{} {} cannot fit in {} remaining bytes",
            name, count, remaining_bytes
        )));
    }
    Ok(())
}

/// Read a GGUF-encoded string: u64 length followed by that many UTF-8 bytes.
fn read_gguf_string(cur: &mut Cursor<&[u8]>) -> Result<String, String> {
    let len = cur
        .read_u64::<LittleEndian>()
        .map_err(|e| format!("string length: {}", e))?;
    if len > 1_048_576 {
        return Err(format!("string length {} exceeds 1 MiB safety limit", len));
    }
    let mut buf = Vec::new();
    buf.try_reserve_exact(len as usize)
        .map_err(|e| format!("allocating string body ({} bytes): {}", len, e))?;
    buf.resize(len as usize, 0);
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
            let elem_type = GgufMetadataValueType::from_raw(elem_type_raw);
            validate_array(cur, elem_type, count, 1)?;
            let mut arr = Vec::new();
            arr.try_reserve_exact(count as usize)
                .map_err(|e| format!("allocating array of {} elements: {}", count, e))?;
            for i in 0..count {
                let v = read_gguf_typed_value(cur, elem_type, 1)
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
    depth: usize,
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
            let elem_type = GgufMetadataValueType::from_raw(elem_type_raw);
            let next_depth = depth
                .checked_add(1)
                .ok_or_else(|| "metadata array nesting depth overflow".to_string())?;
            validate_array(cur, elem_type, count, next_depth)?;
            let mut arr = Vec::new();
            arr.try_reserve_exact(count as usize)
                .map_err(|e| format!("allocating nested array of {} elements: {}", count, e))?;
            for _ in 0..count {
                arr.push(read_gguf_typed_value(cur, elem_type, next_depth)?);
            }
            Ok(GgufValue::Array(arr))
        }
        GgufMetadataValueType::Unknown(t) => {
            Err(format!("unknown type tag {} in array element", t))
        }
    }
}

fn validate_array(
    cur: &Cursor<&[u8]>,
    elem_type: GgufMetadataValueType,
    count: u64,
    depth: usize,
) -> Result<(), String> {
    if depth > MAX_METADATA_ARRAY_DEPTH {
        return Err(format!(
            "metadata array nesting depth {} exceeds {}",
            depth, MAX_METADATA_ARRAY_DEPTH
        ));
    }
    if count > MAX_METADATA_ARRAY_ELEMENTS {
        return Err(format!(
            "array count {} exceeds safety limit {}",
            count, MAX_METADATA_ARRAY_ELEMENTS
        ));
    }
    let element_bytes = match elem_type {
        GgufMetadataValueType::Uint8
        | GgufMetadataValueType::Int8
        | GgufMetadataValueType::Bool => 1,
        GgufMetadataValueType::Uint16 | GgufMetadataValueType::Int16 => 2,
        GgufMetadataValueType::Uint32
        | GgufMetadataValueType::Int32
        | GgufMetadataValueType::Float32 => 4,
        GgufMetadataValueType::Uint64
        | GgufMetadataValueType::Int64
        | GgufMetadataValueType::Float64
        | GgufMetadataValueType::String => 8,
        GgufMetadataValueType::Array => 12,
        GgufMetadataValueType::Unknown(t) => {
            return Err(format!("unknown type tag {} in array element", t));
        }
    };
    let minimum_bytes = count
        .checked_mul(element_bytes)
        .ok_or_else(|| "metadata array minimum byte count overflow".to_string())?;
    let remaining_bytes = cur.get_ref().len().saturating_sub(cur.position() as usize) as u64;
    if minimum_bytes > remaining_bytes {
        return Err(format!(
            "array of {} elements cannot fit in {} remaining bytes",
            count, remaining_bytes
        ));
    }
    Ok(())
}

/// Compute byte length for a tensor given its shape and ggml type.
///
/// For block-quantised types (Q4_0, Q8_0, BitNet I2_S, etc.) the size is
/// computed as `n_elements / block_size * bytes_per_block`.
///
/// For types whose block layout is not implemented, returns `None` so the
/// caller can use raw inter-tensor offsets.
fn compute_tensor_byte_len(shape: &[u64], ggml_type: GgmlType) -> Result<Option<u64>, String> {
    match ggml_type {
        GgmlType::Q4_0 => {
            return TensorView::q4_0_expected_byte_len(shape)
                .map(Some)
                .map_err(|error| error.to_string());
        }
        GgmlType::Q6K => {
            return TensorView::q6k_expected_byte_len(shape)
                .map(Some)
                .map_err(|error| error.to_string());
        }
        GgmlType::Q4K => {
            return TensorView::q4k_expected_byte_len(shape)
                .map(Some)
                .map_err(|error| error.to_string());
        }
        GgmlType::Q8_0 => {
            return TensorView::q8_0_expected_byte_len(shape)
                .map(Some)
                .map_err(|error| error.to_string());
        }
        _ => {}
    }
    let n_elements = shape.iter().try_fold(1u64, |elements, dim| {
        elements.checked_mul(*dim).ok_or_else(|| {
            format!(
                "shape product overflow while multiplying by dimension {}",
                dim
            )
        })
    })?;
    if n_elements == 0 {
        return Ok(Some(0));
    }

    let layout = match ggml_type {
        // ── Scalar types ──
        GgmlType::F32 => Some((1, 4)),
        GgmlType::F16 => Some((1, 2)),
        GgmlType::BF16 => Some((1, 2)),
        GgmlType::F64 => Some((1, 8)),
        GgmlType::I8 => Some((1, 1)),
        GgmlType::I16 => Some((1, 2)),
        GgmlType::I32 => Some((1, 4)),
        GgmlType::I64 => Some((1, 8)),

        // ── Standard quantised types (block_size, bytes_per_block) ──
        GgmlType::Q4_1 => Some((32, 20)),
        GgmlType::Q5_0 => Some((32, 22)),
        GgmlType::Q5_1 => Some((32, 24)),
        GgmlType::Q8_1 => Some((32, 40)),
        GgmlType::Q2K => Some((256, 84)),
        GgmlType::Q3K => Some((256, 110)),
        GgmlType::Q5K => Some((256, 176)),
        GgmlType::Q8K => Some((256, 292)),

        // ── BitNet I2_S: 128 ternary elements per 32-byte block ──
        // Each element uses 2 bits → 128 * 2 bits = 256 bits = 32 bytes per block.
        GgmlType::BitNetI2S => Some((128, 32)),

        // ── BitNet I8_S: 1 byte per element (int8 activations) ──
        GgmlType::BitNetI8S => Some((1, 1)),

        // ── Everything else: infer from raw tensor offsets ──
        _ => None,
    };

    let Some((block_size, bytes_per_block)) = layout else {
        return Ok(None);
    };
    if n_elements % block_size != 0 {
        return Err(format!(
            "element count {} is not divisible by block size {} for type {}",
            n_elements, block_size, ggml_type
        ));
    }
    let blocks = n_elements / block_size;
    blocks
        .checked_mul(bytes_per_block)
        .map(Some)
        .ok_or_else(|| format!("byte length overflow for type {}", ggml_type))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_u32(buf: &mut Vec<u8>, value: u32) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(buf: &mut Vec<u8>, value: u64) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(buf: &mut Vec<u8>, value: &str) {
        push_u64(buf, value.len() as u64);
        buf.extend_from_slice(value.as_bytes());
    }

    fn header(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
        let mut buf = Vec::new();
        push_u32(&mut buf, GGUF_MAGIC);
        push_u32(&mut buf, 3);
        push_u64(&mut buf, tensor_count);
        push_u64(&mut buf, metadata_count);
        buf
    }

    fn build_gguf(infos: &[(&str, &[u64], u32, u64)], tensor_data: &[u8]) -> Vec<u8> {
        let mut buf = header(infos.len() as u64, 0);
        for (name, shape, ggml_type, offset) in infos {
            push_string(&mut buf, name);
            push_u32(&mut buf, shape.len() as u32);
            for dim in *shape {
                push_u64(&mut buf, *dim);
            }
            push_u32(&mut buf, *ggml_type);
            push_u64(&mut buf, *offset);
        }
        while !buf.len().is_multiple_of(GGUF_DEFAULT_ALIGNMENT as usize) {
            buf.push(0);
        }
        buf.extend_from_slice(tensor_data);
        buf
    }

    fn parse_error(data: &[u8]) -> String {
        GgufFile::parse(data).err().unwrap().to_string()
    }

    #[test]
    fn rejects_tensor_with_unknown_byte_layout() {
        let data = build_gguf(&[("unknown", &[7], 999, 0)], &[1, 2, 3, 4, 5, 6, 7]);

        let error = parse_error(&data);
        assert!(error.contains("unsupported type"), "{error}");
        assert!(error.contains("byte length is unknown"), "{error}");
    }

    #[test]
    fn rejects_overlapping_known_tensors() {
        let data = build_gguf(&[("first", &[2], 0, 0), ("second", &[1], 0, 4)], &[0; 8]);

        let error = parse_error(&data);
        assert!(error.contains("overlaps next tensor"), "{error}");
    }

    #[test]
    fn rejects_i2s_scale_block_overlapping_next_tensor() {
        let data = build_gguf(&[("i2s", &[128], 36, 0), ("next", &[1], 0, 48)], &[0; 64]);

        let error = parse_error(&data);
        assert!(error.contains("scale block ends"), "{error}");
        assert!(error.contains("overlapping next tensor"), "{error}");
    }

    #[test]
    fn rejects_shape_product_overflow() {
        let data = build_gguf(&[("bad", &[u64::MAX, 2], 24, 0)], &[]);

        let error = parse_error(&data);
        assert!(error.contains("shape product overflow"), "{error}");
    }

    #[test]
    fn rejects_tensor_byte_length_overflow() {
        let data = build_gguf(&[("bad", &[u64::MAX], 28, 0)], &[]);

        let error = parse_error(&data);
        assert!(error.contains("byte length overflow"), "{error}");
    }

    #[test]
    fn rejects_incomplete_quantized_block() {
        let data = build_gguf(&[("bad", &[31], 2, 0)], &[]);

        let error = parse_error(&data);
        assert!(error.contains("Q4_0 row length 31"), "{error}");
    }

    #[test]
    fn rejects_q8_0_row_with_only_total_block_alignment() {
        let data = build_gguf(&[("bad", &[16, 2], 8, 0)], &[0; 34]);
        let error = parse_error(&data);
        assert!(error.contains("Q8_0 row length 16"), "{error}");
    }

    #[test]
    fn rejects_q4_0_row_with_only_total_block_alignment() {
        let data = build_gguf(&[("bad", &[16, 2], 2, 0)], &[0; 18]);
        let error = parse_error(&data);
        assert!(error.contains("Q4_0 row length 16"), "{error}");
    }

    #[test]
    fn rejects_q4k_row_with_only_total_block_alignment() {
        let data = build_gguf(&[("bad", &[128, 2], 12, 0)], &[0; 144]);
        let error = parse_error(&data);
        assert!(error.contains("Q4_K row length 128"), "{error}");
    }

    #[test]
    fn rejects_excessive_header_counts() {
        let tensor_error = parse_error(&header(MAX_TENSOR_COUNT + 1, 0));
        assert!(tensor_error.contains("tensor_count"), "{tensor_error}");
        assert!(tensor_error.contains("safety limit"), "{tensor_error}");

        let metadata_error = parse_error(&header(0, MAX_METADATA_KV_COUNT + 1));
        assert!(
            metadata_error.contains("metadata_kv_count"),
            "{metadata_error}"
        );
        assert!(metadata_error.contains("safety limit"), "{metadata_error}");
    }

    #[test]
    fn rejects_excessive_tensor_dimensions() {
        let mut data = header(1, 0);
        push_string(&mut data, "bad");
        push_u32(&mut data, MAX_TENSOR_DIMS + 1);
        data.resize(data.len() + 24, 0);

        let error = parse_error(&data);
        assert!(error.contains("n_dims 5 exceeds 4"), "{error}");
    }

    #[test]
    fn rejects_excessive_metadata_array_count_before_allocation() {
        let mut data = header(0, 1);
        push_string(&mut data, "array");
        push_u32(&mut data, 9);
        push_u32(&mut data, 0);
        push_u64(&mut data, MAX_METADATA_ARRAY_ELEMENTS + 1);

        let error = parse_error(&data);
        assert!(error.contains("array count"), "{error}");
        assert!(error.contains("safety limit"), "{error}");
    }
}
