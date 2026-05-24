use crate::error::WillametteError;
use crate::gguf::types::GgmlType;

/// A zero-copy view into a single tensor stored inside a GGUF file.
///
/// The `data` field is a byte slice that points directly into the memory-mapped
/// file. **No copying, no dequantisation, no fake weights.** The slice represents
/// the raw packed representation exactly as stored on disk.
///
/// For I2_S tensors specifically, `data` covers ONLY the packed 2-bit codes
/// area (`byte_len == n_elements / 4`). The 32-byte trailing block that holds
/// the per-tensor `i2_scale` (f32) plus 28 bytes of alignment padding lives at
/// file offset `[offset + byte_len .. offset + byte_len + 32)` — outside this
/// view. See `docs/I2_S_LAYOUT.md` for the upstream citations.
///
/// To perform computation with this tensor, the caller must implement the
/// appropriate unpack / matmul kernel for the tensor's `ggml_type`. If the type
/// is not yet supported, the kernel must return
/// `WillametteError::UnsupportedTensorType` or `WillametteError::NotImplemented`.
#[derive(Debug)]
pub struct TensorView<'a> {
    /// Tensor name as stored in the GGUF tensor info section
    /// (e.g. "blk.0.attn_q.weight").
    pub name: String,

    /// Shape dimensions in GGUF order (innermost-first, i.e. [cols, rows] for
    /// a 2-D weight matrix).
    pub shape: Vec<u64>,

    /// GGML quantisation / data type.
    pub ggml_type: GgmlType,

    /// Byte offset of this tensor's data from the start of the file.
    pub offset: u64,

    /// Number of bytes occupied by this tensor's primary data area.
    ///
    /// For I2_S this is the **packed-codes area only** (`n_elements / 4`);
    /// the trailing 32-byte scale block is NOT included. Use
    /// `i2s_total_disk_bytes()` for the full on-disk footprint.
    pub byte_len: u64,

    /// Zero-copy reference into the memory-mapped file.
    /// This is the raw, packed tensor data — NOT dequantised.
    pub data: &'a [u8],

    /// For I2_S tensors: the 32-byte trailing block immediately after
    /// `data`. The first 4 bytes are an `f32` little-endian per-tensor
    /// scale (`i2_scale` in the BitNet fork). The remaining 28 bytes are
    /// padding (alignment for the next tensor) and MUST NOT be read as
    /// data. `None` for non-I2_S tensors. Populated by
    /// `GgufFile::parse`. See `docs/BITLINEAR_I2S_MATVEC.md` §4.
    pub scale_data: Option<&'a [u8]>,
}

impl<'a> TensorView<'a> {
    /// Total number of logical elements (product of all shape dimensions).
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Pretty-print for the `inspect` CLI command.
    pub fn summary(&self) -> String {
        let shape_str: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
        format!(
            "{:<50} dtype={:<16} shape=[{}]  offset=0x{:X}  bytes={}",
            self.name,
            self.ggml_type.name(),
            shape_str.join(", "),
            self.offset,
            self.byte_len,
        )
    }

    // ── I2_S layout helpers (Stage 3) ──────────────────────────────────────
    //
    // The constants and functions in this section are pure layout math derived
    // from `docs/I2_S_LAYOUT.md`. They do NOT unpack ternary codes, do NOT
    // dereference the scale, and do NOT produce any float outputs. They are
    // here only so Stage 4 (forward pass) can rely on a single source of truth
    // for the I2_S byte map.

    /// I2_S — number of ternary elements packed in one on-disk block.
    /// Source: `dequantize_row_i2_s` in `ggml-quants.c:3897..3927` hard-codes
    /// blocks of 128 elements (4 sub-rows × 32). Independent of host CPU.
    pub const I2S_ELEMENTS_PER_BLOCK: u64 = 128;

    /// I2_S — bytes occupied by one packed block on disk.
    /// 128 elements × 2 bits / 8 bits per byte = 32 bytes.
    pub const I2S_PACKED_BYTES_PER_BLOCK: u64 = 32;

    /// I2_S — bytes appended after the packed-codes area for each tensor.
    ///
    /// First 4 bytes: `i2_scale` as little-endian f32.
    /// Remaining 28 bytes: padding so the next tensor starts on a 32-byte
    /// boundary (matches GGUF `general.alignment = 32`).
    ///
    /// Source: `ggml.c:3485..3492` (`ggml_nbytes` adds `+ 32`) and
    /// `src/ggml-bitnet-mad.cpp:142..149` (`quantize_i2_s` writes the scale
    /// at `(char*)out + n/4` and returns `... / 4 + 32`).
    pub const I2S_TRAILING_SCALE_BLOCK_BYTES: u64 = 32;

    /// For an I2_S tensor of the given shape, return the size of the
    /// packed-codes area (NOT including the trailing scale block).
    ///
    /// Equals `n_elements / 4`. Returns an error if `n_elements` is not a
    /// whole multiple of `I2S_ELEMENTS_PER_BLOCK`.
    pub fn i2s_expected_byte_len(shape: &[u64]) -> Result<u64, WillametteError> {
        let n: u64 = shape.iter().product();
        if !n.is_multiple_of(Self::I2S_ELEMENTS_PER_BLOCK) {
            return Err(WillametteError::GgufParse(format!(
                "I2_S tensor shape product {} is not a multiple of \
                 QK_I2_S={} — block-misaligned tensors are unsupported",
                n,
                Self::I2S_ELEMENTS_PER_BLOCK,
            )));
        }
        Ok(n / 4)
    }

    /// For an I2_S tensor of the given shape, return the total on-disk
    /// footprint: packed area + trailing scale block.
    ///
    /// Equals `n_elements / 4 + 32`.
    pub fn i2s_total_disk_bytes(shape: &[u64]) -> Result<u64, WillametteError> {
        Ok(Self::i2s_expected_byte_len(shape)? + Self::I2S_TRAILING_SCALE_BLOCK_BYTES)
    }

    /// For an I2_S tensor, the file offset where the f32 `i2_scale` is
    /// stored (4 bytes), immediately after this tensor's packed-codes area.
    /// Returns `None` for non-I2_S tensors.
    ///
    /// NOTE: the scale lives OUTSIDE `self.data` — to read it, dereference
    /// the underlying mmap buffer at this absolute file offset, or use
    /// `i2s_scale()` to read it directly from `scale_data`.
    pub fn i2s_scale_file_offset(&self) -> Option<u64> {
        if self.ggml_type == GgmlType::BitNetI2S {
            Some(self.offset + self.byte_len)
        } else {
            None
        }
    }

    /// For an I2_S tensor, read the per-tensor `i2_scale` (single f32)
    /// from the first 4 bytes of `scale_data`. Errors if the tensor is
    /// not I2_S, if `scale_data` is missing, or if the scale is not a
    /// finite f32.
    pub fn i2s_scale(&self) -> Result<f32, WillametteError> {
        if self.ggml_type != GgmlType::BitNetI2S {
            return Err(WillametteError::GgufParse(format!(
                "i2s_scale called on tensor {:?} which is {} (raw {}), not I2_S",
                self.name,
                self.ggml_type.name(),
                self.ggml_type.to_raw()
            )));
        }
        let block = self.scale_data.ok_or_else(|| {
            WillametteError::GgufParse(format!(
                "i2s_scale: tensor {:?} has no scale_data set (was the reader updated?)",
                self.name
            ))
        })?;
        if block.len() < 4 {
            return Err(WillametteError::GgufParse(format!(
                "i2s_scale: scale_data for {:?} too short ({} bytes, need ≥ 4)",
                self.name,
                block.len()
            )));
        }
        let bits = u32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        let scale = f32::from_bits(bits);
        if !scale.is_finite() {
            return Err(WillametteError::GgufParse(format!(
                "i2s_scale: non-finite scale ({}) for tensor {:?}",
                scale, self.name
            )));
        }
        Ok(scale)
    }

    /// Verify that the stored `byte_len` matches the layout-derived expected
    /// value for this tensor's type and shape. Only checks the types that
    /// Stage 3 has confirmed against upstream source.
    ///
    /// Returns `Ok(())` for unverified types (no claim).
    pub fn verify_byte_len(&self) -> Result<(), WillametteError> {
        match self.ggml_type {
            GgmlType::BitNetI2S => {
                let expected = Self::i2s_expected_byte_len(&self.shape)?;
                if self.byte_len != expected {
                    return Err(WillametteError::GgufParse(format!(
                        "tensor {:?} (I2_S): byte_len {} != expected {} \
                         (= n_elements / 4 = {})",
                        self.name,
                        self.byte_len,
                        expected,
                        self.n_elements() / 4,
                    )));
                }
                Ok(())
            }
            GgmlType::F32 => {
                let expected = self.n_elements() * 4;
                if self.byte_len != expected {
                    return Err(WillametteError::GgufParse(format!(
                        "tensor {:?} (F32): byte_len {} != n_elements*4 = {}",
                        self.name, self.byte_len, expected
                    )));
                }
                Ok(())
            }
            GgmlType::F16 => {
                let expected = self.n_elements() * 2;
                if self.byte_len != expected {
                    return Err(WillametteError::GgufParse(format!(
                        "tensor {:?} (F16): byte_len {} != n_elements*2 = {}",
                        self.name, self.byte_len, expected
                    )));
                }
                Ok(())
            }
            // Other types: not yet investigated against upstream source.
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i2s_block_constants() {
        assert_eq!(TensorView::I2S_ELEMENTS_PER_BLOCK, 128);
        assert_eq!(TensorView::I2S_PACKED_BYTES_PER_BLOCK, 32);
        assert_eq!(TensorView::I2S_TRAILING_SCALE_BLOCK_BYTES, 32);
        // 128 elements × 2 bits = 32 bytes exactly
        assert_eq!(
            TensorView::I2S_ELEMENTS_PER_BLOCK * 2 / 8,
            TensorView::I2S_PACKED_BYTES_PER_BLOCK
        );
    }

    #[test]
    fn i2s_expected_byte_len_matches_real_shapes() {
        // Shapes from the official ggml-model-i2_s.gguf (inspect.log).
        let cases = &[
            // (shape, expected packed bytes)
            (vec![6912u64, 2560], 4_423_680u64), // ffn_down
            (vec![2560, 6912], 4_423_680),       // ffn_up / ffn_gate
            (vec![2560, 2560], 1_638_400),       // attn_q / attn_output
            (vec![2560, 640], 409_600),          // attn_k / attn_v (GQA)
        ];
        for (shape, expected) in cases {
            let got = TensorView::i2s_expected_byte_len(shape).unwrap();
            assert_eq!(got, *expected, "shape {:?}", shape);
        }
    }

    #[test]
    fn i2s_total_disk_bytes_adds_32() {
        let shape = vec![2560u64, 640];
        let packed = TensorView::i2s_expected_byte_len(&shape).unwrap();
        let total = TensorView::i2s_total_disk_bytes(&shape).unwrap();
        assert_eq!(total, packed + 32);
    }

    #[test]
    fn i2s_block_misaligned_shape_errors() {
        // 127 is not divisible by 128
        let shape = vec![127u64];
        let result = TensorView::i2s_expected_byte_len(&shape);
        assert!(result.is_err());
    }

    #[test]
    fn i2s_scale_offset_is_none_for_other_types() {
        let v = TensorView {
            name: "x".into(),
            shape: vec![10],
            ggml_type: GgmlType::F32,
            offset: 100,
            byte_len: 40,
            data: &[],
            scale_data: None,
        };
        assert_eq!(v.i2s_scale_file_offset(), None);
    }

    #[test]
    fn i2s_scale_offset_is_offset_plus_byte_len() {
        let v = TensorView {
            name: "x".into(),
            shape: vec![128],
            ggml_type: GgmlType::BitNetI2S,
            offset: 1000,
            byte_len: 32,
            data: &[],
            scale_data: None,
        };
        assert_eq!(v.i2s_scale_file_offset(), Some(1032));
    }

    #[test]
    fn verify_byte_len_passes_for_consistent_view() {
        let v = TensorView {
            name: "test".into(),
            shape: vec![128],
            ggml_type: GgmlType::BitNetI2S,
            offset: 0,
            byte_len: 32, // = 128 / 4
            data: &[],
            scale_data: None,
        };
        assert!(v.verify_byte_len().is_ok());
    }

    #[test]
    fn verify_byte_len_fails_for_wrong_size() {
        let v = TensorView {
            name: "broken".into(),
            shape: vec![128],
            ggml_type: GgmlType::BitNetI2S,
            offset: 0,
            byte_len: 64, // wrong
            data: &[],
            scale_data: None,
        };
        assert!(v.verify_byte_len().is_err());
    }

    #[test]
    fn i2s_scale_reads_first_4_bytes_as_f32() {
        // f32 little-endian for 1.5 = 0x3FC00000 → bytes [0x00, 0x00, 0xC0, 0x3F]
        let block: [u8; 32] = {
            let mut b = [0u8; 32];
            b[..4].copy_from_slice(&1.5_f32.to_le_bytes());
            b
        };
        let v = TensorView {
            name: "i2s".into(),
            shape: vec![128],
            ggml_type: GgmlType::BitNetI2S,
            offset: 0,
            byte_len: 32,
            data: &[],
            scale_data: Some(&block),
        };
        assert_eq!(v.i2s_scale().unwrap(), 1.5);
    }

    #[test]
    fn i2s_scale_errors_for_non_i2s_type() {
        let v = TensorView {
            name: "not-i2s".into(),
            shape: vec![10],
            ggml_type: GgmlType::F32,
            offset: 0,
            byte_len: 40,
            data: &[],
            scale_data: None,
        };
        assert!(v.i2s_scale().is_err());
    }

    #[test]
    fn i2s_scale_errors_for_missing_scale_data() {
        let v = TensorView {
            name: "i2s-no-scale".into(),
            shape: vec![128],
            ggml_type: GgmlType::BitNetI2S,
            offset: 0,
            byte_len: 32,
            data: &[],
            scale_data: None,
        };
        assert!(v.i2s_scale().is_err());
    }

    #[test]
    fn i2s_scale_errors_for_nan_or_inf() {
        let mut block_nan = [0u8; 32];
        block_nan[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        let v_nan = TensorView {
            name: "nan".into(),
            shape: vec![128],
            ggml_type: GgmlType::BitNetI2S,
            offset: 0,
            byte_len: 32,
            data: &[],
            scale_data: Some(&block_nan),
        };
        assert!(v_nan.i2s_scale().is_err());

        let mut block_inf = [0u8; 32];
        block_inf[..4].copy_from_slice(&f32::INFINITY.to_le_bytes());
        let v_inf = TensorView {
            name: "inf".into(),
            shape: vec![128],
            ggml_type: GgmlType::BitNetI2S,
            offset: 0,
            byte_len: 32,
            data: &[],
            scale_data: Some(&block_inf),
        };
        assert!(v_inf.i2s_scale().is_err());
    }
}
