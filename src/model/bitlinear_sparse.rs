//! Prototype: sparsity-aware BitLinear matvec.
//!
//! BitNet b1.58 weights are ternary, and ~42 % of them are exactly 0
//! (measured on the official 2B via `willamette analyze`). A zero
//! weight contributes nothing to the dot product, so in principle we
//! can skip it. This module builds a CSR-like sparse view (non-zero
//! column index + sign per row) offline, then runs a scalar matvec
//! over only the non-zeros.
//!
//! The open question this prototype answers experimentally: does
//! skipping 42 % of the work beat the dense i8 SIMD kernel, given that
//! sparse access is irregular (no 16-wide SSE2 lanes, scalar
//! gather)? `willamette bench` times both for the comparison.
//!
//! This is measurement scaffolding, not (yet) a production path.

// Row-indexed sparse loops read clearer with explicit `j` indexing
// into the CSR row_starts than with iterator zips — same call as the
// dense kernels.
#![allow(clippy::needless_range_loop)]

use crate::error::WillametteError;
use crate::gguf::tensor::TensorView;
use crate::gguf::types::GgmlType;
use crate::model::bitlinear::ternary_from_code;

const QK_I2_S: usize = 128;
const PACKED_BYTES_PER_BLOCK: usize = 32;

/// CSR-like sparse representation of one I2_S BitLinear weight: for
/// each output row, the column indices and signs (`+1` / `-1`) of its
/// non-zero ternary entries. Zeros are dropped.
pub struct SparseWeight {
    pub out_dim: usize,
    pub in_dim: usize,
    pub scale: f32,
    /// `row_starts[j] .. row_starts[j+1]` indexes into `cols` / `signs`.
    pub row_starts: Vec<u32>,
    pub cols: Vec<u16>,
    pub signs: Vec<i8>,
}

impl SparseWeight {
    /// Build the sparse view from a packed I2_S tensor. This is the
    /// "offline preprocessing" step (done once, would live in
    /// willamette-prep) — drops every zero weight.
    pub fn from_i2s(weight: &TensorView<'_>) -> Result<Self, WillametteError> {
        if weight.ggml_type != GgmlType::BitNetI2S {
            return Err(WillametteError::UnsupportedTensorType(
                weight.ggml_type.to_raw(),
            ));
        }
        if weight.shape.len() != 2 {
            return Err(WillametteError::GgufParse(format!(
                "sparse from_i2s: weight {:?} not 2-D",
                weight.name
            )));
        }
        let in_dim = weight.shape[0] as usize;
        let out_dim = weight.shape[1] as usize;
        if in_dim == 0 || !in_dim.is_multiple_of(QK_I2_S) {
            return Err(WillametteError::GgufParse(format!(
                "sparse from_i2s: in_dim {} not a positive multiple of {}",
                in_dim, QK_I2_S
            )));
        }
        if in_dim > u16::MAX as usize {
            return Err(WillametteError::GgufParse(format!(
                "sparse from_i2s: in_dim {} exceeds u16 column index",
                in_dim
            )));
        }
        let scale = weight.i2s_scale()?;
        let packed = weight.data;
        let bytes_per_row = in_dim / 4;
        let blocks_per_row = in_dim / QK_I2_S;

        let mut row_starts = Vec::with_capacity(out_dim + 1);
        let mut cols: Vec<u16> = Vec::new();
        let mut signs: Vec<i8> = Vec::new();
        row_starts.push(0);

        for j in 0..out_dim {
            let row_offset = j * bytes_per_row;
            for bk in 0..blocks_per_row {
                let block_offset = row_offset + bk * PACKED_BYTES_PER_BLOCK;
                let col_base = bk * QK_I2_S;
                for gp in 0..PACKED_BYTES_PER_BLOCK {
                    let b = packed[block_offset + gp];
                    // Same column-stride-32 mapping as the dense path.
                    for (shift, sub) in [(6u8, 0usize), (4, 32), (2, 64), (0, 96)] {
                        let t = ternary_from_code((b >> shift) & 0b11);
                        if t != 0 {
                            cols.push((col_base + sub + gp) as u16);
                            signs.push(t);
                        }
                    }
                }
            }
            row_starts.push(cols.len() as u32);
        }

        Ok(Self {
            out_dim,
            in_dim,
            scale,
            row_starts,
            cols,
            signs,
        })
    }

    /// Total non-zero count — for reporting the realised sparsity.
    pub fn nnz(&self) -> usize {
        self.cols.len()
    }
}

/// Sparse matvec against an int8 activation vector. Scalar: walks only
/// the non-zero entries per row. `out[j] = scale * input_scale *
/// Σ sign·act[col]`.
pub fn sparse_matvec_i8(
    sw: &SparseWeight,
    act_i8: &[i8],
    input_scale: f32,
    out: &mut [f32],
) -> Result<(), WillametteError> {
    if act_i8.len() != sw.in_dim {
        return Err(WillametteError::GgufParse(format!(
            "sparse_matvec_i8: act.len()={} != in_dim={}",
            act_i8.len(),
            sw.in_dim
        )));
    }
    if out.len() != sw.out_dim {
        return Err(WillametteError::GgufParse(format!(
            "sparse_matvec_i8: out.len()={} != out_dim={}",
            out.len(),
            sw.out_dim
        )));
    }
    let combined = sw.scale * input_scale;
    for j in 0..sw.out_dim {
        let start = sw.row_starts[j] as usize;
        let end = sw.row_starts[j + 1] as usize;
        let mut sum: i32 = 0;
        for idx in start..end {
            let c = sw.cols[idx] as usize;
            // sign is +1 / -1 → branchless add/sub.
            sum += (sw.signs[idx] as i32) * (act_i8[c] as i32);
        }
        out[j] = combined * sum as f32;
    }
    Ok(())
}

/// Absmax-per-vector int8 quantisation — same as the SSE2 i8 path.
/// Local copy to keep this prototype self-contained.
pub fn quantize_input_absmax_i8(input: &[f32], out: &mut [i8]) -> f32 {
    let mut max_abs = 0.0_f32;
    for &v in input {
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    if max_abs == 0.0 {
        for slot in out.iter_mut() {
            *slot = 0;
        }
        return 1.0;
    }
    let inv_scale = 127.0_f32 / max_abs;
    for (i, &v) in input.iter().enumerate() {
        out[i] = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
    }
    max_abs / 127.0
}
