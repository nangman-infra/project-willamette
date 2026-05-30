// Inner loops are clearer with explicit index variables.
#![allow(clippy::needless_range_loop)]

//! Scalar LUT BitLinear matvec — step 1 prototype per
//! [`docs/LUT_KERNEL_RFC.md`](../../docs/LUT_KERNEL_RFC.md).
//!
//! ## What this is
//!
//! A pure-Rust (no SIMD) BitLinear matvec that replaces the four
//! per-byte ternary accumulates of [`bitlinear_i2s_matvec_f32_scalar`]
//! with one table lookup per byte. The table is built once per
//! `(block, gp)` pair and amortised across the `out_dim` row scan.
//!
//! ## Why this is RFC step 1
//!
//! This is the **minimum implementation that lets us measure** whether
//! the table-lookup approach earns its keep on our two humble hosts
//! (antix1 SSE2, mbp2012 SSSE3+/AVX). It has no SIMD, no special
//! `pshufb` shortcut — that's RFC step 4. The gate this prototype
//! needs to clear is **≥ 1.3× faster than `bitlinear_i2s_matvec_f32_scalar`**
//! on at least one of antix1 or mbp2012, with byte-identical output.
//!
//! Below 1.3× → recorded as a negative result the same way the KV i4
//! prototypes were; the RFC closes without merging deeper.
//!
//! ## Table shape
//!
//! `table[byte] = Σ(ternary(code_k) · input[col_k])` where `byte` packs
//! four 2-bit codes (`c0 c1 c2 c3`) and `col_k = col_base + gp + 32*k`
//! — the same column-stride-32 map [`bitlinear_i2s_matvec_f32_scalar`]
//! uses. 256 entries × 4 B = **1 KiB**, fits L1 even on Pentium-M
//! (16 KiB L1d).
//!
//! Build cost per `(block, gp)`: with ternary values in {-1, 0, +1},
//! the build reduces to 4 array lookups + 3 additions per byte (no
//! multiplications). The pre-built per-position 4-element arrays
//! (`t0[c]`, `t1[c]`, `t2[c]`, `t3[c]`) cost 4 conditional writes each.
//!
//! Compiled only behind `--cfg willamette_lut` so a default build is
//! untouched by this prototype.

use crate::error::WillametteError;
use crate::gguf::tensor::TensorView;
use crate::gguf::types::GgmlType;

const QK_I2_S: usize = 128;
const PACKED_BYTES_PER_BLOCK: usize = 32;

/// Scalar LUT BitLinear matvec. Numerically equivalent to
/// [`crate::model::bitlinear::bitlinear_i2s_matvec_f32_scalar`] — i.e.
/// produces the same `output` to within f32 reassociation noise
/// (sum-of-products is reordered, so bit-equality is not claimed).
///
/// # Errors
///
/// Same error surface as the scalar reference: wrong `ggml_type`,
/// non-2-D weight, mismatched `input`/`output` lengths, or `in_dim`
/// not a positive multiple of `QK_I2_S = 128`.
pub fn bitlinear_i2s_matvec_f32_lut_scalar(
    weight: &TensorView<'_>,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), WillametteError> {
    validate(weight, input, output)?;

    let in_dim = weight.shape[0] as usize;
    let out_dim = weight.shape[1] as usize;
    let bytes_per_row = in_dim / 4;
    let blocks_per_row = in_dim / QK_I2_S;
    let scale = weight.i2s_scale()?;
    let packed = weight.data;

    for j in 0..out_dim {
        output[j] = 0.0;
    }

    // 256-entry table reused across (block, gp) pairs.
    let mut table = [0.0_f32; 256];
    // Per-code mini-tables — index 0,1,2,3 maps to ternary {-1, 0, +1, 0}.
    let mut t0 = [0.0_f32; 4];
    let mut t1 = [0.0_f32; 4];
    let mut t2 = [0.0_f32; 4];
    let mut t3 = [0.0_f32; 4];

    for bk in 0..blocks_per_row {
        let col_base = bk * QK_I2_S;
        for gp in 0..PACKED_BYTES_PER_BLOCK {
            let x0 = input[col_base + gp];
            let x1 = input[col_base + gp + 32];
            let x2 = input[col_base + gp + 64];
            let x3 = input[col_base + gp + 96];

            // ternary(0b00) = -1, ternary(0b01) = 0, ternary(0b10) = +1, ternary(0b11) = 0
            t0[0] = -x0;
            t0[1] = 0.0;
            t0[2] = x0;
            t0[3] = 0.0;
            t1[0] = -x1;
            t1[1] = 0.0;
            t1[2] = x1;
            t1[3] = 0.0;
            t2[0] = -x2;
            t2[1] = 0.0;
            t2[2] = x2;
            t2[3] = 0.0;
            t3[0] = -x3;
            t3[1] = 0.0;
            t3[2] = x3;
            t3[3] = 0.0;

            for byte_val in 0..256_usize {
                let c0 = (byte_val >> 6) & 0b11;
                let c1 = (byte_val >> 4) & 0b11;
                let c2 = (byte_val >> 2) & 0b11;
                let c3 = byte_val & 0b11;
                table[byte_val] = t0[c0] + t1[c1] + t2[c2] + t3[c3];
            }

            let block_byte_off = bk * PACKED_BYTES_PER_BLOCK + gp;
            for j in 0..out_dim {
                let b = packed[j * bytes_per_row + block_byte_off];
                output[j] += table[b as usize];
            }
        }
    }

    for j in 0..out_dim {
        output[j] *= scale;
    }

    Ok(())
}

fn validate(
    weight: &TensorView<'_>,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), WillametteError> {
    if weight.ggml_type != GgmlType::BitNetI2S {
        return Err(WillametteError::UnsupportedTensorType(
            weight.ggml_type.to_raw(),
        ));
    }
    if weight.shape.len() != 2 {
        return Err(WillametteError::GgufParse(format!(
            "bitlinear_lut_scalar: weight {:?} not 2-D (shape={:?})",
            weight.name, weight.shape
        )));
    }
    let in_dim = weight.shape[0] as usize;
    let out_dim = weight.shape[1] as usize;
    if input.len() != in_dim {
        return Err(WillametteError::GgufParse(format!(
            "bitlinear_lut_scalar: input.len()={} != in_dim={}",
            input.len(),
            in_dim
        )));
    }
    if output.len() != out_dim {
        return Err(WillametteError::GgufParse(format!(
            "bitlinear_lut_scalar: output.len()={} != out_dim={}",
            output.len(),
            out_dim
        )));
    }
    if in_dim == 0 || !in_dim.is_multiple_of(QK_I2_S) {
        return Err(WillametteError::GgufParse(format!(
            "bitlinear_lut_scalar: in_dim {} not a positive multiple of {}",
            in_dim, QK_I2_S
        )));
    }
    let bytes_per_row = in_dim / 4;
    let expected_packed = bytes_per_row * out_dim;
    if weight.data.len() != expected_packed {
        return Err(WillametteError::GgufParse(format!(
            "bitlinear_lut_scalar: weight data.len()={} != expected {} (in_dim/4 × out_dim)",
            weight.data.len(),
            expected_packed
        )));
    }
    Ok(())
}

// Integration parity vs scalar BitLinear lives in
// `tests/bitlinear_lut.rs` — it needs a real I2_S tensor, which
// only the real model file provides. No in-source unit tests
// here because table-build correctness without a real tensor is
// either circular (build a fake tensor that uses the same decode)
// or trivial (validate() error branches).
