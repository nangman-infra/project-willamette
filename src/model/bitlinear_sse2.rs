//! Stage 6-B: 32-bit / 64-bit x86 SSE2 BitLinear matvec.
//!
//! Numerically equivalent to [`super::bitlinear::bitlinear_i2s_matvec_f32_scalar`]
//! within the same per-output tolerance documented in `tests/bitlinear_simd.rs`
//! (`max |Δ| < 1e-2` against the scalar reference). The implementation uses
//! the two-accumulator form `out[j] = scale · (Σ_pos x[i] − Σ_neg x[i])`
//! exactly like the scalar reference, so the only sources of difference
//! are SIMD horizontal-add reduction order and the per-row final
//! `pos − neg` subtraction — both small and bounded.
//!
//! ## SIMD strategy
//!
//! Per row of 128-element blocks:
//!
//! 1. **Unpack** the 32 packed bytes of one block into a local
//!    `[i8; 128]` stack buffer using the same column-stride-32
//!    code mapping the scalar path uses
//!    (`c0 → gp`, `c1 → 32+gp`, `c2 → 64+gp`, `c3 → 96+gp`).
//! 2. **Loop in 4-float chunks** over the unpacked block. For each
//!    chunk:
//!    * Sign-extend the 4 ternary `i8` values to four `i32`s, then
//!      to four `f32`s.
//!    * Build two masks: `is_pos[k] = (signs[k] == +1)`,
//!      `is_neg[k] = (signs[k] == -1)`. Bytes `(0b00 → −1, 0b10 → +1)`
//!      mean the mask is the natural way to express the contract.
//!    * Add `_mm_and_ps(x, is_pos_mask)` into the positive
//!      accumulator and `_mm_and_ps(x, is_neg_mask)` into the
//!      negative one. Multiplying by `±1.0` is never done — the
//!      masked-add form preserves numerical equivalence with the
//!      scalar two-accumulator pattern.
//! 3. After all blocks, horizontal-reduce both accumulators and emit
//!    `out[j] = scale * (pos_sum − neg_sum)`.
//!
//! ## SSE2 only
//!
//! Pentium-M (Banias / Dothan) caps at SSE2 — no SSE3 / SSSE3 / SSE4.x.
//! Sign-extending an `i8` to `f32` therefore needs the manual SSE2
//! sequence `unpacklo_epi8` + `srai_epi16` + `unpacklo_epi16` +
//! `srai_epi32` + `cvtepi32_ps`. `_mm_cvtepi8_epi32` (SSE4.1) is not
//! available here.
//!
//! ## Safety
//!
//! Caller must have verified `is_x86_feature_detected!("sse2")`. The
//! `#[target_feature(enable = "sse2")]` attribute permits the
//! compiler to emit instructions the *target_cpu* might not have, so
//! the runtime detection in `dispatch::select_kernel` is the
//! contract that makes the call sound. All shape / length checks
//! happen in the public entry point.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#![allow(unsafe_op_in_unsafe_fn)] // intrinsic block reads tighter without a per-call unsafe.
#![allow(clippy::needless_range_loop)] // mirrors bitlinear.rs / bitlinear_neon.rs — explicit row × block × chunk indexing is clearer than iterator chains in this hot kernel.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::error::WillametteError;
use crate::gguf::tensor::TensorView;
use crate::gguf::types::GgmlType;

const QK_I2_S: usize = 128;
const PACKED_BYTES_PER_BLOCK: usize = 32;

/// SSE2 BitLinear matvec. Same contract as the scalar reference.
///
/// # Safety
///
/// The caller must ensure `is_x86_feature_detected!("sse2")` is true
/// for this process. `dispatch::select_kernel` is the canonical
/// place that guarantee comes from.
#[target_feature(enable = "sse2")]
pub unsafe fn bitlinear_i2s_matvec_f32_sse2(
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
            "bitlinear_i2s_matvec_f32_sse2: weight {:?} is not 2-D",
            weight.name
        )));
    }
    let in_dim = weight.shape[0] as usize;
    let out_dim = weight.shape[1] as usize;
    if input.len() != in_dim {
        return Err(WillametteError::GgufParse(format!(
            "bitlinear_i2s_matvec_f32_sse2: input.len()={} != in_dim={}",
            input.len(),
            in_dim
        )));
    }
    if output.len() != out_dim {
        return Err(WillametteError::GgufParse(format!(
            "bitlinear_i2s_matvec_f32_sse2: output.len()={} != out_dim={}",
            output.len(),
            out_dim
        )));
    }
    if in_dim == 0 || !in_dim.is_multiple_of(QK_I2_S) {
        return Err(WillametteError::GgufParse(format!(
            "bitlinear_i2s_matvec_f32_sse2: in_dim {} is not a positive multiple of {}",
            in_dim, QK_I2_S
        )));
    }
    let bytes_per_row = in_dim / 4;
    let blocks_per_row = in_dim / QK_I2_S;
    let expected_packed = bytes_per_row * out_dim;
    if weight.data.len() < expected_packed {
        return Err(WillametteError::GgufParse(format!(
            "bitlinear_i2s_matvec_f32_sse2: weight data.len()={} < {} (in_dim/4 × out_dim)",
            weight.data.len(),
            expected_packed
        )));
    }

    let scale = weight.i2s_scale()?;
    let packed = weight.data;
    let pos_mask_target: __m128i = _mm_set1_epi32(1);
    let neg_mask_target: __m128i = _mm_set1_epi32(-1);

    for j in 0..out_dim {
        let row_offset = j * bytes_per_row;
        let mut pos_sum = _mm_setzero_ps();
        let mut neg_sum = _mm_setzero_ps();

        for bk in 0..blocks_per_row {
            let block_offset = row_offset + bk * PACKED_BYTES_PER_BLOCK;
            let col_base = bk * QK_I2_S;

            // Unpack 32 packed bytes → 128 ternary i8 values on the
            // stack. Stays in L1, never allocates.
            let mut unpacked = [0_i8; 128];
            for gp in 0..PACKED_BYTES_PER_BLOCK {
                let b = packed[block_offset + gp];
                unpacked[gp] = code_to_ternary((b >> 6) & 0b11);
                unpacked[gp + 32] = code_to_ternary((b >> 4) & 0b11);
                unpacked[gp + 64] = code_to_ternary((b >> 2) & 0b11);
                unpacked[gp + 96] = code_to_ternary(b & 0b11);
            }

            // Inner SIMD: 32 chunks of 4 f32 each = 128 elements / block.
            let x_block = &input[col_base..col_base + QK_I2_S];
            for c in 0..(QK_I2_S / 4) {
                let chunk_off = c * 4;
                // Load 4 input floats.
                let xv = _mm_loadu_ps(x_block.as_ptr().add(chunk_off));

                // Load 4 ternary i8 codes and sign-extend i8 → i32.
                let four_bytes = i32::from_le_bytes([
                    unpacked[chunk_off] as u8,
                    unpacked[chunk_off + 1] as u8,
                    unpacked[chunk_off + 2] as u8,
                    unpacked[chunk_off + 3] as u8,
                ]);
                let v_i8 = _mm_cvtsi32_si128(four_bytes);
                let v_i16 = _mm_srai_epi16(_mm_unpacklo_epi8(v_i8, v_i8), 8);
                let v_i32 = _mm_srai_epi32(_mm_unpacklo_epi16(v_i16, v_i16), 16);

                // Compare-equal masks: 0xFFFFFFFF where condition holds.
                let pos_mask = _mm_castsi128_ps(_mm_cmpeq_epi32(v_i32, pos_mask_target));
                let neg_mask = _mm_castsi128_ps(_mm_cmpeq_epi32(v_i32, neg_mask_target));

                // Mask-add (no ±1.0 multiplication).
                pos_sum = _mm_add_ps(pos_sum, _mm_and_ps(xv, pos_mask));
                neg_sum = _mm_add_ps(neg_sum, _mm_and_ps(xv, neg_mask));
            }
        }

        // Horizontal-add the two 4-lane accumulators, then combine.
        output[j] = scale * (hsum_ps(pos_sum) - hsum_ps(neg_sum));
    }
    Ok(())
}

#[inline]
fn code_to_ternary(code: u8) -> i8 {
    match code & 0b11 {
        0b00 => -1,
        0b10 => 1,
        _ => 0, // 0b01 + 0b11 (degenerate)
    }
}

/// Horizontal sum of a 4-lane `__m128` — pure SSE2 (`_mm_hadd_ps` is SSE3).
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn hsum_ps(v: __m128) -> f32 {
    let shuf = _mm_shuffle_ps(v, v, 0b00_01_10_11);
    let sums = _mm_add_ps(v, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let final_ = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(final_)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_to_ternary_matches_scalar_helper() {
        assert_eq!(code_to_ternary(0b00), -1);
        assert_eq!(code_to_ternary(0b01), 0);
        assert_eq!(code_to_ternary(0b10), 1);
        assert_eq!(code_to_ternary(0b11), 0);
    }

    #[test]
    fn hsum_ps_basic() {
        // Only run if SSE2 is actually detected on this host.
        if !std::arch::is_x86_feature_detected!("sse2") {
            return;
        }
        unsafe {
            let v = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
            let s = hsum_ps(v);
            assert!((s - 10.0).abs() < 1e-6);
        }
    }
}
