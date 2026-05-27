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
use core::arch::x86::{
    __m128, __m128i, _mm_add_epi32, _mm_add_ps, _mm_add_ss, _mm_and_ps, _mm_and_si128,
    _mm_castsi128_ps, _mm_cmpeq_epi32, _mm_cmpeq_epi8, _mm_cvtsi32_si128, _mm_cvtss_f32,
    _mm_loadu_ps, _mm_loadu_si128, _mm_madd_epi16, _mm_movehl_ps, _mm_or_si128, _mm_set1_epi16,
    _mm_set1_epi32, _mm_set1_epi8, _mm_setzero_ps, _mm_setzero_si128, _mm_shuffle_ps,
    _mm_srai_epi16, _mm_srai_epi32, _mm_storeu_si128, _mm_sub_epi8, _mm_unpackhi_epi8,
    _mm_unpacklo_epi16, _mm_unpacklo_epi8,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128, __m128i, _mm_add_epi32, _mm_add_ps, _mm_add_ss, _mm_and_ps, _mm_and_si128,
    _mm_castsi128_ps, _mm_cmpeq_epi32, _mm_cmpeq_epi8, _mm_cvtsi32_si128, _mm_cvtss_f32,
    _mm_loadu_ps, _mm_loadu_si128, _mm_madd_epi16, _mm_movehl_ps, _mm_or_si128, _mm_set1_epi16,
    _mm_set1_epi32, _mm_set1_epi8, _mm_setzero_ps, _mm_setzero_si128, _mm_shuffle_ps,
    _mm_srai_epi16, _mm_srai_epi32, _mm_storeu_si128, _mm_sub_epi8, _mm_unpackhi_epi8,
    _mm_unpacklo_epi16, _mm_unpacklo_epi8,
};

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
    let (in_dim, out_dim) = validate_inputs(weight, input, output)?;
    let bytes_per_row = in_dim / 4;
    let blocks_per_row = in_dim / QK_I2_S;
    let scale = weight.i2s_scale()?;
    let packed = weight.data;
    let pos_target: __m128i = _mm_set1_epi32(1);
    let neg_target: __m128i = _mm_set1_epi32(-1);

    for j in 0..out_dim {
        let row_offset = j * bytes_per_row;
        let mut pos_sum = _mm_setzero_ps();
        let mut neg_sum = _mm_setzero_ps();
        for bk in 0..blocks_per_row {
            let block_offset = row_offset + bk * PACKED_BYTES_PER_BLOCK;
            let col_base = bk * QK_I2_S;
            let unpacked =
                unpack_block(&packed[block_offset..block_offset + PACKED_BYTES_PER_BLOCK]);
            let x_block = &input[col_base..col_base + QK_I2_S];
            block_accumulate(
                &unpacked,
                x_block,
                pos_target,
                neg_target,
                &mut pos_sum,
                &mut neg_sum,
            );
        }
        output[j] = scale * (hsum_ps(pos_sum) - hsum_ps(neg_sum));
    }
    Ok(())
}

/// All shape / length validations for the SSE2 kernel entry point.
/// Returns `(in_dim, out_dim)` for the caller. Extracted so the hot
/// kernel body stays at one cognitive-complexity-15 level.
fn validate_inputs(
    weight: &TensorView<'_>,
    input: &[f32],
    output: &mut [f32],
) -> Result<(usize, usize), WillametteError> {
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
    let expected_packed = bytes_per_row * out_dim;
    if weight.data.len() < expected_packed {
        return Err(WillametteError::GgufParse(format!(
            "bitlinear_i2s_matvec_f32_sse2: weight data.len()={} < {} (in_dim/4 × out_dim)",
            weight.data.len(),
            expected_packed
        )));
    }
    Ok((in_dim, out_dim))
}

/// Unpack one 32-byte packed block into 128 ternary `i8` values on the
/// stack. Column-stride-32 mapping per `docs/I2_S_LAYOUT.md`:
/// `c0 → gp`, `c1 → 32+gp`, `c2 → 64+gp`, `c3 → 96+gp`.
#[inline]
fn unpack_block(block_bytes: &[u8]) -> [i8; QK_I2_S] {
    let mut unpacked = [0_i8; QK_I2_S];
    for gp in 0..PACKED_BYTES_PER_BLOCK {
        let b = block_bytes[gp];
        unpacked[gp] = code_to_ternary((b >> 6) & 0b11);
        unpacked[gp + 32] = code_to_ternary((b >> 4) & 0b11);
        unpacked[gp + 64] = code_to_ternary((b >> 2) & 0b11);
        unpacked[gp + 96] = code_to_ternary(b & 0b11);
    }
    unpacked
}

/// Inner SIMD loop for one already-unpacked block. Walks 32 chunks of
/// 4 floats each, accumulating `x[i]` into `pos_sum` or `neg_sum`
/// based on the ternary sign of `unpacked[i]`.
///
/// # Safety
///
/// `x_block.len() == QK_I2_S == unpacked.len()`. Caller is responsible
/// for the SSE2 feature gate (called only from the
/// `target_feature(enable = "sse2")` entry point).
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn block_accumulate(
    unpacked: &[i8; QK_I2_S],
    x_block: &[f32],
    pos_target: __m128i,
    neg_target: __m128i,
    pos_sum: &mut __m128,
    neg_sum: &mut __m128,
) {
    for c in 0..(QK_I2_S / 4) {
        let chunk_off = c * 4;
        let xv = _mm_loadu_ps(x_block.as_ptr().add(chunk_off));

        // Sign-extend 4 ternary i8 codes → i32 lanes (pure SSE2).
        let four_bytes = i32::from_le_bytes([
            unpacked[chunk_off] as u8,
            unpacked[chunk_off + 1] as u8,
            unpacked[chunk_off + 2] as u8,
            unpacked[chunk_off + 3] as u8,
        ]);
        let v_i8 = _mm_cvtsi32_si128(four_bytes);
        let v_i16 = _mm_srai_epi16(_mm_unpacklo_epi8(v_i8, v_i8), 8);
        let v_i32 = _mm_srai_epi32(_mm_unpacklo_epi16(v_i16, v_i16), 16);

        let pos_mask = _mm_castsi128_ps(_mm_cmpeq_epi32(v_i32, pos_target));
        let neg_mask = _mm_castsi128_ps(_mm_cmpeq_epi32(v_i32, neg_target));

        *pos_sum = _mm_add_ps(*pos_sum, _mm_and_ps(xv, pos_mask));
        *neg_sum = _mm_add_ps(*neg_sum, _mm_and_ps(xv, neg_mask));
    }
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

// ─────────────────────────────────────────────────────────────────────
// i8 activation path (Stage 6-B follow-up)
//
// Mirror of `bitlinear_neon::bitlinear_i2s_matvec_f32_neon_i8`. Instead
// of keeping the activation in f32 and doing a masked f32 add, we
// quantise the activation to i8 once (absmax-per-vector scale) and run
// the dot product entirely in integer lanes. Two wins on Pentium-M:
//
//   1. No per-element i8 → i32 → f32 sign-extend + convert in the
//      inner loop (that conversion is a big chunk of the f32 kernel's
//      96 %-of-runtime cost — see docs/BENCHMARKS.md profiling).
//   2. 16 i8 lanes per 128-bit register vs 4 f32 lanes.
//
// Ternary weights mean no real multiply: the product is just the
// activation, its negation, or zero — selected by compare-masks.
// Accumulation widens i8 → i16 (sign-extended) and folds into i32 via
// `_mm_madd_epi16(_, 1)`, which can't overflow across a 128-block.
//
// Numerical note: this path is NOT bit-identical to the f32 mask-add
// kernel — the activation is quantised to int8 (the same lossy step
// the production bitnet.cpp CPU path takes). Equivalence is checked at
// the documented tolerance in `tests/bitlinear_sse2_i8.rs`.
// ─────────────────────────────────────────────────────────────────────

/// Quantise an f32 activation vector to i8 via absmax-per-vector scale.
/// Returns `s` such that `f32_x ≈ s * i8_x`. Identical logic to the
/// NEON path's `quantize_input_absmax_i8` (kept local to avoid coupling
/// the two arch modules).
fn quantize_input_absmax_i8(input: &[f32], out: &mut [i8]) -> f32 {
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

/// SSE2 BitLinear matvec with int8 activations. Same contract / shape
/// validation as the f32 entry point.
///
/// # Safety
///
/// Caller must ensure `is_x86_feature_detected!("sse2")`.
#[target_feature(enable = "sse2")]
pub unsafe fn bitlinear_i2s_matvec_f32_sse2_i8(
    weight: &TensorView<'_>,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), WillametteError> {
    let (in_dim, out_dim) = validate_inputs(weight, input, output)?;
    let bytes_per_row = in_dim / 4;
    let blocks_per_row = in_dim / QK_I2_S;
    let w_scale = weight.i2s_scale()?;
    let packed = weight.data;

    // Quantise the activation once for the whole matvec.
    let mut act = vec![0_i8; in_dim];
    let input_scale = quantize_input_absmax_i8(input, &mut act);
    let combined = w_scale * input_scale;

    for j in 0..out_dim {
        let row_offset = j * bytes_per_row;
        let mut dot: i32 = 0;
        for bk in 0..blocks_per_row {
            let block_offset = row_offset + bk * PACKED_BYTES_PER_BLOCK;
            let col_base = bk * QK_I2_S;
            let unpacked =
                unpack_block(&packed[block_offset..block_offset + PACKED_BYTES_PER_BLOCK]);
            dot += dot_i8_ternary(&unpacked, &act[col_base..col_base + QK_I2_S]);
        }
        output[j] = combined * dot as f32;
    }
    Ok(())
}

/// Integer dot product of one 128-element block: ternary weights
/// (`-1 / 0 / +1`) against i8 activations, returning i32. `weight` and
/// `act` are both `QK_I2_S` long.
///
/// # Safety
///
/// SSE2 feature gate is the caller's responsibility.
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn dot_i8_ternary(weight: &[i8; QK_I2_S], act: &[i8]) -> i32 {
    let ones = _mm_set1_epi8(1);
    let neg_ones = _mm_set1_epi8(-1);
    let zero = _mm_setzero_si128();
    let ones16 = _mm_set1_epi16(1);
    let mut acc = _mm_setzero_si128();

    let mut i = 0;
    while i + 16 <= QK_I2_S {
        let w = _mm_loadu_si128(weight.as_ptr().add(i) as *const __m128i);
        let x = _mm_loadu_si128(act.as_ptr().add(i) as *const __m128i);

        // product = +x where w==+1, -x where w==-1, 0 where w==0.
        // Activation is clamped to [-127, 127], so negate never hits
        // the i8 -128 overflow corner.
        let pos = _mm_cmpeq_epi8(w, ones);
        let neg = _mm_cmpeq_epi8(w, neg_ones);
        let neg_x = _mm_sub_epi8(zero, x);
        let prod = _mm_or_si128(_mm_and_si128(x, pos), _mm_and_si128(neg_x, neg));

        // Sign-extend i8 → i16 (low + high halves), then fold to i32.
        // madd_epi16(v, 1) = pairwise i16 adds → i32, overflow-free
        // across a 128-element block (max |sum| ≤ 128·127 < 2^31).
        let lo = _mm_srai_epi16(_mm_unpacklo_epi8(prod, prod), 8);
        let hi = _mm_srai_epi16(_mm_unpackhi_epi8(prod, prod), 8);
        acc = _mm_add_epi32(acc, _mm_madd_epi16(lo, ones16));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(hi, ones16));
        i += 16;
    }

    let mut lanes = [0_i32; 4];
    _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, acc);
    lanes[0] + lanes[1] + lanes[2] + lanes[3]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_arch = "x86")]
    use core::arch::x86::_mm_setr_ps;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::_mm_setr_ps;

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
