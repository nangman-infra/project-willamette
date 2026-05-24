// Hot kernel loops index parallel tensors (weight bytes, input f32s,
// output rows) at the same step; explicit `for j in 0..N` reads better
// than iterator chains here.
#![allow(clippy::needless_range_loop)]

//! Stage 6-C — NEON-accelerated I2_S BitLinear matvec (aarch64).
//!
//! Strategy: per row, scalar-unpack the 2-bit codes into an `i8` scratch
//! buffer of length `in_dim` (values in `{-1, 0, +1}`), then dot the
//! buffer against `input` via 16-element NEON loops with 4 parallel
//! `float32x4_t` accumulators. The dispatcher in [`super::bitlinear`]
//! picks this path on aarch64 hosts where NEON is detected.
//!
//! Numerical contract: the result is **not** bit-identical to the
//! scalar reference because the parallel accumulator reduces in a
//! different order than the scalar two-accumulator form. The absolute
//! per-element difference is bounded by `O(in_dim · ε · max|input|)`;
//! `tests/bitlinear_simd.rs` documents the empirical tolerance we
//! observe against the real GGUF.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;
use std::cell::RefCell;

use crate::error::WillametteError;
use crate::gguf::tensor::TensorView;
use crate::gguf::types::GgmlType;

const QK_I2_S: usize = 128;

thread_local! {
    /// Per-thread reusable scratch buffer for unpacked I2_S row data.
    /// Grows on first need to fit the largest `in_dim` seen by this
    /// thread (max in BitNet b1.58 2B is `feed_forward_length = 6912`),
    /// then stays put — no further allocations.
    static MATVEC_SCRATCH: RefCell<Vec<i8>> = const { RefCell::new(Vec::new()) };
    /// Per-thread reusable scratch buffer for the i8-quantised
    /// activation (Stage 10-D). Same growth strategy as MATVEC_SCRATCH.
    static ACTIVATION_I8_SCRATCH: RefCell<Vec<i8>> = const { RefCell::new(Vec::new()) };
}

/// Quantise an f32 activation vector to i8 via absmax-per-vector scale.
/// Returns the scale `s` such that `f32_x ≈ s * i8_x`. Writes into
/// `out`, which is resized to `input.len()` if needed.
#[inline]
fn quantize_input_absmax_i8(input: &[f32], out: &mut Vec<i8>) -> f32 {
    let mut max_abs: f32 = 0.0;
    for &v in input {
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    if out.len() < input.len() {
        out.resize(input.len(), 0);
    }
    if max_abs == 0.0 {
        for slot in out.iter_mut().take(input.len()) {
            *slot = 0;
        }
        return 1.0;
    }
    let scale = max_abs / 127.0;
    let inv_scale = 127.0_f32 / max_abs;
    for (i, &v) in input.iter().enumerate() {
        let q = (v * inv_scale).round().clamp(-127.0, 127.0);
        out[i] = q as i8;
    }
    scale
}

/// NEON i8×i8 → i32 dot product (Stage 10-D, stable-Rust path).
///
/// Computes `Σᵢ weights[i] * inputs[i]` in `i32`, by widening each
/// 16-element chunk via `vmull_s8` (i8 × i8 → i16, no overflow possible
/// since `|w| ≤ 1` for ternary weights and `|x| ≤ 127`) and then
/// widening + accumulating in two parallel `i32x4` accumulators.
///
/// Why not `vdotq_s32`? Rust stable does not (yet) expose
/// `stdarch_neon_dotprod`, so the 8.4-A `SDOT` instruction is
/// unreachable without nightly. The vmull widening below is
/// ~2× slower than DOTPROD but still wins versus the f32-input path
/// thanks to halved memory bandwidth (2 bytes/elem instead of 5).
///
/// SAFETY: caller guarantees `in_dim` is a multiple of 16, both
/// pointers are valid for `in_dim` elements, and NEON is detected.
#[target_feature(enable = "neon")]
unsafe fn neon_dot_i8_i8_s32(weights: *const i8, inputs: *const i8, in_dim: usize) -> i32 {
    debug_assert_eq!(in_dim % 16, 0);
    let mut acc0 = vdupq_n_s32(0);
    let mut acc1 = vdupq_n_s32(0);
    let mut i: usize = 0;
    while i + 16 <= in_dim {
        let w = vld1q_s8(weights.add(i));
        let x = vld1q_s8(inputs.add(i));

        // 8 × (i8 × i8) → 8 × i16, low and high halves of the 16-lane vectors.
        let prod_lo = vmull_s8(vget_low_s8(w), vget_low_s8(x));
        let prod_hi = vmull_s8(vget_high_s8(w), vget_high_s8(x));

        // Widen 8 × i16 → 8 × i32 via two halves each, accumulate.
        acc0 = vaddq_s32(acc0, vmovl_s16(vget_low_s16(prod_lo)));
        acc0 = vaddq_s32(acc0, vmovl_s16(vget_high_s16(prod_lo)));
        acc1 = vaddq_s32(acc1, vmovl_s16(vget_low_s16(prod_hi)));
        acc1 = vaddq_s32(acc1, vmovl_s16(vget_high_s16(prod_hi)));

        i += 16;
    }
    let sum = vaddq_s32(acc0, acc1);
    vaddvq_s32(sum)
}

/// I2_S matvec via int8 activation quantization (W1.58A8 scheme).
///
/// 1. Quantize f32 input → i8 (absmax-per-vector scale).
/// 2. Per output row: unpack I2_S weights → i8 (in {-1, 0, +1}),
///    compute integer dot with `neon_dot_i8_i8_s32`.
/// 3. Apply combined scale `weight.i2s_scale * input_scale` to the i32
///    sum and write the f32 result.
///
/// This is the same quantization the BitNet paper describes for
/// activation int8. The numerical error vs the scalar f32 path is
/// bounded by `O(in_dim * input_scale / 2)`; on the real model it's
/// ~1e-1 absolute per output element, well below typical logit
/// magnitudes (5-15) so argmax stability is preserved on the Stage
/// 5-E reference prompts.
///
/// # Safety
///
/// Caller must guarantee NEON + DOTPROD CPU features (dispatcher
/// checks via `is_aarch64_feature_detected!`). All shape / length
/// preconditions are re-checked here.
#[target_feature(enable = "neon")]
pub unsafe fn bitlinear_i2s_matvec_f32_neon_i8(
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
            "neon-i8 matvec: weight {:?} is not 2-D",
            weight.name
        )));
    }
    let in_dim = weight.shape[0] as usize;
    let out_dim = weight.shape[1] as usize;
    if input.len() != in_dim {
        return Err(WillametteError::GgufParse(format!(
            "neon-i8 matvec: input.len()={} != in_dim={}",
            input.len(),
            in_dim
        )));
    }
    if output.len() != out_dim {
        return Err(WillametteError::GgufParse(format!(
            "neon-i8 matvec: output.len()={} != out_dim={}",
            output.len(),
            out_dim
        )));
    }
    if in_dim == 0 || !in_dim.is_multiple_of(QK_I2_S) {
        return Err(WillametteError::GgufParse(format!(
            "neon-i8 matvec: in_dim {} is not a positive multiple of {}",
            in_dim, QK_I2_S
        )));
    }
    let bytes_per_row = in_dim / 4;
    let expected = bytes_per_row * out_dim;
    if weight.data.len() != expected {
        return Err(WillametteError::GgufParse(format!(
            "neon-i8 matvec: weight {:?} data.len()={} != expected {}",
            weight.name,
            weight.data.len(),
            expected
        )));
    }

    let w_scale = weight.i2s_scale()?;
    let packed = weight.data;

    // Quantise input once per matvec (not per row).
    let input_scale = ACTIVATION_I8_SCRATCH.with(|cell| {
        let mut buf = cell.borrow_mut();
        quantize_input_absmax_i8(input, &mut buf)
    });
    let combined_scale = w_scale * input_scale;

    // Snapshot the quantised input into a local Vec we can read across
    // rayon workers (the thread_local is not Sync).
    let input_i8: Vec<i8> = ACTIVATION_I8_SCRATCH.with(|cell| {
        let buf = cell.borrow();
        buf[..in_dim].to_vec()
    });

    use rayon::prelude::*;
    output
        .par_chunks_mut(32)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            MATVEC_SCRATCH.with(|cell| {
                let mut unpacked = cell.borrow_mut();
                if unpacked.len() < in_dim {
                    unpacked.resize(in_dim, 0);
                }
                let unpacked: &mut [i8] = &mut unpacked[..in_dim];
                let row_start = chunk_idx * 32;
                for (k, out_val) in out_chunk.iter_mut().enumerate() {
                    let j = row_start + k;
                    let row_byte_start = j * bytes_per_row;
                    let packed_row = &packed[row_byte_start..row_byte_start + bytes_per_row];
                    unpack_row(packed_row, in_dim, unpacked);
                    let dot_i32 = unsafe {
                        neon_dot_i8_i8_s32(unpacked.as_ptr(), input_i8.as_ptr(), in_dim)
                    };
                    *out_val = combined_scale * (dot_i32 as f32);
                }
            });
        });
    Ok(())
}

/// Unpack one packed row (`bytes_per_row = in_dim / 4` bytes) of I2_S
/// into `out` (length `in_dim`, values in `{-1, 0, +1}`). Mirrors the
/// `column-stride-32` byte→element layout from `docs/I2_S_LAYOUT.md` §4.
fn unpack_row(packed_row: &[u8], in_dim: usize, out: &mut [i8]) {
    debug_assert_eq!(packed_row.len(), in_dim / 4);
    debug_assert_eq!(out.len(), in_dim);
    let blocks = in_dim / QK_I2_S;
    for bk in 0..blocks {
        let block = &packed_row[bk * 32..bk * 32 + 32];
        let base = bk * QK_I2_S;
        for gp in 0..32 {
            let b = block[gp];
            let c0 = (b >> 6) & 0b11;
            let c1 = (b >> 4) & 0b11;
            let c2 = (b >> 2) & 0b11;
            let c3 = b & 0b11;
            out[base + gp] = ternary_lut(c0);
            out[base + gp + 32] = ternary_lut(c1);
            out[base + gp + 64] = ternary_lut(c2);
            out[base + gp + 96] = ternary_lut(c3);
        }
    }
}

#[inline(always)]
fn ternary_lut(code: u8) -> i8 {
    match code & 0b11 {
        0b00 => -1,
        0b01 => 0,
        0b10 => 1,
        _ => 0,
    }
}

/// NEON dot product: `sum_{i=0..n} (weight_i8[i] * input_f32[i])`.
///
/// SAFETY: caller must guarantee that `weights_i8.len() == in_dim`,
/// `input.len() == in_dim`, and `in_dim % 16 == 0`. Apple Silicon
/// always has NEON; the dispatcher in [`super::bitlinear`] also runs
/// `is_aarch64_feature_detected!("neon")` before calling this.
#[target_feature(enable = "neon")]
unsafe fn neon_dot_i8_f32(weights_i8: *const i8, input: *const f32, in_dim: usize) -> f32 {
    debug_assert_eq!(in_dim % 16, 0);
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let mut i: usize = 0;
    while i + 16 <= in_dim {
        // 16 weights as int8x16_t
        let w_i8 = vld1q_s8(weights_i8.add(i));

        // Widen i8 → i16 (two halves)
        let w_i16_lo = vmovl_s8(vget_low_s8(w_i8));
        let w_i16_hi = vmovl_s8(vget_high_s8(w_i8));

        // Widen i16 → i32 (four quarters)
        let w_i32_0 = vmovl_s16(vget_low_s16(w_i16_lo));
        let w_i32_1 = vmovl_s16(vget_high_s16(w_i16_lo));
        let w_i32_2 = vmovl_s16(vget_low_s16(w_i16_hi));
        let w_i32_3 = vmovl_s16(vget_high_s16(w_i16_hi));

        // Convert i32 → f32
        let w_f0 = vcvtq_f32_s32(w_i32_0);
        let w_f1 = vcvtq_f32_s32(w_i32_1);
        let w_f2 = vcvtq_f32_s32(w_i32_2);
        let w_f3 = vcvtq_f32_s32(w_i32_3);

        // Load 16 input f32 values
        let x0 = vld1q_f32(input.add(i));
        let x1 = vld1q_f32(input.add(i + 4));
        let x2 = vld1q_f32(input.add(i + 8));
        let x3 = vld1q_f32(input.add(i + 12));

        // Multiply-accumulate
        acc0 = vfmaq_f32(acc0, x0, w_f0);
        acc1 = vfmaq_f32(acc1, x1, w_f1);
        acc2 = vfmaq_f32(acc2, x2, w_f2);
        acc3 = vfmaq_f32(acc3, x3, w_f3);

        i += 16;
    }

    // Reduce: combine 4 accumulators then horizontal sum.
    let sum_a = vaddq_f32(acc0, acc1);
    let sum_b = vaddq_f32(acc2, acc3);
    let sum = vaddq_f32(sum_a, sum_b);
    vaddvq_f32(sum)
}

/// Full I2_S matvec via per-row scalar unpack + NEON dot product.
///
/// # Safety
///
/// The caller must ensure NEON is available on the running CPU
/// (Apple Silicon always satisfies this; on other aarch64 hosts use
/// `std::arch::is_aarch64_feature_detected!("neon")` first — see
/// the dispatcher in [`super::bitlinear::bitlinear_i2s_matvec_f32`]).
/// All length / dtype / shape preconditions are re-checked inside
/// this function and reported as typed errors, not UB.
#[target_feature(enable = "neon")]
pub unsafe fn bitlinear_i2s_matvec_f32_neon(
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
            "neon matvec: weight {:?} is not 2-D (shape={:?})",
            weight.name, weight.shape
        )));
    }
    let in_dim = weight.shape[0] as usize;
    let out_dim = weight.shape[1] as usize;
    if input.len() != in_dim {
        return Err(WillametteError::GgufParse(format!(
            "neon matvec: input.len()={} != in_dim={} ({:?})",
            input.len(),
            in_dim,
            weight.name
        )));
    }
    if output.len() != out_dim {
        return Err(WillametteError::GgufParse(format!(
            "neon matvec: output.len()={} != out_dim={} ({:?})",
            output.len(),
            out_dim,
            weight.name
        )));
    }
    if in_dim == 0 || !in_dim.is_multiple_of(QK_I2_S) {
        return Err(WillametteError::GgufParse(format!(
            "neon matvec: in_dim {} is not a positive multiple of {}",
            in_dim, QK_I2_S
        )));
    }
    let bytes_per_row = in_dim / 4;
    let expected = bytes_per_row * out_dim;
    if weight.data.len() != expected {
        return Err(WillametteError::GgufParse(format!(
            "neon matvec: weight {:?} data.len()={} != expected {}",
            weight.name,
            weight.data.len(),
            expected
        )));
    }

    let scale = weight.i2s_scale()?;
    let packed = weight.data;

    // Stage 10-C: rayon-parallel row dispatch. Each worker thread
    // owns its own MATVEC_SCRATCH (thread_local), so unpack_row +
    // neon_dot_i8_f32 run independently across cores. Chunk size 32
    // keeps per-task work above rayon's ~1-2µs overhead (each row
    // is ~0.7µs of work on M1).
    //
    // SAFETY (NEON): the dispatcher in `super::bitlinear` guarantees
    // `is_aarch64_feature_detected!("neon")` on the calling thread; the
    // rayon worker pool was spawned on the same machine so every
    // worker also has NEON. The `unsafe` block below merely names
    // that guarantee.
    use rayon::prelude::*;
    output
        .par_chunks_mut(32)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            MATVEC_SCRATCH.with(|cell| {
                let mut unpacked = cell.borrow_mut();
                if unpacked.len() < in_dim {
                    unpacked.resize(in_dim, 0);
                }
                let unpacked: &mut [i8] = &mut unpacked[..in_dim];
                let row_start = chunk_idx * 32;
                for (k, out_val) in out_chunk.iter_mut().enumerate() {
                    let j = row_start + k;
                    let row_byte_start = j * bytes_per_row;
                    let packed_row = &packed[row_byte_start..row_byte_start + bytes_per_row];
                    unpack_row(packed_row, in_dim, unpacked);
                    let dot = unsafe {
                        neon_dot_i8_f32(unpacked.as_ptr(), input.as_ptr(), in_dim)
                    };
                    *out_val = scale * dot;
                }
            });
        });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_view<'a>(
        packed: &'a [u8],
        scale_block: &'a [u8],
        in_dim: u64,
        out_dim: u64,
    ) -> TensorView<'a> {
        TensorView {
            name: "test".into(),
            shape: vec![in_dim, out_dim],
            ggml_type: GgmlType::BitNetI2S,
            offset: 0,
            byte_len: packed.len() as u64,
            data: packed,
            scale_data: Some(scale_block),
        }
    }

    #[test]
    fn neon_all_plus_one_matches_sum() {
        let packed = vec![0xAA_u8; 32];
        let mut scale_block = [0u8; 32];
        scale_block[..4].copy_from_slice(&1.0_f32.to_le_bytes());
        let w = make_view(&packed, &scale_block, 128, 1);

        let input: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1).collect();
        let mut out = vec![0.0_f32; 1];
        unsafe {
            bitlinear_i2s_matvec_f32_neon(&w, &input, &mut out).unwrap();
        }
        let expected: f32 = input.iter().sum();
        assert!(
            (out[0] - expected).abs() < 1e-3,
            "all-+1 NEON: got {}, expected {}",
            out[0],
            expected
        );
    }

    #[test]
    fn neon_all_minus_one_matches_negative_sum() {
        let packed = vec![0x00_u8; 32];
        let mut scale_block = [0u8; 32];
        scale_block[..4].copy_from_slice(&1.0_f32.to_le_bytes());
        let w = make_view(&packed, &scale_block, 128, 1);

        let input: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1).collect();
        let mut out = vec![0.0_f32; 1];
        unsafe {
            bitlinear_i2s_matvec_f32_neon(&w, &input, &mut out).unwrap();
        }
        let expected: f32 = -input.iter().sum::<f32>();
        assert!(
            (out[0] - expected).abs() < 1e-3,
            "got {}, expected {}",
            out[0],
            expected
        );
    }

    #[test]
    fn neon_all_zero_weights_yield_zero() {
        let packed = vec![0x55_u8; 32];
        let mut scale_block = [0u8; 32];
        scale_block[..4].copy_from_slice(&1.0_f32.to_le_bytes());
        let w = make_view(&packed, &scale_block, 128, 1);

        let input = vec![1.25_f32; 128];
        let mut out = vec![999.0_f32; 1];
        unsafe {
            bitlinear_i2s_matvec_f32_neon(&w, &input, &mut out).unwrap();
        }
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn neon_scale_applied() {
        let packed = vec![0xAA_u8; 32];
        let mut scale_block = [0u8; 32];
        scale_block[..4].copy_from_slice(&2.5_f32.to_le_bytes());
        let w = make_view(&packed, &scale_block, 128, 1);

        let input = vec![1.0_f32; 128];
        let mut out = vec![0.0_f32; 1];
        unsafe {
            bitlinear_i2s_matvec_f32_neon(&w, &input, &mut out).unwrap();
        }
        // 128 elements × 1.0 × +1 weight × 2.5 scale = 320
        assert!((out[0] - 320.0).abs() < 1e-3, "got {}", out[0]);
    }
}
