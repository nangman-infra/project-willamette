//! SSE2 Q6_K-by-f32 row dot product for pre-SSSE3 x86 hosts.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m128, _mm_add_ps, _mm_add_ss, _mm_cvtepi32_ps, _mm_cvtss_f32, _mm_loadu_ps, _mm_movehl_ps,
    _mm_mul_ps, _mm_set1_ps, _mm_setr_epi32, _mm_setzero_ps, _mm_shuffle_ps,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128, _mm_add_ps, _mm_add_ss, _mm_cvtepi32_ps, _mm_cvtss_f32, _mm_loadu_ps, _mm_movehl_ps,
    _mm_mul_ps, _mm_set1_ps, _mm_setr_epi32, _mm_setzero_ps, _mm_shuffle_ps,
};

use super::primitives::f16_to_f32;

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn accumulate(accumulator: &mut __m128, input: *const f32, quants: [i32; 4], scale: __m128) {
    let quantized = _mm_cvtepi32_ps(_mm_setr_epi32(quants[0], quants[1], quants[2], quants[3]));
    let values = _mm_loadu_ps(input);
    *accumulator = _mm_add_ps(
        *accumulator,
        _mm_mul_ps(_mm_mul_ps(quantized, scale), values),
    );
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn hsum(value: __m128) -> f32 {
    let reversed = _mm_shuffle_ps(value, value, 0b00_01_10_11);
    let pairs = _mm_add_ps(value, reversed);
    let high = _mm_movehl_ps(pairs, pairs);
    _mm_cvtss_f32(_mm_add_ss(pairs, high))
}

/// Dot complete Q6_K blocks against an equal-length f32 input.
///
/// # Safety
///
/// The caller must establish SSE2 support and validate that `row` contains
/// one 210-byte block per 256 input values.
#[target_feature(enable = "sse2")]
pub(super) unsafe fn dot_row_sse2_validated(row: &[u8], input: &[f32]) -> f32 {
    let mut accumulators = [_mm_setzero_ps(); 4];

    for (block_index, block) in row.chunks_exact(210).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let input_block = input.as_ptr().add(block_index * 256);

        for half in 0..2 {
            let ql_base = half * 64;
            let qh_base = 128 + half * 32;
            let scale_base = 192 + half * 8;
            let input_base = input_block.add(half * 128);

            for band in 0..2 {
                let scale_index = band;
                let scales = [
                    _mm_set1_ps(d * block[scale_base + scale_index] as i8 as f32),
                    _mm_set1_ps(d * block[scale_base + scale_index + 2] as i8 as f32),
                    _mm_set1_ps(d * block[scale_base + scale_index + 4] as i8 as f32),
                    _mm_set1_ps(d * block[scale_base + scale_index + 6] as i8 as f32),
                ];

                for chunk in 0..4 {
                    let l = band * 16 + chunk * 4;
                    let mut quants = [[0i32; 4]; 4];
                    for lane in 0..4 {
                        let index = l + lane;
                        let low_13 = block[ql_base + index];
                        let low_24 = block[ql_base + index + 32];
                        let high = block[qh_base + index];
                        quants[0][lane] = ((low_13 & 0x0f) | ((high & 3) << 4)) as i32 - 32;
                        quants[1][lane] = ((low_24 & 0x0f) | (((high >> 2) & 3) << 4)) as i32 - 32;
                        quants[2][lane] = ((low_13 >> 4) | (((high >> 4) & 3) << 4)) as i32 - 32;
                        quants[3][lane] = ((low_24 >> 4) | (((high >> 6) & 3) << 4)) as i32 - 32;
                    }
                    accumulate(
                        &mut accumulators[0],
                        input_base.add(l),
                        quants[0],
                        scales[0],
                    );
                    accumulate(
                        &mut accumulators[1],
                        input_base.add(l + 32),
                        quants[1],
                        scales[1],
                    );
                    accumulate(
                        &mut accumulators[2],
                        input_base.add(l + 64),
                        quants[2],
                        scales[2],
                    );
                    accumulate(
                        &mut accumulators[3],
                        input_base.add(l + 96),
                        quants[3],
                        scales[3],
                    );
                }
            }
        }
    }

    hsum(_mm_add_ps(
        _mm_add_ps(accumulators[0], accumulators[1]),
        _mm_add_ps(accumulators[2], accumulators[3]),
    ))
}
