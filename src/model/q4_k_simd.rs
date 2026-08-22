//! x86 SIMD Q4_K-by-f32 row dot products selected by `q4_k::dot_row`.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#![allow(unsafe_op_in_unsafe_fn)]

use super::{primitives::f16_to_f32, q4_k::scale_min};

const BLOCK_BYTES: usize = 144;
const BLOCK_VALUES: usize = 256;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m128, __m128i, __m256, _mm256_add_ps, _mm256_castps256_ps128, _mm256_cvtepi32_ps,
    _mm256_cvtepu8_epi32, _mm256_extractf128_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps,
    _mm256_setzero_ps, _mm256_sub_ps, _mm_add_ps, _mm_add_ss, _mm_and_si128, _mm_cvtepi32_ps,
    _mm_cvtss_f32, _mm_loadl_epi64, _mm_loadu_ps, _mm_loadu_si128, _mm_movehl_ps, _mm_mul_ps,
    _mm_set1_epi8, _mm_set1_ps, _mm_setzero_ps, _mm_setzero_si128, _mm_shuffle_ps, _mm_srli_epi16,
    _mm_sub_ps, _mm_unpackhi_epi16, _mm_unpackhi_epi8, _mm_unpacklo_epi16, _mm_unpacklo_epi8,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128, __m128i, __m256, _mm256_add_ps, _mm256_castps256_ps128, _mm256_cvtepi32_ps,
    _mm256_cvtepu8_epi32, _mm256_extractf128_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps,
    _mm256_setzero_ps, _mm256_sub_ps, _mm_add_ps, _mm_add_ss, _mm_and_si128, _mm_cvtepi32_ps,
    _mm_cvtss_f32, _mm_loadl_epi64, _mm_loadu_ps, _mm_loadu_si128, _mm_movehl_ps, _mm_mul_ps,
    _mm_set1_epi8, _mm_set1_ps, _mm_setzero_ps, _mm_setzero_si128, _mm_shuffle_ps, _mm_srli_epi16,
    _mm_sub_ps, _mm_unpackhi_epi16, _mm_unpackhi_epi8, _mm_unpacklo_epi16, _mm_unpacklo_epi8,
};

/// Run the AVX2 kernel after checking that this process may execute it.
pub(super) fn dot_row_avx2_validated(row: &[u8], input: &[f32]) -> f32 {
    assert!(std::arch::is_x86_feature_detected!("avx2"));
    debug_assert_eq!(row.len() / BLOCK_BYTES * BLOCK_VALUES, input.len());
    // SAFETY: the runtime check establishes AVX2 support; q4_k::dot_row has
    // validated complete blocks and matching lengths before calling us.
    unsafe { dot_row_avx2_inner(row, input) }
}

/// Run the SSE2 kernel after checking that this process may execute it.
pub(super) fn dot_row_sse2_validated(row: &[u8], input: &[f32]) -> f32 {
    assert!(std::arch::is_x86_feature_detected!("sse2"));
    debug_assert_eq!(row.len() / BLOCK_BYTES * BLOCK_VALUES, input.len());
    // SAFETY: the runtime check establishes SSE2 support; q4_k::dot_row has
    // validated complete blocks and matching lengths before calling us.
    unsafe { dot_row_sse2_inner(row, input) }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_avx2(
    accumulator: &mut __m256,
    quant: __m128i,
    input: *const f32,
    scale: __m256,
    minimum: __m256,
) {
    let quant = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(quant));
    let weights = _mm256_sub_ps(_mm256_mul_ps(quant, scale), minimum);
    *accumulator = _mm256_add_ps(*accumulator, _mm256_mul_ps(weights, _mm256_loadu_ps(input)));
}

#[target_feature(enable = "avx2")]
unsafe fn dot_row_avx2_inner(row: &[u8], input: &[f32]) -> f32 {
    let nibble_mask = _mm_set1_epi8(0x0f);
    let mut sum = _mm256_setzero_ps();

    for (block_index, block) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let quants = block.as_ptr().add(16);
        let inputs = input.as_ptr().add(block_index * BLOCK_VALUES);

        for band in 0..4 {
            let low_group = band * 2;
            let high_group = low_group + 1;
            let (low_scale, low_min) = scale_min(scales, low_group);
            let (high_scale, high_min) = scale_min(scales, high_group);
            let low_scale = _mm256_set1_ps(d * f32::from(low_scale));
            let low_min = _mm256_set1_ps(dmin * f32::from(low_min));
            let high_scale = _mm256_set1_ps(d * f32::from(high_scale));
            let high_min = _mm256_set1_ps(dmin * f32::from(high_min));

            for chunk in [0, 8, 16, 24] {
                let packed = _mm_loadl_epi64(quants.add(band * 32 + chunk).cast::<__m128i>());
                let low = _mm_and_si128(packed, nibble_mask);
                let high = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
                accumulate_avx2(
                    &mut sum,
                    low,
                    inputs.add(low_group * 32 + chunk),
                    low_scale,
                    low_min,
                );
                accumulate_avx2(
                    &mut sum,
                    high,
                    inputs.add(high_group * 32 + chunk),
                    high_scale,
                    high_min,
                );
            }
        }
    }

    let halves = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    hsum_sse2(halves)
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn accumulate_sse2(
    accumulator: &mut __m128,
    quant: __m128i,
    input: *const f32,
    scale: __m128,
    minimum: __m128,
) {
    let quant = _mm_cvtepi32_ps(quant);
    let weights = _mm_sub_ps(_mm_mul_ps(quant, scale), minimum);
    *accumulator = _mm_add_ps(*accumulator, _mm_mul_ps(weights, _mm_loadu_ps(input)));
}

#[target_feature(enable = "sse2")]
unsafe fn dot_row_sse2_inner(row: &[u8], input: &[f32]) -> f32 {
    let zero = _mm_setzero_si128();
    let nibble_mask = _mm_set1_epi8(0x0f);
    let mut sum = _mm_setzero_ps();

    for (block_index, block) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let quants = block.as_ptr().add(16);
        let inputs = input.as_ptr().add(block_index * BLOCK_VALUES);

        for band in 0..4 {
            let low_group = band * 2;
            let high_group = low_group + 1;
            let (low_scale, low_min) = scale_min(scales, low_group);
            let (high_scale, high_min) = scale_min(scales, high_group);
            let low_scale = _mm_set1_ps(d * f32::from(low_scale));
            let low_min = _mm_set1_ps(dmin * f32::from(low_min));
            let high_scale = _mm_set1_ps(d * f32::from(high_scale));
            let high_min = _mm_set1_ps(dmin * f32::from(high_min));

            for chunk in [0, 16] {
                let packed = _mm_loadu_si128(quants.add(band * 32 + chunk).cast::<__m128i>());
                let low = _mm_and_si128(packed, nibble_mask);
                let high = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
                let low_words = [_mm_unpacklo_epi8(low, zero), _mm_unpackhi_epi8(low, zero)];
                let high_words = [_mm_unpacklo_epi8(high, zero), _mm_unpackhi_epi8(high, zero)];

                for half in 0..2 {
                    let low_quants = [
                        _mm_unpacklo_epi16(low_words[half], zero),
                        _mm_unpackhi_epi16(low_words[half], zero),
                    ];
                    let high_quants = [
                        _mm_unpacklo_epi16(high_words[half], zero),
                        _mm_unpackhi_epi16(high_words[half], zero),
                    ];
                    for quarter in 0..2 {
                        let offset = chunk + half * 8 + quarter * 4;
                        accumulate_sse2(
                            &mut sum,
                            low_quants[quarter],
                            inputs.add(low_group * 32 + offset),
                            low_scale,
                            low_min,
                        );
                        accumulate_sse2(
                            &mut sum,
                            high_quants[quarter],
                            inputs.add(high_group * 32 + offset),
                            high_scale,
                            high_min,
                        );
                    }
                }
            }
        }
    }

    hsum_sse2(sum)
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn hsum_sse2(value: __m128) -> f32 {
    let high = _mm_movehl_ps(value, value);
    let pairs = _mm_add_ps(value, high);
    let second = _mm_shuffle_ps(pairs, pairs, 0x55);
    _mm_cvtss_f32(_mm_add_ss(pairs, second))
}
