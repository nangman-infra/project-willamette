//! x86 SIMD Q4_K-by-f32 row dot products selected by `q4_k::dot_row`.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#![allow(unsafe_op_in_unsafe_fn)]

use super::{
    primitives::f16_to_f32,
    q4_k::{dot_row_q8_scalar_validated, scale_min, Q8KBlock},
};

const BLOCK_BYTES: usize = 144;
const BLOCK_VALUES: usize = 256;
const TOKEN_TILE: usize = 4;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m128, __m128i, __m256, __m256i, _mm256_add_epi32, _mm256_add_ps, _mm256_and_si256,
    _mm256_castps256_ps128, _mm256_cvtepi32_ps, _mm256_cvtepu8_epi32, _mm256_extractf128_ps,
    _mm256_loadu_ps, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_mul_ps,
    _mm256_set1_epi16, _mm256_set1_epi8, _mm256_set1_ps, _mm256_setzero_ps, _mm256_setzero_si256,
    _mm256_srai_epi16, _mm256_sub_ps, _mm_add_ps, _mm_add_ss, _mm_and_si128, _mm_cvtepi32_ps,
    _mm_cvtss_f32, _mm_loadl_epi64, _mm_loadu_ps, _mm_loadu_si128, _mm_movehl_ps, _mm_mul_ps,
    _mm_set1_epi8, _mm_set1_ps, _mm_setzero_ps, _mm_setzero_si128, _mm_shuffle_ps, _mm_srli_epi16,
    _mm_sub_ps, _mm_unpackhi_epi16, _mm_unpackhi_epi8, _mm_unpacklo_epi16, _mm_unpacklo_epi8,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128, __m128i, __m256, __m256i, _mm256_add_epi32, _mm256_add_ps, _mm256_and_si256,
    _mm256_castps256_ps128, _mm256_cvtepi32_ps, _mm256_cvtepu8_epi32, _mm256_extractf128_ps,
    _mm256_loadu_ps, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_mul_ps,
    _mm256_set1_epi16, _mm256_set1_epi8, _mm256_set1_ps, _mm256_setzero_ps, _mm256_setzero_si256,
    _mm256_srai_epi16, _mm256_sub_ps, _mm_add_ps, _mm_add_ss, _mm_and_si128, _mm_cvtepi32_ps,
    _mm_cvtss_f32, _mm_loadl_epi64, _mm_loadu_ps, _mm_loadu_si128, _mm_movehl_ps, _mm_mul_ps,
    _mm_set1_epi8, _mm_set1_ps, _mm_setzero_ps, _mm_setzero_si128, _mm_shuffle_ps, _mm_srli_epi16,
    _mm_sub_ps, _mm_unpackhi_epi16, _mm_unpackhi_epi8, _mm_unpacklo_epi16, _mm_unpacklo_epi8,
};

/// The caller must have selected AVX2 and validated matching complete blocks.
pub(super) unsafe fn dot_row_q8_avx2_validated(row: &[u8], input: &[Q8KBlock]) -> f32 {
    debug_assert_eq!(row.len() / BLOCK_BYTES, input.len());
    dot_row_q8_avx2_inner(row, input)
}

pub(super) fn dot_row_q8_sse2_validated(row: &[u8], input: &[Q8KBlock]) -> f32 {
    assert!(std::arch::is_x86_feature_detected!("sse2"));
    debug_assert_eq!(row.len() / BLOCK_BYTES, input.len());
    dot_row_q8_scalar_validated(row, input)
}

#[target_feature(enable = "avx2")]
unsafe fn dot_row_q8_avx2_inner(row: &[u8], input: &[Q8KBlock]) -> f32 {
    let nibble_mask = _mm256_set1_epi8(0x0f);
    let mut scaled_sum = _mm256_setzero_ps();
    let mut minimum_sum = 0.0_f32;

    for (block, activation) in row.chunks_exact(BLOCK_BYTES).zip(input) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let quants = block.as_ptr().add(16);
        let mut block_scaled_sum = _mm256_setzero_si256();
        let mut block_minimum_sum = 0_i32;

        for band in 0..4 {
            let low_group = band * 2;
            let high_group = low_group + 1;
            let packed = _mm256_loadu_si256(quants.add(band * 32).cast::<__m256i>());
            let low_q4 = _mm256_and_si256(packed, nibble_mask);
            let high_q4 = _mm256_and_si256(_mm256_srai_epi16(packed, 4), nibble_mask);
            let low_q8 =
                _mm256_loadu_si256(activation.qs.as_ptr().add(low_group * 32).cast::<__m256i>());
            let high_q8 = _mm256_loadu_si256(
                activation
                    .qs
                    .as_ptr()
                    .add(high_group * 32)
                    .cast::<__m256i>(),
            );
            let (low_scale, low_minimum) = scale_min(scales, low_group);
            let (high_scale, high_minimum) = scale_min(scales, high_group);
            let low_pairs = _mm256_maddubs_epi16(low_q4, low_q8);
            let high_pairs = _mm256_maddubs_epi16(high_q4, high_q8);
            block_scaled_sum = _mm256_add_epi32(
                block_scaled_sum,
                _mm256_madd_epi16(low_pairs, _mm256_set1_epi16(i16::from(low_scale))),
            );
            block_scaled_sum = _mm256_add_epi32(
                block_scaled_sum,
                _mm256_madd_epi16(high_pairs, _mm256_set1_epi16(i16::from(high_scale))),
            );
            block_minimum_sum += i32::from(low_minimum)
                * (i32::from(activation.bsums[low_group * 2])
                    + i32::from(activation.bsums[low_group * 2 + 1]));
            block_minimum_sum += i32::from(high_minimum)
                * (i32::from(activation.bsums[high_group * 2])
                    + i32::from(activation.bsums[high_group * 2 + 1]));
        }

        scaled_sum = _mm256_add_ps(
            scaled_sum,
            _mm256_mul_ps(
                _mm256_cvtepi32_ps(block_scaled_sum),
                _mm256_set1_ps(d * activation.d),
            ),
        );
        minimum_sum += dmin * activation.d * block_minimum_sum as f32;
    }

    let halves = _mm_add_ps(
        _mm256_castps256_ps128(scaled_sum),
        _mm256_extractf128_ps(scaled_sum, 1),
    );
    hsum_sse2(halves) - minimum_sum
}

/// Run the AVX2 kernel after checking that this process may execute it.
pub(super) fn dot_row_avx2_validated(row: &[u8], input: &[f32]) -> f32 {
    assert!(std::arch::is_x86_feature_detected!("avx2"));
    debug_assert_eq!(row.len() / BLOCK_BYTES * BLOCK_VALUES, input.len());
    // SAFETY: the runtime check establishes AVX2 support; q4_k::dot_row has
    // validated complete blocks and matching lengths before calling us.
    unsafe { dot_row_avx2_inner(row, input) }
}

/// Run the batched AVX2 kernel after checking that this process may execute it.
pub(super) fn dot_rows_avx2_validated(
    row: &[u8],
    inputs: &[f32],
    input_dim: usize,
    outputs: &mut [f32],
) {
    assert!(std::arch::is_x86_feature_detected!("avx2"));
    debug_assert_eq!(row.len() / BLOCK_BYTES * BLOCK_VALUES, input_dim);
    debug_assert_eq!(inputs.len(), input_dim * outputs.len());
    // SAFETY: the runtime check establishes AVX2 support; q4_k::dot_rows has
    // validated complete blocks, positive token count, and matching lengths.
    unsafe { dot_rows_avx2_inner(row, inputs, input_dim, outputs) }
}

/// Run the SSE2 kernel after checking that this process may execute it.
pub(super) fn dot_row_sse2_validated(row: &[u8], input: &[f32]) -> f32 {
    assert!(std::arch::is_x86_feature_detected!("sse2"));
    debug_assert_eq!(row.len() / BLOCK_BYTES * BLOCK_VALUES, input.len());
    // SAFETY: the runtime check establishes SSE2 support; q4_k::dot_row has
    // validated complete blocks and matching lengths before calling us.
    unsafe { dot_row_sse2_inner(row, input) }
}

/// Run the batched SSE2 kernel after checking that this process may execute it.
pub(super) fn dot_rows_sse2_validated(
    row: &[u8],
    inputs: &[f32],
    input_dim: usize,
    outputs: &mut [f32],
) {
    assert!(std::arch::is_x86_feature_detected!("sse2"));
    debug_assert_eq!(row.len() / BLOCK_BYTES * BLOCK_VALUES, input_dim);
    debug_assert_eq!(inputs.len(), input_dim * outputs.len());
    // SAFETY: the runtime check establishes SSE2 support; q4_k::dot_rows has
    // validated complete blocks, positive token count, and matching lengths.
    unsafe { dot_rows_sse2_inner(row, inputs, input_dim, outputs) }
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

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_avx2_weights(accumulator: &mut __m256, weights: __m256, input: *const f32) {
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

#[target_feature(enable = "avx2")]
unsafe fn dot_rows_avx2_inner(row: &[u8], inputs: &[f32], input_dim: usize, outputs: &mut [f32]) {
    let nibble_mask = _mm_set1_epi8(0x0f);

    for tile_start in (0..outputs.len()).step_by(TOKEN_TILE) {
        let tile_len = (outputs.len() - tile_start).min(TOKEN_TILE);
        let mut sums = [_mm256_setzero_ps(); TOKEN_TILE];

        for (block_index, block) in row.chunks_exact(BLOCK_BYTES).enumerate() {
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let scales = &block[4..16];
            let quants = block.as_ptr().add(16);
            let block_offset = block_index * BLOCK_VALUES;

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
                    let low = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(low));
                    let high = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(high));
                    let low_weights = _mm256_sub_ps(_mm256_mul_ps(low, low_scale), low_min);
                    let high_weights = _mm256_sub_ps(_mm256_mul_ps(high, high_scale), high_min);

                    for (token, sum) in sums.iter_mut().enumerate().take(tile_len) {
                        let input = inputs.as_ptr().add(
                            (tile_start + token) * input_dim
                                + block_offset
                                + low_group * 32
                                + chunk,
                        );
                        accumulate_avx2_weights(sum, low_weights, input);
                    }
                    for (token, sum) in sums.iter_mut().enumerate().take(tile_len) {
                        let input = inputs.as_ptr().add(
                            (tile_start + token) * input_dim
                                + block_offset
                                + high_group * 32
                                + chunk,
                        );
                        accumulate_avx2_weights(sum, high_weights, input);
                    }
                }
            }
        }

        for token in 0..tile_len {
            let halves = _mm_add_ps(
                _mm256_castps256_ps128(sums[token]),
                _mm256_extractf128_ps(sums[token], 1),
            );
            outputs[tile_start + token] = hsum_sse2(halves);
        }
    }
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

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn accumulate_sse2_weights(accumulator: &mut __m128, weights: __m128, input: *const f32) {
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

#[target_feature(enable = "sse2")]
unsafe fn dot_rows_sse2_inner(row: &[u8], inputs: &[f32], input_dim: usize, outputs: &mut [f32]) {
    let zero = _mm_setzero_si128();
    let nibble_mask = _mm_set1_epi8(0x0f);

    for tile_start in (0..outputs.len()).step_by(TOKEN_TILE) {
        let tile_len = (outputs.len() - tile_start).min(TOKEN_TILE);
        let mut sums = [_mm_setzero_ps(); TOKEN_TILE];

        for (block_index, block) in row.chunks_exact(BLOCK_BYTES).enumerate() {
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let scales = &block[4..16];
            let quants = block.as_ptr().add(16);
            let block_offset = block_index * BLOCK_VALUES;

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
                            let low = _mm_cvtepi32_ps(low_quants[quarter]);
                            let high = _mm_cvtepi32_ps(high_quants[quarter]);
                            let low_weights = _mm_sub_ps(_mm_mul_ps(low, low_scale), low_min);
                            let high_weights = _mm_sub_ps(_mm_mul_ps(high, high_scale), high_min);

                            for (token, sum) in sums.iter_mut().enumerate().take(tile_len) {
                                let input = inputs.as_ptr().add(
                                    (tile_start + token) * input_dim
                                        + block_offset
                                        + low_group * 32
                                        + offset,
                                );
                                accumulate_sse2_weights(sum, low_weights, input);
                            }
                            for (token, sum) in sums.iter_mut().enumerate().take(tile_len) {
                                let input = inputs.as_ptr().add(
                                    (tile_start + token) * input_dim
                                        + block_offset
                                        + high_group * 32
                                        + offset,
                                );
                                accumulate_sse2_weights(sum, high_weights, input);
                            }
                        }
                    }
                }
            }
        }

        for token in 0..tile_len {
            outputs[tile_start + token] = hsum_sse2(sums[token]);
        }
    }
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn hsum_sse2(value: __m128) -> f32 {
    let high = _mm_movehl_ps(value, value);
    let pairs = _mm_add_ps(value, high);
    let second = _mm_shuffle_ps(pairs, pairs, 0x55);
    _mm_cvtss_f32(_mm_add_ss(pairs, second))
}
