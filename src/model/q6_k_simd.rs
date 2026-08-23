//! AVX2 Q6_K-by-Q8_K row dot product for prequantized single-token decode.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#![allow(unsafe_op_in_unsafe_fn)]

use super::{primitives::f16_to_f32, q4_k::Q8KBlock, q6_k::unpack_block_levels};

const BLOCK_BYTES: usize = 210;
const BLOCK_VALUES: usize = 256;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m256i, _mm256_add_epi32, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
    _mm256_setr_epi16, _mm256_setzero_si256, _mm256_storeu_si256,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256i, _mm256_add_epi32, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
    _mm256_setr_epi16, _mm256_setzero_si256, _mm256_storeu_si256,
};

/// The caller must have selected AVX2 once and validated matching complete blocks.
pub(super) unsafe fn dot_row_q8_avx2_validated(row: &[u8], input: &[Q8KBlock]) -> f32 {
    debug_assert_eq!(row.len() / BLOCK_BYTES, input.len());
    dot_row_q8_avx2_inner(row, input)
}

#[target_feature(enable = "avx2")]
unsafe fn dot_row_q8_avx2_inner(row: &[u8], input: &[Q8KBlock]) -> f32 {
    let mut sum = 0.0_f32;
    let mut levels = [0_u8; BLOCK_VALUES];
    for (block, activation) in row.chunks_exact(BLOCK_BYTES).zip(input) {
        unpack_block_levels(block, &mut levels);
        let scales = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let mut dots = _mm256_setzero_si256();
        let mut correction = 0_i32;

        for group in (0..16).step_by(2) {
            let offset = group * 16;
            let q6 = _mm256_loadu_si256(levels.as_ptr().add(offset).cast::<__m256i>());
            let q8 = _mm256_loadu_si256(activation.qs.as_ptr().add(offset).cast::<__m256i>());
            let pairs = _mm256_maddubs_epi16(q6, q8);
            let low_scale = i16::from(scales[group] as i8);
            let high_scale = i16::from(scales[group + 1] as i8);
            let scale = _mm256_setr_epi16(
                low_scale, low_scale, low_scale, low_scale, low_scale, low_scale, low_scale,
                low_scale, high_scale, high_scale, high_scale, high_scale, high_scale, high_scale,
                high_scale, high_scale,
            );
            dots = _mm256_add_epi32(dots, _mm256_madd_epi16(pairs, scale));
            correction += 32
                * (i32::from(scales[group] as i8) * i32::from(activation.bsums[group])
                    + i32::from(scales[group + 1] as i8) * i32::from(activation.bsums[group + 1]));
        }

        let mut lanes = [0_i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), dots);
        let block_sum = lanes.into_iter().sum::<i32>() - correction;
        sum += d * activation.d * block_sum as f32;
    }
    sum
}
