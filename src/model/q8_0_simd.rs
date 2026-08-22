//! SIMD Q8_0 row dot products selected at runtime by `q8_0::dot_row`.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::error::WillametteError;
use crate::model::primitives::f16_to_f32;

const BLOCK_BYTES: usize = 34;
const BLOCK_VALUES: usize = 32;

#[cfg(target_arch = "x86")]
type M128 = core::arch::x86::__m128;
#[cfg(target_arch = "x86_64")]
type M128 = core::arch::x86_64::__m128;

#[inline]
fn scale(block: &[u8]) -> Result<f32, WillametteError> {
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    if !scale.is_finite() {
        return Err(WillametteError::GgufParse(
            "Q8_0 block has a non-finite scale".to_string(),
        ));
    }
    Ok(scale)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dot_row_neon(data: &[u8], input: &[f32]) -> Result<f32, WillametteError> {
    use core::arch::aarch64::{
        vaddvq_f32, vcvtq_f32_s32, vdupq_n_f32, vfmaq_f32, vget_high_s16, vget_high_s8,
        vget_low_s16, vget_low_s8, vld1q_f32, vld1q_s8, vmovl_s16, vmovl_s8,
    };

    let mut total = 0.0_f32;
    for (block_index, block) in data.chunks_exact(BLOCK_BYTES).enumerate() {
        let values = block.as_ptr().add(2).cast::<i8>();
        let input = input.as_ptr().add(block_index * BLOCK_VALUES);
        let mut sum = vdupq_n_f32(0.0);
        for offset in [0, 16] {
            let quant = vld1q_s8(values.add(offset));
            let low = vmovl_s8(vget_low_s8(quant));
            let high = vmovl_s8(vget_high_s8(quant));
            sum = vfmaq_f32(
                sum,
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(low))),
                vld1q_f32(input.add(offset)),
            );
            sum = vfmaq_f32(
                sum,
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(low))),
                vld1q_f32(input.add(offset + 4)),
            );
            sum = vfmaq_f32(
                sum,
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(high))),
                vld1q_f32(input.add(offset + 8)),
            );
            sum = vfmaq_f32(
                sum,
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(high))),
                vld1q_f32(input.add(offset + 12)),
            );
        }
        total += scale(block)? * vaddvq_f32(sum);
    }
    Ok(total)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_row_avx2(data: &[u8], input: &[f32]) -> Result<f32, WillametteError> {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m128i, _mm256_add_ps, _mm256_castps256_ps128, _mm256_cvtepi32_ps, _mm256_cvtepi8_epi32,
        _mm256_extractf128_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm_add_ps,
        _mm_loadl_epi64,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m128i, _mm256_add_ps, _mm256_castps256_ps128, _mm256_cvtepi32_ps, _mm256_cvtepi8_epi32,
        _mm256_extractf128_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm_add_ps,
        _mm_loadl_epi64,
    };

    let mut total = 0.0_f32;
    for (block_index, block) in data.chunks_exact(BLOCK_BYTES).enumerate() {
        let values = block.as_ptr().add(2).cast::<i8>();
        let input = input.as_ptr().add(block_index * BLOCK_VALUES);
        let mut sum = _mm256_setzero_ps();
        for offset in [0, 8, 16, 24] {
            let quant = _mm_loadl_epi64(values.add(offset).cast::<__m128i>());
            let quant = _mm256_cvtepi8_epi32(quant);
            let quant = _mm256_cvtepi32_ps(quant);
            sum = _mm256_add_ps(
                sum,
                _mm256_mul_ps(quant, _mm256_loadu_ps(input.add(offset))),
            );
        }
        let halves = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
        total += scale(block)? * hsum_sse(halves);
    }
    Ok(total)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
pub unsafe fn dot_row_sse2(data: &[u8], input: &[f32]) -> Result<f32, WillametteError> {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        _mm_add_ps, _mm_cvtepi32_ps, _mm_cvtsi32_si128, _mm_loadu_ps, _mm_mul_ps, _mm_setzero_ps,
        _mm_srai_epi16, _mm_srai_epi32, _mm_unpacklo_epi16, _mm_unpacklo_epi8,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        _mm_add_ps, _mm_cvtepi32_ps, _mm_cvtsi32_si128, _mm_loadu_ps, _mm_mul_ps, _mm_setzero_ps,
        _mm_srai_epi16, _mm_srai_epi32, _mm_unpacklo_epi16, _mm_unpacklo_epi8,
    };

    let mut total = 0.0_f32;
    for (block_index, block) in data.chunks_exact(BLOCK_BYTES).enumerate() {
        let values = block.as_ptr().add(2);
        let input = input.as_ptr().add(block_index * BLOCK_VALUES);
        let mut sum = _mm_setzero_ps();
        for offset in (0..BLOCK_VALUES).step_by(4) {
            let packed = i32::from_le_bytes([
                *values.add(offset),
                *values.add(offset + 1),
                *values.add(offset + 2),
                *values.add(offset + 3),
            ]);
            let quant_i8 = _mm_cvtsi32_si128(packed);
            let quant_i16 = _mm_srai_epi16(_mm_unpacklo_epi8(quant_i8, quant_i8), 8);
            let quant_i32 = _mm_srai_epi32(_mm_unpacklo_epi16(quant_i16, quant_i16), 16);
            let quant = _mm_cvtepi32_ps(quant_i32);
            sum = _mm_add_ps(sum, _mm_mul_ps(quant, _mm_loadu_ps(input.add(offset))));
        }
        total += scale(block)? * hsum_sse(sum);
    }
    Ok(total)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn hsum_sse(value: M128) -> f32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{_mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehl_ps, _mm_shuffle_ps};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehl_ps, _mm_shuffle_ps,
    };

    let high = _mm_movehl_ps(value, value);
    let pairs = _mm_add_ps(value, high);
    let second = _mm_shuffle_ps(pairs, pairs, 0x55);
    _mm_cvtss_f32(_mm_add_ss(pairs, second))
}
