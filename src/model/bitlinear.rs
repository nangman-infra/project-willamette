// Hot tensor inner loops are clearer with explicit index variables
// (row j × block bk × byte gp) than with iterator chains.
#![allow(clippy::needless_range_loop)]

//! Stage 4-C — scalar reference BitLinear / I2_S matvec.
//!
//! Single function, used by all seven BitLinear roles
//! (`attn_q`/`attn_k`/`attn_v`/`attn_output`, `ffn_gate`/`ffn_up`/`ffn_down`):
//!
//! ```text
//! out[j] = i2_scale * Σᵢ ternary(W[j,i]) * input[i]
//! ```
//!
//! Pinned semantics: [`docs/BITLINEAR_I2S_MATVEC.md`](../../docs/BITLINEAR_I2S_MATVEC.md).
//!
//! Operates directly on the packed `TensorView::data` bytes and reads the
//! per-tensor `i2_scale` from `TensorView::scale_data`. Never expands the
//! full weight to f32. Does NOT bake `*_sub_norm` into its body — the
//! caller applies any pre-quant RMSNorm.
//!
//! Stage 4-C non-goals: SIMD, int8 activation quantisation, full
//! transformer composition, KV cache, attention, FFN, sampling.

use crate::error::WillametteError;
use crate::gguf::tensor::TensorView;
use crate::gguf::types::GgmlType;

/// QK_I2_S — number of ternary elements per packed block (CPU layout).
/// Mirrors [`TensorView::I2S_ELEMENTS_PER_BLOCK`] for local readability;
/// kept private so callers go through the single source of truth.
const QK_I2_S: usize = 128;

/// Packed bytes per block: 128 elements × 2 bits / 8 bits = 32 bytes.
const PACKED_BYTES_PER_BLOCK: usize = 32;

/// Map a 2-bit code to its ternary integer value.
///
/// `00 → -1`, `01 → 0`, `10 → +1`, `11 → 0`. The quantizer at
/// `src/ggml-bitnet-mad.cpp:65..72` of the pinned commit never produces
/// `11`; the degenerate case is mapped to zero so the function never
/// panics.
#[inline]
pub fn ternary_from_code(code: u8) -> i8 {
    match code & 0b11 {
        0b00 => -1,
        0b01 => 0,
        0b10 => 1,
        _ => 0, // 0b11 — degenerate; map to 0 per ggml-quants.c:3898 (map2bit[3] = 0.0)
    }
}

/// I2_S BitLinear matvec — `output = i2_scale * (W_ternary · input)`.
///
/// See [`docs/BITLINEAR_I2S_MATVEC.md`](../../docs/BITLINEAR_I2S_MATVEC.md) §7
/// for the contract enforced at the boundary.
///
/// Dispatches at runtime to the fastest available backend:
///
/// * On `aarch64` with NEON detected (Apple Silicon is always such a
///   host), routes to [`super::bitlinear_neon::bitlinear_i2s_matvec_f32_neon`].
/// * Otherwise, falls back to [`bitlinear_i2s_matvec_f32_scalar`].
///
/// Numerical contract for the NEON path: the result is **not** bit-identical
/// to the scalar reference (the parallel accumulator reduces in a
/// different order); per-element absolute differences are bounded — see
/// `tests/bitlinear_simd.rs` for the documented tolerance.
pub fn bitlinear_i2s_matvec_f32(
    weight: &TensorView<'_>,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), WillametteError> {
    // Single source of truth for kernel selection — see
    // `src/model/dispatch.rs`. The first call resolves CPU features
    // once and caches the choice; subsequent calls are an atomic
    // pointer load. Dashboard `active_kernel` label comes from the
    // same place, so they can't drift apart.
    match super::dispatch::active_kernel() {
        #[cfg(target_arch = "aarch64")]
        super::dispatch::Kernel::AArch64Neon => {
            // Stage 10-D finding: an int8-activation kernel
            // (`bitlinear_i2s_matvec_f32_neon_i8`) was implemented and is
            // numerically correct (greedy decode produces identical
            // tokens to scalar on Stage 5-E reference prompts). But on
            // stable Rust the `vdotq_s32` SDOT instruction is gated
            // behind the unstable `stdarch_neon_dotprod` feature, forcing
            // a `vmull_s8`-based widening dot product. Across 20-sample
            // decode-step averages on Apple M4 the int8 path ran at 7.82
            // tok/sec vs the f32-input NEON path's 7.91 tok/sec — a
            // measured regression. Default is f32-NEON; switch to the
            // int8 path with `RUSTFLAGS="--cfg willamette_i8_activations"`
            // once nightly dotprod stabilises.
            #[cfg(willamette_i8_activations)]
            return unsafe {
                super::bitlinear_neon::bitlinear_i2s_matvec_f32_neon_i8(weight, input, output)
            };
            #[cfg(not(willamette_i8_activations))]
            unsafe {
                super::bitlinear_neon::bitlinear_i2s_matvec_f32_neon(weight, input, output)
            }
        }
        // Stage 6-B: x86 / x86_64 SSE2 BitLinear matvec. Routed when
        // `dispatch::select_kernel` saw SSE2 advertised on this host.
        // `target_feature(enable = "sse2")` on the callee permits the
        // compiler to emit SSE2 ops; the runtime detection in
        // dispatch::select_kernel is what makes the unsafe call sound.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        super::dispatch::Kernel::X86Sse2 => unsafe {
            // i8 activation kernel is the DEFAULT on x86. Measured on
            // antix1 (Pentium-M): 2.2× faster than the f32 mask-add
            // path (docs/BENCHMARKS.md), and greedy decode on the real
            // 2B model produces byte-identical tokens to f32 (20/20 on
            // "The capital of France is" — int8 quantisation never
            // flipped an argmax). Unlike NEON (where i8 was slower so
            // f32 stays default), x86 i8 wins outright.
            //
            // The f32 mask-add kernel (bit-close to scalar) stays
            // available under `--cfg willamette_sse2_f32` for
            // debugging / numerical reference.
            #[cfg(willamette_sse2_f32)]
            return super::bitlinear_sse2::bitlinear_i2s_matvec_f32_sse2(weight, input, output);
            #[cfg(not(willamette_sse2_f32))]
            super::bitlinear_sse2::bitlinear_i2s_matvec_f32_sse2_i8(weight, input, output)
        },
        _ => bitlinear_i2s_matvec_f32_scalar(weight, input, output),
    }
}

/// Scalar reference matvec (pre-Stage 6 path). Always available; used
/// by tests and as the fallback when no SIMD backend is detected.
pub fn bitlinear_i2s_matvec_f32_scalar(
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
            "bitlinear_i2s_matvec_f32: weight {:?} is not 2-D (shape={:?})",
            weight.name, weight.shape
        )));
    }

    let in_dim = weight.shape[0] as usize;
    let out_dim = weight.shape[1] as usize;

    if input.len() != in_dim {
        return Err(WillametteError::GgufParse(format!(
            "bitlinear_i2s_matvec_f32: input.len()={} != in_dim={} (weight {:?})",
            input.len(),
            in_dim,
            weight.name
        )));
    }
    if output.len() != out_dim {
        return Err(WillametteError::GgufParse(format!(
            "bitlinear_i2s_matvec_f32: output.len()={} != out_dim={} (weight {:?})",
            output.len(),
            out_dim,
            weight.name
        )));
    }
    if in_dim == 0 || !in_dim.is_multiple_of(QK_I2_S) {
        return Err(WillametteError::GgufParse(format!(
            "bitlinear_i2s_matvec_f32: in_dim {} is not a positive multiple of {} (QK_I2_S)",
            in_dim, QK_I2_S
        )));
    }

    let bytes_per_row = in_dim / 4;
    let blocks_per_row = in_dim / QK_I2_S;
    let expected_packed = bytes_per_row * out_dim;
    if weight.data.len() != expected_packed {
        return Err(WillametteError::GgufParse(format!(
            "bitlinear_i2s_matvec_f32: weight {:?} data.len()={} != expected packed {} (= in_dim/4 × out_dim)",
            weight.name,
            weight.data.len(),
            expected_packed
        )));
    }

    let scale = weight.i2s_scale()?;
    let packed = weight.data;

    for j in 0..out_dim {
        let row_offset = j * bytes_per_row;
        // Two-accumulator form avoids `× (±1.0)` multiplications and is
        // numerically more stable than a single running sum.
        let mut pos: f32 = 0.0;
        let mut neg: f32 = 0.0;
        for bk in 0..blocks_per_row {
            let block_offset = row_offset + bk * PACKED_BYTES_PER_BLOCK;
            let col_base = bk * QK_I2_S;
            for gp in 0..PACKED_BYTES_PER_BLOCK {
                let b = packed[block_offset + gp];
                // c0 → col_base + gp + 0
                let c0 = (b >> 6) & 0b11;
                accumulate(&mut pos, &mut neg, c0, input[col_base + gp]);
                // c1 → col_base + gp + 32
                let c1 = (b >> 4) & 0b11;
                accumulate(&mut pos, &mut neg, c1, input[col_base + gp + 32]);
                // c2 → col_base + gp + 64
                let c2 = (b >> 2) & 0b11;
                accumulate(&mut pos, &mut neg, c2, input[col_base + gp + 64]);
                // c3 → col_base + gp + 96
                let c3 = b & 0b11;
                accumulate(&mut pos, &mut neg, c3, input[col_base + gp + 96]);
            }
        }
        output[j] = scale * (pos - neg);
    }

    Ok(())
}

#[inline]
fn accumulate(pos: &mut f32, neg: &mut f32, code: u8, x: f32) {
    match code {
        0b00 => *neg += x, // code 0 → ternary −1
        0b10 => *pos += x, // code 2 → ternary +1
        _ => {}            // code 1 → 0, code 3 → degenerate 0
    }
}

/// Debug helper — unpack one row of an I2_S tensor into a `Vec<i8>` of
/// length `in_dim` whose values are in `{-1, 0, +1}`. Not used by the
/// matvec hot path. Useful for synthetic-fixture tests.
pub fn i2s_unpack_row_to_i8(
    weight: &TensorView<'_>,
    row_idx: usize,
) -> Result<Vec<i8>, WillametteError> {
    if weight.ggml_type != GgmlType::BitNetI2S {
        return Err(WillametteError::UnsupportedTensorType(
            weight.ggml_type.to_raw(),
        ));
    }
    if weight.shape.len() != 2 {
        return Err(WillametteError::GgufParse(format!(
            "i2s_unpack_row_to_i8: weight {:?} is not 2-D",
            weight.name
        )));
    }
    let in_dim = weight.shape[0] as usize;
    let out_dim = weight.shape[1] as usize;
    if row_idx >= out_dim {
        return Err(WillametteError::GgufParse(format!(
            "i2s_unpack_row_to_i8: row_idx {} out of range (out_dim={})",
            row_idx, out_dim
        )));
    }
    if in_dim == 0 || !in_dim.is_multiple_of(QK_I2_S) {
        return Err(WillametteError::GgufParse(format!(
            "i2s_unpack_row_to_i8: in_dim {} is not a positive multiple of {}",
            in_dim, QK_I2_S
        )));
    }
    let bytes_per_row = in_dim / 4;
    let blocks_per_row = in_dim / QK_I2_S;
    let row_offset = row_idx * bytes_per_row;
    if row_offset + bytes_per_row > weight.data.len() {
        return Err(WillametteError::GgufParse(format!(
            "i2s_unpack_row_to_i8: row bytes out of range for weight {:?}",
            weight.name
        )));
    }
    let mut out = vec![0_i8; in_dim];
    let packed = weight.data;
    for bk in 0..blocks_per_row {
        let block_offset = row_offset + bk * PACKED_BYTES_PER_BLOCK;
        let col_base = bk * QK_I2_S;
        for gp in 0..PACKED_BYTES_PER_BLOCK {
            let b = packed[block_offset + gp];
            out[col_base + gp] = ternary_from_code((b >> 6) & 0b11);
            out[col_base + gp + 32] = ternary_from_code((b >> 4) & 0b11);
            out[col_base + gp + 64] = ternary_from_code((b >> 2) & 0b11);
            out[col_base + gp + 96] = ternary_from_code(b & 0b11);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic I2_S TensorView whose packed area is `packed`
    /// and whose scale is `scale`. The caller owns the backing buffers
    /// (returned alongside the view) so they outlive the borrow.
    struct SyntheticI2S {
        packed: Vec<u8>,
        scale_block: [u8; 32],
    }
    impl SyntheticI2S {
        fn new(packed: Vec<u8>, scale: f32) -> Self {
            let mut scale_block = [0u8; 32];
            scale_block[..4].copy_from_slice(&scale.to_le_bytes());
            Self {
                packed,
                scale_block,
            }
        }
        fn view<'a>(&'a self, name: &str, in_dim: u64, out_dim: u64) -> TensorView<'a> {
            TensorView {
                name: name.to_string(),
                shape: vec![in_dim, out_dim],
                ggml_type: GgmlType::BitNetI2S,
                offset: 0,
                byte_len: self.packed.len() as u64,
                data: &self.packed,
                scale_data: Some(&self.scale_block),
            }
        }
    }

    #[test]
    fn ternary_code_table() {
        assert_eq!(ternary_from_code(0b00), -1);
        assert_eq!(ternary_from_code(0b01), 0);
        assert_eq!(ternary_from_code(0b10), 1);
        assert_eq!(ternary_from_code(0b11), 0);
    }

    #[test]
    fn matvec_all_plus_one_weights() {
        // Single row, 128 elements, all +1 → byte pattern 0xAA (code 10 in each pair).
        let packed = vec![0xAA_u8; 32];
        let fx = SyntheticI2S::new(packed, 1.0);
        let w = fx.view("all_plus", 128, 1);

        let input: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1).collect();
        let mut out = vec![0.0_f32; 1];
        bitlinear_i2s_matvec_f32(&w, &input, &mut out).unwrap();

        let expected: f32 = input.iter().sum();
        assert!(
            (out[0] - expected).abs() < 1e-4,
            "all-+1 weights should give Σx: got {}, expected {}",
            out[0],
            expected
        );
    }

    #[test]
    fn matvec_all_minus_one_weights() {
        // Code 0b00 = -1 → byte 0x00.
        let packed = vec![0x00_u8; 32];
        let fx = SyntheticI2S::new(packed, 1.0);
        let w = fx.view("all_minus", 128, 1);

        let input: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1).collect();
        let mut out = vec![0.0_f32; 1];
        bitlinear_i2s_matvec_f32(&w, &input, &mut out).unwrap();

        let expected: f32 = -input.iter().sum::<f32>();
        assert!(
            (out[0] - expected).abs() < 1e-4,
            "all-(-1) weights should give -Σx: got {}, expected {}",
            out[0],
            expected
        );
    }

    #[test]
    fn matvec_all_zero_weights() {
        // Code 0b01 = 0 → byte 0x55 (01_01_01_01).
        let packed = vec![0x55_u8; 32];
        let fx = SyntheticI2S::new(packed, 1.0);
        let w = fx.view("all_zero", 128, 1);

        let input: Vec<f32> = (0..128).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let mut out = vec![123.0_f32; 1]; // sentinel
        bitlinear_i2s_matvec_f32(&w, &input, &mut out).unwrap();

        assert_eq!(
            out[0], 0.0,
            "zero weights × any input must produce exactly 0.0"
        );
    }

    #[test]
    fn matvec_scale_applied_at_end() {
        let packed = vec![0xAA_u8; 32];
        let fx = SyntheticI2S::new(packed, 3.5);
        let w = fx.view("scaled", 128, 1);

        let input = vec![2.0_f32; 128];
        let mut out = vec![0.0_f32; 1];
        bitlinear_i2s_matvec_f32(&w, &input, &mut out).unwrap();

        // all +1 weights, input=2.0, scale=3.5 → 3.5 × (128 × 2.0) = 896
        assert!((out[0] - 896.0).abs() < 1e-3, "got {}", out[0]);
    }

    #[test]
    fn matvec_two_rows_different_patterns() {
        // Row 0: all +1 (0xAA × 32)
        // Row 1: all -1 (0x00 × 32)
        let mut packed = vec![0xAA_u8; 32];
        packed.extend_from_slice(&[0x00_u8; 32]);
        let fx = SyntheticI2S::new(packed, 1.0);
        let w = fx.view("two_rows", 128, 2);

        let input = vec![1.0_f32; 128];
        let mut out = vec![0.0_f32; 2];
        bitlinear_i2s_matvec_f32(&w, &input, &mut out).unwrap();

        assert!((out[0] - 128.0).abs() < 1e-4);
        assert!((out[1] + 128.0).abs() < 1e-4);
    }

    #[test]
    fn matvec_column_stride_32_mapping() {
        // Build a single byte at gp=0 with all four codes distinct:
        //   c0=2 (+1), c1=0 (-1), c2=1 (0), c3=2 (+1)
        // → byte = 10_00_01_10 = 0x86
        // The remaining 31 bytes of the block are 0x55 (all zeros).
        let mut packed = vec![0x55_u8; 32];
        packed[0] = 0x86;
        let fx = SyntheticI2S::new(packed, 1.0);
        let w = fx.view("one_byte", 128, 1);

        // Only positions 0 (c0), 32 (c1), 64 (c2), 96 (c3) matter.
        // Set input so we can read out each code's contribution.
        let mut input = vec![0.0_f32; 128];
        input[0] = 1.0; // contributes +1 × 1.0
        input[32] = 10.0; // contributes -1 × 10.0
        input[64] = 100.0; // contributes 0
        input[96] = 1000.0; // contributes +1 × 1000.0

        let mut out = vec![0.0_f32; 1];
        // Call the scalar kernel directly, NOT the dispatch entry: this
        // test mixes input magnitudes 1.0 … 1000.0 to read out each
        // code's contribution exactly. The x86 default i8 kernel
        // absmax-quantises the activation, so the 1.0 entry rounds to
        // int8 0 next to the 1000.0 entry — correct behaviour for i8,
        // but it would defeat this *mapping*-correctness check. The i8
        // path's numerical fidelity is covered separately by
        // tests/bitlinear_sse2_i8.rs.
        bitlinear_i2s_matvec_f32_scalar(&w, &input, &mut out).unwrap();
        // Expected: 1.0 - 10.0 + 0 + 1000.0 = 991.0
        assert!(
            (out[0] - 991.0).abs() < 1e-4,
            "column-stride-32 mapping wrong: got {}, expected 991.0",
            out[0]
        );
    }

    #[test]
    fn matvec_rejects_non_i2s_type() {
        let v = TensorView {
            name: "wrong".into(),
            shape: vec![128, 1],
            ggml_type: GgmlType::F32,
            offset: 0,
            byte_len: 4 * 128,
            data: &[],
            scale_data: None,
        };
        let input = vec![0.0_f32; 128];
        let mut out = vec![0.0_f32; 1];
        let r = bitlinear_i2s_matvec_f32(&v, &input, &mut out);
        assert!(r.is_err());
    }

    #[test]
    fn matvec_rejects_wrong_input_length() {
        let fx = SyntheticI2S::new(vec![0xAA_u8; 32], 1.0);
        let w = fx.view("x", 128, 1);
        let input = vec![0.0_f32; 64]; // wrong
        let mut out = vec![0.0_f32; 1];
        let r = bitlinear_i2s_matvec_f32(&w, &input, &mut out);
        assert!(r.is_err());
    }

    #[test]
    fn matvec_rejects_wrong_output_length() {
        let fx = SyntheticI2S::new(vec![0xAA_u8; 32], 1.0);
        let w = fx.view("x", 128, 1);
        let input = vec![0.0_f32; 128];
        let mut out = vec![0.0_f32; 2]; // wrong
        let r = bitlinear_i2s_matvec_f32(&w, &input, &mut out);
        assert!(r.is_err());
    }

    #[test]
    fn matvec_rejects_in_dim_not_multiple_of_128() {
        let fx = SyntheticI2S::new(vec![0xAA_u8; 32], 1.0);
        let w = fx.view("x", 64, 1); // 64 < 128, not multiple
        let input = vec![0.0_f32; 64];
        let mut out = vec![0.0_f32; 1];
        let r = bitlinear_i2s_matvec_f32(&w, &input, &mut out);
        assert!(r.is_err());
    }

    #[test]
    fn matvec_is_deterministic_for_same_inputs() {
        let fx = SyntheticI2S::new(vec![0xAA_u8; 32], 1.0);
        let w = fx.view("det", 128, 1);
        let input: Vec<f32> = (0..128).map(|i| (i as f32).sin()).collect();
        let mut a = vec![0.0_f32; 1];
        let mut b = vec![0.0_f32; 1];
        bitlinear_i2s_matvec_f32(&w, &input, &mut a).unwrap();
        bitlinear_i2s_matvec_f32(&w, &input, &mut b).unwrap();
        assert_eq!(a, b, "matvec must be deterministic");
    }

    #[test]
    fn unpack_row_matches_packing() {
        // 128 weights: positions 0,32,64,96 use byte 0 codes 2,0,1,2 (+1,-1,0,+1);
        // remaining bytes all zeros (code 01 → 0 ternary).
        let mut packed = vec![0x55_u8; 32]; // all zeros (code 01)
        packed[0] = 0x86; // c0=2,c1=0,c2=1,c3=2
        let fx = SyntheticI2S::new(packed, 1.0);
        let w = fx.view("u", 128, 1);
        let row = i2s_unpack_row_to_i8(&w, 0).unwrap();
        assert_eq!(row.len(), 128);
        // Position 0  → c0 = 2 → +1
        // Position 32 → c1 = 0 → -1
        // Position 64 → c2 = 1 →  0
        // Position 96 → c3 = 2 → +1
        // All other positions → 0.
        assert_eq!(row[0], 1);
        assert_eq!(row[32], -1);
        assert_eq!(row[64], 0);
        assert_eq!(row[96], 1);
        for &i in &[1usize, 31, 33, 63, 65, 95, 97, 127] {
            assert_eq!(row[i], 0, "position {} should be 0", i);
        }
    }
}
