//! Stage 3 verification — confirm that the I2_S layout derived from the
//! pinned BitNet source (see `docs/I2_S_LAYOUT.md`) matches every I2_S tensor
//! in the official `ggml-model-i2_s.gguf` file.
//!
//! This test does NOT unpack ternary codes, does NOT read the f32 scale, and
//! does NOT touch any forward-pass code. It only checks:
//!
//!   1. The model contains exactly 210 I2_S tensors.
//!   2. The model contains zero `Unknown(_)` tensor types.
//!   3. Each I2_S tensor's stored `byte_len` equals
//!      `TensorView::i2s_expected_byte_len(shape)`.
//!   4. Each I2_S tensor's full on-disk footprint
//!      (`offset + byte_len + I2S_TRAILING_SCALE_BLOCK_BYTES`) lies within
//!      the file and does not overlap the next tensor's data.
//!
//! Tests are skipped (passing with a SKIP message) if the model file is not
//! present at the expected path.

use std::path::Path;

use project_willamette::gguf::reader::GgufFile;
use project_willamette::gguf::tensor::TensorView;
use project_willamette::gguf::types::GgmlType;
use project_willamette::memory::mmap::ModelMmap;

const MODEL_PATH: &str = "./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf";

/// Expected per the Stage 1 inspect report:
/// 30 BitNet blocks × 7 I2_S BitLinear weights per block = 210.
const EXPECTED_I2S_COUNT: usize = 210;

/// Open the real model and run `f` with the parsed `GgufFile`. Returns
/// without asserting when the model file is missing.
fn with_real_gguf<F>(f: F)
where
    F: FnOnce(&GgufFile<'_>, usize),
{
    if !Path::new(MODEL_PATH).exists() {
        eprintln!(
            "SKIP: model file not found at {} — Stage 3 layout tests require it",
            MODEL_PATH
        );
        return;
    }
    let mmap = ModelMmap::open(MODEL_PATH).expect("open model");
    let bytes = mmap.as_bytes();
    let file_len = bytes.len();
    let gguf = GgufFile::parse(bytes).expect("parse model");
    f(&gguf, file_len);
}

#[test]
fn there_are_exactly_210_i2s_tensors() {
    with_real_gguf(|gguf, _| {
        let n_i2s = gguf
            .tensors
            .iter()
            .filter(|t| t.ggml_type == GgmlType::BitNetI2S)
            .count();
        assert_eq!(
            n_i2s, EXPECTED_I2S_COUNT,
            "expected exactly {} I2_S tensors (30 blocks × 7 BitLinear); got {}",
            EXPECTED_I2S_COUNT, n_i2s
        );
    });
}

#[test]
fn zero_unknown_tensor_types() {
    with_real_gguf(|gguf, _| {
        let unknowns: Vec<(&str, GgmlType)> = gguf
            .tensors
            .iter()
            .filter_map(|t| match t.ggml_type {
                GgmlType::Unknown(_) => Some((t.name.as_str(), t.ggml_type)),
                _ => None,
            })
            .collect();
        assert!(
            unknowns.is_empty(),
            "found Unknown ggml_type tensors (would break Stage 3+): {:?}",
            unknowns
        );
    });
}

#[test]
fn every_i2s_byte_len_matches_expected_packed_bytes() {
    with_real_gguf(|gguf, _| {
        let mut checked = 0usize;
        for t in &gguf.tensors {
            if t.ggml_type != GgmlType::BitNetI2S {
                continue;
            }
            let expected = TensorView::i2s_expected_byte_len(&t.shape).expect("shape ok");
            assert_eq!(
                t.byte_len, expected,
                "I2_S tensor {:?} shape={:?}: byte_len {} != expected {}",
                t.name, t.shape, t.byte_len, expected
            );
            // Also exercise the standalone verifier.
            t.verify_byte_len()
                .unwrap_or_else(|e| panic!("verify_byte_len failed on {:?}: {}", t.name, e));
            checked += 1;
        }
        assert_eq!(
            checked, EXPECTED_I2S_COUNT,
            "I2_S byte_len check should run for all {} tensors",
            EXPECTED_I2S_COUNT
        );
    });
}

#[test]
fn every_i2s_total_footprint_is_in_bounds_and_non_overlapping() {
    with_real_gguf(|gguf, file_len| {
        // Build a sorted (by offset) list of (offset, name, end_of_this_tensor)
        // so we can confirm no two tensors overlap.
        let mut entries: Vec<(u64, &str, u64, GgmlType)> = gguf
            .tensors
            .iter()
            .map(|t| {
                let footprint = match t.ggml_type {
                    GgmlType::BitNetI2S => t.byte_len + TensorView::I2S_TRAILING_SCALE_BLOCK_BYTES,
                    _ => t.byte_len,
                };
                let end = t.offset + footprint;
                (t.offset, t.name.as_str(), end, t.ggml_type)
            })
            .collect();
        entries.sort_by_key(|e| e.0);

        // 1. Every footprint fits inside the file.
        for (off, name, end, _) in &entries {
            assert!(
                *end <= file_len as u64,
                "tensor {:?} off={} end={} exceeds file_len={}",
                name,
                off,
                end,
                file_len
            );
        }

        // 2. No overlaps: each tensor's end must be ≤ the next tensor's start.
        //    (For I2_S tensors the gap should be exactly 0 — packed + scale
        //    block consumes the whole 32-byte-aligned slot.)
        for win in entries.windows(2) {
            let (_, a_name, a_end, _) = &win[0];
            let (b_off, b_name, _, _) = &win[1];
            assert!(
                *a_end <= *b_off,
                "tensors overlap: {:?} ends at {} but {:?} starts at {}",
                a_name,
                a_end,
                b_name,
                b_off
            );
        }
    });
}

#[test]
fn i2s_scale_offsets_are_well_formed() {
    with_real_gguf(|gguf, file_len| {
        for t in &gguf.tensors {
            if t.ggml_type != GgmlType::BitNetI2S {
                continue;
            }
            let scale_off = t
                .i2s_scale_file_offset()
                .expect("I2_S tensor must report a scale offset");
            // The scale block is 32 bytes; must fit in file.
            assert!(
                scale_off + TensorView::I2S_TRAILING_SCALE_BLOCK_BYTES <= file_len as u64,
                "I2_S tensor {:?} scale block (off={}..{}) exceeds file_len={}",
                t.name,
                scale_off,
                scale_off + 32,
                file_len
            );
            // Scale offset = tensor.offset + tensor.byte_len, by definition.
            assert_eq!(scale_off, t.offset + t.byte_len);
        }
    });
}

#[test]
fn file_type_metadata_is_40() {
    // LLAMA_FTYPE_MOSTLY_I2_S = 40 per llama.h:183 of the pinned commit.
    with_real_gguf(|gguf, _| {
        let ftype = gguf
            .metadata
            .get("general.file_type")
            .and_then(|v| v.as_u64())
            .expect("general.file_type must be present");
        assert_eq!(
            ftype, 40,
            "expected LLAMA_FTYPE_MOSTLY_I2_S=40; got {}",
            ftype
        );
    });
}

#[test]
fn architecture_is_bitnet_b1_58() {
    with_real_gguf(|gguf, _| {
        let arch = gguf
            .metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .expect("general.architecture must be present");
        assert_eq!(arch, "bitnet-b1.58");
    });
}
