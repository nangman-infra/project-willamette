//! BitNet b1.58 model graph, forward/generation paths, KV cache, sampling, and
//! runtime CPU-kernel dispatch.
//!
//! See [`docs/BITNET_FORWARD_PLAN.md`](../../docs/BITNET_FORWARD_PLAN.md) for
//! the source-pinned topology this module implements.

pub mod architecture;
pub mod attention;
pub mod bitlinear;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod bitlinear_lut;
#[cfg(target_arch = "aarch64")]
pub mod bitlinear_neon;
pub mod bitlinear_sparse;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod bitlinear_sse2;
pub mod block;
pub mod cached_forward;
pub mod config;
pub mod dispatch;
pub mod ffn;
pub mod forward;
pub mod generate;
pub mod graph;
pub mod kv_cache;
pub mod lm_head;
pub mod multi_forward;
pub mod primitives;
pub mod q6_k;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod q6_k_sse2;
pub mod sampler;
pub mod stage_timing;

pub use config::BitNetConfig;
pub use graph::{LayerWeights, ModelGraph};
