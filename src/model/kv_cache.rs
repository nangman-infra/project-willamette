//! Per-layer K/V cache (i8 absmax-per-token storage).
//!
//! Stores rotated K and V tensors for every token already processed by
//! `forward_with_cache`, so each new generation step only has to push
//! one (K, V) pair per layer and reads all past pairs.
//!
//! ## Storage layout
//!
//! Per layer, K and V are kept as two parallel arrays:
//!
//! ```text
//! k_quant:  Vec<i8>    length = position × kv_dim   (1 byte / element)
//! k_scales: Vec<f32>   length = position             (1 scale / token)
//! v_quant:  Vec<i8>    length = position × kv_dim
//! v_scales: Vec<f32>   length = position
//! ```
//!
//! Each token's `kv_dim`-long K vector is **per-token absmax**
//! quantised to i8: `q[i] = round(x[i] · 127 / absmax)`, `scale = absmax / 127`.
//! V uses the same scheme with its own absmax. `kv_dim = n_kv_heads · head_dim`
//! (640 for BitNet b1.58 2B).
//!
//! At ≈4 bytes overhead per token per layer per (K|V), the on-disk
//! penalty is `4 / kv_dim ≈ 0.6 %` and the net memory ratio vs the
//! prior f32 storage is `(1 + 4/kv_dim) / 4 ≈ 0.251` — about **3.97×
//! smaller**. For the 4096-token / 30-layer / 640-kv_dim BitNet 2B that
//! turns ~614 MB of f32 KV cache into ~154 MB at full context.
//!
//! ## Fidelity
//!
//! per-token absmax i8 is the same scheme our BitLinear i8-activation
//! kernel applies once per matvec, so the entire decode data-path is
//! end-to-end i8 (activations, BitLinear product, KV) with f32 only at
//! norm and softmax. Greedy parity vs the f32 KV reference is
//! enforced by `tests/kv_cache_quant.rs` — see also
//! [`docs/KV_CACHE_QUANT.md`](../../docs/KV_CACHE_QUANT.md).

use crate::error::WillametteError;

#[derive(Debug, Clone)]
struct LayerKV {
    k_quant: Vec<i8>,
    k_scales: Vec<f32>,
    v_quant: Vec<i8>,
    v_scales: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct KVCache {
    pub n_layers: usize,
    pub kv_dim: usize,
    pub max_seq_len: usize,
    layers: Vec<LayerKV>,
}

impl KVCache {
    pub fn new(n_layers: usize, kv_dim: usize, max_seq_len: usize) -> Self {
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(LayerKV {
                k_quant: Vec::with_capacity(max_seq_len * kv_dim),
                k_scales: Vec::with_capacity(max_seq_len),
                v_quant: Vec::with_capacity(max_seq_len * kv_dim),
                v_scales: Vec::with_capacity(max_seq_len),
            });
        }
        Self {
            n_layers,
            kv_dim,
            max_seq_len,
            layers,
        }
    }

    /// Number of positions currently stored. All layers are kept in
    /// sync — this returns the count of any one layer.
    pub fn position(&self) -> usize {
        if self.layers.is_empty() {
            0
        } else {
            self.layers[0].k_scales.len()
        }
    }

    /// Append `(K, V)` for the given layer at the next position. Each of
    /// `k` and `v` must be `kv_dim` long; both are quantised to i8
    /// per-token absmax before storage.
    pub fn append(
        &mut self,
        layer_idx: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), WillametteError> {
        if layer_idx >= self.n_layers {
            return Err(WillametteError::GgufParse(format!(
                "KVCache::append: layer_idx {} >= n_layers {}",
                layer_idx, self.n_layers
            )));
        }
        if k.len() != self.kv_dim || v.len() != self.kv_dim {
            return Err(WillametteError::GgufParse(format!(
                "KVCache::append: k.len()={} v.len()={} != kv_dim={}",
                k.len(),
                v.len(),
                self.kv_dim
            )));
        }
        let layer = &mut self.layers[layer_idx];
        let new_pos = layer.k_scales.len();
        if new_pos >= self.max_seq_len {
            return Err(WillametteError::GgufParse(format!(
                "KVCache::append: layer {} already at max_seq_len {}",
                layer_idx, self.max_seq_len
            )));
        }
        append_quantised(k, &mut layer.k_quant, &mut layer.k_scales);
        append_quantised(v, &mut layer.v_quant, &mut layer.v_scales);
        Ok(())
    }

    /// Dequantise the layer's full `(K, V)` cache into the caller's
    /// `out_k` / `out_v` buffers, which are resized to
    /// `position() × kv_dim`. Re-using one pair of buffers across all
    /// layers (as `cached_forward` does) avoids any per-call allocation
    /// once the buffers reach steady-state capacity.
    pub fn read_into(
        &self,
        layer_idx: usize,
        out_k: &mut Vec<f32>,
        out_v: &mut Vec<f32>,
    ) -> Result<(), WillametteError> {
        if layer_idx >= self.n_layers {
            return Err(WillametteError::GgufParse(format!(
                "KVCache::read_into: layer_idx {} >= n_layers {}",
                layer_idx, self.n_layers
            )));
        }
        let layer = &self.layers[layer_idx];
        let n_pos = layer.k_scales.len();
        let len = n_pos * self.kv_dim;
        out_k.clear();
        out_k.resize(len, 0.0);
        out_v.clear();
        out_v.resize(len, 0.0);
        for p in 0..n_pos {
            let s_k = layer.k_scales[p];
            let s_v = layer.v_scales[p];
            let base = p * self.kv_dim;
            for d in 0..self.kv_dim {
                out_k[base + d] = layer.k_quant[base + d] as f32 * s_k;
                out_v[base + d] = layer.v_quant[base + d] as f32 * s_v;
            }
        }
        Ok(())
    }

    /// Clear all cached entries but retain the buffer capacities.
    pub fn reset(&mut self) {
        for l in &mut self.layers {
            l.k_quant.clear();
            l.k_scales.clear();
            l.v_quant.clear();
            l.v_scales.clear();
        }
    }

    /// Total i8 + scale bytes resident in this cache, summed across
    /// layers. Used by the TUI dashboard and the benchmark banner.
    pub fn resident_bytes(&self) -> usize {
        let mut total = 0;
        for l in &self.layers {
            total += l.k_quant.len() + l.v_quant.len();
            total += (l.k_scales.len() + l.v_scales.len()) * std::mem::size_of::<f32>();
        }
        total
    }
}

/// Per-token absmax i8 quantisation. Pushes `x.len()` i8 values plus
/// one f32 scale onto the back of the layer's buffers.
fn append_quantised(x: &[f32], out_quant: &mut Vec<i8>, out_scales: &mut Vec<f32>) {
    let absmax = x.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    if absmax == 0.0 || !absmax.is_finite() {
        out_quant.extend(std::iter::repeat_n(0_i8, x.len()));
        out_scales.push(0.0);
        return;
    }
    let scale = absmax / 127.0;
    let inv = 127.0 / absmax;
    for &v in x {
        let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
        out_quant.push(q);
    }
    out_scales.push(scale);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_slice(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let d = (x - y).abs();
            assert!(d <= tol, "idx {}: {} vs {} (|Δ|={})", i, x, y, d);
        }
    }

    #[test]
    fn new_cache_has_zero_position() {
        let c = KVCache::new(2, 4, 8);
        assert_eq!(c.position(), 0);
        let mut out_k = Vec::new();
        let mut out_v = Vec::new();
        c.read_into(0, &mut out_k, &mut out_v).unwrap();
        assert!(out_k.is_empty());
        assert!(out_v.is_empty());
    }

    #[test]
    fn append_then_dequantise_round_trips_within_absmax_tol() {
        let mut c = KVCache::new(1, 4, 8);
        // K and V have different absmax so they get different scales.
        let k_in = [0.1_f32, -0.2, 0.3, -0.4];
        let v_in = [10.0_f32, -20.0, 30.0, -40.0];
        c.append(0, &k_in, &v_in).unwrap();
        assert_eq!(c.position(), 1);
        let mut k_out = Vec::new();
        let mut v_out = Vec::new();
        c.read_into(0, &mut k_out, &mut v_out).unwrap();
        // Per-token absmax i8 has a worst-case error of scale / 2,
        // i.e. absmax / 254 per element. So tol = absmax / 200 is safe.
        approx_eq_slice(&k_out, &k_in, 0.4 / 200.0);
        approx_eq_slice(&v_out, &v_in, 40.0 / 200.0);
    }

    #[test]
    fn zero_vector_round_trips_exactly() {
        let mut c = KVCache::new(1, 4, 8);
        let zero = [0.0_f32; 4];
        c.append(0, &zero, &zero).unwrap();
        let mut k_out = Vec::new();
        let mut v_out = Vec::new();
        c.read_into(0, &mut k_out, &mut v_out).unwrap();
        assert_eq!(k_out, zero);
        assert_eq!(v_out, zero);
    }

    #[test]
    fn append_to_capacity_then_errors() {
        let mut c = KVCache::new(1, 2, 2);
        c.append(0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        c.append(0, &[5.0, 6.0], &[7.0, 8.0]).unwrap();
        let r = c.append(0, &[9.0, 10.0], &[11.0, 12.0]);
        assert!(r.is_err());
    }

    #[test]
    fn append_rejects_wrong_length() {
        let mut c = KVCache::new(1, 4, 8);
        let r = c.append(0, &[1.0, 2.0], &[3.0, 4.0, 5.0, 6.0]);
        assert!(r.is_err());
    }

    #[test]
    fn append_rejects_invalid_layer_idx() {
        let mut c = KVCache::new(2, 4, 8);
        let r = c.append(5, &[1.0; 4], &[1.0; 4]);
        assert!(r.is_err());
    }

    #[test]
    fn reset_clears_position_but_keeps_capacity() {
        let mut c = KVCache::new(1, 4, 8);
        c.append(0, &[1.0; 4], &[1.0; 4]).unwrap();
        c.append(0, &[1.0; 4], &[1.0; 4]).unwrap();
        assert_eq!(c.position(), 2);
        c.reset();
        assert_eq!(c.position(), 0);
        let mut k_out = Vec::new();
        let mut v_out = Vec::new();
        c.read_into(0, &mut k_out, &mut v_out).unwrap();
        assert!(k_out.is_empty());
    }

    #[test]
    fn resident_bytes_matches_layout() {
        // 2 layers × 3 tokens × kv_dim=4 = 24 i8 each for K and V,
        // plus 3 scales × 4 bytes × 2 (K+V) × 2 layers = 48 bytes.
        let mut c = KVCache::new(2, 4, 8);
        for _ in 0..3 {
            c.append(0, &[1.0; 4], &[2.0; 4]).unwrap();
            c.append(1, &[1.0; 4], &[2.0; 4]).unwrap();
        }
        let expected = 2 * 3 * 4 * 2 /* i8 K+V */ + 2 * 3 * 4 * 2 /* f32 scales */;
        assert_eq!(c.resident_bytes(), expected);
    }
}
