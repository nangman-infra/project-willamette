//! Stage 5-C — per-layer K/V cache.
//!
//! Stores rotated K and V tensors for every token already processed by
//! `forward_with_cache`, so each new generation step only has to push
//! one (K, V) pair per layer and reads all past pairs.
//!
//! Layout: per layer, a flat `Vec<f32>` of length `position × kv_dim`
//! that stores `(K, V)` in token-major order:
//!
//! ```text
//!   k_layer[pos * kv_dim + h_kv * head_dim + d]
//! ```
//!
//! `kv_dim = n_kv_heads * head_dim` (e.g. 640 for our model).
//! The cache enforces `position < max_seq_len`.

use crate::error::WillametteError;

#[derive(Debug, Clone)]
struct LayerKV {
    k: Vec<f32>,
    v: Vec<f32>,
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
                k: Vec::with_capacity(max_seq_len * kv_dim),
                v: Vec::with_capacity(max_seq_len * kv_dim),
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
            self.layers[0].k.len() / self.kv_dim
        }
    }

    /// Append `(K, V)` for the given layer at the next position. Each of
    /// `k` and `v` must be `kv_dim` long.
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
        let new_pos = layer.k.len() / self.kv_dim;
        if new_pos >= self.max_seq_len {
            return Err(WillametteError::GgufParse(format!(
                "KVCache::append: layer {} already at max_seq_len {}",
                layer_idx, self.max_seq_len
            )));
        }
        layer.k.extend_from_slice(k);
        layer.v.extend_from_slice(v);
        Ok(())
    }

    /// Borrow the full `(K, V)` cache slices for a layer. Length =
    /// `position() * kv_dim`.
    pub fn read(&self, layer_idx: usize) -> Result<(&[f32], &[f32]), WillametteError> {
        if layer_idx >= self.n_layers {
            return Err(WillametteError::GgufParse(format!(
                "KVCache::read: layer_idx {} >= n_layers {}",
                layer_idx, self.n_layers
            )));
        }
        Ok((&self.layers[layer_idx].k, &self.layers[layer_idx].v))
    }

    /// Clear all cached entries but retain the buffer capacities.
    pub fn reset(&mut self) {
        for l in &mut self.layers {
            l.k.clear();
            l.v.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_cache_has_zero_position() {
        let c = KVCache::new(2, 4, 8);
        assert_eq!(c.position(), 0);
        let (k, v) = c.read(0).unwrap();
        assert!(k.is_empty());
        assert!(v.is_empty());
    }

    #[test]
    fn append_grows_storage() {
        let mut c = KVCache::new(1, 4, 8);
        let k = [1.0_f32, 2.0, 3.0, 4.0];
        let v = [5.0_f32, 6.0, 7.0, 8.0];
        c.append(0, &k, &v).unwrap();
        assert_eq!(c.position(), 1);
        let (kk, vv) = c.read(0).unwrap();
        assert_eq!(kk, &k);
        assert_eq!(vv, &v);
    }

    #[test]
    fn append_to_capacity_then_errors() {
        let mut c = KVCache::new(1, 2, 2);
        c.append(0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        c.append(0, &[5.0, 6.0], &[7.0, 8.0]).unwrap();
        // Cache full.
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
        let (k, _) = c.read(0).unwrap();
        assert!(k.is_empty());
    }
}
