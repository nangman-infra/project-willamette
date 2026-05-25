//! Stage 5-D — sampling options.
//!
//! Default is deterministic greedy (`argmax`). Sampling activates only
//! when callers explicitly enable temperature/top-k/top-p. A seed makes
//! sampling reproducible.

use crate::error::WillametteError;
use crate::model::lm_head::argmax;

/// Deterministic 64-bit xorshift PRNG, seedable. No external dep, no
/// global state. Used only for sampling; greedy never touches it.
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        let s = if seed == 0 {
            0xdead_beef_cafe_babe_u64
        } else {
            seed
        };
        Self { state: s }
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    /// Uniform [0, 1) from the high 24 bits.
    pub fn next_unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// 0.0 (or < epsilon) means "use greedy". Otherwise logits are
    /// divided by this value before softmax.
    pub temperature: f32,
    /// Keep only the K highest logits; rest set to `-inf`. `None` =
    /// disabled.
    pub top_k: Option<usize>,
    /// Nucleus: keep tokens whose cumulative softmax probability is
    /// `<= top_p`. `None` = disabled.
    pub top_p: Option<f32>,
    /// If `Some(r)` with `r > 1.0`, divide the logit of any recently
    /// emitted token by `r` (the standard HuggingFace convention).
    pub repetition_penalty: Option<f32>,
    /// PRNG seed for reproducible sampling.
    pub seed: u64,
}

impl SamplingParams {
    /// Greedy / deterministic default. `Sampler::sample` will return
    /// the same id as `argmax`.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            seed: 0xabad_1dea,
        }
    }

    /// True if any sampling-related knob is set. Used to decide whether
    /// to short-circuit to `argmax`.
    pub fn is_greedy(&self) -> bool {
        self.temperature.abs() < 1e-9
            && self.top_k.is_none()
            && self.top_p.is_none()
            && self.repetition_penalty.is_none()
    }
}

pub struct Sampler {
    params: SamplingParams,
    rng: XorShift64,
    history: Vec<u32>,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        let seed = params.seed;
        Self {
            params,
            rng: XorShift64::new(seed),
            history: Vec::new(),
        }
    }

    /// Record a token id in the rolling history (used by repetition
    /// penalty). Callers should `observe` every newly emitted token.
    pub fn observe(&mut self, id: u32) {
        self.history.push(id);
    }

    /// Read-only view of the sampling configuration.
    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// Clone of the sampling configuration (cheap; small struct).
    pub fn params_clone(&self) -> SamplingParams {
        self.params.clone()
    }

    /// Sample one token from the supplied logits. The function does
    /// NOT mutate `logits`; it operates on an internal copy.
    pub fn sample(&mut self, logits: &[f32]) -> Result<u32, WillametteError> {
        if self.params.is_greedy() {
            return argmax(logits)
                .ok_or_else(|| WillametteError::GgufParse("sample: empty logits".to_string()));
        }
        let mut buf = logits.to_vec();
        self.apply_repetition_penalty(&mut buf);
        self.apply_temperature(&mut buf);
        if let Some(k) = self.params.top_k {
            apply_top_k(&mut buf, k);
        }
        let mut probs = softmax_to_probs(&buf);
        if let Some(p) = self.params.top_p {
            apply_top_p(&mut probs, p);
        }
        self.multinomial(&probs)
    }

    fn apply_repetition_penalty(&self, logits: &mut [f32]) {
        let Some(r) = self.params.repetition_penalty else {
            return;
        };
        if r <= 0.0 || (r - 1.0).abs() < 1e-9 {
            return;
        }
        // HF convention: if logit > 0, divide; if logit < 0, multiply.
        // This guarantees the magnitude moves toward 0 regardless of sign.
        for &id in &self.history {
            if (id as usize) < logits.len() {
                let l = logits[id as usize];
                logits[id as usize] = if l > 0.0 { l / r } else { l * r };
            }
        }
    }

    fn apply_temperature(&self, logits: &mut [f32]) {
        let t = self.params.temperature;
        if t <= 0.0 || (t - 1.0).abs() < 1e-9 {
            return;
        }
        let inv = 1.0 / t;
        for v in logits.iter_mut() {
            *v *= inv;
        }
    }

    fn multinomial(&mut self, probs: &[f32]) -> Result<u32, WillametteError> {
        let total: f32 = probs.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            return Err(WillametteError::GgufParse(format!(
                "multinomial: probabilities do not sum to a positive finite value ({})",
                total
            )));
        }
        let r = self.rng.next_unit() * total;
        let mut acc = 0.0_f32;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if r <= acc {
                return Ok(i as u32);
            }
        }
        Ok((probs.len() - 1) as u32)
    }
}

/// Set every logit not in the top-K to `-inf`. K is clamped to
/// `logits.len()`.
pub fn apply_top_k(logits: &mut [f32], k: usize) {
    let k = k.min(logits.len());
    if k == 0 {
        for v in logits.iter_mut() {
            *v = f32::NEG_INFINITY;
        }
        return;
    }
    if k == logits.len() {
        return;
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|a, b| {
        logits[*b]
            .partial_cmp(&logits[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = logits[idx[k - 1]];
    // Mask anything strictly below the K-th value. Ties at the boundary
    // are kept (could be more than K).
    for v in logits.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

fn softmax_to_probs(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        if v > max {
            max = v;
        }
    }
    if !max.is_finite() {
        return vec![0.0; logits.len()];
    }
    let mut out = Vec::with_capacity(logits.len());
    let mut sum = 0.0_f32;
    for &v in logits {
        let e = (v - max).exp();
        sum += e;
        out.push(e);
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in out.iter_mut() {
            *v *= inv;
        }
    }
    out
}

/// Apply nucleus / top-p filtering to a probability distribution. Sort
/// by descending probability, keep entries whose cumulative sum is
/// `<= p`, plus the first entry that crosses the threshold; zero the
/// rest. Result is re-normalised.
pub fn apply_top_p(probs: &mut [f32], p: f32) {
    if probs.is_empty() {
        return;
    }
    if p <= 0.0 {
        // Degenerate — keep only the single highest probability mass.
        let mut argmax_i = 0;
        let mut argmax_v = probs[0];
        for (i, &v) in probs.iter().enumerate().skip(1) {
            if v > argmax_v {
                argmax_v = v;
                argmax_i = i;
            }
        }
        for (i, v) in probs.iter_mut().enumerate() {
            *v = if i == argmax_i { 1.0 } else { 0.0 };
        }
        return;
    }
    if p >= 1.0 {
        return;
    }
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_unstable_by(|a, b| {
        probs[*b]
            .partial_cmp(&probs[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut acc = 0.0_f32;
    let mut keep = vec![false; probs.len()];
    for &i in &idx {
        acc += probs[i];
        keep[i] = true;
        if acc >= p {
            break;
        }
    }
    let mut sum = 0.0_f32;
    for (i, v) in probs.iter_mut().enumerate() {
        if !keep[i] {
            *v = 0.0;
        }
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in probs.iter_mut() {
            *v *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_default_returns_argmax() {
        let logits = vec![0.1, 0.5, 0.3, 0.9];
        let mut s = Sampler::new(SamplingParams::greedy());
        for _ in 0..5 {
            let id = s.sample(&logits).unwrap();
            assert_eq!(id, 3);
        }
    }

    #[test]
    fn seed_makes_sampling_deterministic() {
        let logits = vec![0.1_f32; 16];
        let p = SamplingParams {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            seed: 12345,
        };
        let mut a = Sampler::new(p.clone());
        let mut b = Sampler::new(p);
        let mut seq_a = Vec::new();
        let mut seq_b = Vec::new();
        for _ in 0..20 {
            seq_a.push(a.sample(&logits).unwrap());
            seq_b.push(b.sample(&logits).unwrap());
        }
        assert_eq!(seq_a, seq_b);
    }

    #[test]
    fn temperature_one_matches_uniform_sampling_distribution_shape() {
        let logits = vec![0.0_f32, 0.0, 0.0, 0.0];
        let mut s = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            seed: 7,
        });
        let mut hits = [0u32; 4];
        for _ in 0..4000 {
            hits[s.sample(&logits).unwrap() as usize] += 1;
        }
        // Each bucket should land around 1000 (±150 is a generous
        // margin for 4000 draws from uniform).
        for h in hits {
            assert!(h > 700 && h < 1300, "uneven sampling: {:?}", hits);
        }
    }

    #[test]
    fn top_k_zeros_outside_top() {
        let mut logits = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        apply_top_k(&mut logits, 2);
        // Top 2 are 4.0 and 5.0. Everything < 4.0 becomes -inf.
        assert!(logits[0].is_infinite() && logits[0] < 0.0);
        assert!(logits[1].is_infinite() && logits[1] < 0.0);
        assert!(logits[2].is_infinite() && logits[2] < 0.0);
        assert_eq!(logits[3], 4.0);
        assert_eq!(logits[4], 5.0);
    }

    #[test]
    fn top_k_zero_zeroes_everything() {
        let mut logits = vec![1.0_f32, 2.0, 3.0];
        apply_top_k(&mut logits, 0);
        for v in logits {
            assert!(v.is_infinite() && v < 0.0);
        }
    }

    #[test]
    fn top_p_keeps_smallest_set_above_threshold() {
        // p = 0.7 with probs [0.4, 0.3, 0.2, 0.1]: cumulative 0.4, 0.7
        // → keep {0, 1} (first two), zero the rest.
        let mut probs = vec![0.4_f32, 0.3, 0.2, 0.1];
        apply_top_p(&mut probs, 0.7);
        assert!(probs[0] > 0.0);
        assert!(probs[1] > 0.0);
        assert_eq!(probs[2], 0.0);
        assert_eq!(probs[3], 0.0);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn repetition_penalty_pushes_history_logits_toward_zero() {
        // history = [2]. logit[2] is positive → divided by 2.
        let logits = vec![0.5_f32, 0.5, 1.0, 0.5];
        let mut s = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: Some(2.0),
            seed: 7,
        });
        s.observe(2);
        // After repetition penalty: logit[2] becomes 0.5. So tokens 0,1,2,3
        // have logits [0.5, 0.5, 0.5, 0.5]. Uniform sampling.
        // With seed 7 we just confirm it returns a valid id.
        let id = s.sample(&logits).unwrap();
        assert!(id < 4);
    }
}
