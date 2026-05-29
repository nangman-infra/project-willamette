# Phase III RFC — Generic Model Architecture support

*Status: draft, 2026-05-29.*
*Owner: pandora0667. Reviewer: collaborator (Claude).*

## 1. Why this exists

The runtime is hard-coded to one model architecture string —
`bitnet-b1.58` — at every metadata-reading site
(`src/model/config.rs:44-115`). The community has already shipped
fine-tunes of the same Microsoft 2B graph under a *different
architecture name* (`bitnet-25`, GGUF `general.name = "bitnet2b_2501"`)
and our loader refuses to read them:

```
$ willamette run --model aramis.gguf …
Error: model graph load failed: Unsupported architecture: "bitnet-25"
```

Audited evidence that `bitnet-25` is the same graph as `bitnet-b1.58`,
just relabelled:

| field | `bitnet-b1.58` (Microsoft 2B) | `bitnet-25` (Aramis / Bifrost fine-tunes) |
| --- | --- | --- |
| `block_count` | 30 | 30 |
| `embedding_length` | 2560 | 2560 |
| `feed_forward_length` | 6912 | 6912 |
| `attention.head_count_kv` | 5 | 5 |
| `context_length` | 4096 | 4096 |
| `rope.dimension_count` | 128 | 128 |
| `rope.freq_base` | 500 000 | 500 000 |
| `vocab_size` | 128 256 | 128 256 |
| `attention.layer_norm_rms_epsilon` | 1e-5 | 1e-5 |
| Tensor count | 332 | 332 |
| Per-layer tensor names | `attn_{norm,sub_norm,q,k,v,output}` + `ffn_{norm,sub_norm,gate,up,down}` | **identical** |
| On-disk size | 1.106 GiB | 1.106 GiB |

So a *single-line* fix exists (alias `bitnet-25 → bitnet-b1.58`,
prefix-strip the metadata). It is rejected as the recommended path —
see § 4.

## 2. What "Phase III" actually has to cover

The thesis names additional architectures explicitly:

* **BitNet family** (same forward graph, ≠ metadata prefix)
  * `bitnet-b1.58` — Microsoft base 2B, our reference
  * `bitnet-25` — same graph, "BitNet 2.5" relabel (Aramis, Bifrost)
  * `bitnet` — 24 / 26 layer variants from upstream Microsoft (different
    `block_count`, otherwise BitNet b1.58 shape)
* **BitLinear-quantised non-BitNet** — community work that took Llama 2
  or GPT-2 and applied BitNet 1.58 quantisation
  ([`Chris4K/bitnet-gpt2-1.58bit`](https://huggingface.co/Chris4K/bitnet-gpt2-1.58bit),
  [`nijil-k/Bitnet-1.58b-Nous-Llama2-70M`](https://huggingface.co/nijil-k/Bitnet-1.58b-Nous-Llama2-70M)).
  These are *Llama 2 / GPT-2 forward graphs* (no sub_norm) with
  BitLinear weights. Format is safetensors, not GGUF; would need the
  preprocessor (Phase IV) to convert. But the runtime needs to *know
  how to read them* once converted.
* **Vanilla non-quantised mid-models** — Llama 2, Phi-3-mini,
  Gemma-2B, Qwen-1.5B, etc. Different forward graphs (no sub_norm,
  no BitLinear — plain f32/f16 linears or standard GGUF quants like
  Q4_K_M). Kernel layer also has to grow.

These three classes are graph-different in increasing degrees:

```
        same graph            different graph,            different graph,
        different metadata    same BitLinear kernel       different kernel
       ──────────────────  ─────────────────────────  ──────────────────────
   class 1: bitnet-25      class 2: GPT-2 / Llama 2   class 3: vanilla Llama2,
            bitnet (24/26)          + BitLinear                Phi-3, Gemma, Qwen
```

Phase III's job is to make *class 1* trivial (it already is, modulo
the string check) and to lay the ground so *class 2* and *class 3*
are additions of one trait impl, not surgery in twelve files.

## 3. Where the assumption lives today

From the audit (commit `666f293` + `7850769` baseline):

| Site | What's hard-coded | Class-1 impact | Class-2 / 3 impact |
| --- | --- | --- | --- |
| `src/model/config.rs:44` `ARCHITECTURE` const | architecture string | string set | string set |
| `src/model/config.rs:48-49` | reject != ARCHITECTURE | replace with set lookup | replace with trait dispatch |
| `src/model/config.rs:52-65` | every metadata key has `"bitnet-b1.58."` prefix | prefix lookup | per-arch field map |
| `src/model/graph.rs::ModelGraph::from_gguf` | tensor-name lookup is fixed: `blk.{N}.{role}.weight` + `attn_sub_norm` / `ffn_sub_norm` | unchanged | needs per-arch tensor-name set + the sub-norm assumption is BitNet-only |
| `src/model/cached_forward.rs`, `forward.rs`, `multi_forward.rs` | forward graph calls `attn_sub_norm` and `ffn_sub_norm` unconditionally | unchanged | sub-norm calls have to become conditional / per-arch |
| `src/model/bitlinear.rs` | matvec assumes I2_S ternary | unchanged | class 3 needs Linear (Q-quant or f16) matvec — new kernel family |
| `src/synth.rs:396-418` | synth writes `bitnet-b1.58.*` prefix | follow the new namespace | follow |
| `src/chat/dashboard.rs:360` | dashboard placeholder | string | string |

Class 1 is essentially the first two rows. The rest of the table is
class 2/3, but they're listed so the design doesn't paint itself into
a corner.

## 4. Why not the one-line alias

It is tempting to add `"bitnet-25"` to a `match` in `config.rs:48` and
read both prefixes in `from_metadata`. Doing this and shipping is the
**rejected** option, recorded here for the audit trail (per
[[feedback-principled-design]]).

Reasons it is rejected:

1. **The roadmap names class 2 and class 3 explicitly.** Once we add a
   second arch the right way, the "right way" is paid for. Adding the
   alias first and then refactoring again when Llama 2 lands is the
   2× cost path.
2. **The hard-code shows up in five files** (`config.rs`, `synth.rs`,
   `dashboard.rs`, `primitives.rs` docs, `bitnet-b1.58` strings in
   tests). An alias buries the cost in those same files instead of
   removing it.
3. **The sub-norm assumption** in `cached_forward.rs::forward_one_layer`
   is *invisible* to the alias path. The first time someone tries to
   load a Llama 2 weight set, the forward will silently call
   `attn_sub_norm_f32` against tensors that don't exist and fail with
   a generic GgufParse error, not "this architecture has no sub-norm."
   The structural fix surfaces that assumption.

## 5. Proposed design

### 5.1 Core abstraction — `ModelArchitecture` trait

```rust
// src/model/architecture/mod.rs (new module)

/// Static description of a model architecture's GGUF surface and
/// forward graph shape. One trait impl per architecture *family*
/// (BitNet b1.58 + 25, vanilla Llama2, Phi-3, …). The impl itself is
/// stateless; per-model state lives in `ModelGraph` / `BitNetConfig`
/// as before.
pub trait ModelArchitecture: Send + Sync + 'static {
    /// The set of `general.architecture` strings this impl claims.
    /// BitNet impl: `["bitnet-b1.58", "bitnet-25", "bitnet"]`.
    /// Llama2 impl (future): `["llama"]`. Gemma: `["gemma", "gemma2"]`.
    fn architecture_strings(&self) -> &'static [&'static str];

    /// The GGUF metadata key prefix for hyperparams. BitNet b1.58
    /// uses `bitnet-b1.58.*`; bitnet-25 uses `bitnet-25.*`; Llama2
    /// uses `llama.*`. Resolved per-instance because two strings in
    /// `architecture_strings()` may need different prefixes.
    fn metadata_prefix(&self, arch_string: &str) -> &str;

    /// Read the hyperparameter struct from GGUF metadata using this
    /// prefix. Returns the same `ModelConfig` the forward path expects.
    fn config_from_meta(
        &self,
        arch_string: &str,
        meta: &MetadataMap,
    ) -> Result<ModelConfig, WillametteError>;

    /// Per-layer tensor name pattern. BitNet impl returns the eleven
    /// tensors including `attn_sub_norm` and `ffn_sub_norm`. Llama2
    /// impl returns nine (no sub_norms). Used by ModelGraph::from_gguf
    /// instead of the current fixed list.
    fn layer_tensor_names(&self) -> &'static [LayerTensorRole];

    /// Forward-graph variant. Today's BitNet path lives behind one
    /// variant; vanilla Llama2 behind another. Avoids embedding the
    /// sub-norm assumption in `cached_forward.rs` directly.
    fn forward_variant(&self) -> ForwardVariant;
}
```

`ForwardVariant` is a small enum, not a function pointer — the
forward functions in `cached_forward.rs` dispatch on it:

```rust
pub enum ForwardVariant {
    BitNetSubNorm,  // attn_sub_norm + ffn_sub_norm in the residual block
    VanillaLlama,   // no sub_norm, standard pre-norm transformer block
    // Phi-3, Gemma variants land here later
}
```

### 5.2 Registry

```rust
// src/model/architecture/registry.rs

static REGISTRY: OnceLock<Vec<Box<dyn ModelArchitecture>>> = OnceLock::new();

pub fn registry() -> &'static [Box<dyn ModelArchitecture>] {
    REGISTRY.get_or_init(|| vec![
        Box::new(BitNetArchitecture),
        // Future:
        // Box::new(LlamaArchitecture),
        // Box::new(Phi3Architecture),
    ])
}

pub fn resolve(arch_string: &str) -> Option<&'static dyn ModelArchitecture> {
    registry().iter()
        .map(|a| a.as_ref())
        .find(|a| a.architecture_strings().contains(&arch_string))
}
```

`config.rs::BitNetConfig::from_metadata` becomes:

```rust
pub fn from_metadata(meta: &MetadataMap) -> Result<ModelConfig, WillametteError> {
    let arch_string = required_str(meta, "general.architecture")?;
    let arch = resolve(arch_string)
        .ok_or(WillametteError::UnsupportedArchitecture(arch_string.to_string()))?;
    arch.config_from_meta(arch_string, meta)
}
```

### 5.3 First impl — `BitNetArchitecture`

Captures the BitNet family in one struct:

```rust
pub struct BitNetArchitecture;

impl ModelArchitecture for BitNetArchitecture {
    fn architecture_strings(&self) -> &'static [&'static str] {
        &["bitnet-b1.58", "bitnet-25", "bitnet"]
    }
    fn metadata_prefix(&self, arch_string: &str) -> &str {
        // The string is the prefix in all three cases.
        arch_string
    }
    fn config_from_meta(&self, arch_string: &str, meta: &MetadataMap) -> Result<ModelConfig, _> {
        let prefix = self.metadata_prefix(arch_string);
        let block_count       = required_u32(meta, &format!("{prefix}.block_count"))?;
        let embedding_length  = required_u32(meta, &format!("{prefix}.embedding_length"))?;
        // …same fields as today, just templated…
    }
    fn layer_tensor_names(&self) -> &'static [LayerTensorRole] {
        &[
            LayerTensorRole::AttnNorm, LayerTensorRole::AttnSubNorm,
            LayerTensorRole::AttnQ, LayerTensorRole::AttnK,
            LayerTensorRole::AttnV, LayerTensorRole::AttnOutput,
            LayerTensorRole::FfnNorm, LayerTensorRole::FfnSubNorm,
            LayerTensorRole::FfnGate, LayerTensorRole::FfnUp,
            LayerTensorRole::FfnDown,
        ]
    }
    fn forward_variant(&self) -> ForwardVariant { ForwardVariant::BitNetSubNorm }
}
```

### 5.4 Reserved entry point — `LlamaArchitecture` placeholder

Not implemented; merely a documentation + test gate so the next phase
has a named target:

```rust
// src/model/architecture/llama.rs
//
// TODO Phase III-B: when we have a Llama 2 / Llama 3 GGUF to test,
// the impl shape is:
//
//   architecture_strings → ["llama"]
//   metadata_prefix      → "llama" (single)
//   layer_tensor_names   → 9 (no sub_norms)
//   forward_variant      → ForwardVariant::VanillaLlama
//
// Forward graph: cached_forward.rs grows a `match variant` and a
// VanillaLlama arm that skips the two sub-norm calls. BitLinear
// matvec stays as-is for I2_S; new Linear matvec kernel needed for
// F16 / Q4_K_M Llama weights (Phase III-C).
```

This is the principled-design half — the registry knows where the
seam is, so the next arch is "fill in this file" rather than "audit
twelve files."

## 6. Migration plan

Five PR-sized steps, each shippable:

1. **(this RFC) accept the design.** No code change.
2. **`architecture` module skeleton + `BitNetArchitecture` impl.**
   `config.rs` switches to the registry. All current tests still pass
   (Microsoft 2B unchanged). Aramis / Bifrost now load. Run a quality
   sanity on antix1 (the original A-track goal, now actually
   reachable).
3. **`ModelGraph::from_gguf` consults `arch.layer_tensor_names()`.**
   Drops the hard-coded list; behaviour identical for BitNet.
4. **`cached_forward.rs` dispatches on `forward_variant()`.** Body
   unchanged for `BitNetSubNorm`; the other arm panics with a
   `not yet implemented for variant {:?}` until step 5.
5. **`LlamaArchitecture` impl + `VanillaLlama` forward arm + Linear
   matvec kernel.** Phase III-B onwards; not in scope for the first
   Phase III ship.

Steps 2–4 are the actual Phase III deliverable. Step 5 is Phase III-B.

## 7. Tests / acceptance

* `tests/synthetic_model.rs` keeps passing (no behaviour change for
  BitNet b1.58 path).
* New `tests/architecture_registry.rs`: round-trips at least
  `bitnet-b1.58`, `bitnet-25`, `bitnet`; rejects `llama` with
  `UnsupportedArchitecture` until step 5.
* `willamette inspect --model aramis.gguf` no longer errors (the file
  is loadable; the test asserts the `BitNetConfig` round-trips its
  hyperparams).
* `willamette run --model aramis.gguf --prompt "La capitale" …` runs
  end-to-end on antix1 and produces a token stream. Quality is **not**
  asserted (different fine-tune); the assertion is just "finite,
  no NaN, produces ≥ 1 token within wall-clock budget."
* Documentation: `LIMITATIONS.md` updated — BitNet family is fully
  supported; vanilla Llama / Phi / Gemma listed as the next bucket
  with the explicit entry point file referenced.

## 8. Risks and named alternatives

| Risk | Mitigation |
| --- | --- |
| Trait abstraction proves wrong shape when class 2 / 3 lands. | First impl mirrors today's behaviour exactly; the registry is mechanical. Wrong shape becomes visible *at the next impl* (Llama) and is refactored then — cheaper than after copy-paste of three more BitNet variants. |
| `bitnet-25` forward graph is *not* in fact identical (some quiet diff we missed). | Step 2 acceptance includes "Aramis run produces finite output" — if it crashes on tensor-name miss or NaN, we know within one antix1 cycle, not in production. |
| Bigger commit than the one-liner. | Yes — and worth it per [[feedback-principled-design]]. Audit trail in § 4 records the rejected hack. |
| Sub-norm conditional becomes a maintenance burden. | The `ForwardVariant` enum has at most ~3 variants in the foreseeable roadmap (BitNet sub-norm, vanilla pre-norm, possibly Phi-3 partial-rotary). Bounded set, not open-ended. |

## 9. Out of scope (named for clarity)

* **Phase IV preprocessor.** `willamette-prep` converting non-BitNet
  models into the runtime's format is a separate workstream. This
  RFC only covers the runtime *reading* whatever the preprocessor
  emits.
* **New SIMD kernel for Linear (non-ternary) matvec.** Class 3 needs
  it. Phase III-C ticket, not this RFC.
* **Tokenizer generality.** Today's `gpt2` BPE handles BitNet family.
  SentencePiece / Llama-3 BPE pre-tokeniser variants are a separate
  tokenizer-RFC.
