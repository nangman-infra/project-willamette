# Architecture — Project Willamette

High-level block diagram of how a `willamette run` invocation produces
a token, plus the file structure each box maps onto.

The detailed diagram below shows the original BitNet path. The architecture
registry now selects either `BitNetSubNorm` or `VanillaLlama`: classic Llama
uses nine layer roles without sub-norms, normal RoPE, SiLU/SwiGLU,
F16/Q4_0/Q8_0 Linear, and a separate or tied F16/Q4_0/Q8_0 lm-head. Both paths
share mmap loading, attention, KV caching, sampling, and generation control.

## Forward pass dataflow

```mermaid
flowchart TB
    subgraph CLI[CLI layer · src/main.rs]
        run["willamette run --prompt …"]
    end

    subgraph IO["I/O layer · src/memory/mmap.rs · src/gguf/*"]
        mmap[("mmap'd<br/>ggml-model-i2_s.gguf<br/>(1.1 GiB, zero-copy)")]
        parser[GGUF parser]
        tensors[("Vec&lt;TensorView&gt;<br/>332 zero-copy slices")]
    end

    subgraph TOK[Tokenizer · src/tokenizer/*]
        pretok["LLAMA_VOCAB_PRE_TYPE_DEFAULT<br/>3-regex pre-tokenizer"]
        bpe["rank-priority byte-level BPE<br/>(LLaMA 3 vocab, 128256 ids)"]
    end

    subgraph MODEL["Model · src/model/*"]
        cfg[BitNetConfig]
        graph[ModelGraph<br/>332 TensorView refs]
        embed["embedding_gather<br/>(F16 / Q6_K → f32)"]
        layer[/"transformer block × 30<br/>(attention + FFN)"/]
        out_norm[output_norm RMSNorm]
        lm["lm_head = tied token_embd<br/>logits over 128256 vocab"]
    end

    subgraph BIT["BitLinear matvec dispatcher · src/model/bitlinear.rs"]
        disp{{"runtime CPU feature dispatch"}}
        scalar["scalar fallback<br/>(Stage 4-C)"]
        neon["aarch64 NEON kernel"]
        sse2["x86 SSE2 i8 kernel<br/>(SSSE3+ hosts)"]
        lut["x86 scalar LUT<br/>(SSE2-only hosts)"]
    end

    subgraph KV["KV cache · src/model/{kv_cache, cached_forward}.rs"]
        cache[("per-layer K, V buffers<br/>append per token")]
    end

    subgraph SAMP["Sampling · src/model/sampler.rs · src/model/generate.rs"]
        sampler["Sampler<br/>(greedy / temp / top-k / top-p / rep-pen)"]
        loop_ctrl["generate loop<br/>+ EOS / stop ids"]
    end

    run --> parser
    mmap --> parser
    parser --> tensors
    parser --> cfg
    tensors --> graph
    cfg --> graph
    run --> pretok
    pretok --> bpe
    bpe --> embed
    graph --> embed
    embed --> layer
    layer --> out_norm
    out_norm --> lm
    layer -. matvec calls .-> disp
    lm -. F16 / Q6_K row dot product .-> embed
    disp -- aarch64 + NEON --> neon
    disp -- x86 + SSSE3 --> sse2
    disp -- x86 + SSE2 only --> lut
    disp -- fallback --> scalar
    layer <--> cache
    lm --> sampler
    sampler --> loop_ctrl
    loop_ctrl -. next token id .-> embed
```

## File-to-stage map

| Stage | Concept | Files |
| ----- | ------- | ----- |
| 1 | GGUF inspection | `src/gguf/{reader,tensor,types}.rs`, `src/memory/mmap.rs`, `src/main.rs::cmd_inspect` |
| 2 | Tokenizer | `src/tokenizer/{byte_unicode,bpe,pretokenize,mod}.rs`, `src/main.rs::cmd_tokenize` |
| 3 | I2_S layout | `src/gguf/tensor.rs` helpers, `docs/I2_S_LAYOUT.md` |
| 4-A | Config + Graph | `src/model/{config,graph,architecture,mod}.rs` |
| 4-B | f32 primitives | `src/model/primitives.rs` |
| 4-C | BitLinear matvec | `src/model/bitlinear.rs` (scalar) |
| 4-D1 | Attention path | `src/model/attention.rs` |
| 4-D2 | FFN path | `src/model/ffn.rs` |
| 4-D3 | Single block | `src/model/block.rs` |
| 4-D4 | 30-layer forward | `src/model/forward.rs` |
| 4-D5 | Logits | `src/model/lm_head.rs` |
| 5-A | Single-step greedy CLI | `src/main.rs::cmd_run`, `src/model/generate.rs` |
| 5-B | Multi-token no-cache | `src/model/multi_forward.rs` |
| 5-C | KV cache | `src/model/kv_cache.rs`, `src/model/cached_forward.rs` |
| 5-D | Sampling | `src/model/sampler.rs` |
| 5-E | Reference compat | `scripts/run_*_reference.sh`, `scripts/compare_reference.sh`, `docs/REFERENCE_COMPATIBILITY.md`, pre-tokenizer fix |
| 6-A | Scalar bench | `src/main.rs::cmd_bench` |
| 6-C | NEON kernel | `src/model/bitlinear_neon.rs` + dispatch in `bitlinear.rs` |
| 6-B | x86 SSE2 (i8 default + f32 mask-add via `--cfg willamette_sse2_f32`) | `src/model/bitlinear_sse2.rs` + dispatch — validated on antiX Pentium-M (v0.5.0 / v0.7.0). |
| 6-B+ | x86 AVX2 / AVX-512 / LUT (TL2) | _deferred — needs SSSE3+ / AVX2 host validation_ |
| dispatch | Runtime CPU kernel selection (NEON / SSE2-i8 / SSE2-f32 / scalar) | `src/model/dispatch.rs` |
| 6-B-aux | Sparsity prototype (CSR, scalar over non-zeros) | `src/model/bitlinear_sparse.rs` — prototype, not default |
| analyze | Ternary weight distribution (-1/0/+1 fractions) | `src/main.rs::cmd_analyze` |
| synth | Synthetic GGUF builder for benchmarking (tiny/small/medium presets) | `src/synth.rs` |
| 9 | Multi-turn chat + TUI | `src/chat/*`, `src/main.rs::{cmd_chat,cmd_tui}` |
| 10-D | i8 KV cache + i8 activation kernels | `src/model/{kv_cache,bitlinear_neon,bitlinear_sse2}.rs` |
| prep | GGUF artifact planning, relocation, and Q6_K profile | `src/bin/willamette-prep.rs`, `src/gguf/{linker,repack}.rs` |
| Phase III-B | Classic Llama F16/Q4_0/Q8_0 config, graph, tokenizer, and forward path | `src/model/{architecture/llama,linear,q4_0,q8_0,block,forward,multi_forward,cached_forward}.rs`, `src/tokenizer/mod.rs` |

## Memory layout (per token, decode step)

```text
  mmap (immutable, OS page-cached, 1.1 GiB)
    └── packed I2_S row 6912 / 4 = 1728 bytes  ┐
                                                ├── read directly
    └── F32 norm weights (10–28 KiB each)     ┘
    └── F16 token_embd (626 MiB source) or Q6_K (257 MiB artifact)
                                                row gather + lm_head scan

  per-call scratch (allocated then freed)
    ├── x_hidden       : 2560 × f32 = 10 KiB
    ├── q              : 2560 × f32 = 10 KiB
    ├── k              :  640 × f32 =  2.5 KiB
    ├── v              :  640 × f32 =  2.5 KiB
    ├── attn_out       : 2560 × f32 = 10 KiB
    ├── ffn gate/up    : 6912 × f32 = 27 KiB
    └── unpacked_row   : in_dim × i8 ≤ 6912 B    (NEON only)

  KV cache (capacity reserved at session start, length grows with position)
    └── per layer × {K, V} × position × (640 × i8 + one f32 scale)
        ≈  30 × 2 × pos × 644 B = 37.7 KiB / token
```

## Numerical pipeline

```text
token id   ── embed F16/Q6_K→f32 ──► hidden (n_embd = 2560)

repeat 30 times:
  hidden ── RMSNorm(attn_norm) ─► xN
  xN     ── BitLinear wq      ─► Q (2560)
  xN     ── BitLinear wk      ─► K_cur (640)
  xN     ── BitLinear wv      ─► V_cur (640)
  Q      ── NEOX RoPE(pos)    ─► Q'
  K_cur  ── NEOX RoPE(pos)    ─► K_cur'
  cache.append(layer, K_cur', V_cur)
  Q', cache ── causal SDPA + GQA ─► attn_out (2560)
  attn_out ── RMSNorm(attn_sub_norm) ─► subN
  subN   ── BitLinear wo      ─► h1
  hidden ── + h1               ─► hidden  (residual #1)
  hidden ── RMSNorm(ffn_norm) ─► xF
  xF     ── BitLinear ffn_gate ─► gate
  xF     ── BitLinear ffn_up   ─► up
  gate   ── ReLU²              ─► gate'
  fused  ── gate' × up         ─► fused (6912)
  fused  ── RMSNorm(ffn_sub_norm) ─► fusedN
  fusedN ── BitLinear ffn_down ─► h2
  hidden ── + h2               ─► hidden  (residual #2)

hidden ── RMSNorm(output_norm) ─► hidden_final
hidden_final ── dot with tied token_embd (F16/Q6_K) ─► logits (128256)
logits ── Sampler.sample()    ─► next token id
```

Every `BitLinear` arrow above is dispatched to:

* `bitlinear_i2s_matvec_f32_neon` on `aarch64` hosts with NEON
  (Apple Silicon),
* the SSE2 i8 kernel on x86 hosts with SSSE3,
* the scalar LUT kernel on SSE2-only x86 hosts,
* `bitlinear_i2s_matvec_f32_scalar` otherwise.

Both read the same packed I2_S bytes from the mmap; neither expands
the full weight tensor to f32.
