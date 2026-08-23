# Benchmark Result Schema

`willamette bench --format json` writes exactly one JSON object to stdout.
Human-readable output remains the default. Schema version 1 separates cached
transformer forward from complete steady-state token generation so consumers
cannot accidentally compare unlike metrics.

The JSON path supports both BitNet I2_S and classic Llama F16/Q4_0/Q4_K/Q6_K/Q8_0
graphs. `runtime.kernel` identifies the measured linear backend, for example
`Q8_0 NEON`, `Q8_0 AVX2`, or `Q8_0 SSE2`. The detailed human benchmark remains
BitNet-specific.

```json
{
  "schema_version": 1,
  "runtime": {
    "name": "willamette",
    "version": "0.15.0",
    "target_arch": "x86_64",
    "kernel": "x86_64 SSE2 (i8)"
  },
  "model": {
    "path": "model.gguf",
    "bytes": 1187801280,
    "architecture": "bitnet-b1.58",
    "blocks": 30,
    "vocab": 128256
  },
  "config": {
    "decode_steps": 10,
    "rayon_threads": 4,
    "stage_timing": false
  },
  "metrics": {
    "matvec_ms": 1.0,
    "matvec_melements_per_second": 6500.0,
    "forward_no_cache_ms": 350.0,
    "cached_forward_ms": 330.0,
    "lm_head_ms": 90.0,
    "argmax_ms": 0.1,
    "complete_token_ms": 420.1,
    "complete_tokens_per_second": 2.38,
    "token_checksum": 12366
  },
  "stages": []
}
```

## Metric Boundaries

| Field | Boundary |
| --- | --- |
| `cached_forward_ms` | One cached transformer forward only; excludes vocabulary projection and token selection. |
| `lm_head_ms` | Vocabulary projection only. |
| `argmax_ms` | Token selection only. |
| `complete_token_ms` | Sum of cached forward, lm-head, and argmax. |
| `complete_tokens_per_second` | `1000 / complete_token_ms`; use this for steady-state generated-token comparisons. |

`stages` is empty in normal builds. A binary built with
`RUSTFLAGS="--cfg willamette_stage_timing"` reports cached-forward sub-stages;
these stages do not include lm-head or argmax.

## Remote Matrix

`scripts/bench_remote_matrix.sh` consumes a credential-free, pipe-delimited
inventory and stores each host's JSON, stderr, exit status, and host metadata
under ignored `benchmark_outputs/`. It uses SSH aliases and never reads or
stores passwords. Copy `benchmark-hosts.example` to `benchmark-hosts.local`
and replace only remote paths and thread counts.

Raw per-run files are authoritative. Aggregate medians only after preserving
all runs; do not replace individual observations with a single summary value.
