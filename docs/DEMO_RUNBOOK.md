# Demo Runbook

This runbook keeps the Project Willamette demonstration reproducible on antiX,
HP, and mbp2012. Apple Silicon is the build machine, not the demo target.

## Pinned Demo

The responsive interactive model is SmolLM2-360M-Instruct Q8_0:

* Local filename: `models/SmolLM2-360M-Instruct-Q8_0.gguf`
* SHA256: `48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201`
* antiX path: `$HOME/willamette-smollm2-360m/SmolLM2-360M-Instruct-Q8_0.gguf`
* HP path: `$HOME/willamette-smollm2-360m/SmolLM2-360M-Instruct-Q8_0.gguf`
* mbp2012 path: `$HOME/willamette-smollm2-360m/SmolLM2-360M-Instruct-Q8_0.gguf`

The limited-quality comparison model is `models/SmolLM-135M-Instruct-Q8_0.gguf`
with SHA256
`76520babb0daebccb6e17d2f38504ece61356a0ca958d8e8795ef4d23c23c1f0`.
Each host stores it at
`$HOME/willamette-smollm-135m/SmolLM-135M-Instruct-Q8_0.gguf`. Use the 360M
profile for the main conversation.

HP and mbp2012 also expose the higher-quality SmolLM2-1.7B-Instruct Q4_K_M
profile. It is not deployed to antiX:

* Local filename: `models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf`
* SHA256: `decd2598bc2c8ed08c19adc3c8fdd461ee19ed5708679d1c54ef54a5a30d4f33`
* HP/mbp2012 path: `$HOME/willamette-smollm2-1.7b/SmolLM2-1.7B-Instruct-Q4_K_M.gguf`
* Expected dashboard kernel: `Q4_K AVX2` on HP, `Q4_K SSE2` on mbp2012

Physical-host latency is not yet pinned, so keep the 360M profile available as
the known-responsive fallback.

HP alone exposes the Korean-quality Qwen2.5-3B-Instruct Q4_K_M profile. It is
not deployed to the 996 MiB antiX host or mbp2012:

* Local filename: `models/Qwen2.5-3B-Instruct-Q4_K_M.gguf`
* SHA256: `626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d`
* HP path: `$HOME/willamette-qwen2.5-3b/Qwen2.5-3B-Instruct-Q4_K_M.gguf`
* Expected dashboard kernel: `Q4_K AVX2`
* License: Qwen Research License; keep the model artifact separate from the
  MIT/Apache-2.0 Willamette binary

## Build And Deploy

Run from the repository root on the build machine:

```bash
cargo zigbuild --release \
  --target i686-unknown-linux-musl \
  --target x86_64-unknown-linux-musl \
  --bin project-willamette

ssh antix1 'mkdir -p "$HOME/willamette-smollm-135m" "$HOME/willamette-smollm2-360m"'
scp models/SmolLM-135M-Instruct-Q8_0.gguf \
  antix1:willamette-smollm-135m/SmolLM-135M-Instruct-Q8_0.gguf
scp models/SmolLM2-360M-Instruct-Q8_0.gguf \
  antix1:willamette-smollm2-360m/SmolLM2-360M-Instruct-Q8_0.gguf
scp target/i686-unknown-linux-musl/release/project-willamette \
  antix1:willamette-demo-current.new
scp scripts/demo_antix.sh antix1:demo.sh.new
ssh antix1 'chmod +x "$HOME/willamette-demo-current.new" "$HOME/demo.sh.new" && mv "$HOME/willamette-demo-current.new" "$HOME/willamette-demo-current" && mv "$HOME/demo.sh.new" "$HOME/demo.sh"'
```

Deploy the model once if needed, then atomically deploy the x86_64 binary to
the HP host using its configured SSH destination:

```bash
HP_HOST=your-hp-ssh-alias
ssh "${HP_HOST}" 'mkdir -p "$HOME/willamette-smollm-135m" "$HOME/willamette-smollm2-360m" "$HOME/willamette-smollm2-1.7b" "$HOME/willamette-qwen2.5-3b"'
scp models/SmolLM-135M-Instruct-Q8_0.gguf \
  "${HP_HOST}:willamette-smollm-135m/SmolLM-135M-Instruct-Q8_0.gguf"
scp models/SmolLM2-360M-Instruct-Q8_0.gguf \
  "${HP_HOST}:willamette-smollm2-360m/SmolLM2-360M-Instruct-Q8_0.gguf"
scp models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  "${HP_HOST}:willamette-smollm2-1.7b/SmolLM2-1.7B-Instruct-Q4_K_M.gguf"
scp models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  "${HP_HOST}:willamette-qwen2.5-3b/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
scp target/x86_64-unknown-linux-musl/release/project-willamette \
  "${HP_HOST}:willamette-demo-current.new"
ssh "${HP_HOST}" 'chmod +x "$HOME/willamette-demo-current.new" && mv "$HOME/willamette-demo-current.new" "$HOME/willamette-demo-current"'

ssh mbp2012 'mkdir -p "$HOME/willamette-smollm-135m" "$HOME/willamette-smollm2-360m" "$HOME/willamette-smollm2-1.7b"'
scp models/SmolLM-135M-Instruct-Q8_0.gguf \
  mbp2012:willamette-smollm-135m/SmolLM-135M-Instruct-Q8_0.gguf
scp models/SmolLM2-360M-Instruct-Q8_0.gguf \
  mbp2012:willamette-smollm2-360m/SmolLM2-360M-Instruct-Q8_0.gguf
scp models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  mbp2012:willamette-smollm2-1.7b/SmolLM2-1.7B-Instruct-Q4_K_M.gguf
scp target/x86_64-unknown-linux-musl/release/project-willamette \
  mbp2012:willamette-demo-current.new
ssh mbp2012 'chmod +x "$HOME/willamette-demo-current.new" && mv "$HOME/willamette-demo-current.new" "$HOME/willamette-demo-current"'
```

Install the portable menu on HP and mbp2012:

```bash
HP_HOST=your-hp-ssh-alias
scp scripts/demo_host.sh "${HP_HOST}:demo.sh.new"
ssh "${HP_HOST}" 'chmod +x "$HOME/demo.sh.new" && mv "$HOME/demo.sh.new" "$HOME/demo.sh"'
scp scripts/demo_host.sh mbp2012:demo.sh.new
ssh mbp2012 'chmod +x "$HOME/demo.sh.new" && mv "$HOME/demo.sh.new" "$HOME/demo.sh"'
```

Do not add host addresses or credentials to the repository.

## Preflight

Check antiX immediately before a live demonstration:

```bash
ssh antix1 '
  cd "$HOME" &&
  test -x willamette-demo-current && test -x demo.sh &&
  ./willamette-demo-current --version &&
  printf "%s  %s\n%s  %s\n" \
    "76520babb0daebccb6e17d2f38504ece61356a0ca958d8e8795ef4d23c23c1f0" "willamette-smollm-135m/SmolLM-135M-Instruct-Q8_0.gguf" \
    "48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201" "willamette-smollm2-360m/SmolLM2-360M-Instruct-Q8_0.gguf" |
  sha256sum -c -
'
```

Check the HP host with the same pinned digest:

```bash
set -e
HP_HOST=your-hp-ssh-alias
for host in "${HP_HOST}" mbp2012; do
  ssh "${host}" '
    cd "$HOME" &&
    test -x willamette-demo-current && test -x demo.sh &&
    ./willamette-demo-current --version &&
    printf "%s  %s\n%s  %s\n" \
      "76520babb0daebccb6e17d2f38504ece61356a0ca958d8e8795ef4d23c23c1f0" "willamette-smollm-135m/SmolLM-135M-Instruct-Q8_0.gguf" \
      "48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201" "willamette-smollm2-360m/SmolLM2-360M-Instruct-Q8_0.gguf" |
    sha256sum -c -
  '
done
```

The shared loop above checks the portable profiles. Check HP's Qwen profile
separately because mbp2012 intentionally does not carry it:

```bash
HP_HOST=your-hp-ssh-alias
ssh "${HP_HOST}" '
  cd "$HOME" &&
  printf "%s  %s\n" \
    "626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d" \
    "willamette-qwen2.5-3b/Qwen2.5-3B-Instruct-Q4_K_M.gguf" |
  sha256sum -c -
'
```

## Live Demo

Start the antiX menu with a real terminal:

```bash
ssh -t antix1 '$HOME/demo.sh'
```

1. Select `1` for the recommended 360M TUI.
2. Confirm the dashboard shows `SmolLM2-360M-Instruct-Q8_0.gguf`, `Q8_0`, and
   the i686 SSE2 kernel.
3. Ask `What is the capital of France?`
4. Follow with `Answer with only that city again.` to demonstrate incremental
   multi-turn KV-cache reuse.
5. Press `Esc` to leave the TUI.

For a shorter deterministic check, select menu item `4`. The expected answer
contains `The capital of France is Paris.`

The HP host uses the same TUI directly:

```bash
HP_HOST=your-hp-ssh-alias
ssh -t "${HP_HOST}" 'env RAYON_NUM_THREADS=4 "$HOME/willamette-demo-current" tui --model "$HOME/willamette-smollm2-360m/SmolLM2-360M-Instruct-Q8_0.gguf" --max-seq-len 256 --max-new-tokens 24 --temperature 0 --system "You are a concise assistant."'
```

Use the same two prompts and confirm that the HP dashboard reports `Q8_0` and
the x86_64 AVX2 kernel.

HP and mbp2012 can also start their shared menu with:

```bash
ssh -t "${HP_HOST}" '$HOME/demo.sh'
ssh -t mbp2012 '$HOME/demo.sh'
```

On HP, select menu item `1` for the Qwen2.5-3B TUI or item `5` for its
deterministic Korean six-field report. Select item `2` on mbp2012 for the 1.7B
TUI. The portable 1.7B and 360M golden checks remain items `6` and `7`.

## Expected Timing

Pinned greedy two-turn acceptance on 2026-08-22:

| Host | Kernel | First turn | Second turn | Context |
| ---- | ------ | ---------: | ----------: | ------- |
| antiX i686 Pentium-M | Q8_0 SSE2 | 27.0 s | 20.2 s | 34 to 59 tokens |
| HP ProBook x86_64 | Q8_0 AVX2 | 1.1 s | 0.9 s | 34 to 59 tokens |

The portable menu's deterministic Paris check completed in 1.771 seconds on
HP and 3.441 seconds on mbp2012. Both generated the exact pinned sentence.

Before batched prefill, the HP Qwen2.5-3B Korean report check completed in 71.14
seconds with the menu-default eight Rayon threads and 2,013,084 KiB maximum RSS.
The v0.15.0 layer-major/tiled-Q4_K build emitted the same pinned 75 token IDs in
51.58 seconds: 25.31 seconds for 164-token prefill and 26.26 seconds for decode,
with 2,039,504 KiB maximum RSS. This is a 27.5% wall-time reduction, but it does
not meet the experimental 40-second target. Four threads on the old path took
75.35 seconds, so retain the logical-CPU default on this 4C/8T host.

The expanded HP quality pass also completed a factual one-sentence summary and
four context-dependent turns ending at token position 158. The strict table
line-count and missing-field checks failed; these are documented acceptance
limits, not demo claims.

The Qwen TUI was also opened in a 120x40 SSH terminal. Its dashboard reported
`x86_64`, eight logical/four physical cores, and the expected `Q4_K AVX2`
kernel before accepting input.

The antiX pause is expected. Explain that the second turn reuses the existing
KV cache instead of replaying the full transcript.

The antiX 360M Paris proof was rerun on 2026-08-23 and reproduced the exact
seven-token sentence in 39.981 seconds including prompt prefill. Qwen2.5-3B is
intentionally absent from this 996 MiB host.

## Recovery

If the TUI cannot start, confirm the SSH session has a pseudo-terminal and
that `TERM` is not `dumb`. If model verification fails, do not bypass it during
a demonstration; restore the pinned artifact and rerun preflight. The stdio
fallback is:

```bash
ssh -t antix1 'env RAYON_NUM_THREADS=1 "$HOME/willamette-demo-current" chat --model "$HOME/willamette-smollm2-360m/SmolLM2-360M-Instruct-Q8_0.gguf" --max-seq-len 256 --max-new-tokens 24 --temperature 0 --system "You are a concise assistant."'
```

Use the same stdio fallback on HP by replacing `antix1` with `${HP_HOST}` and
setting `RAYON_NUM_THREADS=4`.
