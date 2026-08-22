# Demo Runbook

This runbook keeps the Project Willamette demonstration reproducible on the
two product hosts. Apple Silicon is the build machine, not the demo target.

## Pinned Demo

The recommended interactive model is SmolLM2-360M-Instruct Q8_0:

* Local filename: `models/SmolLM2-360M-Instruct-Q8_0.gguf`
* SHA256: `48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201`
* antiX path: `$HOME/willamette-smollm2-360m/SmolLM2-360M-Instruct-Q8_0.gguf`
* HP path: `$HOME/willamette-smollm2-360m/SmolLM2-360M-Instruct-Q8_0.gguf`

The antiX menu labels the 135M model as a limited-quality comparison. Use the
360M profile for the main conversation.

## Build And Deploy

Run from the repository root on the build machine:

```bash
cargo zigbuild --release \
  --target i686-unknown-linux-musl \
  --target x86_64-unknown-linux-musl \
  --bin project-willamette

scp target/i686-unknown-linux-musl/release/project-willamette \
  antix1:willamette-demo-current.new
scp scripts/demo_antix.sh antix1:demo.sh.new
ssh antix1 'chmod +x "$HOME/willamette-demo-current.new" "$HOME/demo.sh.new" && mv "$HOME/willamette-demo-current.new" "$HOME/willamette-demo-current" && mv "$HOME/demo.sh.new" "$HOME/demo.sh"'
```

Deploy the model once if needed, then atomically deploy the x86_64 binary to
the HP host using its configured SSH destination:

```bash
HP_HOST=your-hp-ssh-alias
ssh "${HP_HOST}" 'mkdir -p "$HOME/willamette-smollm2-360m"'
scp models/SmolLM2-360M-Instruct-Q8_0.gguf \
  "${HP_HOST}:willamette-smollm2-360m/SmolLM2-360M-Instruct-Q8_0.gguf"
scp target/x86_64-unknown-linux-musl/release/project-willamette \
  "${HP_HOST}:willamette-demo-current.new"
ssh "${HP_HOST}" 'chmod +x "$HOME/willamette-demo-current.new" && mv "$HOME/willamette-demo-current.new" "$HOME/willamette-demo-current"'
```

Do not add host addresses or credentials to the repository.

## Preflight

Check antiX immediately before a live demonstration:

```bash
ssh antix1 'cd "$HOME" && test -x willamette-demo-current && test -x demo.sh && ./willamette-demo-current --version && printf "%s  %s\n" "48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201" "willamette-smollm2-360m/SmolLM2-360M-Instruct-Q8_0.gguf" | sha256sum -c -'
```

Check the HP host with the same pinned digest:

```bash
HP_HOST=your-hp-ssh-alias
ssh "${HP_HOST}" 'cd "$HOME" && test -x willamette-demo-current && ./willamette-demo-current --version && printf "%s  %s\n" "48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201" "willamette-smollm2-360m/SmolLM2-360M-Instruct-Q8_0.gguf" | sha256sum -c -'
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

## Expected Timing

Pinned greedy two-turn acceptance on 2026-08-22:

| Host | Kernel | First turn | Second turn | Context |
| ---- | ------ | ---------: | ----------: | ------- |
| antiX i686 Pentium-M | Q8_0 SSE2 | 27.0 s | 20.2 s | 34 to 59 tokens |
| HP ProBook x86_64 | Q8_0 AVX2 | 1.1 s | 0.9 s | 34 to 59 tokens |

The antiX pause is expected. Explain that the second turn reuses the existing
KV cache instead of replaying the full transcript.

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
