#!/usr/bin/env bash
# Stage 5-E — capture Willamette outputs for every comparison prompt.
#
# Output layout:
#   outputs/willamette/<prompt-slug>.tokens.txt   willamette tokenize ...
#   outputs/willamette/<prompt-slug>.logits.txt   willamette logits   ...
#   outputs/willamette/<prompt-slug>.gen5.txt     willamette run --max-new-tokens 5 ...
#
# All runs use deterministic greedy (temperature 0).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${WILLAMETTE_MODEL:-$PROJECT_ROOT/models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf}"
BIN="${WILLAMETTE_BIN:-$PROJECT_ROOT/target/release/project-willamette}"
OUT_DIR="$PROJECT_ROOT/outputs/willamette"
mkdir -p "$OUT_DIR"

if [[ ! -x "$BIN" ]]; then
    echo "error: willamette binary not found at $BIN" >&2
    echo "       run: cargo build --release" >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "error: model not found at $MODEL" >&2
    exit 1
fi

PROMPTS=(
    "hello"
    "안녕하세요"
    "The capital of France is"
    "1 + 1 ="
)

slugify() {
    # Lowercase, replace non-alphanumeric with '-', collapse repeats, trim.
    printf '%s' "$1" \
        | LC_ALL=C tr '[:upper:]' '[:lower:]' \
        | LC_ALL=C tr -c 'a-z0-9' '-' \
        | LC_ALL=C tr -s '-' \
        | sed -e 's/^-//' -e 's/-$//'
}

for p in "${PROMPTS[@]}"; do
    slug="$(slugify "$p")"
    if [[ -z "$slug" ]]; then slug="empty"; fi
    echo "=== Willamette prompt: $p (slug=$slug) ==="
    "$BIN" tokenize --model "$MODEL" --text "$p" \
        > "$OUT_DIR/$slug.tokens.txt" 2>&1 || true
    "$BIN" logits --model "$MODEL" --prompt "$p" --top-k 10 \
        > "$OUT_DIR/$slug.logits.txt" 2>&1 || true
    "$BIN" run --model "$MODEL" --prompt "$p" --max-new-tokens 5 --temperature 0.0 \
        > "$OUT_DIR/$slug.gen5.txt" 2>&1 || true
done

echo "Saved Willamette outputs to: $OUT_DIR"
