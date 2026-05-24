#!/usr/bin/env bash
# Stage 5-E — capture pinned bitnet.cpp reference outputs for the same
# comparison prompts and the same `ggml-model-i2_s.gguf` we feed to
# Willamette. The pinned upstream is `microsoft/BitNet @
# 01eb415772c342d9f20dc42772f1583ae1e5b102` (see ../UPSTREAM_PIN.md).
#
# Prereqs (one-time):
#   * `cmake`, a C++ compiler.
#   * Generate model-specific LUT kernel header:
#       cd /tmp/bitnet-upstream
#       python3 utils/codegen_tl1.py --model bitnet_b1_58-3B \
#           --BM 160,320,320 --BK 64,128,64 --bm 32,64,32
#   * Configure + build llama-cli and llama-tokenize targets:
#       cmake -B build -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF -DBITNET_ARM_TL1=ON
#       cmake --build build --target llama-cli llama-tokenize -j 4
#
# Output layout (mirrors run_willamette_reference.sh):
#   reference_outputs/<slug>.tokens.txt   llama-tokenize -m ... -p ...
#   reference_outputs/<slug>.gen5.txt     llama-cli ... -n 5 --temp 0
#
# Logits are NOT dumped: llama-cli has no top-k logit dump option and we
# don't want to write a custom build. argmax / generated tokens are the
# observable proxy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

UPSTREAM_DIR="${BITNET_UPSTREAM:-/tmp/bitnet-upstream}"
LLAMA_CLI="${LLAMA_CLI:-$UPSTREAM_DIR/build/bin/llama-cli}"
LLAMA_TOKENIZE="${LLAMA_TOKENIZE:-$UPSTREAM_DIR/build/bin/llama-tokenize}"
MODEL="${WILLAMETTE_MODEL:-$PROJECT_ROOT/models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf}"
OUT_DIR="$PROJECT_ROOT/reference_outputs"
mkdir -p "$OUT_DIR"

if [[ ! -x "$LLAMA_CLI" ]]; then
    echo "error: llama-cli not found at $LLAMA_CLI" >&2
    echo "       see prerequisites in the header of this script" >&2
    exit 1
fi
if [[ ! -x "$LLAMA_TOKENIZE" ]]; then
    echo "warning: llama-tokenize not found at $LLAMA_TOKENIZE; skipping tokenize dumps" >&2
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
    printf '%s' "$1" \
        | LC_ALL=C tr '[:upper:]' '[:lower:]' \
        | LC_ALL=C tr -c 'a-z0-9' '-' \
        | LC_ALL=C tr -s '-' \
        | sed -e 's/^-//' -e 's/-$//'
}

for p in "${PROMPTS[@]}"; do
    slug="$(slugify "$p")"
    if [[ -z "$slug" ]]; then slug="empty"; fi
    echo "=== bitnet.cpp prompt: $p (slug=$slug) ==="

    if [[ -x "$LLAMA_TOKENIZE" ]]; then
        # llama-tokenize prints one token per line as "id<TAB>token"
        "$LLAMA_TOKENIZE" -m "$MODEL" -p "$p" \
            > "$OUT_DIR/$slug.tokens.txt" 2>&1 || true
    fi

    # llama-cli greedy. --no-display-prompt keeps the prompt out of the
    # captured generated text, --simple-io reduces ANSI/TTY noise.
    "$LLAMA_CLI" \
        -m "$MODEL" \
        -p "$p" \
        -n 5 \
        --temp 0 \
        --top-k 1 \
        --seed 0 \
        --ctx-size 256 \
        --no-display-prompt \
        --simple-io \
        > "$OUT_DIR/$slug.gen5.txt" 2>&1 || true
done

echo "Saved bitnet.cpp reference outputs to: $OUT_DIR"
