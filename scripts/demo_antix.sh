#!/usr/bin/env bash
# Project Willamette interactive demo launcher for antiX/Pentium-M-class hosts.

set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama2.c}"
LLAMA_BIN="${LLAMA_BIN:-$LLAMA_DIR/run}"
LLAMA_MODEL="${LLAMA_MODEL:-$LLAMA_DIR/stories110M.bin}"

WILLAMETTE_BIN="${WILLAMETTE_BIN:-$HOME/willamette-demo-current}"
SMOLLM_135M="${SMOLLM_135M:-$HOME/willamette-smollm-135m/SmolLM-135M-Instruct-Q8_0.gguf}"
SMOLLM2_360M="${SMOLLM2_360M:-$HOME/willamette-smollm2-360m/SmolLM2-360M-Instruct-Q8_0.gguf}"
BITNET_MODEL="${BITNET_MODEL:-$HOME/models/ggml-model-i2_s.gguf}"
THREADS="${RAYON_NUM_THREADS:-1}"

require() {
    if [[ ! -e "$1" ]]; then
        echo "missing: $1" >&2
        echo "  $2" >&2
        exit 1
    fi
}

show_menu() {
    clear 2>/dev/null || true
    cat <<'EOF'
========================================================
  Project Willamette demo on antiX (Pentium-M 2 GHz)
========================================================

  1) SmolLM2-360M Q8_0 TUI - recommended quality demo
  2) SmolLM-135M Q8_0 TUI  - faster, limited-quality comparison
  3) BitNet b1.58 2B TUI   - historical 2B CPU demonstration
  4) 360M Paris golden     - deterministic one-shot check
  5) llama2.c stories110M  - side-by-side legacy runtime

  q) quit

EOF
}

run_tui() {
    local model="$1"
    local label="$2"
    require "$WILLAMETTE_BIN" "install the current Willamette demo binary"
    require "$model" "copy the pinned GGUF to this host"
    if [[ "${TERM:-dumb}" == "dumb" || -z "${TERM:-}" ]]; then
        echo "warning: the TUI needs a real terminal; connect with ssh -t"
        sleep 2
    fi
    echo
    echo "=== $label ==="
    sleep 1
    exec env RAYON_NUM_THREADS="$THREADS" "$WILLAMETTE_BIN" tui \
        --model "$model" \
        --max-seq-len 1024 \
        --max-new-tokens 96 \
        --system "You are a concise and accurate local assistant."
}

run_paris_golden() {
    require "$WILLAMETTE_BIN" "install the current Willamette demo binary"
    require "$SMOLLM2_360M" "copy the pinned SmolLM2-360M GGUF to this host"
    exec env RAYON_NUM_THREADS="$THREADS" "$WILLAMETTE_BIN" run \
        --model "$SMOLLM2_360M" \
        --prompt "What is the capital of France? Answer in one sentence." \
        --chatml \
        --system "You are a helpful AI assistant named SmolLM, trained by Hugging Face" \
        --max-new-tokens 30 \
        --temperature 0
}

run_llama2c() {
    require "$LLAMA_BIN" "build llama2.c first"
    require "$LLAMA_MODEL" "download stories110M.bin first"
    cd "$LLAMA_DIR"
    exec "$LLAMA_BIN" "$LLAMA_MODEL" -t 0.0 -n 100
}

show_menu
read -r -p "Pick [1/2/3/4/5/q]: " choice

case "$choice" in
    1) run_tui "$SMOLLM2_360M" "SmolLM2-360M Q8_0" ;;
    2) run_tui "$SMOLLM_135M" "SmolLM-135M Q8_0" ;;
    3) run_tui "$BITNET_MODEL" "BitNet b1.58 2B" ;;
    4) run_paris_golden ;;
    5) run_llama2c ;;
    q|Q) echo "bye" ;;
    *) echo "unknown choice: $choice" >&2; exit 1 ;;
esac
