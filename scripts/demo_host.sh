#!/usr/bin/env bash
# Portable Project Willamette SmolLM demo launcher for x86_64 Linux hosts.

set -euo pipefail

WILLAMETTE_BIN="${WILLAMETTE_BIN:-$HOME/willamette-demo-current}"
SMOLLM_135M="${SMOLLM_135M:-$HOME/willamette-smollm-135m/SmolLM-135M-Instruct-Q8_0.gguf}"
SMOLLM2_360M="${SMOLLM2_360M:-$HOME/willamette-smollm2-360m/SmolLM2-360M-Instruct-Q8_0.gguf}"
THREADS="${RAYON_NUM_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')}"

require() {
    if [[ ! -e "$1" ]]; then
        echo "missing: $1" >&2
        echo "  $2" >&2
        exit 1
    fi
}

show_menu() {
    local host_name
    host_name="$(uname -n)"
    clear 2>/dev/null || true
    cat <<EOF
========================================================
  Project Willamette demo on ${host_name} ($(uname -m))
========================================================

  1) SmolLM2-360M Q8_0 TUI - recommended quality demo
  2) SmolLM-135M Q8_0 TUI  - faster, limited-quality comparison
  3) 360M Paris golden     - deterministic one-shot check

  q) quit

EOF
}

run_tui() {
    local model="$1"
    local label="$2"
    require "$WILLAMETTE_BIN" "install the v0.14.0-mvp Willamette binary"
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
    require "$WILLAMETTE_BIN" "install the v0.14.0-mvp Willamette binary"
    require "$SMOLLM2_360M" "copy the pinned SmolLM2-360M GGUF to this host"
    exec env RAYON_NUM_THREADS="$THREADS" "$WILLAMETTE_BIN" run \
        --model "$SMOLLM2_360M" \
        --prompt "What is the capital of France? Answer in one sentence." \
        --chatml \
        --system "You are a helpful AI assistant named SmolLM, trained by Hugging Face" \
        --max-new-tokens 30 \
        --temperature 0
}

show_menu
read -r -p "Pick [1/2/3/q]: " choice

case "$choice" in
    1) run_tui "$SMOLLM2_360M" "SmolLM2-360M Q8_0" ;;
    2) run_tui "$SMOLLM_135M" "SmolLM-135M Q8_0" ;;
    3) run_paris_golden ;;
    q|Q) echo "bye" ;;
    *) echo "unknown choice: $choice" >&2; exit 1 ;;
esac
