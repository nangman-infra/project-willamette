#!/usr/bin/env bash
# Portable Project Willamette SmolLM demo launcher for x86_64 Linux hosts.

set -euo pipefail

WILLAMETTE_BIN="${WILLAMETTE_BIN:-$HOME/willamette-demo-current}"
SMOLLM_135M="${SMOLLM_135M:-$HOME/willamette-smollm-135m/SmolLM-135M-Instruct-Q8_0.gguf}"
SMOLLM2_360M="${SMOLLM2_360M:-$HOME/willamette-smollm2-360m/SmolLM2-360M-Instruct-Q8_0.gguf}"
SMOLLM2_1_7B="${SMOLLM2_1_7B:-$HOME/willamette-smollm2-1.7b/SmolLM2-1.7B-Instruct-Q4_K_M.gguf}"
QWEN2_5_3B="${QWEN2_5_3B:-$HOME/willamette-qwen2.5-3b/Qwen2.5-3B-Instruct-Q4_K_M.gguf}"
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

  1) Qwen2.5-3B Q4_K_M TUI   - HP Korean quality profile
  2) SmolLM2-1.7B Q4_K_M TUI - portable quality profile
  3) SmolLM2-360M Q8_0 TUI   - responsive quality demo
  4) SmolLM-135M Q8_0 TUI    - limited-quality comparison
  5) Qwen 3B Korean report   - deterministic 6-field check
  6) 1.7B Paris golden       - Q4_K_M deterministic check
  7) 360M Paris golden       - Q8_0 deterministic check

  q) quit

EOF
}

run_tui() {
    local model="$1"
    local label="$2"
    require "$WILLAMETTE_BIN" "install a Q4_K-enabled Willamette binary"
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
    local model="$1"
    require "$WILLAMETTE_BIN" "install a Q4_K-enabled Willamette binary"
    require "$model" "copy the pinned SmolLM2 GGUF to this host"
    exec env RAYON_NUM_THREADS="$THREADS" "$WILLAMETTE_BIN" run \
        --model "$model" \
        --prompt "What is the capital of France? Answer in one sentence." \
        --chatml \
        --system "You are a helpful AI assistant named SmolLM, trained by Hugging Face" \
        --max-new-tokens 30 \
        --temperature 0
}

run_korean_report_golden() {
    require "$WILLAMETTE_BIN" "install a Qwen2-enabled Willamette binary"
    require "$QWEN2_5_3B" "copy the pinned Qwen2.5-3B GGUF to the HP host"
    exec env RAYON_NUM_THREADS="$THREADS" "$WILLAMETTE_BIN" run \
        --model "$QWEN2_5_3B" \
        --prompt "다음 정비 메모를 정확히 6줄의 보고서로 변환하세요. 각 줄은 지정된 필드명과 콜론으로 시작하고, 메모에 없는 내용은 추가하지 마세요. 필드 순서: 설비, 시각, 증상, 조치, 작업시간, 결과. 메모: 펌프 P-204. 14:20에 베어링 소음 증가 확인. 전원을 차단하고 체결 볼트를 조였다. 작업시간은 20분. 시험 운전 후 소음이 사라졌다." \
        --chatml \
        --system "You are a concise and accurate local assistant." \
        --max-new-tokens 100 \
        --temperature 0
}

show_menu
read -r -p "Pick [1/2/3/4/5/6/7/q]: " choice

case "$choice" in
    1) run_tui "$QWEN2_5_3B" "Qwen2.5-3B Q4_K_M" ;;
    2) run_tui "$SMOLLM2_1_7B" "SmolLM2-1.7B Q4_K_M" ;;
    3) run_tui "$SMOLLM2_360M" "SmolLM2-360M Q8_0" ;;
    4) run_tui "$SMOLLM_135M" "SmolLM-135M Q8_0" ;;
    5) run_korean_report_golden ;;
    6) run_paris_golden "$SMOLLM2_1_7B" ;;
    7) run_paris_golden "$SMOLLM2_360M" ;;
    q|Q) echo "bye" ;;
    *) echo "unknown choice: $choice" >&2; exit 1 ;;
esac
