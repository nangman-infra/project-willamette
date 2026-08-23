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
  8) Validate release        - public x86_64 artifact + Qwen golden

  q) quit

EOF
}

run_tui() {
    local model="$1"
    local label="$2"
    local max_seq_len="${3:-1024}"
    local max_new_tokens="${4:-96}"
    local system_prompt="${5:-You are a concise and accurate local assistant.}"
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
        --max-seq-len "$max_seq_len" \
        --max-new-tokens "$max_new_tokens" \
        --system "$system_prompt"
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

validate_release_artifact() (
    local tag="${1:-v0.15.1-mvp}"
    local repository="${RELEASE_REPOSITORY:-nangman-infra/project-willamette}"
    local target="x86_64-unknown-linux-musl"
    local model_sha256="626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d"
    local actual_model_sha work_dir name archive base_url binary version output
    local expected_ids expected_text

    if [[ ! "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+-mvp$ ]]; then
        echo "usage: $0 validate-release vX.Y.Z-mvp" >&2
        exit 2
    fi
    if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
        echo "release acceptance requires Linux x86_64" >&2
        exit 2
    fi
    for command in curl sha256sum tar; do
        command -v "$command" >/dev/null 2>&1 || {
            echo "$command is required" >&2
            exit 2
        }
    done
    require "$QWEN2_5_3B" "copy the pinned Qwen2.5-3B GGUF to the HP host"

    actual_model_sha="$(sha256sum "$QWEN2_5_3B")"
    actual_model_sha="${actual_model_sha%% *}"
    if [[ "$actual_model_sha" != "$model_sha256" ]]; then
        echo "Qwen model SHA-256 mismatch: $actual_model_sha" >&2
        exit 1
    fi

    work_dir="$(mktemp -d "${TMPDIR:-/tmp}/willamette-release-acceptance.XXXXXX")"
    trap 'rm -rf "$work_dir"' EXIT
    name="willamette-$tag-$target"
    archive="$name.tar.gz"
    base_url="https://github.com/$repository/releases/download/$tag"

    curl --fail --location --retry 3 --output "$work_dir/$archive" "$base_url/$archive"
    curl --fail --location --retry 3 --output "$work_dir/$archive.sha256" "$base_url/$archive.sha256"
    cd "$work_dir"
    sha256sum --check "$archive.sha256"
    tar -xzf "$archive"

    binary="$work_dir/$name/willamette"
    if [[ ! -x "$binary" ]]; then
        echo "runtime missing from release archive: $binary" >&2
        exit 1
    fi
    version="${tag#v}"
    version="${version%-mvp}"
    if [[ "$("$binary" --version)" != "willamette $version" ]]; then
        echo "runtime version does not match release tag $tag" >&2
        exit 1
    fi

    output="$work_dir/qwen-golden.log"
    env RAYON_NUM_THREADS="$THREADS" "$binary" run \
        --model "$QWEN2_5_3B" \
        --prompt "다음 정비 메모를 정확히 6줄의 보고서로 변환하세요. 각 줄은 지정된 필드명과 콜론으로 시작하고, 메모에 없는 내용은 추가하지 마세요. 필드 순서: 설비, 시각, 증상, 조치, 작업시간, 결과. 메모: 펌프 P-204. 14:20에 베어링 소음 증가 확인. 전원을 차단하고 체결 볼트를 조였다. 작업시간은 20분. 시험 운전 후 소음이 사라졌다." \
        --chatml \
        --system "You are a concise and accurate local assistant." \
        --max-new-tokens 100 \
        --temperature 0 2>&1 | tee "$output"

    expected_ids='Generated 75 token(s): [125624, 70582, 25, 10764, 236, 234, 126445, 393, 12, 17, 15, 19, 198, 29326, 126317, 25, 220, 16, 19, 25, 17, 15, 198, 128844, 55902, 25, 47665, 254, 31079, 136849, 126291, 48431, 132376, 251, 19969, 198, 92817, 59698, 25, 56419, 54321, 129882, 125068, 128355, 48364, 112, 88781, 30520, 120, 28626, 65510, 93701, 198, 67511, 124517, 134745, 25, 220, 17, 15, 79716, 198, 88781, 53680, 25, 44518, 125341, 132028, 65865, 94315, 126291, 48431, 32129, 50340, 144042]'
    expected_text='Generated text:   "설비: 펌프 P-204\n시각: 14:20\n증상: 베어링 소음 증가\n조치: 전원 차단 및 체결 볼트 조임\n작업시간: 20분\n결과: 시험 운전 후 소음 사라짐"'
    grep -Fqx "$expected_ids" "$output" || {
        echo "release artifact failed the pinned 75-token golden" >&2
        exit 1
    }
    grep -Fqx "$expected_text" "$output" || {
        echo "release artifact failed the pinned six-line text golden" >&2
        exit 1
    }

    echo "release artifact accepted: $tag ($target)"
    sha256sum "$work_dir/$archive" "$binary"
)

if [[ "${1:-}" == "validate-release" ]]; then
    validate_release_artifact "${2:-v0.15.1-mvp}"
    exit
elif [[ $# -gt 0 ]]; then
    echo "usage: $0 [validate-release vX.Y.Z-mvp]" >&2
    exit 2
fi

show_menu
read -r -p "Pick [1/2/3/4/5/6/7/8/q]: " choice

case "$choice" in
    1) run_tui "$QWEN2_5_3B" "Qwen2.5-3B Q4_K_M" 32768 8192 "You are an accurate and thorough local assistant. Complete every answer fully without stopping mid-sentence." ;;
    2) run_tui "$SMOLLM2_1_7B" "SmolLM2-1.7B Q4_K_M" ;;
    3) run_tui "$SMOLLM2_360M" "SmolLM2-360M Q8_0" ;;
    4) run_tui "$SMOLLM_135M" "SmolLM-135M Q8_0" ;;
    5) run_korean_report_golden ;;
    6) run_paris_golden "$SMOLLM2_1_7B" ;;
    7) run_paris_golden "$SMOLLM2_360M" ;;
    8) validate_release_artifact v0.15.1-mvp ;;
    q|Q) echo "bye" ;;
    *) echo "unknown choice: $choice" >&2; exit 1 ;;
esac
