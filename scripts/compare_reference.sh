#!/usr/bin/env bash
# Stage 5-E — diff Willamette outputs against bitnet.cpp reference and
# print a compact compatibility report.
#
# Inputs:
#   outputs/willamette/<slug>.{tokens,logits,gen5}.txt
#   reference_outputs/<slug>.{tokens,gen5}.txt
#
# Output: a Markdown table written to compat_report.md and stdout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WIL_DIR="$PROJECT_ROOT/outputs/willamette"
REF_DIR="$PROJECT_ROOT/reference_outputs"
REPORT="$PROJECT_ROOT/compat_report.md"

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

# Extract Willamette token ids from a tokenize-output file. Format:
#   "Token IDs:       [128000, 15339]"
willamette_token_ids() {
    grep -m1 "Token IDs:" "$1" 2>/dev/null \
        | sed -e 's/^.*\[//' -e 's/\].*$//' -e 's/, /,/g'
}

# Extract bitnet.cpp llama-tokenize ids. The format is per line:
#   "    %5d -> '%s'"
# The BOS row in llama-tokenize is left-justified with no leading whitespace,
# so the leading-whitespace requirement of `^\s+` would skip it.
bitnet_token_ids() {
    grep -E "^[[:space:]]*[0-9]+ -> " "$1" 2>/dev/null \
        | awk '{print $1}' \
        | paste -sd, -
}

# Extract Willamette generated token ids (from `run` output's
# "Generated N token(s): [...]" line).
willamette_gen_ids() {
    grep -m1 "Generated [0-9]* token(s):" "$1" 2>/dev/null \
        | sed -e 's/^.*\[//' -e 's/\].*$//' -e 's/, /,/g'
}

# Extract Willamette argmax token id from `logits` output.
willamette_argmax() {
    grep -m1 "argmax_id:" "$1" 2>/dev/null | awk '{print $2}'
}

{
    echo "# Stage 5-E — Willamette vs bitnet.cpp compatibility report"
    echo
    echo "_Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)_"
    echo
    echo "## Tokenizer comparison"
    echo
    echo "| Prompt | Willamette tokens | bitnet.cpp tokens | Match |"
    echo "| ------ | ----------------- | ----------------- | :---: |"
    for p in "${PROMPTS[@]}"; do
        slug="$(slugify "$p")"
        if [[ -z "$slug" ]]; then slug="empty"; fi
        wil="$(willamette_token_ids "$WIL_DIR/$slug.tokens.txt" 2>/dev/null || true)"
        ref="$(bitnet_token_ids "$REF_DIR/$slug.tokens.txt" 2>/dev/null || true)"
        wil="${wil:-MISSING}"
        ref="${ref:-MISSING}"
        if [[ "$wil" = "MISSING" || "$ref" = "MISSING" ]]; then
            mark="?"
        elif [[ "$wil" = "$ref" ]]; then
            mark="OK"
        else
            mark="MISMATCH"
        fi
        printf '| %s | `%s` | `%s` | %s |\n' "$p" "$wil" "$ref" "$mark"
    done
    echo
    echo "## Greedy generation (first 5 tokens)"
    echo
    echo "| Prompt | Willamette first 5 ids | Willamette generated text | bitnet.cpp generated text |"
    echo "| ------ | ---------------------- | ------------------------- | ------------------------- |"
    for p in "${PROMPTS[@]}"; do
        slug="$(slugify "$p")"
        if [[ -z "$slug" ]]; then slug="empty"; fi
        wil_ids="$(willamette_gen_ids "$WIL_DIR/$slug.gen5.txt" 2>/dev/null || true)"
        wil_text="$(grep -m1 "Generated text:" "$WIL_DIR/$slug.gen5.txt" 2>/dev/null \
            | sed -e 's/^.*Generated text:[[:space:]]*//' | tr -d '\n' | tr -d '\r' | head -c 80)"
        # bitnet.cpp llama-cli output between "generate: n_ctx" line and
        # "llama_perf" footer. The lines in between contain ONLY the
        # generated text (no metadata) because we ran with --no-display-prompt.
        ref_text="$(awk '/^generate: n_ctx/{flag=1;next} /^llama_perf/{flag=0} flag' \
            "$REF_DIR/$slug.gen5.txt" 2>/dev/null \
            | tr '\n' ' ' | tr '\r' ' ' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
            | head -c 80)"
        wil_ids="${wil_ids:-MISSING}"
        wil_text="${wil_text:-MISSING}"
        ref_text="${ref_text:-MISSING}"
        # Replace pipe characters in any field so they don't break the
        # markdown table.
        wil_text="${wil_text//|/\\|}"
        ref_text="${ref_text//|/\\|}"
        printf '| %s | `%s` | `%s` | `%s` |\n' "$p" "$wil_ids" "$wil_text" "$ref_text"
    done
    echo
    echo "## Willamette argmax (first new token after the prompt)"
    echo
    echo "| Prompt | Willamette argmax id | Willamette argmax token |"
    echo "| ------ | -------------------: | ----------------------- |"
    for p in "${PROMPTS[@]}"; do
        slug="$(slugify "$p")"
        if [[ -z "$slug" ]]; then slug="empty"; fi
        wil="$(willamette_argmax "$WIL_DIR/$slug.logits.txt" 2>/dev/null || true)"
        wil_str="$(grep -m1 "argmax_str:" "$WIL_DIR/$slug.logits.txt" 2>/dev/null \
            | sed -e 's/^.*argmax_str: //')"
        printf '| %s | %s | %s |\n' "$p" "${wil:-?}" "${wil_str:-?}"
    done
} | tee "$REPORT"

echo
echo "Wrote report to: $REPORT"
