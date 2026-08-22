#!/usr/bin/env bash
# Run versioned Willamette JSON benchmarks through SSH aliases.
# Inventory format: host|binary|model|threads|decode_steps

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INVENTORY="${1:-$ROOT/benchmark-hosts.local}"
OUTPUT_ROOT="${BENCH_OUTPUT_DIR:-$ROOT/benchmark_outputs}"
RUN_ID="${BENCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ ! -f "$INVENTORY" ]]; then
    echo "inventory not found: $INVENTORY" >&2
    echo "copy benchmark-hosts.example and use SSH aliases; do not store passwords" >&2
    exit 2
fi

command -v ssh >/dev/null 2>&1 || {
    echo "ssh is required" >&2
    exit 2
}

RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"

failures=0
while IFS='|' read -r host binary model threads decode_steps extra; do
    [[ -z "${host// }" || "$host" == \#* ]] && continue
    if [[ -n "${extra:-}" || -z "$binary" || -z "$model" || -z "$threads" || -z "$decode_steps" ]]; then
        echo "invalid inventory row for host '$host'" >&2
        failures=$((failures + 1))
        continue
    fi
    if [[ "$host" == *'@'* || "$host" == *' '* ]]; then
        echo "host must be a credential-free SSH alias: $host" >&2
        failures=$((failures + 1))
        continue
    fi

    host_dir="$RUN_DIR/$host"
    mkdir -p "$host_dir"
    metadata="$host_dir/host.txt"
    stdout="$host_dir/result.json"
    stderr="$host_dir/stderr.txt"
    status_file="$host_dir/exit-status.txt"

    if ! ssh -o BatchMode=yes "$host" \
        'hostname; uname -a; getconf _NPROCESSORS_ONLN 2>/dev/null || true' \
        >"$metadata" 2>"$host_dir/host.stderr.txt"; then
        echo "host metadata failed: $host" >&2
        printf '%s\n' 255 >"$status_file"
        failures=$((failures + 1))
        continue
    fi

    set +e
    ssh -o BatchMode=yes "$host" \
        "env RAYON_NUM_THREADS=$threads '$binary' bench --model '$model' --decode-steps '$decode_steps' --format json" \
        >"$stdout" 2>"$stderr"
    status=$?
    set -e
    printf '%s\n' "$status" >"$status_file"
    if [[ $status -ne 0 ]]; then
        echo "benchmark failed: $host (exit $status)" >&2
        failures=$((failures + 1))
        continue
    fi

    if command -v jq >/dev/null 2>&1; then
        if ! jq -e '.schema_version == 1 and .runtime.name == "willamette"' \
            "$stdout" >/dev/null; then
            echo "invalid benchmark JSON: $host" >&2
            failures=$((failures + 1))
        fi
    fi
done <"$INVENTORY"

printf '%s\n' "$RUN_DIR"
[[ $failures -eq 0 ]]
