#!/bin/bash

set -euo pipefail

usage() {
    echo "Usage: $0 -f <function> [-i input_dir] [-o output_dir] [-k \"key=val ...\"]"
    echo
    echo "Functions mirror prepare_multilingual.ipynb blocks:"
    echo "  mimic | casia | padchest | danskcxr | reflacx | danskmri "
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

FUNCTION_NAME="danskmri"
INPUT_DIR="/proc_bd5/bd_wp5/radiology_reports/nlp/neurotekst/neurotekst/data/"
OUTPUT_DIR="/staff/aliceschiavone/mosaic/outputs"
EXTRA_ARGS=()

while getopts "f:i:o:k:h" opt; do
    case $opt in
        f) FUNCTION_NAME="$OPTARG" ;;
        i) INPUT_DIR="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        k) IFS=' ' read -r -a EXTRA_ARGS <<< "$OPTARG" ;;
        h) usage ;;
        ?) usage ;;
    esac
done

if [[ -z "$FUNCTION_NAME" ]]; then
    usage
fi

CMD=(python -m mosaic.core.preprocess_data --function "$FUNCTION_NAME" --input-dir "$INPUT_DIR")

if [[ -n "$OUTPUT_DIR" ]]; then
    CMD+=(--output-dir "$OUTPUT_DIR")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=(--extra-args "${EXTRA_ARGS[*]}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
