#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: translate_danskmri_csv.sh [options]

Translate a CSV of Danish MRI reports to English using mosaic/core/translate.py.

Options:
  -m <model_tag>       Model key from config/models.yaml (default: medgemma-27b)
  -p <models_folder>   Directory that stores model checkpoints (default: outputs)
  -o <output_dir>      Final directory for the translated CSV (default: data/danskmri_eng)
  -s <source_lang>     Source language code (default: dan_Latn)
  -t <target_langs>    Space-separated list of target language codes (default: "eng_Latn")
  -f                   Overwrite existing translated data
  -h                   Show this help message
EOF
    exit 0
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# -------- defaults --------
MODEL_TAG="medgemma-27b"
MODELS_FOLDER="outputs"
FINAL_OUTPUT_DIR="data/danskmri_eng"
SOURCE_LANG="dan_Latn"
TARGET_LANGS="eng_Latn"
FORCE_OVERWRITE=false

# -------- CSV INPUT --------
CSV_PATH="/proc_bd5/bd_wp5/radiology_reports/nlp/data/X_cohort1.csv"
CSV_BASENAME="$(basename "$CSV_PATH" .csv)"
TMP_TRANSLATED_CSV="/proc_bd5/bd_wp5/radiology_reports/nlp/data/X_cohort1_translated.csv"

while getopts ":m:p:o:s:t:fh" opt; do
    case "$opt" in
        m) MODEL_TAG="$OPTARG" ;;
        p) MODELS_FOLDER="$OPTARG" ;;
        o) FINAL_OUTPUT_DIR="$OPTARG" ;;
        s) SOURCE_LANG="$OPTARG" ;;
        t) TARGET_LANGS="$OPTARG" ;;
        f) FORCE_OVERWRITE=true ;;
        h) usage ;;
        :) echo "Error: Option -$OPTARG requires an argument." >&2; usage ;;
        \?) echo "Error: Invalid option -$OPTARG" >&2; usage ;;
    esac
done

# Resolve paths
if [[ "$MODELS_FOLDER" != /* ]]; then
    MODELS_FOLDER="$PROJECT_ROOT/$MODELS_FOLDER"
fi

FINAL_OUTPUT_CSV="/proc_bd5/bd_wp5/radiology_reports/nlp/data/danskmri/X_cohort1_english.csv"

# Overwrite checks
if [[ -f "$FINAL_OUTPUT_CSV" && "$FORCE_OVERWRITE" == false ]]; then
    echo "Error: Output CSV $FINAL_OUTPUT_CSV already exists. Use -f to overwrite."
    exit 1
fi

if [[ -f "$TMP_TRANSLATED_CSV" ]]; then
    if [[ "$FORCE_OVERWRITE" == false ]]; then
        echo "Error: Temporary CSV $TMP_TRANSLATED_CSV already exists. Use -f to overwrite."
        exit 1
    fi
    rm -f "$TMP_TRANSLATED_CSV"
fi

echo "Translating CSV:"
echo "  Input:  $CSV_PATH"
echo "  Output: $FINAL_OUTPUT_CSV"
echo "  Model:  $MODEL_TAG"

python -m mosaic.core.translate \
    --model_tag "$MODEL_TAG" \
    --dataset_name "$CSV_PATH" \
    --models_folder "$MODELS_FOLDER" \
    --output_folder "$PROJECT_ROOT/data/" \
    --source_language "$SOURCE_LANG" \
    --target_languages "$TARGET_LANGS"

if [[ ! -f "$TMP_TRANSLATED_CSV" ]]; then
    echo "Error: Expected translated CSV at $TMP_TRANSLATED_CSV but it was not created." >&2
    exit 1
fi

mv "$TMP_TRANSLATED_CSV" "$FINAL_OUTPUT_CSV"

echo "Saved translated CSV to:"
echo "$FINAL_OUTPUT_CSV"
