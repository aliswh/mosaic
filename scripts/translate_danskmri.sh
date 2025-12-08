#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: translate_danskmri.sh [options]

Translate the Hugging Face danskmri dataset from Danish to English using mosaic/core/translate.py.

Options:
  -m <model_tag>       Model key from config/models.yaml (default: medgemma-4b)
  -n <dataset_name>    Dataset key from config/datasets.yaml (default: danskmri)
  -p <models_folder>   Directory that stores model checkpoints (default: outputs)
  -o <output_dir>      Final directory for the translated dataset (default: data/danskmri_eng)
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

MODEL_TAG="medgemma-4b"
DATASET_NAME="danskmri"
MODELS_FOLDER="outputs"
FINAL_OUTPUT_DIR="data/danskmri_eng"
SOURCE_LANG="dan_Latn"
TARGET_LANGS="eng_Latn"
FORCE_OVERWRITE=false

while getopts ":m:n:p:o:s:t:fh" opt; do
    case "$opt" in
        m) MODEL_TAG="$OPTARG" ;;
        n) DATASET_NAME="$OPTARG" ;;
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

# Resolve paths relative to the project root if needed
if [[ "$MODELS_FOLDER" != /* ]]; then
    MODELS_FOLDER="$PROJECT_ROOT/$MODELS_FOLDER"
fi
if [[ "$FINAL_OUTPUT_DIR" != /* ]]; then
    FINAL_OUTPUT_DIR="$PROJECT_ROOT/$FINAL_OUTPUT_DIR"
fi

DATA_DIR="$PROJECT_ROOT/data/$DATASET_NAME"
if [[ ! -d "$DATA_DIR" ]]; then
    echo "Error: Could not find dataset directory at $DATA_DIR"
    echo "Please preprocess danskmri first (see scripts/preprocess_data.sh)."
    exit 1
fi

mkdir -p "$(dirname "$FINAL_OUTPUT_DIR")"

TMP_TRANSLATED_DIR="$PROJECT_ROOT/data/${DATASET_NAME}_translated"
if [[ -d "$TMP_TRANSLATED_DIR" ]]; then
    if [[ "$FORCE_OVERWRITE" == false ]]; then
        echo "Error: Temporary directory $TMP_TRANSLATED_DIR already exists. Remove it or use -f to overwrite."
        exit 1
    fi
    rm -rf "$TMP_TRANSLATED_DIR"
fi

if [[ -d "$FINAL_OUTPUT_DIR" ]]; then
    if [[ "$FORCE_OVERWRITE" == false ]]; then
        echo "Error: Output directory $FINAL_OUTPUT_DIR already exists. Remove it or use -f to overwrite."
        exit 1
    fi
    rm -rf "$FINAL_OUTPUT_DIR"
fi

echo "Translating dataset '$DATASET_NAME' -> English using model '$MODEL_TAG'"
python -m mosaic.core.translate \
    --model_tag "$MODEL_TAG" \
    --dataset_name "$DATASET_NAME" \
    --models_folder "$MODELS_FOLDER" \
    --output_folder "$PROJECT_ROOT/data/" \
    --source_language "$SOURCE_LANG" \
    --target_languages "$TARGET_LANGS"

if [[ ! -d "$TMP_TRANSLATED_DIR" ]]; then
    echo "Error: Expected translated dataset at $TMP_TRANSLATED_DIR but it was not created." >&2
    exit 1
fi

mv "$TMP_TRANSLATED_DIR" "$FINAL_OUTPUT_DIR"
echo "Saved translated dataset to $FINAL_OUTPUT_DIR"
