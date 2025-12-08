#!/bin/bash

# Ensure imports work regardless of where the script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH}"

# Default values
EXP="danskmri" # configuration file for experiment
MODEL_TAG="mosaic-12b"
WANDB_PROJECT="mosaic-mcd" # Set to None to disable wandb logging by default
TRAIN_DATASETS=("danskmri" "danskmri_eng") # A list like ("mimic" "padchest" "casia")
VALID_DATASETS=("danskmri") # A list like ("mimic" "padchest" "casia")
OUTPUT_DIR="outputs/"

# Help message
usage() {
    echo "Usage: $0 [-e experiment] [-o output_dir] [-p project_name] [-t train_datasets] [-v valid_datasets] [-m model_tag] [-h]"
    echo "Options:"
    echo "  -e : Experiment type (default: m)"
    echo "       Available: m, mpe, mppe, mppec, mppecd"
    echo "  -m : Model tag (default: medgemma-4b)"
    echo "       Available: mosaic-4b, mosaic-12b, medgemma-4b, gemma-12b"
    echo "  -p : Wandb project name (default: None, which disables wandb logging)"
    echo "       Set to None to disable wandb logging"
    echo "  -t : Training datasets (default: mimic)"
    echo "       A space-separated list like \"mimic padchest casia\""
    echo "  -v : Validation datasets (default: mimic)"
    echo "       A space-separated list like \"mimic padchest casia\""
    echo "  -o : Output directory (default: outputs/<model_tag>)"
    echo "  -h : Show this help message"
    exit 1
}

# Parse command line arguments
while getopts "e:o:h:p:t:v:m" opt; do
    case $opt in
        e) EXP="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        p) WANDB_PROJECT="$OPTARG" ;;
        t) IFS=' ' read -r -a TRAIN_DATASETS <<< "$OPTARG" ;;
        v) IFS=' ' read -r -a VALID_DATASETS <<< "$OPTARG" ;;
        m) MODEL_TAG="$OPTARG" ;;
        h) usage ;;
        ?) usage ;;
    esac
done


# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate mosaic

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if data directory exists
DATA_DIR="$PROJECT_ROOT/data/danskmri"
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: dataset directory not found at: $DATA_DIR"
    echo "Please ensure you have:"
    echo "1. Downloaded the dataset"
    echo "2. Preprocessed it into the correct format (HuggingFace datasets)"
    echo "3. Placed it in the correct directory"
    echo "Expected structure:"
    echo "  $DATA_DIR/"
    echo "    ├── train/"
    echo "    ├── val/"
    echo "    └── test/"
    exit 1
fi

# Run the training script
echo "Starting training with:"
echo "  Experiment: $EXP"
echo "  Model: $MODEL_TAG"
echo "  Output directory: $OUTPUT_DIR"
echo "  Data directory: $DATA_DIR"
echo "  Wandb project: $WANDB_PROJECT"
echo "  Training datasets: ${TRAIN_DATASETS[*]}"
echo "  Validation datasets: ${VALID_DATASETS[*]}"
echo

python -m mosaic.core.finetune \
    --model_name "$MODEL_TAG" \
    --config_tag "$EXP" \
    --project_name "$WANDB_PROJECT" \
    --train_dataset_names "${TRAIN_DATASETS[@]}" \
    --valid_dataset_names "${VALID_DATASETS[@]}" \
    --output_dir "$OUTPUT_DIR" 
