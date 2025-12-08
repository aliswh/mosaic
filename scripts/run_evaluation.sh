#!/bin/bash

# Ensure imports work regardless of where the script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH}"

# Default values
EXP="m" # configuration file for experiment
MODEL_TAG="medgemma-4b"
WANDB_PROJECT=None # Set to None to disable wandb logging by default
TRAIN_DATASETS=("mimic") # A list like ("mimic" "padchest" "casia")
TEST_DATASETS=("mimic") # A list like ("mimic" "padchest" "casia")
ZERO_SHOT="off" # "on" or "off"
OUTPUT_DIR="outputs/eval/${MODEL_TAG}"
MODEL_PATH="outputs/" # Path to trained model directory

# Help message
usage() {
    echo "Usage: $0 [-e experiment] [-m model_path]"
    echo "Options:"
    echo "  -e : Experiment type (default: m)"
    echo "       Available: m, mpe, mppe, mppec, mppecd"
    echo "  -o : Output directory (default: outputs/eval/<model_tag>)"
    echo "  -h : Show this help message"
    echo "  -p : Wandb project name (default: None, which disables wandb logging)"
    echo "       Set to None to disable wandb logging"
    echo "  -t : Training datasets (default: mimic)"
    echo "       A space-separated list like \"mimic padchest casia\""
    echo "  -v : Test datasets (default: mimic)"
    echo "       A space-separated list like \"mimic padchest casia\""
    echo "  -z : Zero-shot (default: off)"
    echo "  -m : Path to trained model directory (default: outputs/mosaic-4b)"
    echo "  -h : Show this help message"
    exit 1
}

# Parse command line arguments
while getopts "e:m:h:p:t:v:z" opt; do
    case $opt in
        e) EXP="$OPTARG" ;;
        m) MODEL_PATH="$OPTARG" ;;
        p) WANDB_PROJECT="$OPTARG" ;;
        t) IFS=' ' read -r -a TRAIN_DATASETS <<< "$OPTARG" ;;
        v) IFS=' ' read -r -a TEST_DATASETS <<< "$OPTARG" ;;
        z) ZERO_SHOT="on" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        h) usage ;;
        ?) usage ;;
    esac
done

# Validate experiment type
valid_exps=("m" "mpe" "mppe" "mppec" "mppecd")
if [[ ! " ${valid_exps[@]} " =~ " ${EXP} " ]]; then
    echo "Error: Invalid experiment type '${EXP}'"
    echo "Valid experiments are: ${valid_exps[*]}"
    exit 1
fi

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory does not exist: $MODEL_PATH"
    exit 1
fi

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate mosaic

# Run the evaluation script
echo "Starting evaluation with:"
echo "  Experiment: $EXP"
echo "  Model tag: $MODEL_TAG"
echo "  Training datasets: ${TRAIN_DATASETS[*]}"
echo "  Test datasets: ${TEST_DATASETS[*]}"
echo

# Join dataset arrays for argparse
TRAIN_DATASETS_STR="${TRAIN_DATASETS[*]}"
TEST_DATASETS_STR="${TEST_DATASETS[*]}"

python -m mosaic.core.inference \
    --model_name "$MODEL_TAG" \
    --zeroshot "$ZERO_SHOT" \
    --train_dataset_names "$TRAIN_DATASETS_STR" \
    --test_dataset_names "$TEST_DATASETS_STR" \
    --project_name "$WANDB_PROJECT" \
    --models_folder "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --experiment_tag "_$EXP"
