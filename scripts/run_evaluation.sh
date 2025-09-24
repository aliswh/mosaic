#!/bin/bash

# Default values
EXP="m"
MODEL_PATH="outputs/mosaic-4b"

# Help message
usage() {
    echo "Usage: $0 [-e experiment] [-m model_path]"
    echo "Options:"
    echo "  -e : Experiment type (default: m)"
    echo "       Available: m, mpe, mppe, mppec, mppecd"
    echo "  -m : Path to trained model directory (default: outputs/mosaic-4b)"
    echo "  -h : Show this help message"
    exit 1
}

# Parse command line arguments
while getopts "e:m:h" opt; do
    case $opt in
        e) EXP="$OPTARG" ;;
        m) MODEL_PATH="$OPTARG" ;;
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
echo "  Model path: $MODEL_PATH"
echo

python -m mosaic.core.inference \
    --model_name "mosaic-4b" \
    --config_tag "$EXP" \
    --model_path "$MODEL_PATH" \
    --dataset_names "mimic" \
    --split "test"