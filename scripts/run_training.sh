#!/bin/bash

# Default values
EXP="m"
OUTPUT_DIR="outputs/mosaic-4b"

# Help message
usage() {
    echo "Usage: $0 [-e experiment] [-o output_dir]"
    echo "Options:"
    echo "  -e : Experiment type (default: m)"
    echo "       Available: m, mpe, mppe, mppec, mppecd"
    echo "  -o : Output directory (default: outputs/mosaic-4b)"
    echo "  -h : Show this help message"
    exit 1
}

# Parse command line arguments
while getopts "e:o:h" opt; do
    case $opt in
        e) EXP="$OPTARG" ;;
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

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate mosaic

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the training script
echo "Starting training with:"
echo "  Experiment: $EXP"
echo "  Model: AliceSch/mosaic-4b"
echo "  Output directory: $OUTPUT_DIR"
echo

python -m mosaic.core.finetune \
    --model_name "mosaic-4b" \
    --config_tag "$EXP" \
    --project_name "mosaic-training" \
    --train_dataset_names "mimic" \
    --valid_dataset_names "mimic"