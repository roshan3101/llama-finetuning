#!/bin/bash
# Start training script for GCP - Run this to start training in background

set -e

# Activate virtual environment
source venv/bin/activate

# Set model size (default 8B)
MODEL_SIZE=${1:-8B}
OUTPUT_DIR=${2:-./outputs}

echo "=========================================="
echo "Starting Training on GCP"
echo "=========================================="
echo "Model Size: $MODEL_SIZE"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    if [ -f .env ]; then
        export $(cat .env | grep -v '^#' | xargs)
    else
        echo "⚠ Warning: HF_TOKEN not set. Set it with: export HF_TOKEN='your_token'"
    fi
fi

# Check GPU
echo "Checking GPU..."
nvidia-smi

# Start training in screen session
echo ""
echo "Starting training in screen session..."
echo "To detach: Ctrl+A, then D"
echo "To reattach: screen -r training"
echo ""

screen -S training -dm bash -c "python training/train.py --model_size $MODEL_SIZE --output_dir $OUTPUT_DIR 2>&1 | tee training.log"

echo "Training started in background!"
echo "View logs: tail -f training.log"
echo "Reattach to screen: screen -r training"
echo ""
