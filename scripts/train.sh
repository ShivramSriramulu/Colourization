#!/bin/bash

# Image Colorization Training Script
# This script runs the training process with optimized hyperparameters

echo "Starting Image Colorization Training..."

# Set default values
DATA_DIR=${1:-"./data"}
EPOCHS=${2:-20}
BATCH_SIZE=${3:-32}
LEARNING_RATE=${4:-2e-4}
IMG_SIZE=${5:-256}
OUTPUT_DIR=${6:-"./outputs"}

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
python main.py \
    --mode train \
    --data_dir $DATA_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --img_size $IMG_SIZE \
    --output_dir $OUTPUT_DIR

echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
