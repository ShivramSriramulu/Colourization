#!/bin/bash

# Image Colorization Evaluation Script
# This script evaluates the trained model on test images

echo "Starting Image Colorization Evaluation..."

# Set default values
MODEL_PATH=${1:-"main-model.pt"}
TEST_DIR=${2:-"test_images/"}
OUTPUT_DIR=${3:-"./results"}

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file $MODEL_PATH not found!"
    exit 1
fi

# Check if test directory exists
if [ ! -d "$TEST_DIR" ]; then
    echo "Error: Test directory $TEST_DIR not found!"
    exit 1
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Run evaluation
python main.py \
    --mode eval \
    --model_path $MODEL_PATH \
    --test_dir $TEST_DIR \
    --output_dir $OUTPUT_DIR

echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
