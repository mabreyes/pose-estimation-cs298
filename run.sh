#!/bin/bash
# Run script for violence detection model
# This script sets the necessary environment variables before running the model

# Set environment variable to enable MPS fallback to CPU for unsupported operations
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Define usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --train                   Run in training mode (default)"
    echo "  --infer --input FILE      Run in inference mode with input file"
    echo "  --output FILE             Specify output file for inference (default: inference_results.json)"
    echo "  --model FILE              Specify model file path for inference (default: violence_detection_model.pt)"
    echo "  --help                    Display this help message"
    exit 1
}

# Default values
MODE="train"
INPUT_FILE=""
OUTPUT_FILE="inference_results.json"
MODEL_FILE="violence_detection_model.pt"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --train)
            MODE="train"
            shift
            ;;
        --infer)
            MODE="infer"
            shift
            ;;
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --model)
            MODEL_FILE="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate arguments
if [[ "$MODE" == "infer" && -z "$INPUT_FILE" ]]; then
    echo "Error: Inference mode requires an input file (--input FILE)"
    usage
fi

# Update PYTHONPATH to include src directory
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Run the appropriate mode
if [[ "$MODE" == "train" ]]; then
    echo "Running in training mode..."
    python -W ignore src/train.py
elif [[ "$MODE" == "infer" ]]; then
    echo "Running in inference mode..."
    echo "Input file: $INPUT_FILE"
    echo "Output file: $OUTPUT_FILE"
    echo "Model file: $MODEL_FILE"
    python -W ignore src/inference.py --input_file "$INPUT_FILE" --output_file "$OUTPUT_FILE" --model_path "$MODEL_FILE"
fi

echo "Model execution completed"
