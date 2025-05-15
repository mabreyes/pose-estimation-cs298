#!/bin/bash
# Run script for Violence Detection with Graph Recurrent Neural Networks
# This script provides a simple way to train and run inference with the GRNN model

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --train)
      TRAIN_MODE=true
      shift
      ;;
    --quick-train)
      QUICK_TRAIN=true
      shift
      ;;
    --inference)
      INFERENCE_MODE=true
      shift
      ;;
    --input)
      INPUT_FILE="$2"
      shift
      shift
      ;;
    --output)
      OUTPUT_FILE="$2"
      shift
      shift
      ;;
    --model)
      MODEL_PATH="$2"
      shift
      shift
      ;;
    --help)
      echo "Usage: ./run.sh [options]"
      echo "Options:"
      echo "  --train         Run in training mode"
      echo "  --quick-train   Run training with minimal epochs for testing"
      echo "  --inference     Run in inference mode"
      echo "  --input FILE    Input JSON file for inference"
      echo "  --output FILE   Output JSON file for inference results"
      echo "  --model FILE    Model file path"
      echo "  --help          Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $key"
      echo "Use --help to see available options"
      exit 1
      ;;
  esac
done

# If no mode is specified, show help
if [ -z "$TRAIN_MODE" ] && [ -z "$INFERENCE_MODE" ] && [ -z "$QUICK_TRAIN" ]; then
  echo "No mode specified. Use --train, --quick-train, or --inference."
  echo "Use --help for more information."
  exit 1
fi

# Set up model path with default if not specified
MODEL_PATH=${MODEL_PATH:-"violence_detection_model.pt"}

# Training mode
if [ "$TRAIN_MODE" = true ]; then
  echo "Running in training mode..."
  python src/train.py
  exit $?
fi

# Quick training mode (for testing)
if [ "$QUICK_TRAIN" = true ]; then
  echo "Running in quick training mode for testing..."
  python src/train.py --epochs 1 --batch-size 8 --sample-percentage 10
  exit $?
fi

# Inference mode
if [ "$INFERENCE_MODE" = true ]; then
  if [ -z "$INPUT_FILE" ]; then
    echo "Error: No input file specified for inference mode."
    echo "Use --input FILE to specify an input file."
    exit 1
  fi

  # Set default output file if not specified
  OUTPUT_FILE=${OUTPUT_FILE:-"violence_scores.json"}

  echo "Running inference on $INPUT_FILE..."
  python src/inference.py --model_path "$MODEL_PATH" --input_file "$INPUT_FILE" --output_file "$OUTPUT_FILE"
  exit $?
fi

echo "Script execution completed"
