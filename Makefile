# Violence Detection Model Makefile

# Configuration variables
DATA_DIR = /Volumes/MARCREYES/violence-detection-dataset
VIOLENT_VIDEO_DIR = $(DATA_DIR)/violent/cam1
NONVIOLENT_VIDEO_DIR = $(DATA_DIR)/non-violent/cam1
VIOLENT_PROCESSED_DIR = $(VIOLENT_VIDEO_DIR)/processed
NONVIOLENT_PROCESSED_DIR = $(NONVIOLENT_VIDEO_DIR)/processed
MODEL_FILE = violence_detection_model.pt
METRICS_FILE = training_metrics.png
BATCH_SIZE = 32
NUM_EPOCHS = 50
TRAIN_SCRIPT = src/train.py
INFERENCE_SCRIPT = src/inference.py
RUN_SCRIPT = ./run.sh

# Default target
all: process train test

# Create directories if they don't exist
$(VIOLENT_PROCESSED_DIR):
	mkdir -p $(VIOLENT_PROCESSED_DIR)

$(NONVIOLENT_PROCESSED_DIR):
	mkdir -p $(NONVIOLENT_PROCESSED_DIR)

# Process violent videos to extract pose data
process-violent: $(VIOLENT_PROCESSED_DIR)
	@echo "Processing violent videos to extract pose data..."
	@echo "Note: You need to define your video processing command here"
	@echo "Example: for file in $(VIOLENT_VIDEO_DIR)/*.mp4; do python src/process_video.py --input $$file --output $(VIOLENT_PROCESSED_DIR)/results_$$(basename $$file .mp4).json; done"
	# Add your video processing command here

# Process non-violent videos to extract pose data
process-nonviolent: $(NONVIOLENT_PROCESSED_DIR)
	@echo "Processing non-violent videos to extract pose data..."
	@echo "Note: You need to define your video processing command here"
	@echo "Example: for file in $(NONVIOLENT_VIDEO_DIR)/*.mp4; do python src/process_video.py --input $$file --output $(NONVIOLENT_PROCESSED_DIR)/results_$$(basename $$file .mp4).json; done"
	# Add your video processing command here

# Process all videos
process: process-violent process-nonviolent

# Update training parameters
update-params:
	@echo "Updating training parameters..."
	sed -i '' 's/^NUM_EPOCHS = [0-9]*/NUM_EPOCHS = $(NUM_EPOCHS)/' $(TRAIN_SCRIPT)
	sed -i '' 's/^BATCH_SIZE = [0-9]*/BATCH_SIZE = $(BATCH_SIZE)/' $(TRAIN_SCRIPT)

# Train the model
train: update-params
	@echo "Training the violence detection model..."
	$(RUN_SCRIPT) --train

# Quick training (1 epoch) for testing
quick-train:
	@echo "Quick training with 1 epoch..."
	$(MAKE) train NUM_EPOCHS=1

# Run inference on a specific file
inference:
	@if [ -z "$(INPUT_FILE)" ]; then \
		echo "Error: INPUT_FILE is not set. Use 'make inference INPUT_FILE=/path/to/file.json'"; \
		exit 1; \
	fi
	@echo "Running inference on $(INPUT_FILE)..."
	$(RUN_SCRIPT) --infer --input $(INPUT_FILE) --output $(or $(OUTPUT_FILE),inference_results.json)

# Test on a sample violent file
test-violent:
	@echo "Testing on a sample violent file..."
	$(eval SAMPLE_FILE := $(shell find $(VIOLENT_PROCESSED_DIR) -name "*.json" | head -1))
	@if [ -z "$(SAMPLE_FILE)" ]; then \
		echo "No violent JSON files found!"; \
		exit 1; \
	fi
	@echo "Using sample file: $(SAMPLE_FILE)"
	$(RUN_SCRIPT) --infer --input $(SAMPLE_FILE) --output violent_test_results.json

# Test on a sample non-violent file
test-nonviolent:
	@echo "Testing on a sample non-violent file..."
	$(eval SAMPLE_FILE := $(shell find $(NONVIOLENT_PROCESSED_DIR) -name "*.json" | head -1))
	@if [ -z "$(SAMPLE_FILE)" ]; then \
		echo "No non-violent JSON files found!"; \
		exit 1; \
	fi
	@echo "Using sample file: $(SAMPLE_FILE)"
	$(RUN_SCRIPT) --infer --input $(SAMPLE_FILE) --output nonviolent_test_results.json

# Run tests on both violent and non-violent samples
test: test-violent test-nonviolent

# Process all JSON files in a directory
process-all-json:
	@if [ -z "$(INPUT_DIR)" ]; then \
		echo "Error: INPUT_DIR is not set. Use 'make process-all-json INPUT_DIR=/path/to/dir OUTPUT_DIR=/path/to/output'"; \
		exit 1; \
	fi
	@if [ -z "$(OUTPUT_DIR)" ]; then \
		echo "Error: OUTPUT_DIR is not set. Use 'make process-all-json INPUT_DIR=/path/to/dir OUTPUT_DIR=/path/to/output'"; \
		exit 1; \
	fi
	@echo "Processing all JSON files in $(INPUT_DIR)..."
	@mkdir -p $(OUTPUT_DIR)
	@for file in $(INPUT_DIR)/*.json; do \
		echo "Processing $$file..."; \
		$(RUN_SCRIPT) --infer --input $$file --output $(OUTPUT_DIR)/$$(basename $$file .json)_results.json; \
	done

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	rm -f $(MODEL_FILE) $(METRICS_FILE) *_results.json

# Help command
help:
	@echo "Violence Detection Model Makefile"
	@echo "--------------------------------"
	@echo "Available targets:"
	@echo "  all              Process videos, train model, and run tests"
	@echo "  process          Process all videos to extract pose data"
	@echo "  process-violent  Process only violent videos"
	@echo "  process-nonviolent Process only non-violent videos"
	@echo "  train            Train the model with specified epochs (default: 50)"
	@echo "  quick-train      Train the model with 1 epoch for testing"
	@echo "  inference        Run inference on a specific file (requires INPUT_FILE)"
	@echo "                   Example: make inference INPUT_FILE=/path/to/file.json OUTPUT_FILE=results.json"
	@echo "  test             Run tests on both violent and non-violent samples"
	@echo "  test-violent     Test on a sample violent file"
	@echo "  test-nonviolent  Test on a sample non-violent file"
	@echo "  process-all-json Process all JSON files in a directory"
	@echo "                   Example: make process-all-json INPUT_DIR=/path/to/dir OUTPUT_DIR=/path/to/output"
	@echo "  clean            Remove generated model and results files"
	@echo "  help             Display this help message"
	@echo ""
	@echo "Configuration variables (can be overridden on command line):"
	@echo "  NUM_EPOCHS = $(NUM_EPOCHS)"
	@echo "  BATCH_SIZE = $(BATCH_SIZE)"
	@echo "  DATA_DIR = $(DATA_DIR)"

.PHONY: all process process-violent process-nonviolent train quick-train inference test test-violent test-nonviolent process-all-json clean help update-params
