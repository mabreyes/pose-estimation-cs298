#!/usr/bin/env python3
"""
Violence detection inference script for MMPose JSON files.

This script uses a trained Graph Neural Network model to predict violence
scores from human pose data in MMPose JSON format. It provides:
- Command-line interface for batch processing
- Score interpretation based on configurable thresholds
- Detailed per-frame analytics and overall statistics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

# Import from separate component files
from grnn import create_pose_graph
from model import ViolenceDetectionGNN, get_device

# Constants for inference
DEFAULT_MODEL_PATH = "violence_detection_model.pt"
DEFAULT_OUTPUT_PATH = "violence_scores.json"
THRESHOLD_MARGIN = 0.2  # Margin for interpretation confidence levels
MODEL_IN_CHANNELS = 2
MODEL_HIDDEN_CHANNELS = 64
MODEL_TRANSFORMER_HEADS = 4
MODEL_TRANSFORMER_LAYERS = 2


def interpret_score(score: float, threshold: float) -> Tuple[str, bool]:
    """
    Interpret a violence score based on the threshold.

    Categorizes scores into confidence levels based on their distance from
    the classification threshold, using the defined THRESHOLD_MARGIN.

    Args:
        score: Violence score between 0 and 1
        threshold: Classification threshold

    Returns:
        Tuple of (interpretation string, is_violent boolean)
    """
    is_violent = score >= threshold

    if score < threshold - THRESHOLD_MARGIN:
        return "Likely non-violent", is_violent
    elif score < threshold:
        return "Possibly non-violent", is_violent
    elif score < threshold + THRESHOLD_MARGIN:
        return "Possibly violent", is_violent
    else:
        return "Likely violent", is_violent


def load_and_process_json(json_file: Path) -> List[Tuple[int, List[Data]]]:
    """
    Load and process a single MMPose JSON file for inference.

    Extracts pose keypoints from the MMPose JSON format and converts them
    to graph representations suitable for GNN processing.

    Args:
        json_file: Path to the JSON file

    Returns:
        List of tuples containing (frame_id, list_of_graph_data)
    """
    graphs = []

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Process each frame in the JSON file
    for frame_data in data.get("instance_info", []):
        frame_id = frame_data.get("frame_id")
        instances = frame_data.get("instances", [])

        frame_graphs = []
        for instance in instances:
            keypoints = instance.get("keypoints", [])
            if keypoints:
                # Convert to numpy array
                keypoints_np = np.array(keypoints)

                # Create graph from keypoints
                graph = create_pose_graph(keypoints_np)
                if graph is not None:
                    frame_graphs.append(graph)

        if frame_graphs:
            graphs.append((frame_id, frame_graphs))

    return graphs


def predict_violence(
    model: ViolenceDetectionGNN, graphs: List[Data], device: torch.device
) -> List[float]:
    """
    Predict violence scores for graphs.

    Runs each pose graph through the model to generate a violence score,
    handling batch creation for single-graph inference.

    Args:
        model: Trained GNN model
        graphs: List of graph data objects
        device: Device to run inference on

    Returns:
        List of violence scores between 0 and 1
    """
    model.eval()
    scores = []

    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)

            # Add batch dimension for single graph
            if not hasattr(graph, "batch"):
                graph.batch = torch.zeros(
                    graph.x.shape[0], dtype=torch.long, device=device
                )

            # Forward pass
            score = model(graph.x, graph.edge_index, graph.batch)
            scores.append(score.item())

    return scores


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the inference script.

    Defines and processes command-line arguments for model path, input file,
    output file, classification threshold, and metrics display options.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Violence Detection from MMPose JSON")
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to trained model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to input MMPose JSON file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to output JSON file (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold (0-1). Uses model's threshold if None.",
    )
    parser.add_argument(
        "--show_metrics",
        action="store_true",
        help="Show threshold metrics from the model",
    )
    return parser.parse_args()


def load_model_and_threshold(
    model_path: Path, device: torch.device
) -> Tuple[ViolenceDetectionGNN, float, Optional[Dict]]:
    """
    Load the model and threshold from a saved model file.

    Handles both legacy model format (weights only) and newer format
    with state dict, threshold, and metrics information.

    Args:
        model_path: Path to the saved model file
        device: Device to load the model to

    Returns:
        Tuple of (model, threshold, metrics)
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Create model with standard parameters
    model = ViolenceDetectionGNN(
        in_channels=MODEL_IN_CHANNELS,
        hidden_channels=MODEL_HIDDEN_CHANNELS,
        transformer_heads=MODEL_TRANSFORMER_HEADS,
        transformer_layers=MODEL_TRANSFORMER_LAYERS,
    ).to(device)

    # Default threshold if not found in model
    DEFAULT_THRESHOLD = 0.5

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        threshold = checkpoint.get("threshold", DEFAULT_THRESHOLD)
        metrics = checkpoint.get("metrics", None)
    else:
        model.load_state_dict(checkpoint)
        threshold = DEFAULT_THRESHOLD
        metrics = None

    return model, threshold, metrics


def main() -> None:
    """
    Main inference function to detect violence from pose data.

    This function orchestrates the entire inference process:
    1. Parses command-line arguments
    2. Loads the model and threshold
    3. Processes pose data from input file
    4. Generates predictions for each frame
    5. Calculates overall statistics
    6. Saves results to output file
    """
    args = parse_arguments()

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    model_path = Path(args.model_path)

    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist.")
        return

    device = get_device()
    print(f"Using device: {device}")

    model, model_threshold, metrics = load_model_and_threshold(model_path, device)
    print(f"Model loaded from {model_path}")

    threshold = args.threshold if args.threshold is not None else model_threshold
    source_text = " (from model)" if args.threshold is None else " (user-specified)"
    print(f"Using classification threshold: {threshold}{source_text}")

    if args.show_metrics and metrics:
        print("\nModel threshold metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    print(f"Processing input file: {input_file}")
    graph_data = load_and_process_json(input_file)

    if not graph_data:
        print("No valid pose data found in the input file.")
        return

    results = []
    violent_frame_count = 0

    for frame_id, frame_graphs in graph_data:
        frame_scores = predict_violence(model, frame_graphs, device)
        avg_score = np.mean(frame_scores) if frame_scores else 0.0

        interpretation, is_violent = interpret_score(avg_score, threshold)
        if is_violent:
            violent_frame_count += 1

        results.append(
            {
                "frame_id": frame_id,
                "violence_score": float(avg_score),
                "is_violent": bool(is_violent),
                "interpretation": interpretation,
                "person_scores": [float(score) for score in frame_scores],
            }
        )

    overall_score = np.mean([r["violence_score"] for r in results]) if results else 0.0
    overall_interpretation, is_violent_overall = interpret_score(
        overall_score, threshold
    )

    total_frames = len(results)
    violent_percentage = (
        (violent_frame_count / total_frames) * 100 if total_frames else 0
    )

    violent_stat = f"{violent_frame_count}/{total_frames} ({violent_percentage:.2f}%)"
    print(f"Violent frames: {violent_stat}")

    output_data = {
        "file_name": str(input_file.name),
        "results": results,
        "overall_violence_score": float(overall_score),
        "is_violent_overall": bool(is_violent_overall),
        "violent_frame_percentage": float(violent_percentage),
        "classification_threshold": float(threshold),
        "interpretation": str(overall_interpretation),
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {output_file}")
    print(f"Overall violence score: {overall_score}")
    print(f"Interpretation: {overall_interpretation}")


if __name__ == "__main__":
    main()
