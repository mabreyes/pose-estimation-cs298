#!/usr/bin/env python3
"""
Violence Detection using Graph Neural Networks.

This module implements a GNN model to detect violent behavior from human pose data
in MMPose JSON format. It includes functionality for:
- Data loading and preprocessing from MMPose JSON format
- Graph construction from pose keypoints
- Model training with metrics tracking
- Model evaluation and optimal threshold selection
- Result visualization and model persistence
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import visualization as viz

# Import components from separate files
from gnn import create_pose_graph
from model import ViolenceDetectionGNN, get_device

# Configuration constants
# Data paths
if torch.cuda.is_available():
    # GPU detected paths
    DATA_PATH = Path("./json")
    VIOLENT_PATH_CAM1 = DATA_PATH / "violent/cam1"
    NON_VIOLENT_PATH_CAM1 = DATA_PATH / "non-violent/cam1"
    VIOLENT_PATH_CAM2 = DATA_PATH / "violent/cam2"
    NON_VIOLENT_PATH_CAM2 = DATA_PATH / "non-violent/cam2"
else:
    # Local paths (no GPU)
    DATA_PATH = Path("/Volumes/MARCREYES/violence-detection-dataset")
    VIOLENT_PATH_CAM1 = DATA_PATH / "violent/cam1/processed"
    NON_VIOLENT_PATH_CAM1 = DATA_PATH / "non-violent/cam1/processed"
    VIOLENT_PATH_CAM2 = DATA_PATH / "violent/cam2/processed"
    NON_VIOLENT_PATH_CAM2 = DATA_PATH / "non-violent/cam2/processed"

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 2
LEARNING_RATE = 0.001
SAMPLE_PERCENTAGE = 1  # Percentage of data to use (1-100)

# Model and evaluation constants
MODEL_HIDDEN_CHANNELS = 64
MODEL_TRANSFORMER_HEADS = 4
MODEL_TRANSFORMER_LAYERS = 2
TEST_SPLIT_RATIO = 0.2
VALIDATION_SPLIT_RATIO = 0.25
RANDOM_SEED = 42


def find_optimal_threshold(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate the optimal classification threshold using multiple methods.

    Implements several threshold optimization techniques:
    1. Youden's J statistic (maximizing sensitivity + specificity - 1)
    2. Minimum distance to perfect classifier (0,1) point in ROC space
    3. Maximum F1 score

    The primary method used is Youden's J statistic, which is widely accepted
    in the academic literature for binary classification threshold optimization.

    Args:
        y_true: Ground truth binary labels
        y_score: Predicted scores (probabilities)

    Returns:
        Tuple of (optimal threshold, dictionary of metrics at that threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Calculate Youden's J statistic (J = Sensitivity + Specificity - 1)
    j_scores = tpr - fpr
    optimal_idx_j = np.argmax(j_scores)
    optimal_threshold_j = thresholds[optimal_idx_j]

    # Calculate distance to (0,1) point in ROC space
    distances = np.sqrt((1 - tpr) ** 2 + fpr**2)
    optimal_idx_d = np.argmin(distances)
    optimal_threshold_d = thresholds[optimal_idx_d]

    # Calculate F1 score at different thresholds
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)

    # Calculate F1 for all possible thresholds
    f1_scores = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)

    optimal_idx_f1 = np.argmax(f1_scores)
    optimal_threshold_f1 = thresholds[optimal_idx_f1]

    # Choose Youden's J as the primary method (most common in academic literature)
    optimal_threshold = optimal_threshold_j

    # Calculate confusion matrix at optimal threshold
    y_pred = (y_score >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate various metrics at the optimal threshold
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Create metrics dictionary
    metrics = {
        "threshold_j": optimal_threshold_j,
        "threshold_distance": optimal_threshold_d,
        "threshold_f1": optimal_threshold_f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_val,
        "f1_score": f1_scores[optimal_idx_j],
        "youdens_j": j_scores[optimal_idx_j],
    }

    return optimal_threshold, metrics


def load_mmpose_data(
    violent_path: Path, non_violent_path: Path, sample_percentage: int = 100
) -> Tuple[List[Data], List[float]]:
    """
    Load MMPose JSON files and convert them to graph data.

    Processes JSON files containing pose keypoints from both violent and non-violent
    video frames. Each person instance in a frame is converted to a graph representation
    suitable for GNN processing. The function supports processing a subset of the data
    using the sample_percentage parameter.

    Args:
        violent_path: Path to violent pose JSON files
        non_violent_path: Path to non-violent pose JSON files
        sample_percentage: Percentage of files to process (1-100)

    Returns:
        Tuple of (list of graph Data objects, list of corresponding labels)
    """
    # Validate sample percentage
    if not 1 <= sample_percentage <= 100:
        raise ValueError("sample_percentage must be between 1 and 100")

    all_graphs = []
    all_labels = []

    # Get all JSON files from the violent directory
    violent_files = list(violent_path.glob("*.json"))
    if not violent_files:
        raise ValueError(f"No JSON files found in violent directory: {violent_path}")

    print(f"Found {len(violent_files)} violent JSON files")

    # Calculate number of files to process based on percentage
    num_violent_files = max(1, int(len(violent_files) * sample_percentage / 100))

    # Process violent samples
    for json_file in tqdm(
        violent_files[:num_violent_files], desc="Processing violent samples"
    ):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Process each frame in the JSON file
        for frame_data in data.get("instance_info", []):
            # Get frame ID (not used but kept for consistency)
            _ = frame_data.get("frame_id")
            instances = frame_data.get("instances", [])

            for instance in instances:
                keypoints = instance.get("keypoints", [])
                if keypoints:
                    # Convert to numpy array
                    keypoints_np = np.array(keypoints)

                    # Create graph from keypoints
                    graph = create_pose_graph(keypoints_np)
                    if graph is not None:
                        all_graphs.append(graph)
                        all_labels.append(1.0)  # Violent label

    # Process non-violent data
    non_violent_files = list(non_violent_path.glob("*.json"))
    if not non_violent_files:
        raise ValueError(
            f"No JSON files found in non-violent directory: {non_violent_path}"
        )

    print(f"Found {len(non_violent_files)} non-violent JSON files")

    # Calculate number of files to process based on percentage
    num_nonviolent_files = max(1, int(len(non_violent_files) * sample_percentage / 100))

    # Process non-violent samples
    for json_file in tqdm(
        non_violent_files[:num_nonviolent_files], desc="Processing non-violent samples"
    ):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Process each frame in the JSON file
        for frame_data in data.get("instance_info", []):
            # Get frame ID (not used but kept for consistency)
            _ = frame_data.get("frame_id")
            instances = frame_data.get("instances", [])

            for instance in instances:
                keypoints = instance.get("keypoints", [])
                if keypoints:
                    # Convert to numpy array
                    keypoints_np = np.array(keypoints)

                    # Create graph from keypoints
                    graph = create_pose_graph(keypoints_np)
                    if graph is not None:
                        all_graphs.append(graph)
                        all_labels.append(0.0)  # Non-violent label

    return all_graphs, all_labels


def train_model(
    model: ViolenceDetectionGNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 50,
) -> Dict[str, List[float]]:
    """
    Train the GNN model and track metrics.

    Implements a training loop with both training and validation phases.
    For each epoch, the model is trained on the training set and evaluated
    on the validation set. Metrics including loss and AUC are tracked.

    Args:
        model: The GNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on (CPU/GPU/MPS)
        optimizer: Optimizer for training
        num_epochs: Number of training epochs

    Returns:
        Dictionary of training and validation metrics
    """
    # Training metrics
    metrics: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_auc": []}

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0

        # Process batches
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"
        ):
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            out = model(batch.x, batch.edge_index, batch.batch)
            target = batch.y.view(-1, 1).to(device)

            # Calculate loss
            loss = F.binary_cross_entropy(out, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs

        # Calculate average training loss
        avg_train_loss = total_loss / len(train_loader.dataset)
        metrics["train_loss"].append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"
            ):
                batch = batch.to(device)

                # Forward pass
                out = model(batch.x, batch.edge_index, batch.batch)
                target = batch.y.view(-1, 1).to(device)

                # Calculate loss
                loss = F.binary_cross_entropy(out, target)
                val_loss += loss.item() * batch.num_graphs

                # Store predictions and targets for metrics
                all_preds.extend(out.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())

        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_auc = roc_auc_score(all_targets, all_preds)

        metrics["val_loss"].append(avg_val_loss)
        metrics["val_auc"].append(val_auc)

        # Print epoch results
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val AUC: {val_auc:.4f}")

    return metrics


def evaluate_model(
    model: ViolenceDetectionGNN, test_loader: DataLoader, device: torch.device
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Evaluate the model on the test set.

    Performs a comprehensive evaluation of the trained model on the test set,
    calculating loss, AUC, and finding the optimal classification threshold.

    Args:
        model: The trained GNN model
        test_loader: Test data loader
        device: Device for evaluation

    Returns:
        Tuple of (test_loss, test_auc, optimal_threshold, threshold_metrics)
    """
    model.eval()
    test_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)

            # Forward pass
            out = model(batch.x, batch.edge_index, batch.batch)
            target = batch.y.view(-1, 1).to(device)

            # Calculate loss
            loss = F.binary_cross_entropy(out, target)
            test_loss += loss.item() * batch.num_graphs

            # Store predictions and targets for metrics
            all_preds.extend(out.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())

    avg_test_loss = test_loss / len(test_loader.dataset)
    test_auc = roc_auc_score(all_targets, all_preds)

    # Find optimal classification threshold
    optimal_threshold, threshold_metrics = find_optimal_threshold(
        np.array(all_targets), np.array(all_preds)
    )

    return avg_test_loss, test_auc, optimal_threshold, threshold_metrics


def main() -> None:
    """
    Main function to train and evaluate the violence detection model.

    This function orchestrates the entire training pipeline:
    1. Sets up the device and data paths
    2. Loads and preprocesses data
    3. Splits data into training, validation, and test sets
    4. Trains the model
    5. Evaluates the model and finds optimal classification threshold
    6. Saves the model and generates visualizations
    """
    device = get_device()
    print(f"Using device: {device}")

    # Check if directories exist
    if not VIOLENT_PATH_CAM1.exists():
        print(f"Error: Violent data path (cam1) does not exist: {VIOLENT_PATH_CAM1}")
        return

    if not NON_VIOLENT_PATH_CAM1.exists():
        print(
            f"Error: Non-violent data path (cam1) does not exist: "
            f"{NON_VIOLENT_PATH_CAM1}"
        )
        return

    all_graphs = []
    all_labels = []

    # Load data from cam1
    print("Loading and preprocessing data from cam1...")
    try:
        graphs_cam1, labels_cam1 = load_mmpose_data(
            VIOLENT_PATH_CAM1, NON_VIOLENT_PATH_CAM1, SAMPLE_PERCENTAGE
        )
        all_graphs.extend(graphs_cam1)
        all_labels.extend(labels_cam1)
        print(f"Loaded {len(graphs_cam1)} graphs from cam1")
    except ValueError as e:
        print(f"Error loading data from cam1: {e}")

    # Check if cam2 data exists and load it
    cam2_exists = VIOLENT_PATH_CAM2.exists() and NON_VIOLENT_PATH_CAM2.exists()

    if cam2_exists:
        print("Loading and preprocessing data from cam2...")
        try:
            graphs_cam2, labels_cam2 = load_mmpose_data(
                VIOLENT_PATH_CAM2, NON_VIOLENT_PATH_CAM2, SAMPLE_PERCENTAGE
            )
            all_graphs.extend(graphs_cam2)
            all_labels.extend(labels_cam2)
            print(f"Loaded {len(graphs_cam2)} graphs from cam2")
        except ValueError as e:
            print(f"Error loading data from cam2: {e}")
    else:
        print("Cam2 data not found. Using only cam1 data for training.")

    if not all_graphs:
        print("No valid graphs were created. Check your data.")
        return

    print(f"Total graphs: {len(all_graphs)}")
    print(f"Positive (violent) samples: {sum(all_labels)}")
    print(f"Negative (non-violent) samples: {len(all_labels) - sum(all_labels)}")

    # Assign labels to graphs
    for i, graph in enumerate(all_graphs):
        graph.y = torch.tensor([all_labels[i]], dtype=torch.float)

    # Split data into train, validation, and test sets
    train_graphs, test_graphs = train_test_split(
        all_graphs,
        test_size=TEST_SPLIT_RATIO,
        random_state=RANDOM_SEED,
        stratify=all_labels,
    )
    train_graphs, val_graphs = train_test_split(
        train_graphs, test_size=VALIDATION_SPLIT_RATIO, random_state=RANDOM_SEED
    )

    print(f"Training graphs: {len(train_graphs)}")
    print(f"Validation graphs: {len(val_graphs)}")
    print(f"Test graphs: {len(test_graphs)}")

    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)

    # Get input channel dimension from data
    in_channels = train_graphs[0].x.shape[1]

    # Initialize model
    model = ViolenceDetectionGNN(
        in_channels=in_channels,
        hidden_channels=MODEL_HIDDEN_CHANNELS,
        transformer_heads=MODEL_TRANSFORMER_HEADS,
        transformer_layers=MODEL_TRANSFORMER_LAYERS,
    ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train model
    print("Training model...")
    metrics = train_model(
        model, train_loader, val_loader, device, optimizer, num_epochs=NUM_EPOCHS
    )

    # Evaluate model
    avg_test_loss, test_auc, optimal_threshold, threshold_metrics = evaluate_model(
        model, test_loader, device
    )
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Optimal classification threshold: {optimal_threshold:.4f}")
    print("Threshold metrics:")
    for metric, value in threshold_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save model
    model_path = Path("violence_detection_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "threshold": optimal_threshold,
            "metrics": threshold_metrics,
        },
        model_path,
    )
    print(f"Model saved to {model_path}")

    # Create output directory for plots
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Create test metrics dictionary for visualization
    test_metrics = {
        "loss": avg_test_loss,
        "auc": test_auc,
        "f1": threshold_metrics["f1_score"],
        "threshold": optimal_threshold,
    }

    # Extract all predictions and targets from test set for visualizations
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            all_preds.extend(out.cpu().numpy().flatten())
            all_targets.extend(batch.y.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Generate visualizations
    viz.plot_training_metrics(
        metrics, test_metrics, output_path=plots_dir / "training_metrics.png"
    )

    viz.plot_classification_metrics(
        all_targets,
        all_preds,
        optimal_threshold,
        output_path=plots_dir / "classification_metrics.png",
    )

    viz.plot_learning_curve(metrics, output_path=plots_dir / "learning_curve.png")

    # Sample a pose graph for visualization if available
    if test_graphs:
        sample_idx = 0
        sample_graph = test_graphs[sample_idx]
        keypoints = sample_graph.x.numpy()

        # Extract edges as tuples
        edge_index = sample_graph.edge_index.numpy()
        edges = [
            (edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])
        ]

        # Get label
        is_violent = bool(sample_graph.y.item() > 0.5)

        viz.plot_pose_graph(
            keypoints,
            edges,
            is_violent,
            output_path=plots_dir / "sample_pose_graph.png",
        )

    print(f"All visualizations saved to {plots_dir}")


if __name__ == "__main__":
    main()
