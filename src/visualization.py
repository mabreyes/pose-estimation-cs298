#!/usr/bin/env python3
"""
Visualization utilities for the violence detection model.

This module provides functions to plot and visualize various aspects of the
violence detection model, including training metrics, classification performance,
and model-specific visualizations.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def plot_training_metrics(
    metrics: Dict[str, List[float]],
    test_metrics: Dict[str, float],
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot and save training and validation metrics.

    Args:
        metrics: Dictionary containing training metrics history
        test_metrics: Dictionary containing test metrics
        output_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))

    # Plot loss curves
    plt.subplot(2, 2, 1)
    plt.plot(metrics["train_loss"], label="Train Loss")
    plt.plot(metrics["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)

    # Plot AUC curves
    plt.subplot(2, 2, 2)
    plt.plot(metrics["val_auc"], label="Validation AUC")
    plt.axhline(
        y=test_metrics["auc"],
        color="r",
        linestyle="--",
        label=f"Test AUC: {test_metrics['auc']:.4f}",
    )
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.title("ROC AUC Score")
    plt.grid(True, alpha=0.3)

    # If available, plot additional metrics
    if "val_f1" in metrics:
        plt.subplot(2, 2, 3)
        plt.plot(metrics["val_f1"], label="Validation F1")
        if "f1" in test_metrics:
            plt.axhline(
                y=test_metrics["f1"],
                color="r",
                linestyle="--",
                label=f"Test F1: {test_metrics['f1']:.4f}",
            )
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.title("F1 Score")
        plt.grid(True, alpha=0.3)

    # Plot accuracy if available
    if "val_accuracy" in metrics:
        plt.subplot(2, 2, 4)
        plt.plot(metrics["val_accuracy"], label="Validation Accuracy")
        if "accuracy" in test_metrics:
            plt.axhline(
                y=test_metrics["accuracy"],
                color="r",
                linestyle="--",
                label=f"Test Accuracy: {test_metrics['accuracy']:.4f}",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Training metrics plot saved to {output_path}")
    else:
        plt.show()


def plot_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot comprehensive classification metrics including ROC, PR curves
    and confusion matrix.

    Args:
        y_true: Ground truth binary labels
        y_score: Predicted scores (probabilities)
        threshold: Classification threshold
        output_path: Path to save the plot
    """
    plt.figure(figsize=(18, 10))

    # ROC Curve
    plt.subplot(2, 3, 1)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})")

    # Mark threshold point
    threshold_idx = np.argmin(np.abs(thresholds - threshold))
    plt.plot(
        fpr[threshold_idx],
        tpr[threshold_idx],
        "ro",
        label=f"Threshold = {threshold:.4f}",
    )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Precision-Recall Curve
    plt.subplot(2, 3, 2)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)

    # Find closest threshold value in PR curve
    pr_thresholds = np.append(
        pr_thresholds, 1.0
    )  # Add 1.0 to match precision/recall arrays
    threshold_idx_pr = np.argmin(np.abs(pr_thresholds - threshold))

    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.plot(
        recall[threshold_idx_pr],
        precision[threshold_idx_pr],
        "ro",
        label=f"Threshold = {threshold:.4f}",
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    # Confusion Matrix
    plt.subplot(2, 3, 3)
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.xticks([0.5, 1.5], ["Non-violent", "Violent"])
    plt.yticks([0.5, 1.5], ["Non-violent", "Violent"])

    # Threshold vs. F1 score
    plt.subplot(2, 3, 4)
    f1_scores = []
    threshold_range = np.linspace(0, 1, 100)
    for t in threshold_range:
        y_pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f1_scores.append(f1)

    plt.plot(threshold_range, f1_scores)
    plt.axvline(
        x=threshold,
        color="r",
        linestyle="--",
        label=f"Selected threshold: {threshold:.4f}",
    )
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("Threshold vs. F1 Score")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Metrics at threshold
    plt.subplot(2, 3, 5)
    y_pred = (y_score >= threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    precision_val = precision_score(y_true, y_pred, zero_division=0)
    recall_val = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics_vals = [accuracy, precision_val, recall_val, f1]
    metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score"]

    # Create a horizontal bar chart
    y_pos = np.arange(len(metrics_names))
    plt.barh(y_pos, metrics_vals, align="center")
    plt.yticks(y_pos, metrics_names)
    plt.xlim([0, 1])

    # Add values to bars
    for i, v in enumerate(metrics_vals):
        plt.text(v + 0.01, i, f"{v:.4f}", va="center")

    plt.xlabel("Score")
    plt.title(f"Metrics at Threshold = {threshold:.4f}")
    plt.grid(True, alpha=0.3, axis="x")

    # Distribution of scores
    plt.subplot(2, 3, 6)
    positive_scores = y_score[y_true == 1]
    negative_scores = y_score[y_true == 0]

    plt.hist(negative_scores, bins=20, alpha=0.5, label="Non-violent", color="green")
    plt.hist(positive_scores, bins=20, alpha=0.5, label="Violent", color="red")
    plt.axvline(
        x=threshold, color="k", linestyle="--", label=f"Threshold: {threshold:.4f}"
    )
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Distribution of Scores")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Classification metrics plot saved to {output_path}")
    else:
        plt.show()


def plot_learning_curve(
    metrics: Dict[str, List[float]],
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot learning curve to assess model training process.

    Args:
        metrics: Dictionary containing training metrics history
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(metrics["train_loss"], label="Training Loss")
    plt.plot(metrics["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve - Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot AUC score
    plt.subplot(1, 2, 2)
    plt.plot(metrics["val_auc"], label="Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC Score")
    plt.title("Learning Curve - AUC")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Learning curve plot saved to {output_path}")
    else:
        plt.show()


def plot_pose_graph(
    keypoints: np.ndarray,
    graph_edges: List[Tuple[int, int]],
    is_violent: bool,
    output_path: Optional[Path] = None,
) -> None:
    """
    Visualize a pose graph with its node connections.

    Args:
        keypoints: NumPy array of shape [num_keypoints, 2] containing (x, y) coordinates
        graph_edges: List of tuples representing edges between keypoints
        is_violent: Whether the pose is from a violent sample
        output_path: Path to save the plot
    """
    plt.figure(figsize=(8, 8))

    # Plot keypoints
    plt.scatter(
        keypoints[:, 0],
        keypoints[:, 1],
        c="red" if is_violent else "blue",
        s=100,
        marker="o",
    )

    # Plot edges
    for edge in graph_edges:
        plt.plot(
            [keypoints[edge[0], 0], keypoints[edge[1], 0]],
            [keypoints[edge[0], 1], keypoints[edge[1], 1]],
            "k-",
            alpha=0.6,
        )

    # Add labels for keypoints
    for i, (x, y) in enumerate(keypoints):
        plt.text(x, y, str(i), fontsize=10, ha="center", va="center")

    plt.title(f"Pose Graph ({'Violent' if is_violent else 'Non-violent'})")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path)
        print(f"Pose graph visualization saved to {output_path}")
    else:
        plt.show()
