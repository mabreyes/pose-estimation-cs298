#!/usr/bin/env python3
"""
Violence Detection Model Architecture.

This module defines the main model architecture for violence detection from human pose
data, combining a Graph Neural Network and Transformer components.
It includes:
- Device detection for optimal hardware utilization
- ViolenceDetectionGNN model that combines GNN and Transformer processing
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import components from separate files
from gnn import PoseGNN
from transformer import TransformerEncoder


def get_device() -> torch.device:
    """
    Determine the optimal device for training/inference.

    Checks for CUDA GPU availability first, then Apple Silicon MPS support,
    and falls back to CPU if neither is available.

    Returns:
        torch.device: CUDA if available, MPS if on Apple Silicon, otherwise CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class ViolenceDetectionGNN(nn.Module):
    """
    Full model architecture for violence detection from pose data.

    This model processes pose keypoints using a pipeline of:
    1. Graph Neural Network to process pose graph structure
    2. Transformer to capture contextual patterns
    3. Classifier to produce violence score

    The architecture combines state-of-the-art GNN and transformer techniques
    to effectively process spatial relationships in human pose data and
    classify violent behavior.
    """

    # Constants for the model architecture
    DROPOUT_RATE = 0.3

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
    ):
        """
        Initialize the full model.

        Args:
            in_channels: Number of input features per node
                         (typically 2 for x,y coordinates)
            hidden_channels: Size of hidden representations
            transformer_heads: Number of attention heads in transformer
            transformer_layers: Number of transformer layers
        """
        super(ViolenceDetectionGNN, self).__init__()

        # GNN component
        self.gnn = PoseGNN(in_channels, hidden_channels)

        # Transformer component
        self.transformer = TransformerEncoder(
            input_dim=hidden_channels,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            output_dim=hidden_channels,
        )

        # Final prediction layers
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, 1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the full model.

        Processes pose data through the GNN to capture spatial relationships,
        then through the transformer to capture contextual patterns, and finally
        through classifier layers to produce a violence score.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]

        Returns:
            Violence score between 0 and 1 [batch_size, 1]
        """
        # Process through GNN to get graph embeddings
        x = self.gnn(x, edge_index, batch)

        # Process through transformer to capture contextual patterns
        x = self.transformer(x)

        # Final predictions
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.DROPOUT_RATE, training=self.training)
        x = self.lin2(x)

        # Output violence score between 0 and 1
        return torch.sigmoid(x)
