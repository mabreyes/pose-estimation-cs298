#!/usr/bin/env python3
"""
Violence Detection Model Architecture.

This module defines the main model architecture for violence detection from human pose
data, using a Graph Recurrent Neural Network (GRNN) approach.
It includes:
- Device detection for optimal hardware utilization
- ViolenceDetectionGRNN model that processes spatio-temporal graph data
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Import the GRNN component
from grnn import ViolenceDetectionGRNN


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


# Keeping this class for backwards compatibility
class ViolenceDetectionGNN(nn.Module):
    """
    Compatibility wrapper for the new GRNN-based model.

    This class maintains the same interface as the previous GNN+Transformer model
    but delegates to the new GRNN implementation.

    Note: This is included for backwards compatibility. New code should
    directly use ViolenceDetectionGRNN.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
    ):
        """
        Initialize the compatibility wrapper.

        Args:
            in_channels: Number of input features per node
            hidden_channels: Size of hidden representations
            transformer_heads: Ignored (kept for compatibility)
            transformer_layers: Used as num_layers for GRNN
        """
        super(ViolenceDetectionGNN, self).__init__()

        # Create the actual GRNN model
        self.grnn_model = ViolenceDetectionGRNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=transformer_layers,
            dropout=0.3,
            bidirectional=True,
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass - adapts the old interface to the new GRNN model.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]

        Returns:
            Violence score between 0 and 1 [batch_size, 1]
        """
        # Reshape x for GRNN (which expects sequence data)
        batch_size = torch.max(batch).item() + 1
        # Reshape assuming single time step for compatibility with old interface
        # For the compatibility layer, we treat the input as a sequence of length 1
        x_reshaped = x.view(batch_size, 1, -1, x.size(-1))

        # Process through the GRNN model
        return self.grnn_model(x_reshaped, edge_index, batch)
