#!/usr/bin/env python3
"""
Transformer component for the violence detection system.

This module contains the transformer model that processes embeddings
from the GNN before final classification.
"""
from typing import Optional

import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for processing graph embeddings.

    This model takes embeddings from the GNN and applies self-attention
    to capture temporal and contextual relationships.
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 64,
    ):
        """
        Initialize the transformer encoder.

        Args:
            input_dim: Dimension of input features
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            output_dim: Dimension of output features
        """
        super(TransformerEncoder, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Add batch dimension info for the transformer
        self.position_embedding = nn.Parameter(torch.zeros(1, 1, input_dim))

        # Create transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )

        # Create transformer encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )

        # Output projection if needed
        self.out_projection = None
        if output_dim != input_dim:
            self.out_projection = nn.Linear(input_dim, output_dim)

        self.out_channels = output_dim

        # For storing attention weights
        self.attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Transformed representation [batch_size, output_dim]
        """
        # Add sequence dimension for transformer
        x = x.unsqueeze(1)  # [batch_size, 1, input_dim]

        # Add positional embedding
        x = x + self.position_embedding

        # Apply transformer
        x = self.transformer(x)

        # Remove sequence dimension
        x = x.squeeze(1)  # [batch_size, input_dim]

        # Apply output projection if needed
        if self.out_projection is not None:
            x = self.out_projection(x)

        return x

    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract attention weights from the transformer for visualization.

        This function is a workaround since PyTorch's TransformerEncoder doesn't
        expose attention weights directly. We create a mock attention tensor for
        visualization purposes.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Tensor containing attention weights or None if not available
        """
        # PyTorch doesn't expose attention weights directly from TransformerEncoder
        # We'll create a simplified representation for visualization purposes
        x.shape[0]

        # For actual implementation, you would need to:
        # 1. Register hooks to capture attention weights during forward pass
        # 2. Or modify PyTorch's TransformerEncoder to expose attention weights

        # This is a placeholder implementation that creates mock attention data
        # based on input similarity
        try:
            # Add sequence dimension
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]

            # Normalize inputs
            x_norm = torch.nn.functional.normalize(x, p=2, dim=2)

            # Create mock attention weights based on input similarity
            # This is just for visualization - not actual attention
            mock_attention = torch.bmm(x_norm, x_norm.transpose(1, 2))

            # Expand to match expected dimensions for multiple heads
            # [batch_size, num_heads, seq_len, seq_len]
            mock_attention = mock_attention.unsqueeze(1).expand(
                -1, self.num_heads, -1, -1
            )

            return mock_attention
        except Exception:
            return None

    def __repr__(self) -> str:
        """String representation of the transformer encoder."""
        return (
            f"TransformerEncoder(input_dim={self.input_dim}, "
            f"num_heads={self.num_heads}, num_layers={self.num_layers}, "
            f"output_dim={self.out_channels})"
        )
