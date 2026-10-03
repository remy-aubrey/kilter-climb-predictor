"""Transformer-based climb sequence generator.

Uses a transformer architecture to generate climbing routes.
Can handle long-range dependencies and grade conditioning.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from src.utils.constants import (
    GRADE_TO_INDEX,
    INDEX_TO_GRADE,
    INDEX_TO_HOLD_TYPE,
    INDEX_TO_LED_COLOR,
    RANDOM_SEED,
    NUM_GRADES,
    NUM_HOLD_TYPES,
    NUM_LED_COLORS,
    MAX_HOLDS_PER_CLIMB,
)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class ClimbTransformer(nn.Module):
    """Transformer model for climb generation."""

    def __init__(
        self,
        num_holds: int,
        hold_embedding_dim: int = 64,
        grade_embedding_dim: int = 16,
        hold_type_embedding_dim: int = 16,
        led_color_embedding_dim: int = 16,
        position_dim: int = 2,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_holds = num_holds
        self.d_model = d_model

        # Embedding layers
        self.hold_embedding = nn.Embedding(num_holds, hold_embedding_dim, padding_idx=0)
        self.grade_embedding = nn.Embedding(NUM_GRADES, grade_embedding_dim)
        self.hold_type_embedding = nn.Embedding(NUM_HOLD_TYPES, hold_type_embedding_dim)
        self.led_color_embedding = nn.Embedding(NUM_LED_COLORS, led_color_embedding_dim)

        # Input projection
        input_dim = hold_embedding_dim + position_dim + hold_type_embedding_dim + led_color_embedding_dim + grade_embedding_dim
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads
        self.hold_classifier = nn.Linear(d_model, num_holds)
        self.hold_type_classifier = nn.Linear(d_model, NUM_HOLD_TYPES)
        self.led_color_classifier = nn.Linear(d_model, NUM_LED_COLORS)
        self.position_regressor = nn.Linear(d_model, position_dim)

    def forward(
        self,
        hold_indices: torch.Tensor,
        positions: torch.Tensor,
        hold_type_indices: torch.Tensor,
        led_color_indices: torch.Tensor,
        grade_indices: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            hold_indices: (batch, seq_len) hold ID indices
            positions: (batch, seq_len, 2) x, y coordinates
            hold_type_indices: (batch, seq_len) hold type indices
            led_color_indices: (batch, seq_len) LED color indices
            grade_indices: (batch,) grade indices
            mask: (batch, seq_len) padding mask

        Returns:
            Dict with logits and predictions
        """
        batch_size, seq_len = hold_indices.shape

        # Embed features
        hold_emb = self.hold_embedding(hold_indices)
        hold_type_emb = self.hold_type_embedding(hold_type_indices)
        led_color_emb = self.led_color_embedding(led_color_indices)

        # Grade embedding: expand to sequence length
        grade_emb = self.grade_embedding(grade_indices)
        grade_emb = grade_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate all features
        features = torch.cat([hold_emb, positions, hold_type_emb, led_color_emb, grade_emb], dim=-1)

        # Project to d_model
        features = self.input_projection(features)

        # Add positional encoding
        features = self.pos_encoder(features)

        # Create causal mask (can't see future positions)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(features.device)

        # Transformer
        output = self.transformer(features, mask=causal_mask)

        # Output predictions
        hold_logits = self.hold_classifier(output)
        hold_type_logits = self.hold_type_classifier(output)
        led_color_logits = self.led_color_classifier(output)
        position_pred = self.position_regressor(output)

        return {
            "hold_logits": hold_logits,
            "hold_type_logits": hold_type_logits,
            "led_color_logits": led_color_logits,
            "position_pred": position_pred,
        }

    @torch.no_grad()
    def generate_step(
        self,
        hold_idx: int,
        position: torch.Tensor,
        hold_type_idx: int,
        led_color_idx: int,
        grade_idx: int,
        temperature: float = 1.0,
    ) -> dict:
        """Generate a single next hold (autoregressive step)."""
        # Add batch and sequence dimensions
        hold_indices = torch.tensor([[hold_idx]], dtype=torch.long, device=position.device)
        positions = position.unsqueeze(0).unsqueeze(0)
        hold_type_indices = torch.tensor([[hold_type_idx]], dtype=torch.long, device=position.device)
        led_color_indices = torch.tensor([[led_color_idx]], dtype=torch.long, device=position.device)
        grade_indices = torch.tensor([grade_idx], dtype=torch.long, device=position.device)

        output = self.forward(
            hold_indices, positions, hold_type_indices, led_color_indices, grade_indices
        )

        # Sample next hold
        hold_logits = output["hold_logits"][0, -1] / temperature
        hold_probs = torch.softmax(hold_logits, dim=-1)
        next_hold_idx = torch.multinomial(hold_probs, 1).item()

        # Sample next hold type
        type_logits = output["hold_type_logits"][0, -1] / temperature
        type_probs = torch.softmax(type_logits, dim=-1)
        next_type_idx = torch.multinomial(type_probs, 1).item()

        # Sample next LED color
        color_logits = output["led_color_logits"][0, -1] / temperature
        color_probs = torch.softmax(color_logits, dim=-1)
        next_color_idx = torch.multinomial(color_probs, 1).item()

        # Predict next position
        next_position = output["position_pred"][0, -1]

        return {
            "hold_idx": next_hold_idx,
            "position": next_position,
            "hold_type_idx": next_type_idx,
            "led_color_idx": next_color_idx,
        }


class TransformerClimbGenerator:
    """Transformer-based generator for climbing routes."""

    def __init__(self, model: ClimbTransformer, hold_id_mapping: dict[str, int]):
        self.model = model
        self.hold_id_mapping = hold_id_mapping
        self.idx_to_hold_id = {v: k for k, v in hold_id_mapping.items()}

    def generate(
        self,
        grade: str,
        length: int | None = None,
        temperature: float = 1.0,
        seed: int | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> list[dict]:
        """Generate a climb using the transformer.

        Args:
            grade: Target grade (e.g., "V5")
            length: Number of holds (random if None)
            temperature: Sampling temperature
            seed: Random seed for reproducibility
            device: Device to run on

        Returns:
            List of hold dicts with keys: hold_id, x, y, hold_type, led_color
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        grade_idx = GRADE_TO_INDEX[grade]

        # Determine climb length
        if length is None:
            length = random.randint(7, 20)

        # Start with a random start hold
        start_holds = [
            idx for idx, hold_id in self.idx_to_hold_id.items()
            if idx > 0
        ]
        if not start_holds:
            raise ValueError("No holds available in mapping")

        current_hold_idx = random.choice(start_holds)
        current_position = torch.tensor([74.0, 80.0], dtype=torch.float32, device=device)
        current_type_idx = 0  # start
        current_color_idx = 1  # green

        holds = []
        for step in range(length):
            result = self.model.generate_step(
                hold_idx=current_hold_idx,
                position=current_position,
                hold_type_idx=current_type_idx,
                led_color_idx=current_color_idx,
                grade_idx=grade_idx,
                temperature=temperature,
            )

            current_hold_idx = result["hold_idx"]
            current_position = result["position"]
            current_type_idx = result["hold_type_idx"]
            current_color_idx = result["led_color_idx"]

            hold = {
                "hold_id": self.idx_to_hold_id.get(current_hold_idx, f"hold_{current_hold_idx}"),
                "x": float(current_position[0].item()),
                "y": float(current_position[1].item()),
                "hold_type": INDEX_TO_HOLD_TYPE.get(current_type_idx, "middle"),
                "led_color": INDEX_TO_LED_COLOR.get(current_color_idx, "red"),
            }
            holds.append(hold)

            # Stop if we generated a finish hold
            if hold["hold_type"] == "finish":
                break

        return holds

    def save(self, path: Path) -> None:
        """Save the trained model to disk."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "hold_id_mapping": self.hold_id_mapping,
        }, path)

    def load(self, path: Path, device: torch.device = torch.device("cpu")) -> None:
        """Load a trained model from disk."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.hold_id_mapping = checkpoint["hold_id_mapping"]
        self.idx_to_hold_id = {v: k for k, v in self.hold_id_mapping.items()}
        # Model state dict is loaded separately
