"""LSTM-based climb sequence generator network."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.constants import (
    NUM_GRADES,
    NUM_HOLD_TYPES,
    NUM_LED_COLORS,
    MAX_HOLDS_PER_CLIMB,
)


class ClimbGenerator(nn.Module):
    """LSTM-based sequence generator for climbing routes.

    Architecture:
        - Embedding layers for hold IDs, hold types, LED colors, and grade
        - 2-layer LSTM
        - Linear output heads for each feature

    The model predicts the next hold given the sequence so far and the target grade.
    """

    def __init__(
        self,
        num_holds: int,
        hold_embedding_dim: int = 64,
        grade_embedding_dim: int = 16,
        hold_type_embedding_dim: int = 16,
        led_color_embedding_dim: int = 16,
        position_dim: int = 2,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_holds = num_holds
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layers
        self.hold_embedding = nn.Embedding(num_holds, hold_embedding_dim, padding_idx=0)
        self.grade_embedding = nn.Embedding(NUM_GRADES, grade_embedding_dim)
        self.hold_type_embedding = nn.Embedding(NUM_HOLD_TYPES, hold_type_embedding_dim)
        self.led_color_embedding = nn.Embedding(NUM_LED_COLORS, led_color_embedding_dim)

        # Input dimension: hold_emb + position + hold_type_emb + led_color_emb + grade_emb
        input_dim = hold_embedding_dim + position_dim + hold_type_embedding_dim + led_color_embedding_dim + grade_embedding_dim

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Output heads
        self.hold_classifier = nn.Linear(hidden_size, num_holds)
        self.hold_type_classifier = nn.Linear(hidden_size, NUM_HOLD_TYPES)
        self.led_color_classifier = nn.Linear(hidden_size, NUM_LED_COLORS)
        self.position_regressor = nn.Linear(hidden_size, position_dim)

    def forward(
        self,
        hold_indices: torch.Tensor,
        positions: torch.Tensor,
        hold_type_indices: torch.Tensor,
        led_color_indices: torch.Tensor,
        grade_indices: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            hold_indices: (batch, seq_len) hold ID indices
            positions: (batch, seq_len, 2) x, y coordinates
            hold_type_indices: (batch, seq_len) hold type indices
            led_color_indices: (batch, seq_len) LED color indices
            grade_indices: (batch,) grade indices
            hidden: Optional LSTM hidden state

        Returns:
            Dict with logits and new hidden state
        """
        batch_size, seq_len = hold_indices.shape

        # Embed features
        hold_emb = self.hold_embedding(hold_indices)  # (B, L, hold_emb_dim)
        hold_type_emb = self.hold_type_embedding(hold_type_indices)  # (B, L, type_emb_dim)
        led_color_emb = self.led_color_embedding(led_color_indices)  # (B, L, color_emb_dim)

        # Grade embedding: expand to sequence length
        grade_emb = self.grade_embedding(grade_indices)  # (B, grade_emb_dim)
        grade_emb = grade_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (B, L, grade_emb_dim)

        # Concatenate all features
        features = torch.cat([hold_emb, positions, hold_type_emb, led_color_emb, grade_emb], dim=-1)

        # LSTM
        lstm_out, hidden = self.lstm(features, hidden)

        # Output predictions
        hold_logits = self.hold_classifier(lstm_out)  # (B, L, num_holds)
        hold_type_logits = self.hold_type_classifier(lstm_out)  # (B, L, num_types)
        led_color_logits = self.led_color_classifier(lstm_out)  # (B, L, num_colors)
        position_pred = self.position_regressor(lstm_out)  # (B, L, 2)

        return {
            "hold_logits": hold_logits,
            "hold_type_logits": hold_type_logits,
            "led_color_logits": led_color_logits,
            "position_pred": position_pred,
            "hidden": hidden,
        }

    @torch.no_grad()
    def generate_step(
        self,
        hold_idx: int,
        position: torch.Tensor,
        hold_type_idx: int,
        led_color_idx: int,
        grade_idx: int,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> dict:
        """Generate a single next hold (autoregressive step).

        Args:
            hold_idx: Current hold ID index
            position: (2,) x, y coordinates
            hold_type_idx: Current hold type index
            led_color_idx: Current LED color index
            grade_idx: Target grade index
            hidden: LSTM hidden state
            temperature: Sampling temperature
            top_k: If > 0, sample from top-k most likely holds only

        Returns:
            Dict with predicted hold, type, color, position, and new hidden state
        """
        # Add batch and sequence dimensions
        hold_indices = torch.tensor([[hold_idx]], dtype=torch.long, device=position.device)
        positions = position.unsqueeze(0).unsqueeze(0)  # (1, 1, 2)
        hold_type_indices = torch.tensor([[hold_type_idx]], dtype=torch.long, device=position.device)
        led_color_indices = torch.tensor([[led_color_idx]], dtype=torch.long, device=position.device)
        grade_indices = torch.tensor([grade_idx], dtype=torch.long, device=position.device)

        output = self.forward(
            hold_indices, positions, hold_type_indices, led_color_indices, grade_indices, hidden
        )

        # Sample next hold
        hold_logits = output["hold_logits"][0, -1] / temperature
        if top_k > 0:
            top_k_vals, top_k_indices = torch.topk(hold_logits, top_k)
            hold_probs = torch.softmax(top_k_vals, dim=-1)
            next_hold_idx = top_k_indices[torch.multinomial(hold_probs, 1)].item()
        else:
            hold_probs = torch.softmax(hold_logits, dim=-1)
            next_hold_idx = torch.multinomial(hold_probs, 1).item()

        # Sample next hold type
        type_logits = output["hold_type_logits"][0, -1] / temperature
        if top_k > 0:
            top_k_vals, top_k_indices = torch.topk(type_logits, min(top_k, len(type_logits)))
            type_probs = torch.softmax(top_k_vals, dim=-1)
            next_type_idx = top_k_indices[torch.multinomial(type_probs, 1)].item()
        else:
            type_probs = torch.softmax(type_logits, dim=-1)
            # Add small epsilon to prevent collapse
            type_probs = type_probs + 1e-6
            type_probs = type_probs / type_probs.sum()
            next_type_idx = torch.multinomial(type_probs, 1).item()

        # Sample next LED color
        color_logits = output["led_color_logits"][0, -1] / temperature
        if top_k > 0:
            top_k_vals, top_k_indices = torch.topk(color_logits, min(top_k, len(color_logits)))
            color_probs = torch.softmax(top_k_vals, dim=-1)
            next_color_idx = top_k_indices[torch.multinomial(color_probs, 1)].item()
        else:
            color_probs = torch.softmax(color_logits, dim=-1)
            next_color_idx = torch.multinomial(color_probs, 1).item()

        # Predict next position
        next_position = output["position_pred"][0, -1]

        return {
            "hold_idx": next_hold_idx,
            "position": next_position,
            "hold_type_idx": next_type_idx,
            "led_color_idx": next_color_idx,
            "hidden": output["hidden"],
        }
