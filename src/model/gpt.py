"""Decoder-only transformer (GPT-style) climb generator.

Fixes applied:
1. One token per (hold, role) pair with one output head
2. Sequence format: [GRADE] start start middle ... finish feet... [END]
3. Decoder-only model with causal mask
4. Input = seq[:-1], target = seq[1:], padding ignored
5. No class weights
6. Rule-based masking during sampling
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.utils.constants import (
    GRADE_TO_INDEX,
    INDEX_TO_GRADE,
    RANDOM_SEED,
)


# ---------- Vocabulary ----------
PAD, END = 0, 1
GRADES = list(range(0, 18))  # V0..V17
GRADE_TOK = {g: 2 + i for i, g in enumerate(GRADES)}


class ClimbGPT(nn.Module):
    """Decoder-only transformer with causal mask."""

    def __init__(self, vocab_size: int, max_len: int, d_model: int = 128, nhead: int = 4, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.position_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal mask.

        Args:
            x: (batch, seq_len) token indices

        Returns:
            (batch, seq_len, vocab_size) logits
        """
        seq_len = x.size(1)

        # Embed tokens and positions
        h = self.token_embedding(x) + self.position_embedding(torch.arange(seq_len, device=x.device))

        # Create causal mask (True = can't look)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)

        # Transformer with causal mask and padding mask
        h = self.transformer(h, mask=causal_mask, src_key_padding_mask=(x == PAD))

        # Output logits
        return self.output_head(self.layer_norm(h))


def encode_sequence(holds: list[dict], grade_idx: int, tok_of: dict, pair_of: dict) -> list[int]:
    """Encode a climb sequence into tokens.

    Format: [GRADE] start start middle ... finish feet... [END]
    """
    # Separate hands and feet, sort by y (then x)
    hands = sorted(
        [h for h in holds if h["hold_type"] != "foot"],
        key=lambda h: (h["y"], h["x"])
    )
    feet = sorted(
        [h for h in holds if h["hold_type"] == "foot"],
        key=lambda h: (h["y"], h["x"])
    )

    # Build sequence
    seq = [GRADE_TOK[grade_idx]]
    for h in hands + feet:
        # Convert hole_id to string to match vocabulary keys
        pair = (str(h["hole_id"]), h["hold_type"])
        if pair in tok_of:
            seq.append(tok_of[pair])
    seq.append(END)

    return seq


def build_vocab(hold_id_mapping: dict[str, int], hold_positions: dict[int, tuple[float, float]]) -> tuple[dict, dict]:
    """Build vocabulary of (hold_id, hold_type) pairs.

    Only includes holds that appear in hold_positions (training data).

    Returns:
        tok_of: dict mapping (hold_id, hold_type) -> token index
        pair_of: dict mapping token index -> (hold_id, hold_type)
    """
    # Get all unique (hold_id, hold_type) pairs from training data
    # Only include holds that are in hold_positions
    pairs = set()
    for hole_id in hold_positions.keys():
        for hold_type in ["start", "middle", "finish", "foot"]:
            pairs.add((str(hole_id), hold_type))

    # Build vocabulary
    tok_of = {}
    pair_of = {}
    for i, pair in enumerate(sorted(pairs)):
        tok = 2 + len(GRADES) + i
        tok_of[pair] = tok
        pair_of[tok] = pair

    return tok_of, pair_of


def train_model(
    model: ClimbGPT,
    train_sequences: list[list[int]],
    val_sequences: list[list[int]],
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Train the model.

    Args:
        model: The model to train
        train_sequences: List of training sequences
        val_sequences: List of validation sequences
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        device: Device to train on
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Pad sequences to same length
    max_len = max(len(s) for s in train_sequences + val_sequences)

    def pad_sequences(sequences: list[list[int]]) -> torch.Tensor:
        padded = torch.full((len(sequences), max_len), PAD, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = torch.tensor(seq)
        return padded

    train_padded = pad_sequences(train_sequences).to(device)
    val_padded = pad_sequences(val_sequences).to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        # Shuffle training data
        indices = torch.randperm(len(train_padded))

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = train_padded[batch_indices]

            # Input = seq[:-1], target = seq[1:]
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1),
                ignore_index=PAD,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for i in range(0, len(val_padded), batch_size):
                batch = val_padded[i:i + batch_size]
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch[:, 1:].reshape(-1),
                    ignore_index=PAD,
                )
                val_loss += loss.item()
                val_batches += 1

        print(f"Epoch {epoch + 1}/{epochs} - train_loss={total_loss / num_batches:.4f} val_loss={val_loss / val_batches:.4f}")


@torch.no_grad()
def generate_climb(
    model: ClimbGPT,
    grade_idx: int,
    tok_of: dict,
    pair_of: dict,
    hold_positions: dict[str, tuple[float, float]],
    temperature: float = 0.9,
    top_k: int = 20,
    max_reach: float = 140.0,
    min_hands: int = 5,
    max_len: int = 30,
    device: torch.device = torch.device("cpu"),
) -> list[dict]:
    """Generate a climb using rule-based masking.

    Args:
        model: The trained model
        grade_idx: Target grade index
        tok_of: dict mapping (hold_id, hold_type) -> token index
        pair_of: dict mapping token index -> (hold_id, hold_type)
        hold_positions: dict mapping hold_id -> (x, y)
        temperature: Sampling temperature
        top_k: Top-k sampling
        max_reach: Maximum reach distance (cm)
        min_hands: Minimum number of hand holds before finish
        max_len: Maximum sequence length
        device: Device to run on

    Returns:
        List of hold dicts with keys: hold_id, x, y, hold_type
    """
    model.eval()

    # Compute height thresholds
    all_y = [pos[1] for pos in hold_positions.values()]
    top_y = np.percentile(all_y, 85)
    bottom_y = np.percentile(all_y, 35)

    seq = [GRADE_TOK[grade_idx]]
    used = set()
    hands = []
    in_feet = False

    for _ in range(max_len - 1):
        # Get logits from model
        logits = model(torch.tensor([seq], device=device))[0, -1] / temperature

        # Apply rule-based masking
        allowed = torch.zeros(len(tok_of) + 2 + len(GRADES), dtype=torch.bool, device=device)

        n_start = sum(1 for h in hands if h["hold_type"] == "start")
        last = hands[-1] if hands else None
        done_hands = any(h["hold_type"] == "finish" for h in hands)

        for tok, (hold_id, hold_type) in pair_of.items():
            # Convert string hold_id to int for lookup
            hold_id_int = int(hold_id)
            if hold_id_int in used:
                continue

            x, y = hold_positions[hold_id_int]

            if done_hands:
                # After finish: feet only
                ok = hold_type == "foot"
            elif len(hands) < 1:
                # First hold must be a low start
                ok = hold_type == "start" and y <= bottom_y
            elif hold_type == "start":
                ok = n_start < 2 and y <= bottom_y
            elif hold_type in ("middle", "finish"):
                if last is None:
                    continue
                lx, ly = last["x"], last["y"]
                ok = y >= ly and math.hypot(x - lx, y - ly) <= max_reach
                if hold_type == "finish":
                    ok = ok and y >= top_y and len(hands) >= min_hands - 1
            else:
                # Foot before finish: not allowed in this ordering
                ok = False

            allowed[tok] = bool(ok)

        # END token: only allowed after finish
        allowed[END] = done_hands

        if not allowed.any():
            break  # Painted into a corner

        # Apply top-k sampling
        logits[~allowed] = -float("inf")
        if top_k:
            top_k_vals = torch.topk(logits, min(top_k, int(allowed.sum())))
            logits[logits < top_k_vals.values[-1]] = -float("inf")

        # Sample
        probs = F.softmax(logits, dim=-1)
        tok = torch.multinomial(probs, 1).item()

        if tok == END:
            break

        hold_id, hold_type = pair_of[tok]
        used.add(hold_id)
        seq.append(tok)

        hold = {
            "hold_id": hold_id,
            "x": hold_positions[int(hold_id)][0],
            "y": hold_positions[int(hold_id)][1],
            "hold_type": hold_type,
        }

        if hold_type != "foot":
            hands.append(hold)

    # Convert to output format
    result = []
    for tok in seq[1:]:
        if tok == END:
            break
        hold_id, hold_type = pair_of[tok]
        result.append({
            "hold_id": hold_id,
            "x": hold_positions[int(hold_id)][0],
            "y": hold_positions[int(hold_id)][1],
            "hold_type": hold_type,
        })

    return result


def generate_valid_climb(
    model: ClimbGPT,
    grade_idx: int,
    tok_of: dict,
    pair_of: dict,
    hold_positions: dict[str, tuple[float, float]],
    tries: int = 50,
    **kwargs,
) -> list[dict] | None:
    """Generate a climb that has a finish hold.

    Keeps sampling until we get a climb with a finish hold.
    """
    for _ in range(tries):
        climb = generate_climb(model, grade_idx, tok_of, pair_of, hold_positions, **kwargs)
        if any(h["hold_type"] == "finish" for h in climb):
            return climb
    return None
