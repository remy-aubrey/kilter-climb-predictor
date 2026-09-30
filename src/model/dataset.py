"""PyTorch Dataset for climb sequences."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.utils.constants import (
    GRADE_TO_INDEX,
    HOLD_TYPE_TO_INDEX,
    LED_COLOR_TO_INDEX,
    MAX_HOLDS_PER_CLIMB,
)


class ClimbDataset(Dataset):
    """PyTorch Dataset for climb sequences.

    Each sample is a sequence of holds with features:
        - hold_id_idx (int)
        - x, y (float, normalized 0-1)
        - hold_type_idx (int)
        - led_color_idx (int)

    Plus a grade index for conditioning.
    """

    def __init__(self, data_path: Path | str, max_length: int = MAX_HOLDS_PER_CLIMB):
        """Load preprocessed data from JSON file.

        Args:
            data_path: Path to a JSON file (train.json, val.json, or test.json).
            max_length: Maximum sequence length (pad/truncate to this).
        """
        self.max_length = max_length

        with open(data_path) as f:
            records = json.load(f)

        self.samples = []
        for record in records:
            sequence = record["sequence"]
            grade_idx = record["grade_idx"]

            # Truncate if needed
            if len(sequence) > max_length:
                sequence = sequence[:max_length]

            # Build feature tensors
            hold_indices = []
            positions = []
            hold_type_indices = []
            led_color_indices = []

            for hold in sequence:
                hold_indices.append(hold["hold_id_idx"])
                positions.append([hold["x"], hold["y"]])
                hold_type_indices.append(hold["hold_type_idx"])
                led_color_indices.append(hold["led_color_idx"])

            self.samples.append({
                "hold_indices": torch.tensor(hold_indices, dtype=torch.long),
                "positions": torch.tensor(positions, dtype=torch.float32),
                "hold_type_indices": torch.tensor(hold_type_indices, dtype=torch.long),
                "led_color_indices": torch.tensor(led_color_indices, dtype=torch.long),
                "grade_idx": torch.tensor(grade_idx, dtype=torch.long),
                "length": len(sequence),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict:
    """Collate function for padding batches.

    Pads sequences to the max length in the batch and returns masks.
    """
    batch_size = len(batch)
    max_len = max(sample["length"] for sample in batch)

    # Get feature dimensions
    pos_dim = batch[0]["positions"].shape[1]  # should be 2 (x, y)

    # Initialize padded tensors
    hold_indices = torch.zeros(batch_size, max_len, dtype=torch.long)
    positions = torch.zeros(batch_size, max_len, pos_dim, dtype=torch.float32)
    hold_type_indices = torch.zeros(batch_size, max_len, dtype=torch.long)
    led_color_indices = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    grade_indices = torch.tensor([s["grade_idx"] for s in batch], dtype=torch.long)
    lengths = torch.tensor([s["length"] for s in batch], dtype=torch.long)

    for i, sample in enumerate(batch):
        length = sample["length"]
        hold_indices[i, :length] = sample["hold_indices"]
        positions[i, :length] = sample["positions"]
        hold_type_indices[i, :length] = sample["hold_type_indices"]
        led_color_indices[i, :length] = sample["led_color_indices"]
        mask[i, :length] = True

    return {
        "hold_indices": hold_indices,
        "positions": positions,
        "hold_type_indices": hold_type_indices,
        "led_color_indices": led_color_indices,
        "mask": mask,
        "grade_indices": grade_indices,
        "lengths": lengths,
    }
