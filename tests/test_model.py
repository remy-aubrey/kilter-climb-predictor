"""Tests for src/model/ modules."""

import json
from pathlib import Path

import torch

from src.model.dataset import ClimbDataset, collate_fn
from src.model.network import ClimbGenerator


def _create_test_data(path: Path, num_samples: int = 5) -> None:
    """Create minimal test data JSON."""
    samples = []
    for i in range(num_samples):
        seq_len = 3 + i % 3
        sequence = []
        for j in range(seq_len):
            sequence.append({
                "hold_id": f"h{j}",
                "hold_id_idx": j,
                "x": float(j * 10),
                "y": float(j * 20),
                "hold_type": "jug",
                "hold_type_idx": 0,
                "led_color": "red",
                "led_color_idx": 0,
            })
        samples.append({
            "climb_id": i,
            "grade": "V3",
            "grade_idx": 3,
            "sequence": sequence,
            "length": seq_len,
        })

    with open(path, "w") as f:
        json.dump(samples, f)


def test_climb_dataset(tmp_path: Path):
    data_path = tmp_path / "test_data.json"
    _create_test_data(data_path)

    dataset = ClimbDataset(data_path)

    assert len(dataset) == 5

    sample = dataset[0]
    assert "hold_indices" in sample
    assert "positions" in sample
    assert "grade_idx" in sample
    assert sample["hold_indices"].dtype == torch.long
    assert sample["positions"].dtype == torch.float32


def test_collate_fn(tmp_path: Path):
    data_path = tmp_path / "test_data.json"
    _create_test_data(data_path)

    dataset = ClimbDataset(data_path)
    batch = [dataset[i] for i in range(3)]
    collated = collate_fn(batch)

    assert collated["hold_indices"].shape[0] == 3  # batch size
    assert collated["hold_indices"].shape[1] == 5  # max length in batch
    assert collated["mask"].shape == (3, 5)
    assert collated["grade_indices"].shape == (3,)


def test_climb_generator_forward():
    num_holds = 100
    model = ClimbGenerator(
        num_holds=num_holds,
        hold_embedding_dim=32,
        grade_embedding_dim=8,
        hold_type_embedding_dim=8,
        led_color_embedding_dim=8,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
    )

    batch_size = 4
    seq_len = 5

    hold_indices = torch.randint(0, num_holds, (batch_size, seq_len))
    positions = torch.rand(batch_size, seq_len, 2)
    hold_type_indices = torch.randint(0, 4, (batch_size, seq_len))  # 4 hold types: start, middle, finish, foot
    led_color_indices = torch.randint(0, 4, (batch_size, seq_len))  # 4 LED colors
    grade_indices = torch.randint(0, 18, (batch_size,))

    output = model(hold_indices, positions, hold_type_indices, led_color_indices, grade_indices)

    assert output["hold_logits"].shape == (batch_size, seq_len, num_holds)
    assert output["hold_type_logits"].shape == (batch_size, seq_len, 4)
    assert output["led_color_logits"].shape == (batch_size, seq_len, 4)
    assert output["position_pred"].shape == (batch_size, seq_len, 2)
    assert output["hidden"] is not None


def test_climb_generator_generate_step():
    num_holds = 100
    model = ClimbGenerator(
        num_holds=num_holds,
        hold_embedding_dim=32,
        grade_embedding_dim=8,
        hold_type_embedding_dim=8,
        led_color_embedding_dim=8,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
    )
    model.eval()

    position = torch.tensor([0.5, 0.5])
    result = model.generate_step(
        hold_idx=0,
        position=position,
        hold_type_idx=0,
        led_color_idx=0,
        grade_idx=3,
        temperature=1.0,
    )

    assert "hold_idx" in result
    assert "position" in result
    assert "hold_type_idx" in result
    assert "led_color_idx" in result
    assert "hidden" in result
    assert 0 <= result["hold_idx"] < num_holds
