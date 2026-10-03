"""Training script for the GPT-style climb generator.

Usage:
    python -m src.model.train_gpt
    python -m src.model.train_gpt --epochs 20 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from rich.console import Console

from src.model.gpt import ClimbGPT, build_vocab, encode_sequence, train_model
from src.utils.constants import RANDOM_SEED

console = Console()

DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_MODEL_DIR = Path("models")


def load_data(data_dir: Path) -> tuple[list[list[int]], list[list[int]], dict, dict, dict]:
    """Load and encode training data.

    Returns:
        train_sequences: List of encoded training sequences
        val_sequences: List of encoded validation sequences
        tok_of: dict mapping (hold_id, hold_type) -> token index
        pair_of: dict mapping token index -> (hold_id, hold_type)
        hold_positions: dict mapping hold_id -> (x, y)
    """
    # Load training data
    with open(data_dir / "train.json") as f:
        train_data = json.load(f)

    with open(data_dir / "val.json") as f:
        val_data = json.load(f)

    with open(data_dir / "hold_id_mapping.json") as f:
        hold_id_mapping = json.load(f)

    # Build hold positions mapping from training data
    hold_positions = {}
    for climb in train_data:
        for hold in climb["sequence"]:
            hold_positions[int(hold["hole_id"])] = (hold["x"], hold["y"])

    # Build vocabulary (only holds that appear in training data)
    tok_of, pair_of = build_vocab(hold_id_mapping, hold_positions)

    # Build hold positions mapping from hold_id_mapping
    # hold_id_mapping maps hole_id (int) -> hold_id_idx (int)
    # We need to get the actual hole positions from the database
    # For now, use the training data to build positions
    hold_positions = {}
    for climb in train_data:
        for hold in climb["sequence"]:
            hold_positions[int(hold["hole_id"])] = (hold["x"], hold["y"])

    # Encode sequences
    train_sequences = []
    for climb in train_data:
        seq = encode_sequence(climb["sequence"], climb["grade_idx"], tok_of, pair_of)
        train_sequences.append(seq)

    val_sequences = []
    for climb in val_data:
        seq = encode_sequence(climb["sequence"], climb["grade_idx"], tok_of, pair_of)
        val_sequences.append(seq)

    return train_sequences, val_sequences, tok_of, pair_of, hold_positions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GPT-style climb generator.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    args = parser.parse_args()

    # Set seed
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Load data
    console.print(f"[bold]Loading data from {args.data_dir}...[/bold]")
    train_sequences, val_sequences, tok_of, pair_of, hold_positions = load_data(args.data_dir)

    console.print(f"  Train sequences: {len(train_sequences)}")
    console.print(f"  Val sequences: {len(val_sequences)}")
    console.print(f"  Vocabulary size: {len(tok_of) + 2 + 18}")  # +2 for PAD/END, +18 for grades

    # Create model
    vocab_size = len(tok_of) + 2 + 18  # +2 for PAD/END, +18 for grades
    max_len = max(len(s) for s in train_sequences + val_sequences)

    model = ClimbGPT(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    )

    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"  Model parameters: {total_params:,}")

    # Train
    console.print(f"\n[bold]Starting training...[/bold]")
    train_model(
        model,
        train_sequences,
        val_sequences,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Save model
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "climb_gpt.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "tok_of": tok_of,
        "pair_of": {str(k): v for k, v in pair_of.items()},
        "hold_positions": hold_positions,
    }, model_path)

    console.print(f"\n[green]Training complete![/green]")
    console.print(f"  Model saved to: {model_path}")


if __name__ == "__main__":
    main()
