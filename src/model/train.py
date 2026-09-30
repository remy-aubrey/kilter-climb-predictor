"""Training pipeline for the climb generator model.

Usage:
    python -m src.model.train
    python -m src.model.train --epochs 50 --batch-size 32 --lr 1e-3
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from torch.utils.data import DataLoader

from src.model.dataset import ClimbDataset, collate_fn
from src.model.network import ClimbGenerator
from src.utils.constants import RANDOM_SEED

console = Console()

DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_OUTPUT_DIR = Path("outputs")


@dataclass
class TrainConfig:
    """Configuration for training."""
    data_dir: Path = DEFAULT_DATA_DIR
    model_dir: Path = DEFAULT_MODEL_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    hold_embedding_dim: int = 64
    grade_embedding_dim: int = 16
    hold_type_embedding_dim: int = 16
    led_color_embedding_dim: int = 16
    weight_decay: float = 1e-5
    patience: int = 5
    max_grad_norm: float = 5.0
    num_workers: int = 0
    seed: int = RANDOM_SEED


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_loss(
    output: dict,
    batch: dict,
    hold_type_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute multi-task loss.

    Combines:
        - Cross-entropy for hold ID prediction
        - Cross-entropy for hold type prediction (with class weights)
        - Cross-entropy for LED color prediction
        - MSE for position regression
    """
    hold_logits = output["hold_logits"]  # (B, L, num_holds)
    hold_type_logits = output["hold_type_logits"]  # (B, L, num_types)
    led_color_logits = output["led_color_logits"]  # (B, L, num_colors)
    position_pred = output["position_pred"]  # (B, L, 2)

    targets_hold = batch["hold_indices"]  # (B, L)
    targets_type = batch["hold_type_indices"]  # (B, L)
    targets_color = batch["led_color_indices"]  # (B, L)
    targets_pos = batch["positions"]  # (B, L, 2)
    mask = batch["mask"]  # (B, L)

    # Cross-entropy losses (only on valid positions)
    loss_hold = nn.functional.cross_entropy(
        hold_logits.view(-1, hold_logits.size(-1)),
        targets_hold.view(-1),
        reduction="none",
    )
    loss_type = nn.functional.cross_entropy(
        hold_type_logits.view(-1, hold_type_logits.size(-1)),
        targets_type.view(-1),
        weight=hold_type_weights,
        reduction="none",
    )
    loss_color = nn.functional.cross_entropy(
        led_color_logits.view(-1, led_color_logits.size(-1)),
        targets_color.view(-1),
        reduction="none",
    )

    # Apply mask
    mask_flat = mask.view(-1).float()
    loss_hold = (loss_hold * mask_flat).sum() / mask_flat.sum().clamp(min=1)
    loss_type = (loss_type * mask_flat).sum() / mask_flat.sum().clamp(min=1)
    loss_color = (loss_color * mask_flat).sum() / mask_flat.sum().clamp(min=1)

    # Position MSE (only on valid positions)
    loss_pos = ((position_pred - targets_pos) ** 2).sum(dim=-1)  # (B, L)
    loss_pos = (loss_pos * mask).sum() / mask.sum().clamp(min=1)

    # Combined loss
    total_loss = loss_hold + loss_type + loss_color + 0.1 * loss_pos

    return total_loss


def train_epoch(
    model: ClimbGenerator,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float,
    hold_type_weights: torch.Tensor | None = None,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        output = model(
            batch["hold_indices"],
            batch["positions"],
            batch["hold_type_indices"],
            batch["led_color_indices"],
            batch["grade_indices"],
        )
        loss = compute_loss(output, batch, hold_type_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate(
    model: ClimbGenerator,
    dataloader: DataLoader,
    device: torch.device,
    hold_type_weights: torch.Tensor | None = None,
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(
                batch["hold_indices"],
                batch["positions"],
                batch["hold_type_indices"],
                batch["led_color_indices"],
                batch["grade_indices"],
            )
            loss = compute_loss(output, batch, hold_type_weights)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def load_hold_id_mapping(data_dir: Path) -> dict[str, int]:
    """Load the hold ID mapping to determine vocabulary size."""
    mapping_path = data_dir / "hold_id_mapping.json"
    with open(mapping_path) as f:
        mapping = json.load(f)
    return mapping


def compute_hold_type_weights(dataset: ClimbDataset, num_types: int, device: torch.device) -> torch.Tensor:
    """Compute class weights for hold type prediction based on frequency.

    Weights are inversely proportional to class frequency.
    """
    counts = torch.zeros(num_types)
    for sample in dataset.samples:
        for idx in sample["hold_type_indices"]:
            counts[idx] += 1

    # Inverse frequency weighting
    weights = 1.0 / counts.clamp(min=1)
    # Normalize so mean weight is 1.0
    weights = weights / weights.mean()
    return weights.to(device)


def train(config: TrainConfig) -> Path:
    """Run the full training pipeline."""
    set_seed(config.seed)
    device = get_device()

    console.print(f"[bold]Training Configuration:[/bold]")
    console.print(f"  Device: {device}")
    console.print(f"  Epochs: {config.epochs}")
    console.print(f"  Batch size: {config.batch_size}")
    console.print(f"  Learning rate: {config.learning_rate}")
    console.print(f"  Hidden size: {config.hidden_size}")
    console.print(f"  Num layers: {config.num_layers}")
    console.print(f"  Dropout: {config.dropout}")

    # Load data
    console.print(f"\n[bold]Loading data from {config.data_dir}...[/bold]")
    train_dataset = ClimbDataset(config.data_dir / "train.json")
    val_dataset = ClimbDataset(config.data_dir / "val.json")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )

    console.print(f"  Train samples: {len(train_dataset)}")
    console.print(f"  Val samples: {len(val_dataset)}")

    # Load hold ID mapping
    hold_id_mapping = load_hold_id_mapping(config.data_dir)
    num_holds = len(hold_id_mapping)
    console.print(f"  Unique holds: {num_holds}")

    # Compute hold type class weights
    from src.utils.constants import NUM_HOLD_TYPES
    hold_type_weights = compute_hold_type_weights(train_dataset, NUM_HOLD_TYPES, device)
    console.print(f"  Hold type weights: {[f'{w:.2f}' for w in hold_type_weights.tolist()]}")

    # Create model
    model = ClimbGenerator(
        num_holds=num_holds,
        hold_embedding_dim=config.hold_embedding_dim,
        grade_embedding_dim=config.grade_embedding_dim,
        hold_type_embedding_dim=config.hold_type_embedding_dim,
        led_color_embedding_dim=config.led_color_embedding_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"  Model parameters: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Training loop
    config.model_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = config.model_dir / "climb_generator.pt"

    console.print(f"\n[bold]Starting training...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.fields[loss]}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Training", total=config.epochs, loss="—")

        for epoch in range(config.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device, config.max_grad_norm, hold_type_weights)
            val_loss = validate(model, val_loader, device, hold_type_weights)

            progress.update(task, advance=1, loss=f"train={train_loss:.4f} val={val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": asdict(config),
                }, best_model_path)
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= config.patience:
                console.print(f"\n[yellow]Early stopping at epoch {epoch + 1}[/yellow]")
                break

    console.print(f"\n[green]Training complete![/green]")
    console.print(f"  Best validation loss: {best_val_loss:.4f}")
    console.print(f"  Model saved to: {best_model_path}")

    # Log experiment
    experiment_log = config.output_dir / "experiments.csv"
    experiment_log.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(asdict(config).keys()) + ["best_val_loss", "total_params"]
    row = asdict(config)
    row["best_val_loss"] = best_val_loss
    row["total_params"] = total_params

    file_exists = experiment_log.exists()
    with open(experiment_log, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    console.print(f"  Experiment logged to: {experiment_log}")

    return best_model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the climb generator model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    config = TrainConfig(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        patience=args.patience,
    )

    train(config)


if __name__ == "__main__":
    main()
