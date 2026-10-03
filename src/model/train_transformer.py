"""Training script for the transformer climb generator.

Usage:
    python -m src.model.train_transformer
    python -m src.model.train_transformer --epochs 10 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from torch.utils.data import DataLoader

from src.model.dataset import ClimbDataset, collate_fn
from src.model.transformer import ClimbTransformer, TransformerClimbGenerator
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
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    hold_embedding_dim: int = 64
    grade_embedding_dim: int = 16
    hold_type_embedding_dim: int = 16
    led_color_embedding_dim: int = 16
    weight_decay: float = 1e-5
    patience: int = 3
    max_grad_norm: float = 5.0
    num_workers: int = 0
    seed: int = RANDOM_SEED


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_loss(
    output: dict,
    batch: dict,
    hold_type_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute multi-task loss with class weights for hold types."""
    hold_logits = output["hold_logits"]
    hold_type_logits = output["hold_type_logits"]
    led_color_logits = output["led_color_logits"]
    position_pred = output["position_pred"]

    targets_hold = batch["hold_indices"]
    targets_type = batch["hold_type_indices"]
    targets_color = batch["led_color_indices"]
    targets_pos = batch["positions"]
    mask = batch["mask"]

    # Cross-entropy losses
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

    # Position MSE
    loss_pos = ((position_pred - targets_pos) ** 2).sum(dim=-1)
    loss_pos = (loss_pos * mask).sum() / mask.sum().clamp(min=1)

    return loss_hold + loss_type + loss_color + 0.1 * loss_pos


def compute_hold_type_weights(dataloader: DataLoader, num_types: int, device: torch.device) -> torch.Tensor:
    """Compute class weights for hold type prediction based on frequency."""
    counts = torch.zeros(num_types)
    for batch in dataloader:
        for idx in batch["hold_type_indices"]:
            for i in idx:
                counts[i] += 1

    # Inverse frequency weighting
    weights = 1.0 / counts.clamp(min=1)
    # Normalize so mean weight is 1.0
    weights = weights / weights.mean()
    return weights.to(device)


def train_epoch(
    model: ClimbTransformer,
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
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        output = model(
            batch["hold_indices"],
            batch["positions"],
            batch["hold_type_indices"],
            batch["led_color_indices"],
            batch["grade_indices"],
            batch["mask"],
        )
        loss = compute_loss(output, batch, hold_type_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate(
    model: ClimbTransformer,
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
                batch["mask"],
            )
            loss = compute_loss(output, batch, hold_type_weights)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def train(config: TrainConfig) -> Path:
    """Run the full training pipeline."""
    set_seed(config.seed)
    device = get_device()

    # Setup logging
    log_file = config.output_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    console.print(f"[bold]Training Configuration:[/bold]")
    console.print(f"  Device: {device}")
    console.print(f"  Epochs: {config.epochs}")
    console.print(f"  Batch size: {config.batch_size}")
    console.print(f"  Learning rate: {config.learning_rate}")
    console.print(f"  d_model: {config.d_model}")
    console.print(f"  Num layers: {config.num_layers}")

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
    with open(config.data_dir / "hold_id_mapping.json") as f:
        hold_id_mapping = json.load(f)
    num_holds = len(hold_id_mapping)
    console.print(f"  Unique holds: {num_holds}")

    # Create model
    model = ClimbTransformer(
        num_holds=num_holds,
        hold_embedding_dim=config.hold_embedding_dim,
        grade_embedding_dim=config.grade_embedding_dim,
        hold_type_embedding_dim=config.hold_type_embedding_dim,
        led_color_embedding_dim=config.led_color_embedding_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"  Model parameters: {total_params:,}")

    # Compute hold type class weights
    from src.utils.constants import NUM_HOLD_TYPES
    hold_type_weights = compute_hold_type_weights(train_loader, NUM_HOLD_TYPES, device)
    console.print(f"  Hold type weights: {[f'{w:.2f}' for w in hold_type_weights.tolist()]}")

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
    best_model_path = config.model_dir / "climb_transformer.pt"

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
            logger.info(f"Epoch {epoch + 1}/{config.epochs} - train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

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
                    "hold_id_mapping": hold_id_mapping,
                }, best_model_path)
            else:
                patience_counter += 1

            # Save checkpoint every epoch
            checkpoint_path = config.model_dir / f"climb_transformer_epoch_{epoch + 1}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "config": asdict(config),
                "hold_id_mapping": hold_id_mapping,
            }, checkpoint_path)
            console.print(f"  [dim]Checkpoint saved: {checkpoint_path}[/dim]")

            # Early stopping
            if patience_counter >= config.patience:
                console.print(f"\n[yellow]Early stopping at epoch {epoch + 1}[/yellow]")
                break

    console.print(f"\n[green]Training complete![/green]")
    console.print(f"  Best validation loss: {best_val_loss:.4f}")
    console.print(f"  Model saved to: {best_model_path}")

    return best_model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the transformer climb generator.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    args = parser.parse_args()

    config = TrainConfig(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        d_model=args.d_model,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
    )

    train(config)


if __name__ == "__main__":
    main()
