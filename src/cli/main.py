"""CLI interface for the Kilter Climb Predictor.

Usage:
    kilter-gen generate --grade V5 --count 3
    kilter-gen train --epochs 50
    kilter-gen download
    kilter-gen stats
    kilter-gen visualize --grade V5
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from src.utils.constants import GRADE_TO_INDEX

app = typer.Typer(
    name="kilter-gen",
    help="ML-powered climbing route generator for the Kilter Board.",
    no_args_is_help=True,
)
console = Console()


def validate_grade(grade: str) -> str:
    """Validate and normalize a grade string."""
    grade = grade.upper()
    if grade not in GRADE_TO_INDEX:
        valid = ", ".join(sorted(GRADE_TO_INDEX.keys(), key=lambda g: GRADE_TO_INDEX[g]))
        raise typer.BadParameter(f"Invalid grade '{grade}'. Valid grades: {valid}")
    return grade


@app.command()
def generate(
    grade: str = typer.Option(..., "--grade", "-g", help="Target grade (V0-V17)"),
    count: int = typer.Option(3, "--count", "-n", help="Number of climbs to generate"),
    temperature: float = typer.Option(1.0, "--temperature", "-t", help="Sampling temperature"),
    data_dir: Path = typer.Option(Path("data/processed"), "--data-dir", help="Preprocessed data directory"),
):
    """Generate new climbs at the specified grade."""
    grade = validate_grade(grade)

    if count < 1:
        raise typer.BadParameter("Count must be a positive integer")

    import torch
    from src.model.gpt import ClimbGPT, generate_valid_climb
    from src.cli.visualize import visualize_climb

    model_path = Path("models/climb_gpt.pt")
    if not model_path.exists():
        console.print(f"[red]Model not found: {model_path}[/red]")
        console.print("Train a model first: kilter-gen train")
        raise typer.Exit(1)

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    tok_of = checkpoint["tok_of"]
    pair_of = {int(k): tuple(v) for k, v in checkpoint["pair_of"].items()}
    hold_positions = checkpoint["hold_positions"]

    # Create model
    vocab_size = len(tok_of) + 2 + 18  # +2 for PAD/END, +18 for grades
    max_len = 32  # Must match training max_len
    model = ClimbGPT(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=128,
        nhead=4,
        num_layers=4,
    )

    # Load trained weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Generate climbs
    from src.utils.constants import GRADE_TO_INDEX
    grade_idx = GRADE_TO_INDEX[grade]

    for i in range(count):
        climb = generate_valid_climb(
            model,
            grade_idx,
            tok_of,
            pair_of,
            hold_positions,
            temperature=temperature,
        )
        if climb:
            visualize_climb(climb, grade)
        else:
            console.print(f"[yellow]Failed to generate climb {i + 1}[/yellow]")


@app.command()
def train(
    data_dir: Path = typer.Option(Path("data/processed"), "--data-dir", help="Preprocessed data directory"),
    model_dir: Path = typer.Option(Path("models"), "--model-dir", help="Where to save model checkpoints"),
    output_dir: Path = typer.Option(Path("outputs"), "--output-dir", help="Where to save logs"),
    epochs: int = typer.Option(50, "--epochs", help="Number of training epochs"),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size"),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate"),
    hidden_size: int = typer.Option(256, "--hidden-size", help="LSTM hidden size"),
    dropout: float = typer.Option(0.2, "--dropout", help="Dropout rate"),
    patience: int = typer.Option(5, "--patience", help="Early stopping patience"),
):
    """Train the climb generator model."""
    from src.model.train import TrainConfig, train

    config = TrainConfig(
        data_dir=data_dir,
        model_dir=model_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        hidden_size=hidden_size,
        dropout=dropout,
        patience=patience,
    )

    train(config)


@app.command()
def download(
    output: Path = typer.Option(Path("data/kilter.db"), "--output", "-o", help="Output path"),
    username: str = typer.Option(None, "--username", "-u", help="Kilter Board app username"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing database"),
):
    """Download the Kilter Board database."""
    from src.data.download import download_database

    download_database(output_path=output, username=username, force=force)


@app.command()
def stats(
    data_dir: Path = typer.Option(Path("data/processed"), "--data-dir", help="Preprocessed data directory"),
):
    """Show dataset statistics."""
    stats_path = data_dir / "stats.json"

    if not stats_path.exists():
        console.print(f"[red]Stats file not found: {stats_path}[/red]")
        console.print("Run preprocessing first: python -m src.data.preprocess")
        raise typer.Exit(1)

    with open(stats_path) as f:
        data = json.load(f)

    console.print(f"\n[bold cyan]Dataset Statistics[/bold cyan]")
    console.print(f"  Total climbs: {data['total_climbs']:,}")
    console.print(f"  Unique holds: {data['total_unique_holds']:,}")
    console.print(f"  Avg sequence length: {data['avg_sequence_length']:.1f}")
    console.print(f"  Min/Max length: {data['min_sequence_length']}/{data['max_sequence_length']}")

    if "split_sizes" in data:
        console.print(f"\n[bold]Split Sizes:[/bold]")
        for split, size in data["split_sizes"].items():
            console.print(f"  {split}: {size:,}")

    console.print(f"\n[bold]Grade Distribution:[/bold]")
    for grade, count in sorted(data["grade_distribution"].items(), key=lambda x: GRADE_TO_INDEX.get(x[0], 99)):
        console.print(f"  {grade}: {count:,}")


@app.command()
def visualize(
    grade: str = typer.Option("V5", "--grade", "-g", help="Grade to visualize"),
    model: Path = typer.Option(Path("models/climb_generator.pt"), "--model", help="Model checkpoint path"),
    data_dir: Path = typer.Option(Path("data/processed"), "--data-dir", help="Preprocessed data directory"),
):
    """Generate and visualize a climb as ASCII art."""
    from src.model.generate import load_model, generate_climb
    from src.cli.visualize import visualize_climb
    import torch

    grade = validate_grade(grade)
    device = torch.device("cpu")

    if not model.exists():
        console.print(f"[red]Model not found: {model}[/red]")
        console.print("Train a model first: kilter-gen train")
        raise typer.Exit(1)

    model_obj, hold_id_mapping = load_model(model, data_dir, device)
    climb = generate_climb(model_obj, grade, hold_id_mapping, device)

    visualize_climb(climb, grade)


if __name__ == "__main__":
    app()
