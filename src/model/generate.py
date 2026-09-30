"""Generation and inference: generate new climbs from the trained model.

Usage:
    python -m src.model.generate --grade V5 --count 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from rich.console import Console

from src.model.network import ClimbGenerator
from src.utils.constants import (
    GRADE_TO_INDEX,
    INDEX_TO_GRADE,
    INDEX_TO_HOLD_TYPE,
    INDEX_TO_LED_COLOR,
    REACHABILITY,
    MAX_GENERATION_HOLDS,
    BOARD_WIDTH_CM,
    BOARD_HEIGHT_CM,
)

console = Console()

DEFAULT_MODEL_PATH = Path("models/climb_generator.pt")
DEFAULT_DATA_DIR = Path("data/processed")


def load_model(model_path: Path, data_dir: Path | str, device: torch.device) -> ClimbGenerator:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Load hold ID mapping to get vocab size
    mapping_path = Path(data_dir) / "hold_id_mapping.json"
    with open(mapping_path) as f:
        hold_id_mapping = json.load(f)
    num_holds = len(hold_id_mapping)

    model = ClimbGenerator(
        num_holds=num_holds,
        hold_embedding_dim=config.get("hold_embedding_dim", 64),
        grade_embedding_dim=config.get("grade_embedding_dim", 16),
        hold_type_embedding_dim=config.get("hold_type_embedding_dim", 16),
        led_color_embedding_dim=config.get("led_color_embedding_dim", 16),
        hidden_size=config.get("hidden_size", 256),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.2),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, hold_id_mapping


def generate_climb(
    model: ClimbGenerator,
    grade: str,
    hold_id_mapping: dict[str, int],
    device: torch.device,
    temperature: float = 1.0,
    max_holds: int = MAX_GENERATION_HOLDS,
    top_k: int = 0,
) -> list[dict]:
    """Generate a single climb autoregressively with hard constraints.

    Constraints:
        - Maximum 2 start holds
        - Minimum 1 foot hold for climbs > 5 holds
        - Finish hold in upper third of board
        - Stop at max_holds or when finish hold is generated

    Returns a list of hold dicts with keys:
        hold_id, x, y, hold_type, led_color
    """
    if grade not in GRADE_TO_INDEX:
        raise ValueError(f"Invalid grade: {grade}. Must be V0-V17.")

    grade_idx = GRADE_TO_INDEX[grade]
    idx_to_hold_id = {v: k for k, v in hold_id_mapping.items()}

    # Start with a random start hold in the lower third of the board
    start_holds = [
        idx for idx, hid in idx_to_hold_id.items()
        if idx > 0  # skip padding
    ]
    if not start_holds:
        raise ValueError("No holds available in mapping")

    # Start from a random hold position in the lower third of the board
    # Board coordinates: x ∈ [-56, 204], y ∈ [-12, 291]
    current_hold_idx = start_holds[torch.randint(0, len(start_holds), (1,)).item()]
    current_position = torch.tensor([74.0, 80.0], dtype=torch.float32, device=device)
    current_type_idx = 0  # start
    current_color_idx = 1  # green

    # Use a state machine for hold types (model's type prediction is too biased)
    # Only use the model for hold ID and position prediction
    target_length = torch.randint(5, 12, (1,)).item()  # Random climb length 5-11

    holds = []
    hidden = None

    for step in range(target_length):
        result = model.generate_step(
            hold_idx=current_hold_idx,
            position=current_position,
            hold_type_idx=current_type_idx,
            led_color_idx=current_color_idx,
            grade_idx=grade_idx,
            hidden=hidden,
            temperature=temperature,
            top_k=top_k,
        )

        hidden = result["hidden"]
        current_hold_idx = result["hold_idx"]
        current_position = result["position"]

        # Determine hold type from state machine
        if step < 2:
            hold_type = "start"
            current_type_idx = 0
        elif step == target_length - 1:
            hold_type = "finish"
            current_type_idx = 2
        elif step == target_length // 2:
            hold_type = "foot"
            current_type_idx = 3
        else:
            hold_type = "middle"
            current_type_idx = 1

        current_color_idx = current_type_idx

        hold = {
            "hold_id": idx_to_hold_id.get(current_hold_idx, f"hold_{current_hold_idx}"),
            "x": float(current_position[0].item()),
            "y": float(current_position[1].item()),
            "hold_type": hold_type,
            "led_color": INDEX_TO_LED_COLOR.get(current_color_idx, "red"),
        }
        holds.append(hold)

    # Post-generation: ensure minimum constraints are met
    # If climb has > 5 holds but no foot holds, insert one
    if len(holds) > 5 and foot_count == 0:
        # Insert a foot hold in the middle of the climb
        insert_pos = len(holds) // 2
        holds.insert(insert_pos, {
            "hold_id": "forced_foot",
            "x": (holds[insert_pos - 1]["x"] + holds[insert_pos]["x"]) / 2,
            "y": (holds[insert_pos - 1]["y"] + holds[insert_pos]["y"]) / 2,
            "hold_type": "foot",
            "led_color": "FFA500",
        })

    return holds


def validate_climb(holds: list[dict]) -> tuple[bool, list[str]]:
    """Validate a generated climb against reachability constraints.

    Returns:
        (is_valid, list_of_violations)
    """
    violations = []

    if len(holds) < 3:
        violations.append(f"Too few holds: {len(holds)} (min 3)")

    # Check consecutive hand-to-hand distances
    max_hand_dist = REACHABILITY["max_hand_to_hand_cm"]
    for i in range(len(holds) - 1):
        dx = holds[i + 1]["x"] - holds[i]["x"]
        dy = holds[i + 1]["y"] - holds[i]["y"]
        dist = (dx ** 2 + dy ** 2) ** 0.5
        if dist > max_hand_dist:
            violations.append(
                f"Hand-to-hand distance {dist:.1f}cm exceeds max {max_hand_dist}cm at position {i}"
            )

    # Check foot-to-hand distances
    max_foot_hand = REACHABILITY["max_foot_to_hand_cm"]
    min_foot_hand = REACHABILITY["min_foot_to_hand_cm"]
    foot_holds = [h for h in holds if h["hold_type"] == "foothold"]
    hand_holds = [h for h in holds if h["hold_type"] != "foothold"]

    if len(holds) > 5 and len(foot_holds) < 2:
        violations.append(f"Climb has {len(holds)} holds but only {len(foot_holds)} foot holds")

    for foot in foot_holds:
        min_dist_to_hand = min(
            ((foot["x"] - hand["x"]) ** 2 + (foot["y"] - hand["y"]) ** 2) ** 0.5
            for hand in hand_holds
        ) if hand_holds else float("inf")

        if min_dist_to_hand > max_foot_hand:
            violations.append(
                f"Foot hold at ({foot['x']:.0f}, {foot['y']:.0f}) is {min_dist_to_hand:.1f}cm from nearest hand (max {max_foot_hand}cm)"
            )

    # Check at least one foot hold is close enough for tension
    if foot_holds and hand_holds:
        has_close_foot = any(
            ((foot["x"] - hand["x"]) ** 2 + (foot["y"] - hand["y"]) ** 2) ** 0.5 <= min_foot_hand
            for foot in foot_holds
            for hand in hand_holds
        )
        if not has_close_foot:
            violations.append("No foot hold within tension distance of a hand hold")

    # Check start hold is in lower third
    if holds:
        start_y = holds[0]["y"]
        if start_y > BOARD_HEIGHT_CM * 0.4:
            violations.append(f"Start hold at y={start_y:.0f}cm is not in lower third of board")

    # Check finish hold is in upper third
    if holds:
        finish_y = holds[-1]["y"]
        if finish_y < BOARD_HEIGHT_CM * 0.6:
            violations.append(f"Finish hold at y={finish_y:.0f}cm is not in upper third of board")

    return len(violations) == 0, violations


def generate_multiple(
    model: ClimbGenerator,
    grade: str,
    hold_id_mapping: dict[str, int],
    device: torch.device,
    count: int = 5,
    temperature: float = 1.0,
    max_retries: int = 10,
    top_k: int = 0,
) -> list[list[dict]]:
    """Generate multiple valid climbs, retrying invalid ones."""
    climbs = []
    attempts = 0

    while len(climbs) < count and attempts < count * max_retries:
        attempts += 1
        holds = generate_climb(model, grade, hold_id_mapping, device, temperature, top_k=top_k)
        is_valid, violations = validate_climb(holds)

        if is_valid:
            climbs.append(holds)
        else:
            console.print(f"  [dim]Attempt {attempts} invalid: {violations[0]}[/dim]")

    if len(climbs) < count:
        console.print(f"[yellow]Only generated {len(climbs)}/{count} valid climbs after {attempts} attempts[/yellow]")

    return climbs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate new climbs.")
    parser.add_argument("--grade", type=str, default="V5", help="Target grade (V0-V17)")
    parser.add_argument("--count", type=int, default=5, help="Number of climbs to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=0, help="Sample from top-k most likely holds (0 = disabled)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    device = torch.device("cpu")
    model, hold_id_mapping = load_model(args.model, args.data_dir, device)

    climbs = generate_multiple(
        model, args.grade, hold_id_mapping, device,
        count=args.count, temperature=args.temperature, top_k=args.top_k,
    )

    for i, climb in enumerate(climbs):
        console.print(f"\n[bold]Climb {i + 1} ({args.grade}):[/bold]")
        for j, hold in enumerate(climb):
            console.print(
                f"  {j + 1}. {hold['hold_type']:10s} "
                f"({hold['x']:5.1f}, {hold['y']:5.1f}) "
                f"[{hold['led_color']}]"
            )


if __name__ == "__main__":
    main()
