"""Template-based climb generator.

Picks a random real climb from the training data and modifies it slightly.
This gives realistic climbs with proper positions without the complexity
of a learned model.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.utils.constants import (
    GRADE_TO_INDEX,
    INDEX_TO_GRADE,
    INDEX_TO_HOLD_TYPE,
    INDEX_TO_LED_COLOR,
    RANDOM_SEED,
)


class MarkovClimbGenerator:
    """Template-based generator for climbing routes.

    Picks a random real climb from the training data and modifies it slightly.
    This is much simpler than an LSTM and produces realistic results.
    """

    def __init__(self):
        self.climbs_by_grade: dict[int, list[list[dict]]] = defaultdict(list)  # grade_idx -> [sequence, ...]
        self.hold_id_mapping: dict[str, int] = {}
        self.idx_to_hold_id: dict[int, str] = {}
        self.transitions_with_pos: dict[int, list[tuple[int, float, float]]] = defaultdict(list)  # current_idx -> [(next_idx, x, y), ...]

    def train(self, data_dir: Path) -> None:
        """Load training data and index by grade."""
        with open(data_dir / "train.json") as f:
            train_data = json.load(f)

        with open(data_dir / "hold_id_mapping.json") as f:
            self.hold_id_mapping = json.load(f)

        self.idx_to_hold_id = {v: k for k, v in self.hold_id_mapping.items()}

        for climb in train_data:
            grade_idx = climb["grade_idx"]
            self.climbs_by_grade[grade_idx].append(climb["sequence"])

        print(f"Trained on {len(train_data)} climbs")
        print(f"Unique holds: {len(self.hold_id_mapping)}")
        for grade_idx in sorted(self.climbs_by_grade.keys()):
            print(f"  {INDEX_TO_GRADE.get(grade_idx, f'V{grade_idx}')}: {len(self.climbs_by_grade[grade_idx])} climbs")

    def generate(
        self,
        grade: str,
        length: int | None = None,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> list[dict]:
        """Generate a climb by picking a random template and modifying it.

        Args:
            grade: Target grade (e.g., "V5")
            length: Number of holds (random if None)
            temperature: Sampling temperature
            seed: Random seed for reproducibility

        Returns:
            List of hold dicts with keys: hold_id, x, y, hold_type, led_color
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        grade_idx = GRADE_TO_INDEX[grade]

        # Pick a random climb from the training data at this grade
        climbs = self.climbs_by_grade.get(grade_idx, [])
        if not climbs:
            return []

        template = random.choice(climbs)

        # Determine climb length
        if length is None:
            length = random.randint(5, min(12, len(template)))

        # Start with a random start hold from the template
        start_holds = [h for h in template if h["hold_type"] == "start"]
        if not start_holds:
            start_holds = template[:2]

        current_hold = random.choice(start_holds)
        current_idx = current_hold["hold_id_idx"]
        current_x, current_y = current_hold["x"], current_hold["y"]

        holds = []
        for step in range(length):
            # Determine hold type
            if step < 2:
                hold_type = "start"
            elif step == length - 1:
                hold_type = "finish"
            elif step == length // 2:
                hold_type = "foot"
            else:
                hold_type = "middle"

            holds.append({
                "hold_id": self.idx_to_hold_id.get(current_idx, f"hold_{current_idx}"),
                "x": current_x,
                "y": current_y,
                "hold_type": hold_type,
                "led_color": INDEX_TO_LED_COLOR.get(0, "red"),
            })

            # Sample next hold with position awareness
            if step < length - 1:
                result = self._sample_next_position_aware(
                    current_idx, current_x, current_y, grade_idx, temperature
                )
                if result is None:
                    break
                next_idx, next_x, next_y = result
                current_idx = next_idx
                current_x, current_y = next_x, next_y

        return holds

    def _sample_next_position_aware(
        self,
        current_idx: int,
        current_x: float,
        current_y: float,
        grade_idx: int,
        temperature: float,
        max_distance: float = 140.0,
    ) -> tuple[int, float, float] | None:
        """Sample the next hold given the current hold position.

        Looks through all climbs at this grade to find instances where
        the current hold is followed by another hold, then filters to
        only include next holds within max_distance of the current position.

        Returns:
            (next_idx, next_x, next_y) or None if no valid transition
        """
        # Collect all (next_idx, next_x, next_y) from climbs at this grade
        # where current_idx appears and is followed by another hold
        candidates = []
        for sequence in self.climbs_by_grade.get(grade_idx, []):
            for i, hold in enumerate(sequence):
                if hold["hold_id_idx"] == current_idx and i < len(sequence) - 1:
                    next_hold = sequence[i + 1]
                    candidates.append((
                        next_hold["hold_id_idx"],
                        next_hold["x"],
                        next_hold["y"],
                    ))

        if not candidates:
            return None

        # Filter by distance
        valid_candidates = []
        for next_idx, next_x, next_y in candidates:
            dist = ((next_x - current_x) ** 2 + (next_y - current_y) ** 2) ** 0.5
            if dist <= max_distance:
                valid_candidates.append((next_idx, next_x, next_y, dist))

        if not valid_candidates:
            return None

        # Weight by inverse distance (closer holds more likely)
        weights = [1.0 / (dist + 1.0) for _, _, _, dist in valid_candidates]

        # Apply temperature
        if temperature != 1.0:
            weights = [w ** (1.0 / temperature) for w in weights]

        # Normalize
        total = sum(weights)
        weights = [w / total for w in weights]

        # Sample
        r = random.random()
        cumulative = 0.0
        for (next_idx, next_x, next_y, _), weight in zip(valid_candidates, weights):
            cumulative += weight
            if r <= cumulative:
                return next_idx, next_x, next_y

        # Fallback to last candidate
        next_idx, next_x, next_y, _ = valid_candidates[-1]
        return next_idx, next_x, next_y

    def _sample_next(
        self,
        current_idx: int,
        grade_idx: int,
        temperature: float,
    ) -> int | None:
        """Sample the next hold given the current hold and grade."""
        # Try grade-conditioned transitions first
        grade_transitions = self.grade_conditioned.get(grade_idx, {}).get(current_idx, {})

        if not grade_transitions:
            # Fall back to global transitions
            grade_transitions = self.transitions.get(current_idx, {})

        if not grade_transitions:
            return None

        # Apply temperature
        if temperature != 1.0:
            # Convert counts to probabilities with temperature
            items = list(grade_transitions.items())
            counts = np.array([count for _, idx in items], dtype=float)
            probs = counts / temperature
            probs = np.exp(probs - np.max(probs))  # Numerical stability
            probs = probs / probs.sum()
            next_idx = np.random.choice([idx for _, idx in items], p=probs)
        else:
            # Sample directly from counts
            items = list(grade_transitions.items())
            total = sum(count for _, count in items)
            r = random.randint(1, total)
            cumulative = 0
            for idx, count in items:
                cumulative += count
                if r <= cumulative:
                    return idx
            return items[-1][0]

        return next_idx

    def save(self, path: Path) -> None:
        """Save the trained model to disk."""
        data = {
            "climbs_by_grade": {
                str(grade_idx): [
                    [
                        {
                            "hold_id_idx": h["hold_id_idx"],
                            "x": h["x"],
                            "y": h["y"],
                            "hold_type": h["hold_type"],
                            "led_color_idx": h["led_color_idx"],
                        }
                        for h in sequence
                    ]
                    for sequence in sequences
                ]
                for grade_idx, sequences in self.climbs_by_grade.items()
            },
            "hold_id_mapping": self.hold_id_mapping,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: Path) -> None:
        """Load a trained model from disk."""
        with open(path) as f:
            data = json.load(f)

        self.climbs_by_grade = defaultdict(list)
        for grade_idx_str, sequences in data["climbs_by_grade"].items():
            grade_idx = int(grade_idx_str)
            for sequence in sequences:
                self.climbs_by_grade[grade_idx].append(sequence)

        self.hold_id_mapping = data["hold_id_mapping"]
        self.idx_to_hold_id = {v: k for k, v in self.hold_id_mapping.items()}
