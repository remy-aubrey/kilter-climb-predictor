"""Markov chain climb generator.

Uses position-aware transitions learned from training data.
Fixes applied:
1. Sort holds by y (then x) before counting transitions
2. Build chain from hand holds only, add feet afterwards
3. Stop based on height, not chance
4. Resample instead of stopping when constraints violated
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
    """Markov chain generator for climbing routes.

    Builds a transition matrix from training data, then generates climbs
    by sampling from the transition distribution.
    """

    def __init__(self):
        self.climbs_by_grade: dict[int, list[list[dict]]] = defaultdict(list)
        self.hold_id_mapping: dict[str, int] = {}
        self.idx_to_hold_id: dict[int, str] = {}
        self.transitions: dict[int, list[tuple[int, float, float]]] = defaultdict(list)

    def train(self, data_dir: Path) -> None:
        """Build transition matrix from preprocessed training data.

        Fixes applied:
        1. Sort each climb's holds by y (then x) before counting transitions
        2. Build chain from hand holds only (start, middle, finish)
        6. Pool transitions across all angles (hold positions are angle-independent)
        """
        with open(data_dir / "train.json") as f:
            train_data = json.load(f)

        with open(data_dir / "hold_id_mapping.json") as f:
            self.hold_id_mapping = json.load(f)

        self.idx_to_hold_id = {v: k for k, v in self.hold_id_mapping.items()}

        for climb in train_data:
            grade_idx = climb["grade_idx"]
            sequence = climb["sequence"]

            # Fix 1: Sort holds by y (then x) to get climbing order
            sorted_sequence = sorted(sequence, key=lambda h: (h["y"], h["x"]))

            # Fix 2: Only use hand holds (start, middle, finish) for transitions
            hand_holds = [h for h in sorted_sequence if h["hold_type"] != "foot"]

            self.climbs_by_grade[grade_idx].append(hand_holds)

            # Fix 6: Build transitions from all angles (not just 40-degree)
            # Hold positions are the same across all angles
            for i in range(len(hand_holds) - 1):
                current = hand_holds[i]
                next_hold = hand_holds[i + 1]
                self.transitions[current["hold_id_idx"]].append((
                    next_hold["hold_id_idx"],
                    next_hold["x"],
                    next_hold["y"],
                ))

        # Fix 7: Smooth transitions for holds with few transitions
        # For each hold with < 5 transitions, borrow transitions from nearby holds
        self._smooth_transitions()

        print(f"Trained on {len(train_data)} climbs")
        print(f"Unique holds: {len(self.hold_id_mapping)}")
        print(f"Transition pairs: {sum(len(v) for v in self.transitions.values())}")
        for grade_idx in sorted(self.climbs_by_grade.keys()):
            print(f"  {INDEX_TO_GRADE.get(grade_idx, f'V{grade_idx}')}: {len(self.climbs_by_grade[grade_idx])} climbs")

    def _smooth_transitions(self, min_transitions: int = 5, max_distance: float = 100.0):
        """Smooth transitions for holds with few transitions.

        For each hold with fewer than min_transitions transitions,
        borrow transitions from nearby holds (within max_distance).
        """
        # Build a mapping from hold_idx to position
        hold_positions = {}
        for climb in self.climbs_by_grade.get(5, []):  # Use V5 climbs for positions
            for hold in climb:
                idx = hold["hold_id_idx"]
                if idx not in hold_positions:
                    hold_positions[idx] = (hold["x"], hold["y"])

        # For each hold with few transitions, borrow from nearby holds
        for hold_idx, transitions in list(self.transitions.items()):
            if len(transitions) >= min_transitions:
                continue

            if hold_idx not in hold_positions:
                continue

            hx, hy = hold_positions[hold_idx]

            # Find nearby holds
            for other_idx, (ox, oy) in hold_positions.items():
                if other_idx == hold_idx:
                    continue

                dist = ((ox - hx) ** 2 + (oy - hy) ** 2) ** 0.5
                if dist <= max_distance:
                    # Borrow transitions from nearby hold
                    for next_idx, next_x, next_y in self.transitions.get(other_idx, []):
                        self.transitions[hold_idx].append((next_idx, next_x, next_y))

        print(f"Smoothed transitions for holds with < {min_transitions} transitions")

    def generate(
        self,
        grade: str,
        length: int | None = None,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> list[dict]:
        """Generate a climb using position-aware Markov chain.

        Fixes applied:
        3. Stop based on height (y > 242), not chance
        4. Resample instead of stopping when constraints violated
        5. Add feet as a separate step after generating hand sequence
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

        # Determine target length from real climbs at this grade
        if length is None:
            min_len = min(7, len(template))
            max_len = min(20, len(template))
            length = random.randint(min_len, max_len) if min_len < max_len else max_len

        # Start with a random start hold from the template
        start_holds = [h for h in template if h["hold_type"] == "start"]
        if not start_holds:
            start_holds = template[:2]

        current_hold = random.choice(start_holds)
        current_idx = current_hold["hold_id_idx"]
        current_x, current_y = current_hold["x"], current_hold["y"]

        holds = []
        max_attempts = 100  # Prevent infinite loops
        attempts = 0

        # Fix 3: Stop based on height, not chance
        while len(holds) < length and attempts < max_attempts:
            attempts += 1

            # Determine hold type
            if len(holds) < 2:
                hold_type = "start"
            elif len(holds) == length - 1:
                hold_type = "finish"
            else:
                hold_type = "middle"

            holds.append({
                "hold_id": self.idx_to_hold_id.get(current_idx, f"hold_{current_idx}"),
                "x": current_x,
                "y": current_y,
                "hold_type": hold_type,
                "led_color": INDEX_TO_LED_COLOR.get(0, "red"),
            })

            # Check if we've reached the top zone
            if current_y >= 242 and len(holds) >= 7:
                break

            # Fix 4: Resample instead of stopping when constraints violated
            result = self._sample_next_position_aware(
                current_idx, current_x, current_y, grade_idx, temperature
            )
            if result is None:
                # Try again with relaxed constraints
                result = self._sample_next_relaxed(
                    current_idx, current_x, current_y, grade_idx
                )
                if result is None:
                    break

            next_idx, next_x, next_y = result
            current_idx = next_idx
            current_x, current_y = next_x, next_y

        # Fix 5: Add feet as a separate step
        holds = self._add_feet(holds, grade_idx)

        # Fix 8: Template finish fallback - ensure climb reaches top 3 rows
        if holds and holds[-1]["y"] < 242:
            # Find the highest finish hold in the template
            template_finish = [h for h in template if h["hold_type"] == "finish"]
            if template_finish:
                # Use the highest finish hold
                finish_hold = max(template_finish, key=lambda h: h["y"])
                holds.append({
                    "hold_id": self.idx_to_hold_id.get(finish_hold["hold_id_idx"], "finish"),
                    "x": finish_hold["x"],
                    "y": finish_hold["y"],
                    "hold_type": "finish",
                    "led_color": INDEX_TO_LED_COLOR.get(2, "red"),
                })

        return holds

    def _sample_next_position_aware(
        self,
        current_idx: int,
        current_x: float,
        current_y: float,
        grade_idx: int,
        temperature: float,
        max_distance: float = 500.0,
    ) -> tuple[int, float, float] | None:
        """Sample next hold with directional and distance constraints.

        Only considers holds at or above the current row (next_y >= current_y).
        """
        candidates = self.transitions.get(current_idx, [])
        if not candidates:
            return None

        # Filter by distance and direction
        valid_candidates = []
        for next_idx, next_x, next_y in candidates:
            dist = ((next_x - current_x) ** 2 + (next_y - current_y) ** 2) ** 0.5
            if dist <= max_distance and next_y >= current_y:
                valid_candidates.append((next_idx, next_x, next_y, dist))

        if not valid_candidates:
            return None

        # Weight by inverse distance
        weights = [1.0 / (dist + 1.0) for _, _, _, dist in valid_candidates]

        if temperature != 1.0:
            weights = [w ** (1.0 / temperature) for w in weights]

        total = sum(weights)
        weights = [w / total for w in weights]

        r = random.random()
        cumulative = 0.0
        for (next_idx, next_x, next_y, _), weight in zip(valid_candidates, weights):
            cumulative += weight
            if r <= cumulative:
                return next_idx, next_x, next_y

        next_idx, next_x, next_y, _ = valid_candidates[-1]
        return next_idx, next_x, next_y

    def _sample_next_relaxed(
        self,
        current_idx: int,
        current_x: float,
        current_y: float,
        grade_idx: int,
    ) -> tuple[int, float, float] | None:
        """Relaxed sampling when strict constraints fail.

        Allows some downward movement and longer distances.
        """
        candidates = self.transitions.get(current_idx, [])
        if not candidates:
            return None

        # Relaxed: allow some downward movement (within 50cm)
        valid_candidates = []
        for next_idx, next_x, next_y in candidates:
            dist = ((next_x - current_x) ** 2 + (next_y - current_y) ** 2) ** 0.5
            if dist <= 800 and next_y >= current_y - 50:
                valid_candidates.append((next_idx, next_x, next_y, dist))

        if not valid_candidates:
            return None

        # Simple random choice
        next_idx, next_x, next_y, _ = random.choice(valid_candidates)
        return next_idx, next_x, next_y

    def _add_feet(self, holds: list[dict], grade_idx: int) -> list[dict]:
        """Add foot holds to the climb as a separate step.

        Insert foot holds at positions where they make sense:
        - After the first 2 start holds
        - Before the finish hold
        - At positions where the climb is relatively flat
        """
        if len(holds) < 5:
            return holds

        # Find positions to insert feet
        foot_positions = []
        for i in range(2, len(holds) - 1):
            # Insert foot if the climb is relatively flat here
            prev_y = holds[i - 1]["y"]
            curr_y = holds[i]["y"]
            next_y = holds[i + 1]["y"]

            # Flat section: current hold is between prev and next
            if prev_y <= curr_y <= next_y or next_y <= curr_y <= prev_y:
                foot_positions.append(i)

        # Add at most 2 feet
        for pos in foot_positions[:2]:
            # Create a foot hold at a position near the current hold
            foot_hold = {
                "hold_id": "foot",
                "x": holds[pos]["x"] + random.uniform(-20, 20),
                "y": holds[pos]["y"] - random.uniform(10, 30),
                "hold_type": "foot",
                "led_color": INDEX_TO_LED_COLOR.get(3, "red"),
            }
            holds.insert(pos, foot_hold)

        return holds

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
            "transitions": {str(k): v for k, v in self.transitions.items()},
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

        self.transitions = defaultdict(list)
        for k, v in data["transitions"].items():
            self.transitions[int(k)] = v
