"""Constants for the Kilter Climb Predictor.

Grade mappings, hold types, LED colors, and reachability constraints.
All values are configurable for different body types.
"""

from __future__ import annotations

# ── Grade Mappings ───────────────────────────────────────────────────────────

GRADE_TO_INDEX: dict[str, int] = {
    f"V{i}": i for i in range(18)  # V0–V17
}
INDEX_TO_GRADE: dict[int, str] = {v: k for k, v in GRADE_TO_INDEX.items()}
NUM_GRADES = len(GRADE_TO_INDEX)

# ── Hold Types ───────────────────────────────────────────────────────────────
# Kilter Board hold types as they appear in the database (placement_roles.name).
# These are categorical — we embed them, not one-hot encode.

HOLD_TYPES: list[str] = [
    "start",
    "middle",
    "finish",
    "foot",
]
HOLD_TYPE_TO_INDEX: dict[str, int] = {ht: i for i, ht in enumerate(HOLD_TYPES)}
INDEX_TO_HOLD_TYPE: dict[int, str] = {v: k for k, v in HOLD_TYPE_TO_INDEX.items()}
NUM_HOLD_TYPES = len(HOLD_TYPES)

# ── LED Colors ───────────────────────────────────────────────────────────────
# Categorical LED colors used on the Kilter Board (placement_roles.led_color).
# Stored as hex strings in the database.

LED_COLORS: list[str] = [
    "00FF00",  # green (start)
    "00FFFF",  # cyan (middle)
    "FF00FF",  # magenta (finish)
    "FFA500",  # orange (foot)
]
LED_COLOR_TO_INDEX: dict[str, int] = {c: i for i, c in enumerate(LED_COLORS)}
INDEX_TO_LED_COLOR: dict[int, str] = {v: k for k, v in LED_COLOR_TO_INDEX.items()}
NUM_LED_COLORS = len(LED_COLORS)

# ── Reachability Constraints ─────────────────────────────────────────────────
# Based on user measurements (height: 157cm, wingspan: 157cm).
# Override these in code for different body types.

REACHABILITY = {
    "max_hand_to_hand_cm": 140,       # 89% of wingspan — full stretch
    "comfortable_hand_to_hand_cm": 110,  # 70% of wingspan — controlled
    "max_foot_to_hand_cm": 120,       # Beyond this, cannot maintain tension
    "max_foot_to_foot_cm": 100,      # Feet should stay within reasonable base
    "min_foot_to_hand_cm": 100,      # At least one foot hold this close (for tension)
}

# ── Board Geometry ───────────────────────────────────────────────────────────
# Kilter Board dimensions (40-degree angle, standard layout).
# Coordinates are in centimeters from the bottom-left corner.
# Actual coordinate range from database: x ∈ [-56, 204], y ∈ [-12, 291]

BOARD_WIDTH_CM = 260
BOARD_HEIGHT_CM = 303
BOARD_ANGLE_DEGREES = 40

# ── Sequence Constraints ─────────────────────────────────────────────────────

MIN_HOLDS_PER_CLIMB = 3
MAX_HOLDS_PER_CLIMB = 30
MAX_GENERATION_HOLDS = 20

# ── Data Splits ──────────────────────────────────────────────────────────────

TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1

# ── Random Seed ──────────────────────────────────────────────────────────────

RANDOM_SEED = 42
