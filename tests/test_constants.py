"""Tests for src/utils/constants.py."""

from src.utils.constants import (
    GRADE_TO_INDEX,
    INDEX_TO_GRADE,
    NUM_GRADES,
    HOLD_TYPES,
    HOLD_TYPE_TO_INDEX,
    LED_COLORS,
    LED_COLOR_TO_INDEX,
    REACHABILITY,
    MIN_HOLDS_PER_CLIMB,
    MAX_HOLDS_PER_CLIMB,
    RANDOM_SEED,
)


def test_grade_mappings_are_bidirectional():
    for grade, idx in GRADE_TO_INDEX.items():
        assert INDEX_TO_GRADE[idx] == grade


def test_grade_count():
    assert NUM_GRADES == 18  # V0–V17
    assert len(GRADE_TO_INDEX) == 18


def test_hold_types_nonempty():
    assert len(HOLD_TYPES) > 0
    assert len(HOLD_TYPE_TO_INDEX) == len(HOLD_TYPES)


def test_led_colors_nonempty():
    assert len(LED_COLORS) > 0
    assert len(LED_COLOR_TO_INDEX) == len(LED_COLORS)


def test_reachability_constraints_exist():
    assert "max_hand_to_hand_cm" in REACHABILITY
    assert "max_foot_to_hand_cm" in REACHABILITY
    assert "max_foot_to_foot_cm" in REACHABILITY
    assert REACHABILITY["max_hand_to_hand_cm"] == 140


def test_sequence_constraints():
    assert MIN_HOLDS_PER_CLIMB >= 2
    assert MAX_HOLDS_PER_CLIMB <= 50
    assert MIN_HOLDS_PER_CLIMB < MAX_HOLDS_PER_CLIMB


def test_random_seed():
    assert RANDOM_SEED == 42
