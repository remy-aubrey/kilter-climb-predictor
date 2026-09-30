"""Tests for src/data/preprocess.py."""

import sqlite3
from pathlib import Path

import pandas as pd

from src.data.preprocess import (
    PreprocessConfig,
    load_raw_data,
    filter_climbs,
    encode_features,
    create_sequences,
    stratified_split,
    compute_stats,
    parse_frames,
    difficulty_to_grade,
)


def _create_test_db(path: Path) -> None:
    """Create a minimal test SQLite database matching the real Kilter schema."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE climbs (
            uuid TEXT PRIMARY KEY,
            name TEXT,
            angle INTEGER,
            frames TEXT
        );
        CREATE TABLE climb_stats (
            climb_uuid TEXT,
            angle INTEGER,
            display_difficulty REAL,
            difficulty_average REAL
        );
        CREATE TABLE holes (
            id INTEGER PRIMARY KEY,
            x REAL,
            y REAL,
            name TEXT
        );
        CREATE TABLE placement_roles (
            id INTEGER PRIMARY KEY,
            name TEXT,
            led_color TEXT
        );

        -- Roles: 12=start, 13=middle, 14=finish, 15=foot
        INSERT INTO placement_roles (id, name, led_color) VALUES
            (12, 'start', 'green'),
            (13, 'middle', 'red'),
            (14, 'finish', 'purple'),
            (15, 'foot', 'orange');

        -- Holes
        INSERT INTO holes (id, x, y, name) VALUES
            (100, 10.0, 10.0, 'hold1'),
            (101, 20.0, 30.0, 'hold2'),
            (102, 30.0, 50.0, 'hold3'),
            (103, 40.0, 70.0, 'hold4'),
            (104, 50.0, 90.0, 'hold5'),
            (105, 15.0, 10.0, 'hold6'),
            (106, 25.0, 35.0, 'hold7'),
            (107, 35.0, 60.0, 'hold8'),
            (108, 45.0, 85.0, 'hold9');

        -- Climb 1: V3 (difficulty 16), 5 holds
        INSERT INTO climbs (uuid, name, angle, frames) VALUES
            ('climb1', 'Test V3', 40, 'p100r12p101r13p102r13p103r13p104r14');
        INSERT INTO climb_stats (climb_uuid, angle, display_difficulty, difficulty_average) VALUES
            ('climb1', 40, 16.0, 16.0);

        -- Climb 2: V5 (difficulty 20), 4 holds
        INSERT INTO climbs (uuid, name, angle, frames) VALUES
            ('climb2', 'Test V5', 40, 'p105r12p106r13p107r13p108r14');
        INSERT INTO climb_stats (climb_uuid, angle, display_difficulty, difficulty_average) VALUES
            ('climb2', 40, 20.0, 20.0);

        -- Climb 3: V3 (difficulty 16), 2 holds (too few, should be filtered)
        INSERT INTO climbs (uuid, name, angle, frames) VALUES
            ('climb3', 'Too Short', 40, 'p100r12p104r14');
        INSERT INTO climb_stats (climb_uuid, angle, display_difficulty, difficulty_average) VALUES
            ('climb3', 40, 16.0, 16.0);

        -- Climb 4: V3 (difficulty 16), 25 degrees (wrong angle, should be filtered)
        INSERT INTO climbs (uuid, name, angle, frames) VALUES
            ('climb4', 'Wrong Angle', 25, 'p100r12p101r13p102r13p103r13p104r14');
        INSERT INTO climb_stats (climb_uuid, angle, display_difficulty, difficulty_average) VALUES
            ('climb4', 25, 16.0, 16.0);
    """)
    conn.commit()
    conn.close()


def test_parse_frames():
    frames = "p100r12p101r13p102r14"
    pairs = parse_frames(frames)
    assert pairs == [(100, 12), (101, 13), (102, 14)]


def test_parse_frames_empty():
    assert parse_frames("") == []
    assert parse_frames(None) == []


def test_difficulty_to_grade():
    assert difficulty_to_grade(1.0) == "V0"
    assert difficulty_to_grade(12.0) == "V0"
    assert difficulty_to_grade(13.0) == "V1"
    assert difficulty_to_grade(16.0) == "V3"
    assert difficulty_to_grade(20.0) == "V5"
    assert difficulty_to_grade(22.0) == "V6"
    assert difficulty_to_grade(34.0) == "V17"
    assert difficulty_to_grade(35.0) is None  # V18, out of range


def test_load_raw_data(tmp_path: Path):
    db_path = tmp_path / "test.db"
    _create_test_db(db_path)

    df = load_raw_data(db_path, angle=40)

    # Should only have 40-degree climbs (climb1, climb2, climb3 — climb4 is 25 degrees)
    assert set(df["climb_uuid"].unique()) == {"climb1", "climb2", "climb3"}
    assert len(df) == 11  # 5 + 4 + 2 holds


def test_filter_climbs(tmp_path: Path):
    db_path = tmp_path / "test.db"
    _create_test_db(db_path)

    df = load_raw_data(db_path, angle=40)
    df = filter_climbs(df, min_holds=3, max_holds=30)

    # climb3 has only 2 holds, should be filtered out
    assert set(df["climb_uuid"].unique()) == {"climb1", "climb2"}


def test_encode_features(tmp_path: Path):
    db_path = tmp_path / "test.db"
    _create_test_db(db_path)

    df = load_raw_data(db_path, angle=40)
    df = filter_climbs(df, min_holds=3, max_holds=30)
    df, hold_id_mapping = encode_features(df)

    assert "hold_id_idx" in df.columns
    assert "hold_type_idx" in df.columns
    assert "led_color_idx" in df.columns
    assert "grade_idx" in df.columns
    assert len(hold_id_mapping) > 0


def test_create_sequences(tmp_path: Path):
    db_path = tmp_path / "test.db"
    _create_test_db(db_path)

    df = load_raw_data(db_path, angle=40)
    df = filter_climbs(df, min_holds=3, max_holds=30)
    df, _ = encode_features(df)
    sequences = create_sequences(df)

    assert len(sequences) == 2
    assert "sequence" in sequences.columns
    assert "grade" in sequences.columns
    assert "length" in sequences.columns


def test_stratified_split(tmp_path: Path):
    db_path = tmp_path / "test.db"
    _create_test_db(db_path)

    df = load_raw_data(db_path, angle=40)
    df = filter_climbs(df, min_holds=3, max_holds=30)
    df, _ = encode_features(df)
    sequences = create_sequences(df)

    train, val, test = stratified_split(sequences, 0.8, 0.1, 0.1, seed=42)

    total = len(train) + len(val) + len(test)
    assert total == len(sequences)


def test_compute_stats(tmp_path: Path):
    db_path = tmp_path / "test.db"
    _create_test_db(db_path)

    df = load_raw_data(db_path, angle=40)
    df = filter_climbs(df, min_holds=3, max_holds=30)
    df, hold_id_mapping = encode_features(df)
    sequences = create_sequences(df)

    stats = compute_stats(sequences, hold_id_mapping)

    assert stats["total_climbs"] == 2
    assert stats["total_unique_holds"] > 0
    assert "grade_distribution" in stats
