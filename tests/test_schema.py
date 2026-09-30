"""Tests for src/data/schema.py."""

import sqlite3
from pathlib import Path

from src.data.schema import explore_schema, get_table_counts


def _create_test_db(path: Path) -> None:
    """Create a minimal test SQLite database."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE climbs (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE holds (id INTEGER PRIMARY KEY, climb_id INTEGER)")
    conn.execute("INSERT INTO climbs (id, name) VALUES (1, 'Test Climb')")
    conn.execute("INSERT INTO holds (id, climb_id) VALUES (1, 1)")
    conn.commit()
    conn.close()


def test_explore_schema(tmp_path: Path):
    db_path = tmp_path / "test.db"
    _create_test_db(db_path)

    schema = explore_schema(db_path)

    assert "climbs" in schema
    assert "holds" in schema
    assert len(schema["climbs"]) == 2  # id, name
    assert len(schema["holds"]) == 2   # id, climb_id


def test_get_table_counts(tmp_path: Path):
    db_path = tmp_path / "test.db"
    _create_test_db(db_path)

    counts = get_table_counts(db_path)

    assert counts["climbs"] == 1
    assert counts["holds"] == 1
