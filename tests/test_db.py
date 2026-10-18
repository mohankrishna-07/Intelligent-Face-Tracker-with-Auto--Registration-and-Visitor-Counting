"""Unit tests for SQLite WAL DatabaseManager (schema, transactions, retry logic)."""

import tempfile
import pytest
import numpy as np
from pathlib import Path

from app.config import StorageConfig
from app.db import DatabaseManager
from app.visit_state import VisitEvent


@pytest.fixture
def temp_db(tmp_path):
    storage_cfg = StorageConfig(
        db_path=str(tmp_path / "test_faces.db"),
        log_dir=str(tmp_path / "logs"),
        log_file=str(tmp_path / "logs" / "events.log")
    )
    return DatabaseManager(storage_cfg)


def test_db_wal_mode_and_tables(temp_db):
    """Verify database initializes in WAL mode and creates required tables."""
    with temp_db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]
        assert mode.lower() == "wal"

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        assert "faces" in tables
        assert "face_embeddings" in tables
        assert "events" in tables


def test_register_and_count_visitors(temp_db):
    """Test registering faces and retrieving unique count."""
    assert temp_db.get_unique_visitor_count() == 0

    emb = np.random.randn(512).astype(np.float32)
    temp_db.register_or_update_face("FACE_0001", emb, "2026-10-18T10:00:00", quality=80.0)
    assert temp_db.get_unique_visitor_count() == 1

    # Update same face (re-entry)
    temp_db.register_or_update_face("FACE_0001", emb, "2026-10-18T10:05:00", quality=85.0)
    assert temp_db.get_unique_visitor_count() == 1, "Count must remain 1 for same face"

    # Register second face
    temp_db.register_or_update_face("FACE_0002", emb, "2026-10-18T10:06:00", quality=90.0)
    assert temp_db.get_unique_visitor_count() == 2


def test_log_event_and_history(temp_db):
    """Test persisting entry and exit events."""
    emb = np.random.randn(512).astype(np.float32)
    temp_db.register_or_update_face("FACE_0001", emb, "2026-10-18T10:00:00")

    ev_entry = VisitEvent(
        event_type="entry",
        face_id="FACE_0001",
        track_id=1,
        timestamp="2026-10-18T10:00:01",
        confidence=0.92,
        visit_id="visit_1"
    )
    temp_db.log_event(ev_entry, "logs/entries/2026-10-18/FACE_0001_entry.jpg")

    ev_exit = VisitEvent(
        event_type="exit",
        face_id="FACE_0001",
        track_id=1,
        timestamp="2026-10-18T10:01:00",
        confidence=0.92,
        visit_id="visit_1"
    )
    temp_db.log_event(ev_exit, "logs/exits/2026-10-18/FACE_0001_exit.jpg")

    history = temp_db.get_event_history(limit=10)
    assert len(history) == 2
    types = [h["event_type"] for h in history]
    assert "entry" in types
    assert "exit" in types
