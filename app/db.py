"""SQLite persistence layer with WAL mode, transactions, foreign keys, and retry resilience."""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from app.config import StorageConfig
from app.visit_state import VisitEvent

logger = logging.getLogger("face_tracker.db")


def retry_on_lock(max_retries: int = 5, delay: float = 0.1):
    """Decorator to retry SQLite operations if locked."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() or "busy" in str(e).lower():
                        last_err = e
                        time.sleep(delay * (2 ** attempt))
                    else:
                        raise
            raise last_err or sqlite3.OperationalError("Database retry limit reached.")
        return wrapper
    return decorator


class DatabaseManager:
    """Manages SQLite connection in WAL mode with robust crash-resiliency."""

    def __init__(self, config: StorageConfig, base_dir: Optional[Path] = None):
        self.config = config
        self.base_dir = base_dir
        self.db_path = str(
            (base_dir / self.config.db_path).resolve()
            if base_dir
            else Path(self.config.db_path).resolve()
        )
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with WAL mode and foreign key constraints enabled."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.row_factory = sqlite3.Row
        return conn

    @retry_on_lock()
    def _init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        logger.info(f"Initializing SQLite database at: {self.db_path}")
        with self._get_connection() as conn:
            # 1. Master faces table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    face_id TEXT PRIMARY KEY,
                    first_seen TIMESTAMP NOT NULL,
                    last_seen TIMESTAMP NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_count INTEGER DEFAULT 1,
                    notes TEXT
                );
            """)

            # 2. Multi-embedding gallery table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    quality REAL NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY(face_id) REFERENCES faces(face_id) ON DELETE CASCADE
                );
            """)

            # 3. Events log table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id TEXT NOT NULL,
                    event_type TEXT NOT NULL CHECK(event_type IN ('entry', 'exit')),
                    timestamp TIMESTAMP NOT NULL,
                    image_path TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    visit_id TEXT,
                    FOREIGN KEY(face_id) REFERENCES faces(face_id) ON DELETE RESTRICT
                );
            """)

            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_face_id ON events(face_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_face_embeddings_face ON face_embeddings(face_id);")
            conn.commit()

        logger.info("Database schema verified with WAL mode enabled.")

    @retry_on_lock()
    def register_or_update_face(
        self,
        face_id: str,
        embedding: np.ndarray,
        timestamp: str,
        quality: float = 1.0
    ) -> None:
        """Insert a newly registered face or update last_seen and append to exemplar bank."""
        emb_bytes = embedding.astype(np.float32).tobytes()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding_count FROM faces WHERE face_id = ?", (face_id,))
            row = cursor.fetchone()

            if row is None:
                # Insert master face record
                cursor.execute("""
                    INSERT INTO faces (face_id, first_seen, last_seen, embedding, embedding_count)
                    VALUES (?, ?, ?, ?, 1)
                """, (face_id, timestamp, timestamp, emb_bytes))
            else:
                # Update last seen and count
                cursor.execute("""
                    UPDATE faces
                    SET last_seen = ?, embedding_count = embedding_count + 1
                    WHERE face_id = ?
                """, (timestamp, face_id))

            # Insert into multi-embedding table
            cursor.execute("""
                INSERT INTO face_embeddings (face_id, embedding, quality, created_at)
                VALUES (?, ?, ?, ?)
            """, (face_id, emb_bytes, quality, timestamp))

            conn.commit()

    @retry_on_lock()
    def log_event(self, event: VisitEvent, image_path: str) -> int:
        """Persist an idempotent entry or exit event with image crop path."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO events (face_id, event_type, timestamp, image_path, track_id, confidence, visit_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event.face_id,
                event.event_type,
                event.timestamp,
                image_path,
                event.track_id,
                event.confidence,
                event.visit_id
            ))
            
            # Update last_seen in faces table
            cursor.execute("""
                UPDATE faces SET last_seen = ? WHERE face_id = ?
            """, (event.timestamp, event.face_id))

            conn.commit()
            return cursor.lastrowid

    def get_unique_visitor_count(self) -> int:
        """Direct retrieval of unique visitor count from SQLite source of truth."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM faces;")
            return int(cursor.fetchone()[0])

    def load_all_identities(self) -> List[Tuple[str, str, str, bytes, int]]:
        """Retrieve all registered identities on startup."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT face_id, first_seen, last_seen, embedding, embedding_count
                FROM faces
                ORDER BY first_seen ASC;
            """)
            rows = cursor.fetchall()
            return [(r["face_id"], r["first_seen"], r["last_seen"], r["embedding"], r["embedding_count"]) for r in rows]

    def get_event_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch recent events with details."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, face_id, event_type, timestamp, image_path, track_id, confidence, visit_id
                FROM events
                ORDER BY timestamp DESC
                LIMIT ?;
            """, (limit,))
            return [dict(r) for r in cursor.fetchall()]
