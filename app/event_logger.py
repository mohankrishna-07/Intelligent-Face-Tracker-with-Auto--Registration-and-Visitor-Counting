"""Structured event logging module with rotating file handler, console output, and disk image crop store."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from app.config import StorageConfig
from app.visit_state import VisitEvent


def setup_logger(config: StorageConfig, base_dir: Optional[Path] = None) -> logging.Logger:
    """Set up structured rotating file and console logging."""
    log_file_path = (
        (base_dir / config.log_file).resolve()
        if base_dir
        else Path(config.log_file).resolve()
    )
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("face_tracker")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Prevent duplicate handlers
    if not logger.handlers:
        # Standardized format easy to grep
        formatter = logging.Formatter(
            fmt="[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Rotating file handler (10 MB, 5 backups)
        file_handler = RotatingFileHandler(
            str(log_file_path),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger


class EventDiskStore:
    """Manages saving and structuring cropped face images for entry and exit events."""

    def __init__(self, config: StorageConfig, base_dir: Optional[Path] = None):
        self.config = config
        self.base_dir = base_dir
        self.log_dir = (
            (base_dir / config.log_dir).resolve()
            if base_dir
            else Path(config.log_dir).resolve()
        )
        self.entries_dir = self.log_dir / "entries"
        self.exits_dir = self.log_dir / "exits"
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        self.exits_dir.mkdir(parents=True, exist_ok=True)

    def save_event_crop(self, event: VisitEvent, frame: Optional[np.ndarray] = None) -> str:
        """Save cropped face image to structured local folder: logs/{entries|exits}/YYYY-MM-DD/.
        
        Filename format: FACE_0001_entry_20261016_153012_123.jpg
        
        Returns:
            Relative image path for database record.
        """
        # Parse timestamp for date folder and file string
        try:
            dt = datetime.fromisoformat(event.timestamp)
        except Exception:
            dt = datetime.now()

        date_folder = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%Y%m%d_%H%M%S_%f")[:19]  # millisecond precision

        target_dir = (self.entries_dir if event.event_type == "entry" else self.exits_dir) / date_folder
        target_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{event.face_id}_{event.event_type}_{time_str}.jpg"
        abs_path = target_dir / filename

        # Prefer event.face_crop; if missing, create placeholder or crop from frame
        crop_to_save = event.face_crop
        if crop_to_save is None or crop_to_save.size == 0:
            # Create a blank fallback badge
            crop_to_save = np.zeros((150, 150, 3), dtype=np.uint8)
            cv2.putText(
                crop_to_save,
                f"{event.face_id}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        cv2.imwrite(str(abs_path), crop_to_save)
        
        # Return path relative to base directory if possible
        if self.base_dir:
            try:
                return str(abs_path.relative_to(self.base_dir)).replace("\\", "/")
            except ValueError:
                pass
        return str(abs_path).replace("\\", "/")
