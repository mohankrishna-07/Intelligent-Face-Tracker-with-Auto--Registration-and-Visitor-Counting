"""Configuration loader and schema validator for Intelligent Face Tracker."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class InputConfig:
    source: str = "video"  # "video" or "rtsp"
    video_path: str = "data/sample.mp4"
    rtsp_url: str = ""
    reconnect_delay_sec: float = 2.0
    max_reconnect_attempts: int = 10
    queue_size: int = 3

    def validate(self, base_dir: Path) -> None:
        if self.source not in ("video", "rtsp"):
            raise ValueError(f"Invalid input source: '{self.source}'. Must be 'video' or 'rtsp'.")
        if self.source == "video":
            full_path = (base_dir / self.video_path).resolve()
            if not full_path.exists():
                raise FileNotFoundError(f"Video file not found at: {full_path}")
        elif self.source == "rtsp" and not self.rtsp_url:
            raise ValueError("source='rtsp' specified but 'rtsp_url' is empty.")


@dataclass
class DetectionConfig:
    model_path: str = "models/yolov8n-face.pt"
    frame_skip: int = 3
    conf_threshold: float = 0.5
    min_face_size: int = 40
    blur_threshold: float = 60.0
    device: str = "auto"  # "auto", "cpu", "cuda", "0", etc.

    def validate(self, base_dir: Path) -> None:
        full_path = (base_dir / self.model_path).resolve()
        if not full_path.exists():
            raise FileNotFoundError(f"Detection model weights not found at: {full_path}")
        if self.frame_skip < 1:
            raise ValueError(f"frame_skip must be >= 1, got {self.frame_skip}")
        if not (0.0 < self.conf_threshold <= 1.0):
            raise ValueError(f"conf_threshold must be in (0.0, 1.0], got {self.conf_threshold}")
        if self.min_face_size < 10:
            raise ValueError(f"min_face_size must be >= 10, got {self.min_face_size}")


@dataclass
class RecognitionConfig:
    model_name: str = "buffalo_l"
    recognition_threshold: float = 0.45
    max_embeddings_per_face: int = 5
    reverify_every_n_frames: int = 30
    device: str = "auto"

    def validate(self) -> None:
        if not (0.0 < self.recognition_threshold <= 1.0):
            raise ValueError(f"recognition_threshold must be in (0.0, 1.0], got {self.recognition_threshold}")
        if self.max_embeddings_per_face < 1:
            raise ValueError(f"max_embeddings_per_face must be >= 1, got {self.max_embeddings_per_face}")
        if self.reverify_every_n_frames < 1:
            raise ValueError(f"reverify_every_n_frames must be >= 1, got {self.reverify_every_n_frames}")


@dataclass
class TrackingConfig:
    tracker: str = "bytetrack"
    min_confirm_frames: int = 3
    exit_timeout_frames: int = 30

    def validate(self) -> None:
        if self.min_confirm_frames < 1:
            raise ValueError(f"min_confirm_frames must be >= 1, got {self.min_confirm_frames}")
        if self.exit_timeout_frames < 1:
            raise ValueError(f"exit_timeout_frames must be >= 1, got {self.exit_timeout_frames}")


@dataclass
class StorageConfig:
    db_path: str = "data/faces.db"
    log_dir: str = "logs"
    log_file: str = "logs/events.log"

    def validate(self, base_dir: Path) -> None:
        # Ensure parent directories exist
        (base_dir / self.db_path).parent.mkdir(parents=True, exist_ok=True)
        (base_dir / self.log_dir / "entries").mkdir(parents=True, exist_ok=True)
        (base_dir / self.log_dir / "exits").mkdir(parents=True, exist_ok=True)
        (base_dir / self.log_file).parent.mkdir(parents=True, exist_ok=True)


@dataclass
class DisplayConfig:
    show_window: bool = True
    draw_boxes: bool = True
    save_video_output: bool = False
    output_video_path: str = "data/output.mp4"


@dataclass
class AppConfig:
    input: InputConfig = field(default_factory=InputConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    base_dir: Path = field(default_factory=lambda: Path.cwd())

    @classmethod
    def load(cls, config_path: str | Path = "config.json") -> AppConfig:
        """Load configuration from JSON file with fallback defaults and schema validation."""
        path = Path(config_path).resolve()
        base_dir = path.parent

        data: Dict[str, Any] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            print(f"[Config] Warning: {config_path} not found. Using default configurations.")

        cfg = cls(
            input=InputConfig(**data.get("input", {})),
            detection=DetectionConfig(**data.get("detection", {})),
            recognition=RecognitionConfig(**data.get("recognition", {})),
            tracking=TrackingConfig(**data.get("tracking", {})),
            storage=StorageConfig(**data.get("storage", {})),
            display=DisplayConfig(**data.get("display", {})),
            base_dir=base_dir,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Validate all nested configuration sections."""
        self.input.validate(self.base_dir)
        self.detection.validate(self.base_dir)
        self.recognition.validate()
        self.tracking.validate()
        self.storage.validate(self.base_dir)

    def resolve_path(self, relative_path: str) -> str:
        """Return absolute path string relative to application base directory."""
        return str((self.base_dir / relative_path).resolve())
