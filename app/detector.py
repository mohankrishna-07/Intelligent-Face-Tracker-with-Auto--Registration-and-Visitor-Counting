"""YOLO Face Detector module with multi-metric quality gating (blur, size, confidence)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

from app.config import DetectionConfig

logger = logging.getLogger("face_tracker.detector")


@dataclass
class FaceDetection:
    """Represents a single face detection and its quality evaluation."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    conf: float
    blur_score: float
    face_crop: np.ndarray
    is_quality_pass: bool
    landmarks: Optional[np.ndarray] = None  # (5, 2) facial keypoints if available
    rejection_reason: Optional[str] = None


class FaceDetector:
    """YOLOv8-Face detector with strict quality gating."""

    def __init__(self, config: DetectionConfig, base_dir: Optional[Path] = None):
        self.config = config
        self.base_dir = base_dir
        self.model_path = str(
            (base_dir / self.config.model_path).resolve()
            if base_dir
            else Path(self.config.model_path).resolve()
        )
        self.device = self._resolve_device(self.config.device)
        self.model = None
        self._load_model()

    def _resolve_device(self, req_device: str) -> str:
        """Auto-detect CUDA availability or fallback to CPU."""
        if req_device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("CUDA GPU detected. Using CUDA for face detection.")
                    return "cuda"
            except ImportError:
                pass
            logger.info("Using CPU for face detection.")
            return "cpu"
        return req_device

    def _load_model(self) -> None:
        """Load YOLO model weights using ultralytics."""
        from ultralytics import YOLO

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"YOLO Face model weights not found at: {self.model_path}")

        logger.info(f"Loading YOLO Face model from {self.model_path} on {self.device}...")
        self.model = YOLO(self.model_path)
        logger.info("YOLO Face model loaded successfully.")

    @staticmethod
    def calculate_blur(crop: np.ndarray) -> float:
        """Estimate image sharpness using the variance of the Laplacian operator."""
        if crop is None or crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces in a frame and evaluate quality criteria."""
        if self.model is None or frame is None or frame.size == 0:
            return []

        h_img, w_img = frame.shape[:2]
        
        # Run inference
        results = self.model.predict(
            source=frame,
            conf=self.config.conf_threshold,
            device=self.device,
            verbose=False
        )

        detections: List[FaceDetection] = []

        if not results or len(results) == 0:
            return detections

        res = results[0]
        boxes = res.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        # Extract bounding boxes and confidences
        xyxy_arr = boxes.xyxy.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        
        # Check if landmarks (keypoints) are present
        has_keypoints = hasattr(res, "keypoints") and res.keypoints is not None and res.keypoints.xy is not None
        keypoints_arr = res.keypoints.xy.cpu().numpy() if has_keypoints else None

        for idx in range(len(xyxy_arr)):
            x1, y1, x2, y2 = xyxy_arr[idx].astype(int)
            conf = float(conf_arr[idx])

            # Clamp coordinates to frame boundaries
            x1 = max(0, min(x1, w_img - 1))
            y1 = max(0, min(y1, h_img - 1))
            x2 = max(0, min(x2, w_img))
            y2 = max(0, min(y2, h_img))

            w_box = x2 - x1
            h_box = y2 - y1

            if w_box <= 0 or h_box <= 0:
                continue

            crop = frame[y1:y2, x1:x2].copy()
            blur_score = self.calculate_blur(crop)

            # Quality Gating
            rejection = None
            if min(w_box, h_box) < self.config.min_face_size:
                rejection = f"Too small: min({w_box},{h_box}) < {self.config.min_face_size}px"
            elif blur_score < self.config.blur_threshold:
                rejection = f"Too blurry: blur={blur_score:.1f} < {self.config.blur_threshold}"

            is_quality_pass = (rejection is None)
            
            lms = None
            if keypoints_arr is not None and idx < len(keypoints_arr):
                lms = keypoints_arr[idx]

            detections.append(
                FaceDetection(
                    bbox=(x1, y1, x2, y2),
                    conf=conf,
                    blur_score=blur_score,
                    face_crop=crop,
                    is_quality_pass=is_quality_pass,
                    landmarks=lms,
                    rejection_reason=rejection,
                )
            )

        return detections
