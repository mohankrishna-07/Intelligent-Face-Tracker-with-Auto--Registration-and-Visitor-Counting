"""InsightFace ArcFace (buffalo_l) feature extraction module for 512-d embeddings."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

from app.config import RecognitionConfig

logger = logging.getLogger("face_tracker.recognizer")


class FaceRecognizer:
    """ArcFace 512-d embedding extractor with landmark alignment."""

    def __init__(self, config: RecognitionConfig, base_dir: Optional[Path] = None):
        self.config = config
        self.base_dir = base_dir
        self.model_name = self.config.model_name
        self.device = self._resolve_device(self.config.device)
        self.handler = None
        self._init_model()

    def _resolve_device(self, req_device: str) -> str:
        """Auto-detect CUDA availability or CPU fallback."""
        if req_device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("CUDA detected. Using CUDAExecutionProvider for InsightFace.")
                    return "cuda"
            except ImportError:
                pass
            logger.info("Using CPUExecutionProvider for InsightFace.")
            return "cpu"
        return req_device

    def _init_model(self) -> None:
        """Initialize InsightFace ArcFace model."""
        import insightface
        from insightface.app import FaceAnalysis

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )

        logger.info(f"Initializing InsightFace ({self.model_name}) with providers: {providers}")
        self.app = FaceAnalysis(name=self.model_name, providers=providers)
        ctx_id = 0 if self.device == "cuda" else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        
        # Access ArcFace recognition model directly from models dictionary
        self.rec_model = self.app.models.get("recognition", None)
        if self.rec_model is None:
            logger.warning("Recognition model not found in FaceAnalysis.models; will use fallback.")
        logger.info("InsightFace FaceRecognizer initialized successfully.")

    def extract_embedding(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        landmarks: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Extract a 512-d normalized embedding for a detected face.
        
        Args:
            frame: Original BGR frame.
            bbox: (x1, y1, x2, y2)
            landmarks: Optional 5-point facial landmarks for affine normalization.
            
        Returns:
            Normalized 512-d float32 numpy vector or None if extraction fails.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        # If landmarks available, use standard InsightFace affine alignment to 112x112
        if landmarks is not None and len(landmarks) >= 5:
            try:
                from insightface.utils import face_align
                aligned = face_align.norm_crop(frame, landmark=landmarks[:5], image_size=112)
            except Exception as e:
                logger.debug(f"Landmark alignment failed ({e}); falling back to bbox crop.")
                crop = frame[y1:y2, x1:x2]
                aligned = cv2.resize(crop, (112, 112))
        else:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            aligned = cv2.resize(crop, (112, 112))

        # Extract features using ArcFace recognition model
        try:
            if self.rec_model is not None:
                feat = self.rec_model.get_feat(aligned)
                if feat is not None:
                    feat = feat.flatten()
                    norm = np.linalg.norm(feat)
                    if norm > 0:
                        return (feat / norm).astype(np.float32)
            
            # Fallback: run app.get on crop
            crop = frame[y1:y2, x1:x2]
            faces = self.app.get(crop)
            if faces and len(faces) > 0 and faces[0].embedding is not None:
                feat = faces[0].embedding.flatten()
                norm = np.linalg.norm(feat)
                if norm > 0:
                    return (feat / norm).astype(np.float32)
        except Exception as e:
            logger.error(f"Error during ArcFace feature extraction: {e}")

        return None

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two unit-normalized embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        # For unit vectors, cosine similarity is simply the dot product
        return float(np.dot(emb1, emb2))
