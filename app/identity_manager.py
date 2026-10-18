"""Identity Manager maintaining the gallery of unique registered faces and multi-embedding exemplars."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from app.config import RecognitionConfig

logger = logging.getLogger("face_tracker.identity_manager")


@dataclass
class FaceIdentity:
    """Represents a unique visitor identity in the gallery."""
    face_id: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    first_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    visit_count: int = 1
    best_crop: Optional[np.ndarray] = None

    def add_embedding(self, emb: np.ndarray, quality: float, max_k: int = 5) -> None:
        """Add an embedding exemplar, keeping the top-K highest quality vectors."""
        if len(self.embeddings) < max_k:
            self.embeddings.append(emb)
            self.quality_scores.append(quality)
        else:
            # Replace lowest quality embedding if current sample is higher quality
            min_idx = int(np.argmin(self.quality_scores))
            if quality > self.quality_scores[min_idx]:
                self.embeddings[min_idx] = emb
                self.quality_scores[min_idx] = quality

    def match_score(self, query_emb: np.ndarray) -> float:
        """Calculate max cosine similarity across all stored exemplars."""
        if not self.embeddings or query_emb is None:
            return 0.0
        scores = [float(np.dot(query_emb, e)) for e in self.embeddings]
        return max(scores)


class IdentityManager:
    """Manages one-shot auto-registration, gallery matching, and collision prevention."""

    def __init__(self, config: RecognitionConfig):
        self.config = config
        self.identities: Dict[str, FaceIdentity] = {}
        self.id_counter: int = 0
        
        # Track-to-face active assignments for the current frame to prevent duplicate binding
        # Dict[track_id -> face_id]
        self.track_to_face: Dict[int, str] = {}
        # Reverse mapping: Dict[face_id -> track_id]
        self.face_to_track: Dict[str, int] = {}

    @property
    def unique_visitor_count(self) -> int:
        """Total distinct visitors registered in the system."""
        return len(self.identities)

    def load_from_db(self, records: List[Tuple[str, str, str, bytes, int]]) -> None:
        """Hydrate existing identities from database on startup."""
        for face_id, first_seen, last_seen, emb_bytes, count in records:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            
            identity = FaceIdentity(
                face_id=face_id,
                embeddings=[emb],
                quality_scores=[1.0],
                first_seen=first_seen,
                last_seen=last_seen,
                visit_count=count
            )
            self.identities[face_id] = identity
            
            # Update counter to prevent ID collision
            if face_id.startswith("FACE_"):
                try:
                    num = int(face_id.split("_")[1])
                    if num > self.id_counter:
                        self.id_counter = num
                except ValueError:
                    pass
        logger.info(f"Loaded {len(self.identities)} identities from database. Next counter: {self.id_counter + 1}")

    def _generate_new_id(self) -> str:
        """Generate next incremental ID (e.g. FACE_0001)."""
        self.id_counter += 1
        return f"FACE_{self.id_counter:04d}"

    def match_or_register(
        self,
        query_emb: np.ndarray,
        quality_score: float,
        track_id: int,
        crop: Optional[np.ndarray] = None
    ) -> Tuple[str, float, bool]:
        """Match query embedding against gallery or auto-register a new visitor.
        
        Args:
            query_emb: 512-d normalized embedding.
            quality_score: blur or confidence metric.
            track_id: spatial track ID.
            crop: face image crop.
            
        Returns:
            (face_id, similarity_score, is_new_registration)
        """
        now_iso = datetime.now().isoformat()
        best_face_id: Optional[str] = None
        best_score = -1.0

        # Find best match in gallery
        for fid, identity in self.identities.items():
            # If this face_id is already bound to another active track in this frame, skip to prevent collision
            if fid in self.face_to_track and self.face_to_track[fid] != track_id:
                continue

            score = identity.match_score(query_emb)
            if score > best_score:
                best_score = score
                best_face_id = fid

        # Evaluate against recognition threshold
        if best_face_id is not None and best_score >= self.config.recognition_threshold:
            # Recognized existing visitor
            identity = self.identities[best_face_id]
            identity.last_seen = now_iso
            identity.add_embedding(query_emb, quality_score, max_k=self.config.max_embeddings_per_face)
            if crop is not None:
                identity.best_crop = crop
                
            self._bind_track_face(track_id, best_face_id)
            logger.info(
                f"[RECOGNIZED] Face ID: {best_face_id} (Track: {track_id}, "
                f"Similarity: {best_score:.3f} >= {self.config.recognition_threshold})"
            )
            return best_face_id, best_score, False
        else:
            # Auto-register new visitor
            new_id = self._generate_new_id()
            identity = FaceIdentity(
                face_id=new_id,
                embeddings=[query_emb],
                quality_scores=[quality_score],
                first_seen=now_iso,
                last_seen=now_iso,
                visit_count=1,
                best_crop=crop
            )
            self.identities[new_id] = identity
            self._bind_track_face(track_id, new_id)
            logger.info(
                f"[REGISTERED] New Face ID: {new_id} (Track: {track_id}, "
                f"Highest similarity was {best_score:.3f} < {self.config.recognition_threshold})"
            )
            return new_id, float(best_score if best_score >= 0 else 0.0), True

    def _bind_track_face(self, track_id: int, face_id: str) -> None:
        """Bind spatial track to persistent Face ID with mutual exclusion."""
        # Unbind prior assignments if any
        if track_id in self.track_to_face:
            old_fid = self.track_to_face[track_id]
            if self.face_to_track.get(old_fid) == track_id:
                del self.face_to_track[old_fid]

        self.track_to_face[track_id] = face_id
        self.face_to_track[face_id] = track_id

    def release_track(self, track_id: int) -> Optional[str]:
        """Release track binding when track exits or is purged."""
        if track_id in self.track_to_face:
            face_id = self.track_to_face.pop(track_id)
            if self.face_to_track.get(face_id) == track_id:
                del self.face_to_track[face_id]
            return face_id
        return None

    def get_face_id_for_track(self, track_id: int) -> Optional[str]:
        """Get currently assigned Face ID for given track ID."""
        return self.track_to_face.get(track_id, None)
