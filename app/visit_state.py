"""Visit State Machine managing deterministic, idempotent Entry and Exit lifecycle transitions."""

from __future__ import annotations

import enum
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set
import numpy as np

from app.config import TrackingConfig

logger = logging.getLogger("face_tracker.visit_state")


class TrackStatus(enum.Enum):
    TENTATIVE = "TENTATIVE"  # Newly created, accumulating hits
    ACTIVE = "ACTIVE"        # Confirmed and identified; ENTRY emitted
    LOST = "LOST"            # Not seen in current frame, but within exit_timeout_frames
    EXITED = "EXITED"        # Missing frames exceeded threshold; EXIT emitted


@dataclass
class VisitEvent:
    """Represents an immutable Entry or Exit lifecycle event."""
    event_type: str  # "entry" or "exit"
    face_id: str
    track_id: int
    timestamp: str
    confidence: float
    face_crop: Optional[np.ndarray] = None
    visit_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class VisitorTrack:
    """Tracks state and lifecycle transitions for a single physical visit."""
    track_id: int
    face_id: Optional[str] = None
    status: TrackStatus = TrackStatus.TENTATIVE
    consecutive_hits: int = 0
    consecutive_misses: int = 0
    confidence: float = 0.0
    best_crop: Optional[np.ndarray] = None
    entry_emitted: bool = False
    exit_emitted: bool = False
    visit_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_seen_at: str = field(default_factory=lambda: datetime.now().isoformat())


class VisitStateMachine:
    """State machine governing TENTATIVE -> ACTIVE -> LOST -> EXITED transitions.
    
    Guarantees:
    - Idempotency: exact-once entry and exit event emissions per visit.
    - Graceful flush on shutdown: emits exit for any active tracks.
    - Seamless recovery for brief occlusions within exit_timeout_frames.
    """

    def __init__(
        self,
        config: TrackingConfig,
        on_event: Optional[Callable[[VisitEvent], None]] = None
    ):
        self.config = config
        self.on_event = on_event
        self.tracks: Dict[int, VisitorTrack] = {}
        self.active_face_ids: Set[str] = set()

    def update_track(
        self,
        track_id: int,
        face_id: Optional[str],
        confidence: float,
        crop: Optional[np.ndarray] = None
    ) -> List[VisitEvent]:
        """Process detection/continuation hit for a track."""
        events: List[VisitEvent] = []
        now_iso = datetime.now().isoformat()

        if track_id not in self.tracks:
            self.tracks[track_id] = VisitorTrack(
                track_id=track_id,
                face_id=face_id,
                confidence=confidence,
                best_crop=crop,
                consecutive_hits=1,
                consecutive_misses=0,
                last_seen_at=now_iso
            )
        else:
            vt = self.tracks[track_id]
            vt.consecutive_hits += 1
            vt.consecutive_misses = 0
            vt.last_seen_at = now_iso
            vt.confidence = max(vt.confidence, confidence)
            if crop is not None:
                vt.best_crop = crop
            if face_id and not vt.face_id:
                vt.face_id = face_id

        vt = self.tracks[track_id]

        # Transition: TENTATIVE -> ACTIVE
        if vt.status in (TrackStatus.TENTATIVE, TrackStatus.LOST):
            # Check if track is now confirmed and identified
            if vt.consecutive_hits >= self.config.min_confirm_frames and vt.face_id:
                if not vt.entry_emitted:
                    vt.status = TrackStatus.ACTIVE
                    vt.entry_emitted = True
                    self.active_face_ids.add(vt.face_id)
                    
                    event = VisitEvent(
                        event_type="entry",
                        face_id=vt.face_id,
                        track_id=track_id,
                        timestamp=now_iso,
                        confidence=vt.confidence,
                        face_crop=vt.best_crop,
                        visit_id=vt.visit_id
                    )
                    events.append(event)
                    logger.info(
                        f"[STATE ENTRY] face_id={vt.face_id} track_id={track_id} "
                        f"hits={vt.consecutive_hits} visit_id={vt.visit_id[:8]}"
                    )
                    if self.on_event:
                        self.on_event(event)
                else:
                    # Re-acquired after temporary LOST state within timeout window
                    vt.status = TrackStatus.ACTIVE

        return events

    def update_misses(self, active_track_ids: Set[int]) -> List[VisitEvent]:
        """Increment misses for tracks missing from the current frame and trigger EXITED when timeout exceeded."""
        events: List[VisitEvent] = []
        now_iso = datetime.now().isoformat()
        to_delete = []

        for tid, vt in list(self.tracks.items()):
            if tid not in active_track_ids:
                vt.consecutive_misses += 1

                # Transition: ACTIVE -> LOST
                if vt.status == TrackStatus.ACTIVE:
                    vt.status = TrackStatus.LOST

                # Transition: LOST / TENTATIVE -> EXITED
                if vt.consecutive_misses > self.config.exit_timeout_frames:
                    if vt.entry_emitted and not vt.exit_emitted and vt.face_id:
                        vt.status = TrackStatus.EXITED
                        vt.exit_emitted = True
                        if vt.face_id in self.active_face_ids:
                            self.active_face_ids.remove(vt.face_id)

                        event = VisitEvent(
                            event_type="exit",
                            face_id=vt.face_id,
                            track_id=tid,
                            timestamp=now_iso,
                            confidence=vt.confidence,
                            face_crop=vt.best_crop,
                            visit_id=vt.visit_id
                        )
                        events.append(event)
                        logger.info(
                            f"[STATE EXIT] face_id={vt.face_id} track_id={tid} "
                            f"misses={vt.consecutive_misses} visit_id={vt.visit_id[:8]}"
                        )
                        if self.on_event:
                            self.on_event(event)

                    to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

        return events

    def flush_all(self) -> List[VisitEvent]:
        """Emit EXIT events for all remaining active tracks on stream end or shutdown."""
        events: List[VisitEvent] = []
        now_iso = datetime.now().isoformat()

        logger.info(f"Flushing all remaining active tracks (count={len(self.tracks)})...")
        for tid, vt in list(self.tracks.items()):
            if vt.entry_emitted and not vt.exit_emitted and vt.face_id:
                vt.status = TrackStatus.EXITED
                vt.exit_emitted = True
                event = VisitEvent(
                    event_type="exit",
                    face_id=vt.face_id,
                    track_id=tid,
                    timestamp=now_iso,
                    confidence=vt.confidence,
                    face_crop=vt.best_crop,
                    visit_id=vt.visit_id
                )
                events.append(event)
                logger.info(f"[FLUSH EXIT] face_id={vt.face_id} track_id={tid}")
                if self.on_event:
                    self.on_event(event)

        self.tracks.clear()
        self.active_face_ids.clear()
        return events
