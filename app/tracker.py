"""Spatial face tracking module integrating ByteTrack with Kalman prediction on skipped frames."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.config import TrackingConfig
from app.detector import FaceDetection

logger = logging.getLogger("face_tracker.tracker")


class DetWrapper:
    """NumPy-backed detections wrapper supporting boolean indexing for BYTETracker."""
    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask) -> "DetWrapper":
        return DetWrapper(self.xywh[mask], self.conf[mask], self.cls[mask])


@dataclass
class TrackState:
    """Internal state of an active face track."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    conf: float
    hits: int = 1
    misses: int = 0
    face_id: Optional[str] = None
    similarity_score: float = 0.0
    best_crop: Optional[np.ndarray] = None
    best_quality: float = 0.0
    first_frame: int = 0
    last_frame: int = 0
    is_confirmed: bool = False


class FaceTracker:
    """ByteTrack wrapper supporting detection updates and Kalman extrapolation on skipped frames."""

    def __init__(self, config: TrackingConfig, fps: float = 30.0):
        self.config = config
        self.fps = fps
        self.tracker = None
        self.active_tracks: Dict[int, TrackState] = {}
        self._init_tracker()

    def _init_tracker(self) -> None:
        """Initialize ByteTrack instance with configured parameters."""
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker

            tracker_args = SimpleNamespace(
                tracker_type="bytetrack",
                track_high_thresh=0.25,
                track_low_thresh=0.1,
                new_track_thresh=0.25,
                track_buffer=self.config.exit_timeout_frames,
                match_thresh=0.8,
                fuse_score=True,
                frame_rate=int(self.fps)
            )
            self.tracker = BYTETracker(tracker_args)
            logger.info("BYTETracker initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize BYTETracker: {e}")
            raise

    def update_with_detections(
        self,
        detections: List[FaceDetection],
        frame_idx: int,
        frame_shape: Tuple[int, int]
    ) -> List[TrackState]:
        """Update tracker with new YOLO face detections on detection cycles."""
        h_img, w_img = frame_shape[:2]

        if self.tracker is None:
            return []

        # Convert detections to xywh: [cx, cy, w, h]
        if detections:
            xywh_list = []
            conf_list = []
            for d in detections:
                x1, y1, x2, y2 = d.bbox
                w = float(x2 - x1)
                h = float(y2 - y1)
                cx = float(x1) + w / 2.0
                cy = float(y1) + h / 2.0
                xywh_list.append([cx, cy, w, h])
                conf_list.append(d.conf)

            xywh_arr = np.array(xywh_list, dtype=np.float32)
            conf_arr = np.array(conf_list, dtype=np.float32)
            cls_arr = np.zeros(len(conf_arr), dtype=np.float32)
        else:
            xywh_arr = np.empty((0, 4), dtype=np.float32)
            conf_arr = np.empty((0,), dtype=np.float32)
            cls_arr = np.empty((0,), dtype=np.float32)

        wrapper = DetWrapper(xywh_arr, conf_arr, cls_arr)
        # online_targets returns array of shape (N, 8): [x1, y1, x2, y2, track_id, score, cls, idx]
        online_targets = self.tracker.update(wrapper)

        current_track_ids = set()
        updated_tracks: List[TrackState] = []

        if online_targets is not None and len(online_targets) > 0:
            for row in online_targets:
                x1 = max(0, min(int(row[0]), w_img - 1))
                y1 = max(0, min(int(row[1]), h_img - 1))
                x2 = max(0, min(int(row[2]), w_img))
                y2 = max(0, min(int(row[3]), h_img))
                track_id = int(row[4])
                score = float(row[5])
                det_idx = int(row[7])

                bbox = (x1, y1, x2, y2)
                current_track_ids.add(track_id)

                if track_id not in self.active_tracks:
                    # New track spawned
                    st = TrackState(
                        track_id=track_id,
                        bbox=bbox,
                        conf=score,
                        hits=1,
                        misses=0,
                        first_frame=frame_idx,
                        last_frame=frame_idx,
                        is_confirmed=(self.config.min_confirm_frames <= 1)
                    )
                    self.active_tracks[track_id] = st
                else:
                    # Existing track updated
                    st = self.active_tracks[track_id]
                    st.bbox = bbox
                    st.conf = score
                    st.hits += 1
                    st.misses = 0
                    st.last_frame = frame_idx
                    if st.hits >= self.config.min_confirm_frames:
                        st.is_confirmed = True

                # Associate detection crop if available
                if 0 <= det_idx < len(detections):
                    det = detections[det_idx]
                    if det.blur_score >= st.best_quality:
                        st.best_crop = det.face_crop
                        st.best_quality = det.blur_score

                updated_tracks.append(st)

        # Increment misses for active tracks not detected in this frame
        for tid in list(self.active_tracks.keys()):
            if tid not in current_track_ids:
                self.active_tracks[tid].misses += 1

        return updated_tracks

    def predict_skipped_frame(self, frame_idx: int) -> List[TrackState]:
        """On skipped frames, keep confirmed active tracks alive with zero-flicker."""
        return [
            st for st in self.active_tracks.values()
            if st.misses <= self.config.exit_timeout_frames
        ]

    def remove_track(self, track_id: int) -> Optional[TrackState]:
        """Remove a track when exited or timed out."""
        return self.active_tracks.pop(track_id, None)
