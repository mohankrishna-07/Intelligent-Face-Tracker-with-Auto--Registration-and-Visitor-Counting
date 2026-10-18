"""Live visual HUD overlay renderer for face bounding boxes, track badges, and system statistics."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.tracker import TrackState
from app.visit_state import TrackStatus


class HUDOverlay:
    """Renders professional bounding boxes, visitor badges, and live metric HUDs."""

    # Colors (BGR)
    COLOR_CONFIRMED = (46, 204, 113)    # Emerald Green
    COLOR_TENTATIVE = (52, 152, 219)    # Blue
    COLOR_LOST = (231, 76, 60)          # Red
    COLOR_HEADER_BG = (20, 25, 30)      # Dark Charcoal
    COLOR_TEXT = (255, 255, 255)        # White

    @classmethod
    def draw_banner(
        cls,
        frame: np.ndarray,
        unique_count: int,
        active_count: int,
        fps: float,
        frame_idx: int,
        total_frames: int = -1
    ) -> np.ndarray:
        """Draw a sleek header banner with live system metrics."""
        h, w = frame.shape[:2]
        banner_h = 60

        # Semi-transparent overlay banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), cls.COLOR_HEADER_BG, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Draw metrics
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 1. Unique visitor count (highlighted in Green)
        cv2.putText(frame, "UNIQUE VISITORS:", (20, 38), font, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{unique_count}", (245, 40), font, 1.1, (46, 204, 113), 3, cv2.LINE_AA)

        # 2. Active in frame
        cv2.putText(frame, f"Active Faces: {active_count}", (350, 38), font, 0.75, (230, 230, 230), 2, cv2.LINE_AA)

        # 3. Processing FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (580, 38), font, 0.75, (230, 230, 230), 2, cv2.LINE_AA)

        # 4. Frame progress
        if total_frames > 0:
            pct = (frame_idx / total_frames) * 100.0
            cv2.putText(frame, f"Frame: {frame_idx}/{total_frames} ({pct:.0f}%)", (w - 380, 38), font, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"Frame: {frame_idx} (LIVE)", (w - 280, 38), font, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        return frame

    @classmethod
    def draw_tracks(
        cls,
        frame: np.ndarray,
        tracks: List[TrackState],
        statuses: Dict[int, TrackStatus]
    ) -> np.ndarray:
        """Draw bounding boxes and identity badges for active tracks."""
        h_frame, w_frame = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        for trk in tracks:
            x1, y1, x2, y2 = trk.bbox
            status = statuses.get(trk.track_id, TrackStatus.TENTATIVE)

            if status == TrackStatus.ACTIVE:
                color = cls.COLOR_CONFIRMED
                label = f"{trk.face_id or 'IDENTIFYING'} [Track {trk.track_id}]"
            elif status == TrackStatus.LOST:
                color = cls.COLOR_LOST
                label = f"LOST [Track {trk.track_id}]"
            else:
                color = cls.COLOR_TENTATIVE
                label = f"TENTATIVE [Track {trk.track_id}] ({trk.hits})"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Draw badge background
            (tw, th), baseline = cv2.getTextSize(label, font, 0.55, 2)
            badge_y1 = max(0, y1 - th - 10)
            badge_y2 = y1
            cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 10, badge_y2), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 5, badge_y2 - 5),
                font,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

        return frame
