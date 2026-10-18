"""Main application orchestrator for Intelligent Face Tracker with Auto-Registration and Visitor Counting."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Set
import cv2
import numpy as np

from app.config import AppConfig
from app.db import DatabaseManager
from app.detector import FaceDetector
from app.event_logger import EventDiskStore, setup_logger
from app.identity_manager import IdentityManager
from app.overlay import HUDOverlay
from app.recognizer import FaceRecognizer
from app.tracker import FaceTracker
from app.video_source import VideoSource
from app.visit_state import TrackStatus, VisitEvent, VisitStateMachine

logger = logging.getLogger("face_tracker.main")


class FaceTrackerApp:
    """Core end-to-end video pipeline manager."""

    def __init__(self, config_path: str = "config.json", headless: bool = False):
        self.project_root = Path(__file__).resolve().parent.parent
        self.config = AppConfig.load(self.project_root / config_path)
        if headless:
            self.config.display.show_window = False

        # Setup structured logger
        setup_logger(self.config.storage, base_dir=self.project_root)
        logger.info("=" * 60)
        logger.info("Starting Intelligent Face Tracker Pipeline")
        logger.info("=" * 60)

        # Initialize persistence & disk store
        self.db = DatabaseManager(self.config.storage, base_dir=self.project_root)
        self.disk_store = EventDiskStore(self.config.storage, base_dir=self.project_root)

        # Initialize AI Models
        logger.info("Initializing detection and recognition models...")
        self.detector = FaceDetector(self.config.detection, base_dir=self.project_root)
        self.recognizer = FaceRecognizer(self.config.recognition, base_dir=self.project_root)
        self.identity_mgr = IdentityManager(self.config.recognition)

        # Load existing identities from SQLite to resume state safely
        existing_records = self.db.load_all_identities()
        if existing_records:
            self.identity_mgr.load_from_db(existing_records)

        # Setup State Machine
        self.visit_state = VisitStateMachine(
            self.config.tracking,
            on_event=self._handle_visit_event
        )

        # Video source & tracker
        self.video_source = VideoSource(self.config.input, base_dir=self.project_root)
        self.tracker = FaceTracker(self.config.tracking, fps=self.video_source.fps)

        # Video writer for output recording if enabled
        self.video_writer = None
        self.is_running = False

        # Signal handlers for graceful flush on Ctrl+C / SIGTERM
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _handle_visit_event(self, event: VisitEvent) -> None:
        """Callback triggered when an ENTRY or EXIT event occurs."""
        # 1. Save crop to disk store
        rel_img_path = self.disk_store.save_event_crop(event)

        # 2. Persist event to SQLite database
        self.db.log_event(event, rel_img_path)

        # 3. Log to events.log in grep-friendly format
        log_msg = (
            f"[{event.event_type.upper()}] face_id={event.face_id} track_id={event.track_id} "
            f"conf={event.confidence:.2f} visit_id={event.visit_id[:8]} img={rel_img_path}"
        )
        logger.info(log_msg)

    def _signal_handler(self, sig, frame):
        """Handle interrupt signals by triggering graceful exit flush."""
        logger.info(f"Received termination signal ({sig}). Initiating graceful shutdown...")
        self.is_running = False

    def run(self, max_frames: Optional[int] = None) -> int:
        """Execute the primary video processing loop."""
        self.is_running = True
        self.video_source.start()

        fps_timer = time.time()
        fps_actual = 0.0
        frame_count = 0

        # Video recording setup
        if self.config.display.save_video_output:
            out_path = str((self.project_root / self.config.display.output_video_path).resolve())
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # Downscale output to 1080p for manageable file size if original is 4K
            w_out = min(1920, self.video_source.frame_width or 1920)
            h_out = min(1080, self.video_source.frame_height or 1080)
            self.video_writer = cv2.VideoWriter(out_path, fourcc, 25.0, (w_out, h_out))
            logger.info(f"Recording output video to: {out_path} ({w_out}x{h_out})")

        logger.info("Pipeline processing loop active. Press 'q' to exit.")

        try:
            while self.is_running:
                ret, frame, frame_idx = self.video_source.read(timeout=2.0)
                if not ret or frame is None:
                    if self.video_source.is_eof:
                        logger.info("Reached End of Stream (EOS).")
                    break

                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    logger.info(f"Reached requested max_frames limit: {max_frames}")
                    break

                t_start = time.time()
                is_detection_frame = ((frame_idx - 1) % self.config.detection.frame_skip == 0)

                if is_detection_frame:
                    # 1. Detection cycle
                    detections = self.detector.detect(frame)

                    # 2. Update ByteTrack
                    active_tracks = self.tracker.update_with_detections(
                        detections, frame_idx, frame.shape
                    )

                    # 3. For each track, check if embedding extraction is needed
                    for trk in active_tracks:
                        # Extract embedding if:
                        # a) track is not yet identified
                        # b) or scheduled for periodic re-verification
                        needs_reverify = (
                            trk.hits % self.config.recognition.reverify_every_n_frames == 0
                        )
                        if trk.face_id is None or needs_reverify:
                            # Find best quality crop matching this track
                            crop_for_emb = trk.best_crop if trk.best_crop is not None else frame[trk.bbox[1]:trk.bbox[3], trk.bbox[0]:trk.bbox[2]]
                            if crop_for_emb is not None and crop_for_emb.size > 0:
                                emb = self.recognizer.extract_embedding(frame, trk.bbox)
                                if emb is not None:
                                    fid, sim, is_new = self.identity_mgr.match_or_register(
                                        emb, trk.conf, trk.track_id, crop=crop_for_emb
                                    )
                                    trk.face_id = fid
                                    trk.similarity_score = sim

                                    # Update SQLite with new/updated face
                                    self.db.register_or_update_face(
                                        fid, emb, datetime_now_iso(), quality=trk.conf
                                    )
                else:
                    # Skipped frame: Kalman track extrapolation
                    active_tracks = self.tracker.predict_skipped_frame(frame_idx)

                # 4. Advance State Machine
                current_tids: Set[int] = set()
                statuses: Dict[int, TrackStatus] = {}

                for trk in active_tracks:
                    current_tids.add(trk.track_id)
                    # Update track in state machine
                    self.visit_state.update_track(
                        track_id=trk.track_id,
                        face_id=trk.face_id,
                        confidence=trk.conf,
                        crop=trk.best_crop
                    )
                    vt = self.visit_state.tracks.get(trk.track_id)
                    if vt:
                        statuses[trk.track_id] = vt.status

                # Handle missed tracks & exit timeouts
                self.visit_state.update_misses(current_tids)

                # Compute FPS
                dt = time.time() - t_start
                if dt > 0:
                    instant_fps = 1.0 / dt
                    fps_actual = 0.9 * fps_actual + 0.1 * instant_fps if fps_actual > 0 else instant_fps

                # 5. Visual Rendering & Display
                if self.config.display.show_window or self.video_writer is not None:
                    vis_frame = frame.copy()
                    if self.config.display.draw_boxes:
                        vis_frame = HUDOverlay.draw_tracks(vis_frame, active_tracks, statuses)
                    
                    vis_frame = HUDOverlay.draw_banner(
                        vis_frame,
                        unique_count=self.identity_mgr.unique_visitor_count,
                        active_count=len(active_tracks),
                        fps=fps_actual,
                        frame_idx=frame_count,
                        total_frames=self.video_source.total_frames
                    )

                    # Write frame if recording
                    if self.video_writer is not None:
                        h_o, w_o = vis_frame.shape[:2]
                        if w_o != 1920 or h_o != 1080:
                            small = cv2.resize(vis_frame, (1920, 1080))
                            self.video_writer.write(small)
                        else:
                            self.video_writer.write(vis_frame)

                    # Show OpenCV window
                    if self.config.display.show_window:
                        # Downscale for display screen fit
                        h_d, w_d = vis_frame.shape[:2]
                        disp_w = min(1280, w_d)
                        disp_h = int(h_d * (disp_w / w_d))
                        disp_img = cv2.resize(vis_frame, (disp_w, disp_h))
                        cv2.imshow("Intelligent Face Tracker", disp_img)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            logger.info("User requested termination ('q').")
                            break

        finally:
            self._teardown()

        return self.identity_mgr.unique_visitor_count

    def _teardown(self) -> None:
        """Clean teardown: flush all remaining active tracks to emit EXIT events."""
        logger.info("Initiating teardown: flushing all active visitor tracks...")
        self.visit_state.flush_all()

        if self.video_source:
            self.video_source.stop()
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        if self.config.display.show_window:
            cv2.destroyAllWindows()

        final_count = self.db.get_unique_visitor_count()
        logger.info("=" * 60)
        logger.info(f"PROCESSING COMPLETED. FINAL UNIQUE VISITORS: {final_count}")
        logger.info("=" * 60)


def datetime_now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat()


def main():
    parser = argparse.ArgumentParser(description="Intelligent Face Tracker with Auto-Registration.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--source", type=str, choices=["video", "rtsp"], default=None, help="Input source override")
    parser.add_argument("--video-path", type=str, default=None, help="Video file path override")
    parser.add_argument("--rtsp-url", type=str, default=None, help="RTSP URL override")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Process max frames then stop")
    args = parser.parse_args()

    app = FaceTrackerApp(config_path=args.config, headless=args.headless)

    # CLI parameter overrides
    if args.source:
        app.config.input.source = args.source
    if args.video_path:
        app.config.input.video_path = args.video_path
    if args.rtsp_url:
        app.config.input.rtsp_url = args.rtsp_url

    final_unique_count = app.run(max_frames=args.max_frames)
    print(f"\n[FINAL RESULT] Total Unique Visitors Count: {final_unique_count}\n")


if __name__ == "__main__":
    main()
