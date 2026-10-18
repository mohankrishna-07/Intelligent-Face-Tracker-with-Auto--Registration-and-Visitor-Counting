"""Threaded video capture module for video files and live RTSP streams with auto-reconnect."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from app.config import InputConfig

logger = logging.getLogger("face_tracker.video_source")


class VideoSource:
    """Producer-consumer threaded video capture.
    
    Guarantees low latency on live RTSP feeds by dropping stale frames when the queue is full.
    Provides exponential backoff auto-reconnect for RTSP dropouts and clean EOF handling for files.
    """

    def __init__(self, config: InputConfig, base_dir=None):
        self.config = config
        self.base_dir = base_dir
        self.source_target: str = (
            str(self.base_dir / self.config.video_path)
            if (self.config.source == "video" and self.base_dir)
            else (self.config.video_path if self.config.source == "video" else self.config.rtsp_url)
        )
        self.is_rtsp = self.config.source == "rtsp"
        
        # Bounded frame queue
        self.queue: queue.Queue[Tuple[int, np.ndarray]] = queue.Queue(maxsize=self.config.queue_size)
        
        # Thread control flags
        self.running = False
        self.stopped = False
        self.is_eof = False
        self.thread: Optional[threading.Thread] = None
        
        # Video metadata
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 30.0
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.total_frames: int = -1
        self.frame_counter: int = 0

    def _open_capture(self) -> bool:
        """Attempt to open cv2.VideoCapture with optimized flags."""
        logger.info(f"Opening video source: {self.source_target} (is_rtsp={self.is_rtsp})")
        
        if self.is_rtsp:
            # Low latency RTSP options via OpenCV FFMPEG backend
            # st_timeout is microsecond timeout (5s = 5000000)
            self.cap = cv2.VideoCapture(self.source_target, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.cap = cv2.VideoCapture(self.source_target)

        if not self.cap or not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source_target}")
            return False

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_rtsp else -1
        logger.info(
            f"Video source initialized successfully: {self.frame_width}x{self.frame_height} "
            f"@ {self.fps:.2f} FPS (Total frames: {self.total_frames})"
        )
        return True

    def start(self) -> "VideoSource":
        """Start the background frame ingestion thread."""
        if not self._open_capture():
            if not self.is_rtsp:
                raise RuntimeError(f"Cannot open video file: {self.source_target}")
        
        self.running = True
        self.stopped = False
        self.thread = threading.Thread(target=self._capture_worker, name="VideoSourceThread", daemon=True)
        self.thread.start()
        return self

    def _capture_worker(self) -> None:
        """Background worker thread continuously grabbing frames and populating the queue."""
        reconnect_attempts = 0
        delay = self.config.reconnect_delay_sec

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                if not self.is_rtsp:
                    self.is_eof = True
                    break
                
                # Attempt RTSP reconnection
                if reconnect_attempts >= self.config.max_reconnect_attempts:
                    logger.critical("Exceeded maximum RTSP reconnect attempts. Terminating capture thread.")
                    break

                reconnect_attempts += 1
                logger.warning(
                    f"RTSP stream disconnected. Attempting reconnect {reconnect_attempts}/"
                    f"{self.config.max_reconnect_attempts} in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay = min(delay * 1.5, 30.0)  # Exponential backoff capped at 30s
                
                if self._open_capture():
                    reconnect_attempts = 0
                    delay = self.config.reconnect_delay_sec
                continue

            ret, frame = self.cap.read()
            if not ret:
                if self.is_rtsp:
                    logger.warning("RTSP read() returned False. Triggering reconnect loop.")
                    if self.cap:
                        self.cap.release()
                    self.cap = None
                    continue
                else:
                    logger.info("Reached End of File (EOF) on video source.")
                    self.is_eof = True
                    break

            self.frame_counter += 1
            
            # For RTSP: drop oldest frame if queue is full to prevent lag accumulation
            if self.is_rtsp and self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.queue.put((self.frame_counter, frame), timeout=0.1)
            except queue.Full:
                # If offline video file and queue is full, wait a bit so consumer can catch up
                time.sleep(0.005)

        logger.info("Video capture thread exiting.")

    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray], int]:
        """Fetch the next available frame from the queue.
        
        Returns:
            (success, frame, frame_number)
        """
        while self.running or not self.queue.empty():
            try:
                frame_idx, frame = self.queue.get(timeout=timeout)
                return True, frame, frame_idx
            except queue.Empty:
                if self.is_eof:
                    return False, None, self.frame_counter
                if not self.running:
                    return False, None, self.frame_counter
                # RTSP might be reconnecting
                continue
        return False, None, self.frame_counter

    def stop(self) -> None:
        """Stop capture worker and release system camera resources."""
        self.running = False
        self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("VideoSource stopped and resources released.")

    def __enter__(self) -> "VideoSource":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
