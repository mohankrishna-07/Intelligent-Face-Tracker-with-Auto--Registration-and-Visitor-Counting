"""Phase 1 Verification Script: Test YOLOv8-Face Detection & Quality Gating on data/sample.mp4."""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppConfig
from app.detector import FaceDetector


def main():
    print("=" * 60)
    print("PHASE 1 VERIFICATION: YOLOv8-Face Detector & Quality Gating")
    print("=" * 60)

    cfg = AppConfig.load(PROJECT_ROOT / "config.json")
    print(f"[Config] Model path: {cfg.detection.model_path}")
    print(f"[Config] Confidence threshold: {cfg.detection.conf_threshold}")
    print(f"[Config] Min face size: {cfg.detection.min_face_size}px")
    print(f"[Config] Blur threshold: {cfg.detection.blur_threshold}")
    print(f"[Config] Frame skip: {cfg.detection.frame_skip}")

    detector = FaceDetector(cfg.detection, base_dir=PROJECT_ROOT)

    video_path = str(PROJECT_ROOT / cfg.input.video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Video] Source: {video_path}")
    print(f"[Video] Resolution: {w}x{h}, FPS: {fps:.2f}, Total frames: {total_frames}")

    frame_idx = 0
    detected_count = 0
    passed_quality_count = 0
    inference_times = []
    sample_annotated_frame = None

    start_total = time.time()
    # Process up to first 60 frames for quick verification
    max_test_frames = min(total_frames, 60)

    while cap.isOpened() and frame_idx < max_test_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if (frame_idx - 1) % cfg.detection.frame_skip != 0:
            continue

        t0 = time.time()
        detections = detector.detect(frame)
        t_infer = (time.time() - t0) * 1000.0
        inference_times.append(t_infer)

        detected_count += len(detections)
        for det in detections:
            if det.is_quality_pass:
                passed_quality_count += 1
            print(
                f"  Frame {frame_idx:03d} | Box: {det.bbox} | Conf: {det.conf:.2f} | "
                f"Blur: {det.blur_score:.1f} | Quality Pass: {det.is_quality_pass} "
                f"{f'({det.rejection_reason})' if not det.is_quality_pass else ''}"
            )

        # Annotate a representative frame for verification
        if len(detections) > 0 and sample_annotated_frame is None:
            vis = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                color = (0, 255, 0) if det.is_quality_pass else (0, 0, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
                label = f"Conf:{det.conf:.2f} Blur:{det.blur_score:.0f}"
                cv2.putText(vis, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            sample_annotated_frame = vis

    cap.release()
    total_elapsed = time.time() - start_total

    print("-" * 60)
    print(f"Processed {frame_idx} frames in {total_elapsed:.2f}s")
    if inference_times:
        print(f"Average Detection Latency: {np.mean(inference_times):.2f} ms per detection cycle")
        print(f"Total face detections: {detected_count} (Quality passed: {passed_quality_count})")

    if sample_annotated_frame is not None:
        out_img_path = PROJECT_ROOT / "data" / "phase1_detection_sample.jpg"
        cv2.imwrite(str(out_img_path), sample_annotated_frame)
        print(f"[Output] Saved annotated sample detection to: {out_img_path}")

    print("=" * 60)
    print("PHASE 1 DETECTION VERIFICATION COMPLETE: SUCCESS")
    print("=" * 60)


if __name__ == "__main__":
    main()
