# AI Engineering Prompts Log

This document records the structured engineering prompts, system directives, and design requirements utilized during the development and evaluation of the **Intelligent Face Tracker with Auto-Registration and Unique Visitor Counting** system.

---

## 1. System Planning & Architecture Design Prompt

```text
Act as a Principal Computer Vision & Distributed Systems Engineer. Design a modular, production-ready Python 3.10+ application for 'Intelligent Face Tracker with Auto-Registration and Unique Visitor Counting'.
Requirements:
- Resilient threaded ingestion (supporting MP4 files and live RTSP with drop-stale-frame low-latency buffering and exponential backoff auto-reconnect).
- High-efficiency face detection using dedicated YOLOv8-face models with configurable frame skipping (N=3) and Kalman track extrapolation across skipped frames.
- Multi-metric quality gating (Laplacian blur score variance, minimum face bounding box dimension, detection confidence) prior to embedding inference.
- High-accuracy face recognition using InsightFace (ArcFace / buffalo_l 512-dimensional embeddings) with landmark affine normalization.
- One-shot dynamic visitor auto-registration, maintaining an exemplar bank of up to K embeddings per identity with cosine similarity matching.
- Multi-object tracking via ByteTrack with track-to-face identity mutual exclusion to eliminate track swaps and duplicate assignments.
- Deterministic, idempotent visit state machine: TENTATIVE -> ACTIVE -> LOST -> EXITED with exact-once ENTRY/EXIT emissions, occlusion recovery, and graceful shutdown flushing.
- SQLite WAL mode database persistence with foreign keys, busy-timeout retry resilience, rotating events.log, and local disk cropped face image store.
- Comprehensive CLI analytics report tool verifying DB counts against events.log.
```

---

## 2. Face Detection & Quality Gating Prompt

```text
Implement `app/detector.py` using Ultralytics YOLO with `yolov8n-face.pt` weights.
Ensure:
1. GPU auto-detection (CUDA) with CPU fallback.
2. Blur estimation using the variance of the Laplacian operator: cv2.Laplacian(gray, cv2.CV_64F).var().
3. Strict quality filtering rejecting faces below blur_threshold, min_face_size, or conf_threshold.
4. Structured `FaceDetection` dataclass capturing bounding box, confidence, blur score, crop, and quality status.
```

---

## 3. Facial Recognition & Identity Gallery Prompt

```text
Implement `app/recognizer.py` and `app/identity_manager.py` using InsightFace ArcFace (buffalo_l):
1. Extract unit-normalized 512-d float32 embeddings from detected faces with 112x112 affine normalization.
2. Maintain a memory gallery `FaceIdentity` with an exemplar bank storing top-K high-quality embeddings.
3. Cosine similarity matching: max dot product against gallery. If max_score >= recognition_threshold, match existing ID and update last_seen; otherwise auto-register incremental ID (e.g., FACE_0001).
4. Enforce strict mutual exclusion: one Face ID cannot be assigned to two simultaneous tracks in the same frame.
5. Hydrate identities from SQLite on startup to resume state after interruptions.
```

---

## 4. ByteTrack & Spatial Tracking Integration Prompt

```text
Implement `app/tracker.py` wrapping Ultralytics BYTETracker:
1. Accept detections and convert coordinates to xywh format [cx, cy, w, h] with confidence scores.
2. Supply NumPy-backed DetWrapper supporting boolean indexing for BYTETracker.update().
3. Track hits, misses, bounding boxes, and detection crops.
4. On skipped frames, extrapolate confirmed active tracks using internal Kalman filter predictions to guarantee zero-flicker visualization.
```

---

## 5. Deterministic Visit State Machine Prompt

```text
Implement `app/visit_state.py` with states: TENTATIVE -> ACTIVE -> LOST -> EXITED.
Requirements:
1. Exact-once ENTRY logged when hits >= min_confirm_frames and identity is bound.
2. Exact-once EXIT logged when misses > exit_timeout_frames.
3. Temporary occlusions within timeout remain the same visit (no duplicate events).
4. Re-entry of returning visitor logs a new visit under the same Face ID without increasing the unique visitor count.
5. Graceful shutdown flush: flush_all() emits EXIT events for all open active tracks upon EOF, SIGINT, or SIGTERM.
```

---

## 6. Persistence & SQLite WAL Optimization Prompt

```text
Implement `app/db.py` and `app/event_logger.py`:
1. SQLite WAL mode configuration:
   PRAGMA journal_mode = WAL;
   PRAGMA synchronous = NORMAL;
   PRAGMA foreign_keys = ON;
   PRAGMA busy_timeout = 5000;
2. Master `faces` table, `face_embeddings` table, and `events` table with index optimization.
3. Retry decorator for write transactions to handle database lock contention.
4. Rotating file handler for `logs/events.log` (10MB, 5 backups) with grep-friendly structured log lines.
5. Disk crop saver saving cropped faces to `logs/entries/YYYY-MM-DD/` and `logs/exits/YYYY-MM-DD/`.
```

---

## 7. Verification, Analytics & Test Suite Prompt

```text
Write automated pytest test suites covering:
1. `tests/test_visit_state.py`: ENTRY exact-once, EXIT timeout, occlusion recovery, re-entry, and flush on shutdown.
2. `tests/test_identity_manager.py`: One-shot auto-registration, cosine similarity re-ID, multi-embedding bank capping, and collision prevention.
3. `tests/test_db.py`: SQLite WAL mode, schema verification, visitor count accuracy, and event history retrieval.
Implement `app/report.py` to query DB metrics and cross-verify with events.log.
```
