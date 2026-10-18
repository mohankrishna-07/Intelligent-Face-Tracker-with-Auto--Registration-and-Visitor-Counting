# Intelligent Face Tracker with Auto-Registration and Unique Visitor Counting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8 Face](https://img.shields.io/badge/Detector-YOLOv8--Face-green.svg)](https://github.com/ultralytics/ultralytics)
[![InsightFace](https://img.shields.io/badge/Recognition-InsightFace%20ArcFace-orange.svg)](https://github.com/deepinsight/insightface)
[![SQLite WAL](https://img.shields.io/badge/Database-SQLite%20WAL-lightgrey.svg)](https://www.sqlite.org/wal.html)
[![Tests Passing](https://img.shields.io/badge/Tests-12%2F12%20Passing-brightgreen.svg)]()

A modular, production-quality computer vision system that ingests video streams (offline MP4 or live RTSP), detects faces, generates 512-dimensional ArcFace embeddings, auto-registers new identities, tracks individuals with ByteTrack, deterministically logs every entry and exit event exactly once, and accurately tallies unique visitors.

---

## 📽️ Demo Video
> **Explanatory Demo Video Link:**
> [Watch Solution Demonstration & Architecture Walkthrough (YouTube / Loom)](#) *(Placeholder: Insert your recorded Loom/YouTube URL here)*

![Live Tracking HUD Demo](data/demo_overlay.jpg)

---

## 🏗️ System Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Ingestion ["Thread 1: Resilient Stream Ingestion"]
        A[Offline Video / Live RTSP] --> B[VideoSource: Bounded Queue]
        B -->|Stale Frame Dropping| C[(Queue cap=3)]
    end

    subgraph CorePipeline ["Thread 2: Face Vision Pipeline"]
        C -->|Get Frame| D{Frame % Skip == 0?}
        
        %% Detection Path
        D -->|Yes: Frame k| E[YOLOv8-Face Detection]
        E --> F{Quality Gate:<br>Blur >= 25 & Size >= 40px}
        F -->|Pass| G[InsightFace ArcFace 512-d Extraction]
        F -->|Fail| H[Retain Spatial Track]
        
        %% Identity Management
        G --> I[IdentityManager: Cosine Similarity]
        I -->|Score >= 0.45| J[Associate Existing Face ID]
        I -->|Score < 0.45| K[Auto-Register New Face ID]
        
        %% Skipped Frame Path
        D -->|No: Skipped Frame| L[ByteTrack Kalman Extrapolation]
        
        %% Tracking & Spatial Association
        E --> M[ByteTrack Association]
        L --> M
        J --> N[Track-to-Face Binding & Collision Guard]
        K --> N
        H --> N
        M --> N
        
        %% State Machine
        N --> O[VisitStateMachine]
        O -->|Hits >= min_confirm_frames| P[Event: ENTRY]
        O -->|Misses > exit_timeout_frames| Q[Event: EXIT]
        O -->|Stream End / Ctrl+C| R[Flush: Final EXITs]
    end

    subgraph Persistence ["Persistence & Audit Trail"]
        P --> S[(SQLite WAL DB)]
        Q --> S
        R --> S
        P --> T[Disk Crop Store: logs/entries/]
        Q --> T[Disk Crop Store: logs/exits/]
        R --> T
        P --> U[Rotating events.log & Console]
        Q --> U
        R --> U
    end

    subgraph Analytics ["Analytics & UI"]
        S --> V[CLI Report: python -m app.report]
        CorePipeline --> W[Live HUD Overlay Banner]
    end
```

---

## 📋 AI Planning Document & Feature Roadmap

### 1. Key Features
- **Threaded Stream Ingestion**: Decoupled producer-consumer capture with bounded queue (`cap=3`), stale frame dropping for RTSP, and exponential backoff auto-reconnection.
- **Dedicated YOLOv8-Face Model**: Utilizes specialized face weights (`yolov8n-face.pt`) rather than generic COCO person models.
- **Dynamic Frame Skipping**: YOLO detection runs every $N$ frames (`frame_skip=3`). On skipped frames, ByteTrack's Kalman filter extrapolates track positions with zero visual flicker.
- **Multi-Metric Quality Gating**: Evaluates Laplacian blur variance ($\text{Var}(\nabla^2 I) \ge 25.0$), bounding box minimum dimension ($\ge 40$px), and detection confidence ($\ge 0.50$) to keep noisy samples from polluting identity vectors.
- **InsightFace ArcFace 512-d Embeddings**: Employs landmark-based affine alignment to 112x112 and unit-normalized 512-d float vectors.
- **One-Shot Auto-Registration**: First-seen faces are automatically registered as `FACE_0001`, `FACE_0002`, etc.
- **Multi-Embedding Exemplar Bank**: Retains up to $K$ (e.g. 5) highest-quality embeddings per person to remain robust against head poses and lighting variations.
- **Collision Prevention**: Enforces mutual exclusion to prevent assigning the same Face ID to two simultaneous tracks in the same frame.
- **Deterministic State Machine**: Governs `TENTATIVE` $\rightarrow$ `ACTIVE` $\rightarrow$ `LOST` $\rightarrow$ `EXITED` transitions. Guarantees exact-once ENTRY/EXIT emissions, smooth occlusion recovery within timeout, and a graceful flush on shutdown.
- **SQLite WAL Mode Storage**: Full ACID persistence with Write-Ahead Logging (`PRAGMA journal_mode=WAL`), foreign keys, and write-lock retry resilience.
- **Grep-Friendly Logging & Disk Storage**: Timestamped face crops saved to `logs/entries/YYYY-MM-DD/` and `logs/exits/YYYY-MM-DD/` alongside structured lines in `logs/events.log`.
- **CLI Analytics Report**: `python -m app.report` displays visitor history and cross-verifies DB counts against `events.log`.

### 2. Compute Load & Resource Benchmarks
| Metric | CPU (Intel Core i7 / Ryzen 7, 8-Core) | GPU (NVIDIA RTX 3060 / 4060, CUDA) |
| :--- | :--- | :--- |
| **YOLO Face Detection Latency** | ~20 - 30 ms / cycle | ~4 - 6 ms / cycle |
| **ArcFace Embedding Latency** | ~45 - 65 ms (event-driven only) | ~8 - 12 ms |
| **Frame Skip ($N=3$) Effective FPS** | **28 - 35 FPS (Real-Time)** | **60+ FPS** |
| **System RAM Usage** | ~650 MB - 1.1 GB | ~800 MB |
| **GPU VRAM Usage** | N/A | ~1.5 - 2.0 GB |

---

## ⚙️ Configuration Design (`config.json`)

```json
{
  "input": {
    "source": "video",
    "video_path": "data/sample.mp4",
    "rtsp_url": "rtsp://admin:password@192.168.1.100:554/stream1",
    "reconnect_delay_sec": 2.0,
    "max_reconnect_attempts": 10,
    "queue_size": 3
  },
  "detection": {
    "model_path": "models/yolov8n-face.pt",
    "frame_skip": 3,
    "conf_threshold": 0.5,
    "min_face_size": 40,
    "blur_threshold": 25.0,
    "device": "auto"
  },
  "recognition": {
    "model_name": "buffalo_l",
    "recognition_threshold": 0.45,
    "max_embeddings_per_face": 5,
    "reverify_every_n_frames": 30,
    "device": "auto"
  },
  "tracking": {
    "tracker": "bytetrack",
    "min_confirm_frames": 3,
    "exit_timeout_frames": 30
  },
  "storage": {
    "db_path": "data/faces.db",
    "log_dir": "logs",
    "log_file": "logs/events.log"
  },
  "display": {
    "show_window": true,
    "draw_boxes": true
  }
}
```

### Parameter Explanations
- `input.source`: Ingestion mode (`"video"` for file, `"rtsp"` for live RTSP stream).
- `input.queue_size`: Bounded queue capacity; drops stale frames on RTSP to maintain real-time low latency.
- `detection.frame_skip`: Frequency of YOLO detection inference (1 = every frame, 3 = every 3rd frame).
- `detection.blur_threshold`: Laplacian variance threshold. Faces with blur score below this are marked as blurry.
- `recognition.recognition_threshold`: Cosine similarity cutoff (0.45). Scores $\ge 0.45$ match an existing visitor; $< 0.45$ registers a new visitor.
- `recognition.max_embeddings_per_face`: Maximum exemplar embeddings preserved in bank per visitor.
- `tracking.min_confirm_frames`: Number of consecutive detection hits required before promoting a track to `ACTIVE` and emitting an `ENTRY` event.
- `tracking.exit_timeout_frames`: Frame threshold without detections before promoting a track to `EXITED` and emitting an `EXIT` event.
- `storage.db_path`: Location of the SQLite database.

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/mohankrishna-07/Intelligent-Face-Tracker-with-Auto--Registration-and-Visitor-Counting.git
cd Intelligent-Face-Tracker-with-Auto--Registration-and-Visitor-Counting

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Weights Download
The `models/yolov8n-face.pt` weight and `buffalo_l` recognition models are automatically downloaded or can be manually verified:
```bash
# Verify YOLOv8-face model
curl -L -o models/yolov8n-face.pt https://huggingface.co/junjiang/GestureFace/resolve/main/yolov8n-face.pt
```

### 4. Running the Application

#### Option A: Running on a Video File
```bash
# Standard run with live OpenCV GUI window
python -m app.main

# Headless mode (ideal for servers or batch benchmarking)
python -m app.main --headless

# Override video path via CLI
python -m app.main --video-path data/sample.mp4
```

#### Option B: Running on Live RTSP Stream
```bash
# Using CLI override:
python -m app.main --source rtsp --rtsp-url "rtsp://admin:pass@192.168.1.100:554/stream"

# Or configure config.json:
# "input": {"source": "rtsp", "rtsp_url": "rtsp://..."}
```

### 5. Generating Analytics & Audit Report
```bash
python -m app.report
```

### 6. Running Automated Unit Tests
```bash
python -m pytest tests/ -v
```

---

## 📊 Evaluation & Verification Results

Running on the 4K surveillance sample dataset (`data/sample.mp4`):
```text
======================================================================
      INTELLIGENT FACE TRACKER - VISITOR ANALYTICS REPORT
======================================================================

[SUMMARY METRICS]
  • Total Unique Visitors Registered: 64

[LOG FILE CROSS-VERIFICATION]
  - Log File Location:        logs/events.log
  - Logged Auto-Registrations: 64
  - Logged Re-Identifications: 0
  - Logged ENTRY events:       64
  - Logged EXIT events:        64
  - DB Unique Faces Count:     64
  [VERIFIED] DB faces count (64) matches logged auto-registrations exactly!
======================================================================
```

---

## 🧪 Unit Tests Suite
The test suite validates core domain logic across 12 automated unit tests:
- `tests/test_visit_state.py`:
  - `test_entry_emitted_exactly_once_after_confirm`: Verifies exact-once entry emission.
  - `test_exit_emitted_after_timeout`: Verifies timeout-based exit transition.
  - `test_temporary_occlusion_recovery_same_visit`: Ensures brief occlusions do not spawn duplicate visits.
  - `test_reentry_after_full_exit`: Tests returning visitors logging a new visit under the same Face ID.
  - `test_flush_all_on_shutdown`: Validates graceful shutdown flush of all active tracks.
- `tests/test_identity_manager.py`:
  - `test_one_shot_auto_registration`: Incremental `FACE_XXXX` generation.
  - `test_reidentification_does_not_increment_unique_count`: Invariance of unique visitor tally.
  - `test_multi_embedding_exemplar_bank_cap`: Quality-based exemplar replacement.
  - `test_collision_prevention_mutual_exclusion`: Mutual exclusion guard.
- `tests/test_db.py`:
  - `test_db_wal_mode_and_tables`: Schema and WAL validation.
  - `test_register_and_count_visitors`: Integrity of unique count queries.
  - `test_log_event_and_history`: Event logging audit trail.

---

## 📝 Assumptions Made
1. **Surveillance Angle**: For high-angle surveillance cameras, faces appear smaller; hence `blur_threshold` is calibrated to 25.0 and `min_face_size` to 40px to capture passing visitors accurately.
2. **Offline vs. Live Ingestion**: For offline video files, EOF signals the natural termination of the stream, triggering the state machine flush. For RTSP, transient read failures trigger exponential backoff reconnection.
3. **Identity Memory**: Face embeddings are preserved both in an in-memory cosine gallery and persistently in SQLite, allowing the system to resume state across application restarts.

---

This project is a part of a hackathon run by https://katomaran.com
