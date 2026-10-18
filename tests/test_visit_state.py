"""Unit tests for the Visit State Machine (Entry/Exit lifecycle, idempotency, flush)."""

import pytest
import numpy as np
from types import SimpleNamespace

from app.config import TrackingConfig
from app.visit_state import TrackStatus, VisitEvent, VisitStateMachine


@pytest.fixture
def tracking_config():
    return TrackingConfig(
        tracker="bytetrack",
        min_confirm_frames=3,
        exit_timeout_frames=5
    )


def test_entry_emitted_exactly_once_after_confirm(tracking_config):
    """Test that ENTRY event is emitted exactly once when track reaches min_confirm_frames and has face_id."""
    emitted_events = []
    sm = VisitStateMachine(tracking_config, on_event=lambda e: emitted_events.append(e))

    dummy_crop = np.zeros((50, 50, 3), dtype=np.uint8)

    # Frame 1: Tentative (hits=1)
    ev1 = sm.update_track(track_id=1, face_id="FACE_0001", confidence=0.8, crop=dummy_crop)
    assert len(ev1) == 0
    assert sm.tracks[1].status == TrackStatus.TENTATIVE

    # Frame 2: Tentative (hits=2)
    ev2 = sm.update_track(track_id=1, face_id="FACE_0001", confidence=0.85, crop=dummy_crop)
    assert len(ev2) == 0
    assert sm.tracks[1].status == TrackStatus.TENTATIVE

    # Frame 3: Confirmed (hits=3 >= min_confirm_frames) -> Triggers ENTRY
    ev3 = sm.update_track(track_id=1, face_id="FACE_0001", confidence=0.9, crop=dummy_crop)
    assert len(ev3) == 1
    assert ev3[0].event_type == "entry"
    assert ev3[0].face_id == "FACE_0001"
    assert ev3[0].track_id == 1
    assert sm.tracks[1].status == TrackStatus.ACTIVE
    assert sm.tracks[1].entry_emitted is True

    # Frame 4: Continues ACTIVE -> Must NOT emit another ENTRY
    ev4 = sm.update_track(track_id=1, face_id="FACE_0001", confidence=0.92, crop=dummy_crop)
    assert len(ev4) == 0
    assert len(emitted_events) == 1, "ENTRY event must be strictly idempotent"


def test_exit_emitted_after_timeout(tracking_config):
    """Test that EXIT event is emitted exactly once after missing frames exceed exit_timeout_frames."""
    emitted_events = []
    sm = VisitStateMachine(tracking_config, on_event=lambda e: emitted_events.append(e))
    dummy_crop = np.zeros((50, 50, 3), dtype=np.uint8)

    # Bring track 1 to ACTIVE
    for _ in range(3):
        sm.update_track(track_id=1, face_id="FACE_0001", confidence=0.9, crop=dummy_crop)

    assert len(emitted_events) == 1
    assert emitted_events[0].event_type == "entry"

    # Now track 1 is missing for 5 frames (timeout is 5)
    for miss in range(1, 6):
        events = sm.update_misses(active_track_ids=set())
        assert len(events) == 0, f"Must not emit exit before timeout: miss {miss}"
        assert sm.tracks[1].status == TrackStatus.LOST

    # Frame 6 of misses (> 5) -> Triggers EXIT
    events = sm.update_misses(active_track_ids=set())
    assert len(events) == 1
    assert events[0].event_type == "exit"
    assert events[0].face_id == "FACE_0001"
    assert events[0].track_id == 1
    assert 1 not in sm.tracks, "Track should be purged after exit"


def test_temporary_occlusion_recovery_same_visit(tracking_config):
    """Test that disappearing and reappearing within timeout preserves the same visit without duplicate events."""
    emitted_events = []
    sm = VisitStateMachine(tracking_config, on_event=lambda e: emitted_events.append(e))
    dummy_crop = np.zeros((50, 50, 3), dtype=np.uint8)

    # Confirm track 1
    for _ in range(3):
        sm.update_track(track_id=1, face_id="FACE_0001", confidence=0.9, crop=dummy_crop)

    initial_visit_id = sm.tracks[1].visit_id

    # Missing for 3 frames (< 5 timeout)
    for _ in range(3):
        sm.update_misses(active_track_ids=set())
    assert sm.tracks[1].status == TrackStatus.LOST

    # Re-appears in frame 4
    events = sm.update_track(track_id=1, face_id="FACE_0001", confidence=0.88, crop=dummy_crop)
    assert len(events) == 0, "No new entry event should be emitted on re-acquisition"
    assert sm.tracks[1].status == TrackStatus.ACTIVE
    assert sm.tracks[1].visit_id == initial_visit_id, "Visit ID must remain the same"
    assert len(emitted_events) == 1, "Only the single initial ENTRY should exist"


def test_reentry_after_full_exit(tracking_config):
    """Test that a visitor returning after exit logs a new visit for the same Face ID."""
    emitted_events = []
    sm = VisitStateMachine(tracking_config, on_event=lambda e: emitted_events.append(e))
    dummy_crop = np.zeros((50, 50, 3), dtype=np.uint8)

    # Visit 1
    for _ in range(3):
        sm.update_track(track_id=1, face_id="FACE_0001", confidence=0.9, crop=dummy_crop)
    # Timeout visit 1
    for _ in range(6):
        sm.update_misses(active_track_ids=set())

    assert len(emitted_events) == 2  # Entry + Exit
    assert emitted_events[0].event_type == "entry"
    assert emitted_events[1].event_type == "exit"

    # Visit 2 (new spatial track_id 2, but recognized as same FACE_0001)
    for _ in range(3):
        sm.update_track(track_id=2, face_id="FACE_0001", confidence=0.91, crop=dummy_crop)

    assert len(emitted_events) == 3
    assert emitted_events[2].event_type == "entry"
    assert emitted_events[2].face_id == "FACE_0001"
    assert emitted_events[2].visit_id != emitted_events[0].visit_id


def test_flush_all_on_shutdown(tracking_config):
    """Test that flush_all emits EXIT for all remaining active tracks."""
    emitted_events = []
    sm = VisitStateMachine(tracking_config, on_event=lambda e: emitted_events.append(e))
    dummy_crop = np.zeros((50, 50, 3), dtype=np.uint8)

    for _ in range(3):
        sm.update_track(track_id=10, face_id="FACE_0010", confidence=0.85, crop=dummy_crop)
        sm.update_track(track_id=20, face_id="FACE_0020", confidence=0.87, crop=dummy_crop)

    assert len(emitted_events) == 2  # Two entries

    # Stream terminates abruptly -> flush
    flush_events = sm.flush_all()
    assert len(flush_events) == 2
    assert {e.face_id for e in flush_events} == {"FACE_0010", "FACE_0020"}
    assert all(e.event_type == "exit" for e in flush_events)
    assert len(sm.tracks) == 0
