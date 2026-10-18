"""Unit tests for the IdentityManager (one-shot auto-registration, re-ID, exemplar bank, collision prevention)."""

import pytest
import numpy as np

from app.config import RecognitionConfig
from app.identity_manager import IdentityManager


@pytest.fixture
def recognition_config():
    return RecognitionConfig(
        model_name="buffalo_l",
        recognition_threshold=0.5,
        max_embeddings_per_face=3,
        reverify_every_n_frames=30
    )


def create_random_embedding(seed: int = 42) -> np.ndarray:
    """Generate a reproducible normalized 512-d unit vector."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(512).astype(np.float32)
    return vec / np.linalg.norm(vec)


def test_one_shot_auto_registration(recognition_config):
    """Test that a new visitor is automatically registered with incremented ID."""
    im = IdentityManager(recognition_config)
    assert im.unique_visitor_count == 0

    emb1 = create_random_embedding(seed=100)
    face_id1, score1, is_new1 = im.match_or_register(emb1, quality_score=75.0, track_id=1)

    assert is_new1 is True
    assert face_id1 == "FACE_0001"
    assert im.unique_visitor_count == 1

    # Second distinct person (low similarity to emb1)
    emb2 = create_random_embedding(seed=200)
    face_id2, score2, is_new2 = im.match_or_register(emb2, quality_score=80.0, track_id=2)

    assert is_new2 is True
    assert face_id2 == "FACE_0002"
    assert im.unique_visitor_count == 2


def test_reidentification_does_not_increment_unique_count(recognition_config):
    """Test that recognizing an existing visitor matches their ID and keeps unique count constant."""
    im = IdentityManager(recognition_config)
    emb_base = create_random_embedding(seed=300)

    # Register visitor initially
    fid, _, is_new = im.match_or_register(emb_base, quality_score=70.0, track_id=1)
    assert is_new is True
    assert fid == "FACE_0001"
    assert im.unique_visitor_count == 1

    # Release track 1 (visitor walked out of frame)
    im.release_track(1)

    # Returning visitor with slight noise (high cosine similarity ~0.95+)
    noise = np.random.RandomState(42).randn(512).astype(np.float32) * 0.005
    noisy_emb = emb_base + noise
    noisy_emb = noisy_emb / np.linalg.norm(noisy_emb)

    matched_id, score, is_new_again = im.match_or_register(noisy_emb, quality_score=85.0, track_id=5)

    assert is_new_again is False
    assert matched_id == "FACE_0001"
    assert score >= 0.90
    assert im.unique_visitor_count == 1, "Re-identification must NEVER increment unique visitor count"


def test_multi_embedding_exemplar_bank_cap(recognition_config):
    """Test that exemplar bank caps at max_embeddings_per_face and prioritizes high quality."""
    im = IdentityManager(recognition_config)
    base_emb = create_random_embedding(seed=400)

    im.match_or_register(base_emb, quality_score=50.0, track_id=1)
    im.release_track(1)

    # Add 2 more embeddings to reach cap (max=3)
    im.match_or_register(base_emb, quality_score=60.0, track_id=1)
    im.release_track(1)
    im.match_or_register(base_emb, quality_score=70.0, track_id=1)
    im.release_track(1)

    ident = im.identities["FACE_0001"]
    assert len(ident.embeddings) == 3
    assert set(ident.quality_scores) == {50.0, 60.0, 70.0}

    # Add a 4th embedding with higher quality (85.0) -> should replace lowest (50.0)
    im.match_or_register(base_emb, quality_score=85.0, track_id=1)
    assert len(ident.embeddings) == 3
    assert 50.0 not in ident.quality_scores
    assert 85.0 in ident.quality_scores


def test_collision_prevention_mutual_exclusion(recognition_config):
    """Test that two simultaneous tracks in the same frame cannot be assigned the same Face ID."""
    im = IdentityManager(recognition_config)
    emb_alice = create_random_embedding(seed=500)

    # Track 1 assigned to Alice
    fid1, _, _ = im.match_or_register(emb_alice, quality_score=90.0, track_id=1)
    assert fid1 == "FACE_0001"

    # Track 2 presents a very similar embedding (e.g. crossing paths or twin)
    # Because Track 1 currently holds FACE_0001, Track 2 MUST NOT steal or share FACE_0001
    fid2, _, is_new2 = im.match_or_register(emb_alice, quality_score=80.0, track_id=2)
    assert fid2 != "FACE_0001"
    assert fid2 == "FACE_0002"
    assert is_new2 is True
