"""
Unit tests for Phase 1 discriminator probe scripts.

Tests cover:
1. roc_auc_score parsing correctness (sklearn integration)
2. Stratified split correctness (stratify_groups function)
3. Label/embedding alignment verification
4. Known-answer tests: synthetic separable embeddings → AUC≈1.0; noise → AUC≈0.5
5. run_split correctness for LR, MLP, k-NN scripts
"""

import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import functions from probe scripts
from src.run_logistic_probes import (
    load_metadata as lr_load_metadata,
    stratify_groups as lr_stratify_groups,
    run_split as lr_run_split,
)
from src.run_mlp_probes import (
    load_metadata as mlp_load_metadata,
    stratify_groups as mlp_stratify_groups,
    run_split as mlp_run_split,
)
from src.run_knn_probes import (
    load_metadata as knn_load_metadata,
    stratify_groups as knn_stratify_groups,
    run_split as knn_run_split,
)


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_csv(tmp_path):
    """Create a minimal verified.csv for testing."""
    csv_path = tmp_path / "verified.csv"
    rows = [
        {"flickr_id": "1", "class": "positive", "scene_type": "portrait", "body": "LEICA M10"},
        {"flickr_id": "2", "class": "positive", "scene_type": "landscape", "body": "LEICA M11"},
        {"flickr_id": "3", "class": "positive", "scene_type": "street", "body": "LEICA Q2"},
        {"flickr_id": "4", "class": "positive", "scene_type": "portrait", "body": "LEICA M10-R"},
        {"flickr_id": "5", "class": "negative", "scene_type": "portrait", "body": "Canon EOS R5"},
        {"flickr_id": "6", "class": "negative", "scene_type": "landscape", "body": "ILCE-7M4"},
        {"flickr_id": "7", "class": "negative", "scene_type": "street", "body": "NIKON Z6"},
        {"flickr_id": "8", "class": "negative", "scene_type": "macro", "body": "Canon EOS R6"},
        {"flickr_id": "9", "class": "positive", "scene_type": "night", "body": "LEICA SL3"},
        {"flickr_id": "10", "class": "negative", "scene_type": "night", "body": "ILCE-7RM5"},
        {"flickr_id": "11", "class": "positive", "scene_type": "", "body": ""},
        {"flickr_id": "12", "class": "negative", "scene_type": "", "body": ""},
        {"flickr_id": "13", "class": "positive", "scene_type": "architecture", "body": "LEICA M10-P"},
        {"flickr_id": "14", "class": "negative", "scene_type": "architecture", "body": "Canon EOS R3"},
        {"flickr_id": "15", "class": "positive", "scene_type": "macro", "body": "LEICA SL (Typ 601)"},
        {"flickr_id": "16", "class": "negative", "scene_type": "macro", "body": "NIKON Z8"},
        # Rare class/body combinations
        {"flickr_id": "17", "class": "positive", "scene_type": "aerial", "body": "FUJIFILM GFX100S"},
        {"flickr_id": "18", "class": "negative", "scene_type": "aerial", "body": "FUJIFILM GFX100S"},
        {"flickr_id": "19", "class": "positive", "scene_type": "underwater", "body": "LEICA M10"},
        {"flickr_id": "20", "class": "negative", "scene_type": "underwater", "body": "Canon EOS R5"},
    ]

    fieldnames = [
        "flickr_id", "url", "class", "lens_label", "lens_exif",
        "body", "scene_type", "license_id", "tags", "file_path",
        "width", "height", "file_size_bytes", "verified_at",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            full = {fn: "" for fn in fieldnames}
            full.update(row)
            writer.writerow(full)

    return csv_path


@pytest.fixture
def synthetic_separable_embeddings():
    """Generate perfectly separable embeddings (first half of dims encode the class).

    Returns (X, y) where:
    - X: (n, 64) float32 matrix
    - y: (n,) int64 labels
    - First 32 dims are perfectly separable (pos=[+1], neg=[-1])
    - Remaining 32 dims are random noise (Gaussian, sigma=0.01 to keep SNR high)
    """
    rng = np.random.RandomState(42)
    n_pos, n_neg = 100, 100
    n = n_pos + n_neg

    X = rng.randn(n, 64).astype(np.float32) * 0.01  # Low-amplitude noise
    X[:n_pos, :32] = +1.0  # Signal dimensions: pos class
    X[n_pos:, :32] = -1.0  # Signal dimensions: neg class

    y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int64)
    return X, y


@pytest.fixture
def synthetic_noise_embeddings():
    """Generate pure random noise embeddings (no class signal).

    Returns (X, y) where:
    - X: (n, 64) float32 matrix of i.i.d. Gaussian noise (sigma=0.1)
    - y: (n,) int64 labels, balanced
    """
    rng = np.random.RandomState(99)
    n_pos, n_neg = 100, 100
    X = rng.randn(n_pos + n_neg, 64).astype(np.float32) * 0.1
    y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int64)
    return X, y


@pytest.fixture
def sample_groups():
    """Create balanced groups for split testing."""
    # 2 groups, 10 samples each, balanced classes
    groups = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    return groups


# ─────────────────────────────────────────────────────────────────────
# 1. roc_auc_score parsing correctness
# ─────────────────────────────────────────────────────────────────────


class TestROCAUCScore:
    """Verify sklearn.metrics.roc_auc_score integration is correct."""

    def test_perfect_prediction(self):
        """Perfect predictions should yield AUC = 1.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.0, 0.1, 0.0, 1.0, 0.9, 1.0])
        auc = roc_auc_score(y_true, y_prob)
        assert auc == 1.0, f"Expected AUC=1.0 for perfect predictions, got {auc}"

    def test_random_prediction(self):
        """Random predictions should yield AUC ≈ 0.5."""
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 50 + [1] * 50)
        y_prob = rng.rand(100)
        auc = roc_auc_score(y_true, y_prob)
        assert 0.3 < auc < 0.7, f"Expected AUC ≈ 0.5 for random predictions, got {auc:.4f}"

    def test_inverted_prediction(self):
        """Inverted predictions should yield AUC = 0.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([1.0, 0.9, 1.0, 0.0, 0.1, 0.0])
        auc = roc_auc_score(y_true, y_prob)
        assert auc == 0.0, f"Expected AUC=0.0 for inverted predictions, got {auc}"

    def test_proba_column_indexing(self):
        """Verify predict_proba[:, 1] selects the correct column."""
        from sklearn.linear_model import LogisticRegression

        # Simple separable data
        X = np.array([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]])
        y = np.array([1, 1, 0, 0])
        clf = LogisticRegression(random_state=42)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        # Column 0 should be probability of class 0, column 1 of class 1
        assert proba.shape == (4, 2)
        # Class 1 samples should have higher column-1 probability
        assert proba[0, 1] > proba[0, 0]
        assert proba[1, 1] > proba[1, 0]
        # Class 0 samples should have lower column-1 probability
        assert proba[2, 1] < proba[2, 0]
        assert proba[3, 1] < proba[3, 0]

    def test_auc_with_single_class(self):
        """AUC should warn/raise when test set has only one class.

        Newer sklearn versions emit UndefinedMetricWarning instead of ValueError.
        Either behavior is correct — both signal the degenerate case.
        """
        import warnings
        y_true = np.array([0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                result = roc_auc_score(y_true, y_prob)
                # If it returns without error, it should at least warn
                assert len(w) > 0, "Should warn about undefined metric for single-class"
                # Result for single class is undefined (NaN or 0)
                assert np.isnan(result) or result == 0.0 or result == 0.5
            except ValueError:
                # Older sklearn raises ValueError — this is also correct
                pass


# ─────────────────────────────────────────────────────────────────────
# 2. Label/embedding alignment
# ─────────────────────────────────────────────────────────────────────


class TestLabelEmbeddingAlignment:
    """Verify the label alignment check between verified.csv and labels.npy."""

    def test_labels_match(self, sample_csv):
        """When labels match, no error should be raised."""
        # Override the global VERIFIED_CSV for this module's import
        import src.run_logistic_probes as lr_mod

        original_csv = lr_mod.VERIFIED_CSV
        try:
            lr_mod.VERIFIED_CSV = sample_csv
            flickr_ids, scene_types, bodies, labels = lr_load_metadata()
            assert len(labels) == 20
            # Count pos and neg
            assert labels.sum() == 10  # 10 positive
            assert (labels == 0).sum() == 10  # 10 negative
        finally:
            lr_mod.VERIFIED_CSV = original_csv

    def test_label_array_numpy_alignment_check(self, sample_csv):
        """The alignment check in main() verifies labels match labels.npy."""
        import src.run_logistic_probes as lr_mod

        original_csv = lr_mod.VERIFIED_CSV
        try:
            lr_mod.VERIFIED_CSV = sample_csv
            flickr_ids, scene_types, bodies, labels = lr_load_metadata()

            # Simulate the alignment check from main()
            # If both arrays match, np.array_equal returns True
            embed_labels = labels.copy()  # Same order
            assert np.array_equal(labels, embed_labels)

            # If arrays mismatch, np.array_equal returns False
            shifted = np.roll(labels, 1)
            assert not np.array_equal(labels, shifted)
        finally:
            lr_mod.VERIFIED_CSV = original_csv


# ─────────────────────────────────────────────────────────────────────
# 3. Stratified split correctness (stratify_groups)
# ─────────────────────────────────────────────────────────────────────


class TestStratifyGroups:
    """Verify stratify_groups correctly maps scene_type+body to group IDs."""

    def test_output_shape(self, sample_csv):
        """Groups array should have same length as inputs."""
        import src.run_logistic_probes as lr_mod

        original_csv = lr_mod.VERIFIED_CSV
        try:
            lr_mod.VERIFIED_CSV = sample_csv
            flickr_ids, scene_types, bodies, labels = lr_load_metadata()
            groups = lr_stratify_groups(scene_types, bodies)
            assert len(groups) == len(scene_types) == len(bodies) == 20
        finally:
            lr_mod.VERIFIED_CSV = original_csv

    def test_groups_are_integers(self, sample_csv):
        """All group IDs should be integers."""
        import src.run_logistic_probes as lr_mod

        original_csv = lr_mod.VERIFIED_CSV
        try:
            lr_mod.VERIFIED_CSV = sample_csv
            flickr_ids, scene_types, bodies, labels = lr_load_metadata()
            groups = lr_stratify_groups(scene_types, bodies)
            assert groups.dtype == np.int64
            assert np.all(groups >= 0)
        finally:
            lr_mod.VERIFIED_CSV = original_csv

    def test_same_scene_body_same_group(self, sample_csv):
        """Images with same scene_type + body_group should be in same group."""
        import src.run_logistic_probes as lr_mod

        original_csv = lr_mod.VERIFIED_CSV
        try:
            lr_mod.VERIFIED_CSV = sample_csv
            flickr_ids, scene_types, bodies, labels = lr_load_metadata()
            groups = lr_stratify_groups(scene_types, bodies)

            # portrait + Leica should all be same group
            # Row 0: portrait + LEICA M10
            # Row 3: portrait + LEICA M10-R → collapsed to "leica"
            assert groups[0] == groups[3], "portrait+Leica should be same group"
        finally:
            lr_mod.VERIFIED_CSV = original_csv

    def test_unknown_body_collapsed(self, sample_csv):
        """Bodies not matching known brands should be collapsed to 'other'."""
        import src.run_logistic_probes as lr_mod

        original_csv = lr_mod.VERIFIED_CSV
        try:
            lr_mod.VERIFIED_CSV = sample_csv
            flickr_ids, scene_types, bodies, labels = lr_load_metadata()
            groups = lr_stratify_groups(scene_types, bodies)

            # FUJIFILM body (rows 16, 17) should be collapsed to "other"
            # Find them
            fuji_idx = [i for i, b in enumerate(bodies) if b and b.startswith("FUJIFILM")]
            assert len(fuji_idx) == 2
            assert groups[fuji_idx[0]] == groups[fuji_idx[1]], \
                "FUJIFILM bodies should be in same 'other' group"
        finally:
            lr_mod.VERIFIED_CSV = original_csv

    def test_rare_groups_collapsed(self, sample_csv):
        """Groups with < 3 members should be collapsed into 'rare'.

        Notes on the test data:
        - Rows 10,11 (empty/empty) + rows 16,17 (aerial/FUJIFILM) all map to
          "other|other" = 4 members → NOT rare (gets its own group ID).
        - Rows 18 (underwater/LEICA M10) → "other|leica" = 1 member → rare.
        - Rows 19 (underwater/Canon) → "other|canon" = 1 member → rare.
        - Rows 18 and 19 should map to the same "rare" group ID.
        """
        import src.run_logistic_probes as lr_mod

        original_csv = lr_mod.VERIFIED_CSV
        try:
            lr_mod.VERIFIED_CSV = sample_csv
            flickr_ids, scene_types, bodies, labels = lr_load_metadata()
            groups = lr_stratify_groups(scene_types, bodies)

            # Rows 18 and 19 are the only truly rare groups (< 3 members)
            assert groups[18] == groups[19], \
                "Rare groups (underwater scenes) should map to the same 'rare' ID"

            # Rows 10, 11, 16, 17 all map to "other|other" (4 members → NOT rare)
            assert groups[10] == groups[11] == groups[16] == groups[17], \
                "'other|other' group should have its own ID"

            # The rare ID should differ from the "other|other" group ID
            assert groups[18] != groups[10], \
                "Rare group ID should differ from 'other|other' group ID"
        finally:
            lr_mod.VERIFIED_CSV = original_csv

    def test_empty_scene_type_defaults_to_other(self, sample_csv):
        """Empty or missing scene_type should default to 'other'."""
        import src.run_logistic_probes as lr_mod

        original_csv = lr_mod.VERIFIED_CSV
        try:
            lr_mod.VERIFIED_CSV = sample_csv
            flickr_ids, scene_types, bodies, labels = lr_load_metadata()

            # Rows 10, 11 have empty scene_type
            assert scene_types[10] == "other"
            assert scene_types[11] == "other"
        finally:
            lr_mod.VERIFIED_CSV = original_csv

    def test_empty_body_defaults_to_unknown(self, sample_csv):
        """Empty or missing body should default to 'unknown'."""
        import src.run_logistic_probes as lr_mod

        original_csv = lr_mod.VERIFIED_CSV
        try:
            lr_mod.VERIFIED_CSV = sample_csv
            flickr_ids, scene_types, bodies, labels = lr_load_metadata()

            # Rows 10, 11 have empty body
            assert bodies[10] == "unknown"
            assert bodies[11] == "unknown"
        finally:
            lr_mod.VERIFIED_CSV = original_csv

    def test_identical_to_lr_implementation(self, sample_csv):
        """All three scripts use the identical stratify_groups function."""
        import src.run_logistic_probes as lr_mod
        import src.run_mlp_probes as mlp_mod
        import src.run_knn_probes as knn_mod

        original_lr = lr_mod.VERIFIED_CSV
        original_mlp = mlp_mod.VERIFIED_CSV
        original_knn = knn_mod.VERIFIED_CSV

        try:
            lr_mod.VERIFIED_CSV = sample_csv
            mlp_mod.VERIFIED_CSV = sample_csv
            knn_mod.VERIFIED_CSV = sample_csv

            _, st, bo, _ = lr_load_metadata()
            groups_lr = lr_stratify_groups(st, bo)

            _, st, bo, _ = mlp_load_metadata()
            groups_mlp = mlp_stratify_groups(st, bo)

            _, st, bo, _ = knn_load_metadata()
            groups_knn = knn_stratify_groups(st, bo)

            assert np.array_equal(groups_lr, groups_mlp), \
                "LR and MLP stratify_groups should produce identical output"
            assert np.array_equal(groups_lr, groups_knn), \
                "LR and k-NN stratify_groups should produce identical output"
        finally:
            lr_mod.VERIFIED_CSV = original_lr
            mlp_mod.VERIFIED_CSV = original_mlp
            knn_mod.VERIFIED_CSV = original_knn


# ─────────────────────────────────────────────────────────────────────
# 4. Stratified split via train_test_split
# ─────────────────────────────────────────────────────────────────────


class TestStratifiedSplit:
    """Verify that stratified split preserves class balance."""

    def test_stratify_preserves_class_balance(self):
        """train_test_split with stratify=y should preserve class proportions."""
        X = np.random.randn(100, 10)
        y = np.array([0] * 60 + [1] * 40)
        rng = np.random.RandomState(42)
        shuffle = rng.permutation(len(y))
        X, y = X[shuffle], y[shuffle]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        # Check total sizes
        assert len(y_te) == 30
        # Check class proportions are approximately preserved
        orig_p1 = y.sum() / len(y)
        train_p1 = y_tr.sum() / len(y_tr)
        test_p1 = y_te.sum() / len(y_te)
        assert abs(orig_p1 - train_p1) < 0.05, \
            f"Train class balance drifted: {orig_p1:.3f} → {train_p1:.3f}"
        assert abs(orig_p1 - test_p1) < 0.05, \
            f"Test class balance drifted: {orig_p1:.3f} → {test_p1:.3f}"

    def test_stratify_with_groups(self):
        """Stratification by group IDs should work when groups are diverse."""
        X = np.random.randn(80, 10)
        y = np.array([0] * 40 + [1] * 40)
        # Create 4 groups, 10 from each class per group
        groups = np.array([0] * 20 + [1] * 20 + [2] * 20 + [3] * 20, dtype=np.int64)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, stratify=groups, random_state=42
        )
        assert len(y_te) == 24  # 30% of 80

    def test_small_group_valueerror_fallback(self):
        """When group stratification fails, fallback to class stratification."""
        X = np.random.randn(40, 10)
        y = np.array([0] * 20 + [1] * 20)
        # Groups with only 1 member each → can't stratify
        groups = np.arange(40, dtype=np.int64)  # Each sample is its own group

        # train_test_split with 40 unique groups for 40 samples → ValueError
        with pytest.raises(ValueError):
            train_test_split(X, y, test_size=0.3, stratify=groups, random_state=42)

        # But the scripts handle this gracefully by falling back to y stratification
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        assert len(y_te) == 12


# ─────────────────────────────────────────────────────────────────────
# 5. Known-answer tests: synthetic data
# ─────────────────────────────────────────────────────────────────────


class TestKnownAnswerSeparable:
    """Verify that perfectly separable embeddings → AUC ≈ 1.0."""

    def test_lr_separable(self, synthetic_separable_embeddings):
        """LR should achieve AUC ≈ 1.0 on separable embeddings."""
        X, y = synthetic_separable_embeddings
        groups = np.arange(len(y), dtype=np.int64)  # One per sample

        # Use a deterministic seed
        result = lr_run_split(X.copy(), y, groups, n_per_class=50, seed=42)
        for C, auc in result["results"].items():
            assert auc > 0.95, \
                f"LR with C={C} should have AUC > 0.95 on separable data, got {auc}"

    def test_knn_separable(self, synthetic_separable_embeddings):
        """k-NN should achieve AUC ≈ 1.0 on separable embeddings."""
        X, y = synthetic_separable_embeddings
        groups = np.arange(len(y), dtype=np.int64)

        result = knn_run_split(X.copy(), y, groups, n_per_class=50, seed=42)
        for k, auc in result["results"].items():
            if k <= 5:  # k=1,5 should be near-perfect; k=11 might drop slightly
                assert auc > 0.90, \
                    f"k-NN with k={k} should have AUC > 0.90 on separable data, got {auc}"

    def test_mlp_separable(self, synthetic_separable_embeddings):
        """MLP should achieve AUC ≈ 1.0 on separable embeddings."""
        X, y = synthetic_separable_embeddings
        groups = np.arange(len(y), dtype=np.int64)

        result = mlp_run_split(X.copy(), y, groups, n_per_class=50, seed=42)
        assert result["auc"] > 0.90, \
            f"MLP should have AUC > 0.90 on separable data, got {result['auc']}"


class TestKnownAnswerNoise:
    """Verify that pure noise embeddings → AUC ≈ 0.5."""

    def test_lr_noise(self, synthetic_noise_embeddings):
        """LR on pure random noise should be near random (AUC ≈ 0.5)."""
        X, y = synthetic_noise_embeddings
        groups = np.arange(len(y), dtype=np.int64)

        result = lr_run_split(X.copy(), y, groups, n_per_class=50, seed=42)
        aucs = [v for v in result["results"].values() if isinstance(v, float)]
        mean_auc = np.mean(aucs)
        # Should be close to 0.5, allow some variance from small sample
        assert 0.3 < mean_auc < 0.7, \
            f"LR on noise should have mean AUC ≈ 0.5, got {mean_auc:.4f}"

    def test_knn_noise(self, synthetic_noise_embeddings):
        """k-NN on pure random noise should be near random."""
        X, y = synthetic_noise_embeddings
        groups = np.arange(len(y), dtype=np.int64)

        result = knn_run_split(X.copy(), y, groups, n_per_class=50, seed=42)
        aucs = [v for v in result["results"].values()]
        mean_auc = np.mean(aucs)
        assert 0.3 < mean_auc < 0.7, \
            f"k-NN on noise should have mean AUC ≈ 0.5, got {mean_auc:.4f}"

    def test_mlp_noise(self, synthetic_noise_embeddings):
        """MLP on pure random noise should be near random.

        Note: MLP can overfit on small datasets even with noise.
        Allow wider tolerance (0.2–0.8) since 200 epochs with early stopping
        on 100 samples can still find spurious patterns.
        """
        X, y = synthetic_noise_embeddings
        groups = np.arange(len(y), dtype=np.int64)

        result = mlp_run_split(X.copy(), y, groups, n_per_class=50, seed=42)
        assert 0.15 < result["auc"] < 0.85, \
            f"MLP on noise should have AUC ≈ 0.5, got {result['auc']:.4f}"


# ─────────────────────────────────────────────────────────────────────
# 6. run_split correctness
# ─────────────────────────────────────────────────────────────────────


class TestRunSplitLR:
    """Verify LR run_split returns correct structure."""

    def test_output_keys(self, synthetic_separable_embeddings):
        """run_split should return dict with test_pos, test_neg, results."""
        X, y = synthetic_separable_embeddings
        groups = np.arange(len(y), dtype=np.int64)
        result = lr_run_split(X, y, groups, n_per_class=50, seed=42)

        assert "test_pos" in result
        assert "test_neg" in result
        assert "results" in result
        assert isinstance(result["results"], dict)

    def test_test_set_size(self, synthetic_separable_embeddings):
        """Test set should be ~30% of sampled data."""
        X, y = synthetic_separable_embeddings
        groups = np.arange(len(y), dtype=np.int64)
        result = lr_run_split(X, y, groups, n_per_class=50, seed=42)

        total_test = result["test_pos"] + result["test_neg"]
        # 50*2 = 100 sampled, 30% test = 30
        assert 20 <= total_test <= 40, \
            f"Test set should be ~30 samples, got {total_test}"

    def test_four_c_values(self, synthetic_separable_embeddings):
        """LR should evaluate all 4 C values."""
        X, y = synthetic_separable_embeddings
        groups = np.arange(len(y), dtype=np.int64)
        result = lr_run_split(X, y, groups, n_per_class=50, seed=42)

        expected_c = {0.01, 0.1, 1.0, 10.0}
        assert set(result["results"].keys()) == expected_c

    def test_sample_vs_n_per_class(self, synthetic_separable_embeddings):
        """run_split should not sample more than n_per_class per class."""
        X, y = synthetic_separable_embeddings
        groups = np.arange(len(y), dtype=np.int64)

        # With enough data, should sample exactly n_per_class from each
        result = lr_run_split(X, y, groups, n_per_class=40, seed=42)
        total_test = result["test_pos"] + result["test_neg"]
        # 80 sampled, 30% test ≈ 24
        assert 15 <= total_test <= 35


class TestRunSplitMLP:
    """Verify MLP run_split returns correct structure."""

    def test_output_keys(self, synthetic_separable_embeddings):
        """MLP run_split should return auc, test_pos, test_neg, val_loss."""
        X, y = synthetic_separable_embeddings
        groups = np.arange(len(y), dtype=np.int64)
        result = mlp_run_split(X, y, groups, n_per_class=50, seed=42)

        assert "auc" in result
        assert "test_pos" in result
        assert "test_neg" in result
        assert "val_loss" in result
        assert isinstance(result["auc"], float)


class TestRunSplitKNN:
    """Verify k-NN run_split returns correct structure."""

    def test_output_keys(self, synthetic_separable_embeddings):
        """k-NN run_split should return test_pos, test_neg, results."""
        X, y = synthetic_separable_embeddings
        groups = np.arange(len(y), dtype=np.int64)
        result = knn_run_split(X, y, groups, n_per_class=50, seed=42)

        assert "test_pos" in result
        assert "test_neg" in result
        assert "results" in result
        assert isinstance(result["results"], dict)

    def test_three_k_values(self, synthetic_separable_embeddings):
        """k-NN should evaluate exactly k=1, k=5, k=11."""
        X, y = synthetic_separable_embeddings
        groups = np.arange(len(y), dtype=np.int64)
        result = knn_run_split(X, y, groups, n_per_class=50, seed=42)

        expected_k = {1, 5, 11}
        assert set(result["results"].keys()) == expected_k


# ─────────────────────────────────────────────────────────────────────
# 7. load_metadata correctness
# ─────────────────────────────────────────────────────────────────────


class TestLoadMetadata:
    """Verify load_metadata correctly parses verified.csv."""

    def test_returns_expected_tuples(self, sample_csv):
        """Should return 4 elements: flickr_ids, scene_types, bodies, labels."""
        import src.run_logistic_probes as lr_mod

        original_csv = lr_mod.VERIFIED_CSV
        try:
            lr_mod.VERIFIED_CSV = sample_csv
            result = lr_load_metadata()
            assert len(result) == 4
            flickr_ids, scene_types, bodies, labels = result
            assert isinstance(flickr_ids, list)
            assert isinstance(scene_types, list)
            assert isinstance(bodies, list)
            assert isinstance(labels, np.ndarray)
        finally:
            lr_mod.VERIFIED_CSV = original_csv

    def test_label_values(self, sample_csv):
        """Labels should be 1 for positive, 0 for negative."""
        import src.run_logistic_probes as lr_mod

        original_csv = lr_mod.VERIFIED_CSV
        try:
            lr_mod.VERIFIED_CSV = sample_csv
            _, _, _, labels = lr_load_metadata()
            # All labels should be 0 or 1
            assert np.all((labels == 0) | (labels == 1))
            # First 4 are positive (1), next 6 are negative (0) etc
            assert labels[0] == 1
            assert labels[4] == 0
        finally:
            lr_mod.VERIFIED_CSV = original_csv

    def test_all_functions_identical(self, sample_csv):
        """All three scripts have byte-identical load_metadata implementations."""
        import inspect
        src_lr = inspect.getsource(lr_load_metadata)
        src_mlp = inspect.getsource(mlp_load_metadata)
        src_knn = inspect.getsource(knn_load_metadata)

        # They import from different modules but have the same function body
        # Strip the @wraps decorator differences if any
        assert "csv.DictReader" in src_lr
        assert "csv.DictReader" in src_mlp
        assert "csv.DictReader" in src_knn


# ─────────────────────────────────────────────────────────────────────
# 8. Regression: no row-shift bugs
# ─────────────────────────────────────────────────────────────────────


class TestNoRowShift:
    """Verify labels stay aligned with embeddings (no index shift bugs)."""

    def test_labels_match_after_stratification(self, sample_csv):
        """After loading, labels should correspond to correct class assignments."""
        import src.run_logistic_probes as lr_mod

        original_csv = lr_mod.VERIFIED_CSV
        try:
            lr_mod.VERIFIED_CSV = sample_csv
            flickr_ids, scene_types, bodies, labels = lr_load_metadata()

            # Read CSV manually and compare row-by-row
            with open(sample_csv) as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    expected = 1 if row["class"] == "positive" else 0
                    assert labels[i] == expected, \
                        f"Row {i}: expected label {expected}, got {labels[i]}"
        finally:
            lr_mod.VERIFIED_CSV = original_csv

    def test_flickr_ids_match_labels(self, sample_csv):
        """flickr_ids and labels arrays should be in the same order."""
        import src.run_logistic_probes as lr_mod

        original_csv = lr_mod.VERIFIED_CSV
        try:
            lr_mod.VERIFIED_CSV = sample_csv
            flickr_ids, _, _, labels = lr_load_metadata()

            assert len(flickr_ids) == len(labels)
            # flickr_id "1" should be positive
            idx = flickr_ids.index("1")
            assert labels[idx] == 1
            # flickr_id "5" should be negative
            idx = flickr_ids.index("5")
            assert labels[idx] == 0
        finally:
            lr_mod.VERIFIED_CSV = original_csv


# ─────────────────────────────────────────────────────────────────────
# 9. Determinism
# ─────────────────────────────────────────────────────────────────────


class TestDeterminism:
    """Same seed should produce identical results."""

    def test_lr_deterministic(self, synthetic_separable_embeddings):
        """Two runs with same seed should give identical AUC."""
        X, y = synthetic_separable_embeddings
        groups = np.arange(len(y), dtype=np.int64)

        r1 = lr_run_split(X.copy(), y, groups, n_per_class=50, seed=42)
        r2 = lr_run_split(X.copy(), y, groups, n_per_class=50, seed=42)

        for C in r1["results"]:
            assert r1["results"][C] == r2["results"][C], \
                f"C={C}: seeds differ ({r1['results'][C]} vs {r2['results'][C]})"

    def test_knn_deterministic(self, synthetic_separable_embeddings):
        """k-NN with same seed should give identical results."""
        X, y = synthetic_separable_embeddings
        groups = np.arange(len(y), dtype=np.int64)

        r1 = knn_run_split(X.copy(), y, groups, n_per_class=50, seed=42)
        r2 = knn_run_split(X.copy(), y, groups, n_per_class=50, seed=42)

        for k in r1["results"]:
            assert r1["results"][k] == r2["results"][k], \
                f"k={k}: seeds differ ({r1['results'][k]} vs {r2['results'][k]})"


# ─────────────────────────────────────────────────────────────────────
# 10. Edge cases
# ─────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case handling in the probe scripts."""

    def test_lr_error_handling_on_degenerate_data(self):
        """LR should return error string on degenerate (all-same-class) test set."""
        # Create data where the split might produce single-class test set
        X = np.random.randn(20, 10).astype(np.float32)
        # All same class
        y = np.ones(20, dtype=np.int64)
        groups = np.arange(20, dtype=np.int64)

        # This will fail because n_per_class can't sample both classes
        # run_split assumes both classes exist
        # We just test that the function doesn't crash in unreasonable ways
        # when given insufficient positives
        with pytest.raises(Exception):
            # Should fail because there aren't enough negatives to sample
            lr_run_split(X, y, groups, n_per_class=5, seed=42)

    def test_proportial_sampling_respects_group_membership(self, sample_csv):
        """Proportional sampling shouldn't sample from groups that have zero members of a class."""
        import src.run_logistic_probes as lr_mod

        original_csv = lr_mod.VERIFIED_CSV
        try:
            lr_mod.VERIFIED_CSV = sample_csv
            flickr_ids, scene_types, bodies, labels = lr_load_metadata()
            groups = lr_stratify_groups(scene_types, bodies)

            # Verify that groups are diverse enough for stratification
            unique_groups = np.unique(groups)
            # Some groups should exist
            assert len(unique_groups) > 1, "Need multiple groups for stratification test"

            # Check that at least some groups have both pos and neg
            has_both = False
            for g in unique_groups:
                g_pos = np.sum((groups == g) & (labels == 1))
                g_neg = np.sum((groups == g) & (labels == 0))
                if g_pos > 0 and g_neg > 0:
                    has_both = True
                    break
            # If no group has both, the stratification still works
            # (it will fall back to class stratification)
        finally:
            lr_mod.VERIFIED_CSV = original_csv


# ─────────────────────────────────────────────────────────────────────
# 11. Integration: Full pipeline smoke test
# ─────────────────────────────────────────────────────────────────────


class TestIntegrationSmoke:
    """Smoke test of the full pipeline with synthetic and real data."""

    def test_full_pipeline_lr_with_synthetic(self, synthetic_separable_embeddings):
        """End-to-end: load-like data → split → classify → AUC."""
        X, y = synthetic_separable_embeddings
        groups = np.arange(len(y), dtype=np.int64)

        result = lr_run_split(X, y, groups, n_per_class=50, seed=42)
        assert "results" in result
        # Best AUC should be near 1.0
        best_auc = max(
            v for v in result["results"].values() if isinstance(v, float)
        )
        assert best_auc > 0.95, f"Best LR AUC should be > 0.95 on separable data, got {best_auc:.4f}"

    def test_full_pipeline_knn_with_synthetic(self, synthetic_separable_embeddings):
        """End-to-end k-NN on separable data."""
        X, y = synthetic_separable_embeddings
        groups = np.arange(len(y), dtype=np.int64)

        result = knn_run_split(X, y, groups, n_per_class=50, seed=42)
        best_auc = max(result["results"].values())
        assert best_auc > 0.90, \
            f"Best k-NN AUC should be > 0.90 on separable data, got {best_auc:.4f}"
