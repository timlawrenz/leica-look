#!/usr/bin/env python3
"""
Attention-map analysis for leica-look Phase 1.5 (Issue #12).
Extract DINOv2 CLS-to-patch attention maps and analyze radial profiles
to determine center-weighted (content) vs edge-weighted (lens) signal.

Strategy:
1. Load DINOv2-S/14 (fast, AUC 0.93) embeddings + run 5-fold CV LR for per-image predictions
2. Select ~50 correctly classified + ~50 misclassified images
3. Load DINOv2-S/14 model with output_attentions=True
4. Extract last-layer CLS-to-patch attention for selected images
5. Compute radial attention profiles (attention vs distance from center)
6. Compare center/edge weighting between correct and incorrect classifications
7. Optionally validate with DINOv2-g/14

Output: experiments/discriminator-attention/
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

os.environ["HF_HOME"] = "/mnt/nas-ai-models/huggingface-cache"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # leica-look/
VERIFIED_CSV = REPO_ROOT / "data/registry/verified.csv"
EMBEDDINGS_DIR = Path("/mnt/nas-ai-models/training-data/leica-look/embeddings")
OUTPUT_DIR = REPO_ROOT / "experiments/discriminator-attention"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "dinov2-small"  # Primary: fast (AUC 0.93, 21M params)
MODEL_REPO = "facebook/dinov2-small"
IMG_SIZE = 224
N_CORRECT_TARGET = 50
N_INCORRECT_TARGET = 50
N_CV_FOLDS = 5
RANDOM_SEED = 42
C_VALUE = 1.0
ATTN_LAYER = -1  # Last transformer layer
BATCH_SIZE = 4    # Attention extraction batch size

# For validation sweep
VALIDATION_MODEL_NAME = "dinov2-giant"
VALIDATION_MODEL_REPO = "facebook/dinov2-giant"
VALIDATION_SAMPLE = 20  # Subset for validation (giant is slow)

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_metadata():
    """Load verified.csv. Returns (ids, scene_types, bodies, labels, file_paths)."""
    flickr_ids, scene_types, bodies, labels, file_paths = [], [], [], [], []
    with open(VERIFIED_CSV, "r") as f:
        for row in csv.DictReader(f):
            flickr_ids.append(row["flickr_id"])
            scene_types.append(row["scene_type"] or "other")
            bodies.append(row["body"] or "unknown")
            labels.append(1 if row["class"] == "positive" else 0)
            file_paths.append(row["file_path"])
    return flickr_ids, scene_types, bodies, np.array(labels, dtype=np.int64), file_paths


def get_model(repo_name: str):
    """Load a DINOv2 model with attention output support.
    
    Must set config.output_attentions=True BEFORE loading,
    then pass output_attentions=True in the forward call.
    """
    from transformers import AutoModel, AutoConfig
    config = AutoConfig.from_pretrained(
        repo_name, cache_dir="/mnt/nas-ai-models/huggingface-cache",
        trust_remote_code=True
    )
    config.output_attentions = True
    model = AutoModel.from_pretrained(
        repo_name, config=config,
        cache_dir="/mnt/nas-ai-models/huggingface-cache",
        trust_remote_code=True
    )
    model.eval()
    model.to(DEVICE)
    return model


def preprocess_image(path: str, img_size: int = 224) -> torch.Tensor:
    """
    Identical preprocessing to extract_embeddings.py:
    resize shortest→img_size, center crop, ImageNet normalize.
    Returns (3, H, W) tensor.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = img_size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - img_size) // 2
    top = (new_h - img_size) // 2
    img = img.crop((left, top, left + img_size, top + img_size))

    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1)


def extract_attention_maps(model, image_paths: list, batch_size: int = 4):
    """
    Extract last-layer CLS-to-patch attention for a list of images.
    Returns (N, N_patches) numpy array.
    """
    all_attn_maps = []
    n_total = len(image_paths)
    n_batches = (n_total + batch_size - 1) // batch_size

    pbar = tqdm(total=n_total, desc="  Attention extraction", unit="img")
    for i in range(0, n_total, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        actual_bs = len(batch_paths)

        # Preprocess batch
        batch_tensors = []
        valid_indices = []
        for bj, p in enumerate(batch_paths):
            try:
                x = preprocess_image(p, IMG_SIZE)
                batch_tensors.append(x)
                valid_indices.append(bj)
            except Exception as e:
                print(f"\n  ⚠️  Failed to load {p}: {e}")
                all_attn_maps.append(np.zeros(1))  # placeholder

        if not batch_tensors:
            pbar.update(actual_bs)
            continue

        x = torch.stack(batch_tensors, dim=0).to(DEVICE)

        with torch.no_grad():
            try:
                out = model(x, output_attentions=True)
            except Exception as e:
                print(f"\n  ⚠️  Model forward failed: {e}")
                for _ in range(actual_bs):
                    all_attn_maps.append(np.zeros(1))
                pbar.update(actual_bs)
                continue

        # out.attentions: tuple of (B, num_heads, N+1, N+1) per layer
        if not hasattr(out, "attentions") or out.attentions is None:
            print("\n  ⚠️  Model returned no attention maps! Falling back to zeros.")
            for _ in range(actual_bs):
                all_attn_maps.append(np.zeros(1))
            pbar.update(actual_bs)
            continue

        last_attn = out.attentions[ATTN_LAYER]  # (B, num_heads, N+1, N+1)

        # Average across heads
        avg_attn = last_attn.mean(dim=1)  # (B, N+1, N+1)

        # CLS-to-patch attention: row 0, columns 1..N
        cls_to_patch = avg_attn[:, 0, 1:]  # (B, N_patches)

        # Map back to original batch order
        result_map = {}
        for bj_out, bj_orig in enumerate(valid_indices):
            result_map[bj_orig] = cls_to_patch[bj_out].cpu().numpy()

        for bj in range(actual_bs):
            if bj in result_map:
                all_attn_maps.append(result_map[bj])
            # else: placeholder already added above

        pbar.update(actual_bs)

    pbar.close()

    # Filter out placeholders (zeros with shape (1,) from failed loads)
    valid_maps = [m for m in all_attn_maps if m.shape != (1,) or m[0] != 0]
    # But we need to maintain alignment with selected_idx...

    return np.array(all_attn_maps)


def compute_radial_profile(attn_map_1d: np.ndarray, grid_size: int) -> dict:
    """
    Compute average attention vs distance from image center.
    Returns dict with bin_centers, bin_values, center_weight, edge_weight, center_edge_ratio.
    """
    # Skip invalid maps
    if attn_map_1d.shape == (1,) and attn_map_1d[0] == 0:
        return {
            "bin_centers": [0.0] * 10,
            "bin_values": [0.0] * 10,
            "center_weight": 0.0,
            "edge_weight": 0.0,
            "center_edge_ratio": 0.0,
            "valid": False,
        }

    n_patches = len(attn_map_1d)
    actual_grid = int(np.sqrt(n_patches))
    attn_2d = attn_map_1d.reshape(actual_grid, actual_grid)
    h, w = attn_2d.shape
    center_y, center_x = (h - 1) / 2.0, (w - 1) / 2.0

    # Distance grid
    ys, xs = np.mgrid[0:h, 0:w]
    distances = np.sqrt((ys - center_y) ** 2 + (xs - center_x) ** 2)
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)

    # Radial bins
    n_bins = 10
    bin_edges = np.linspace(0, max_dist * 1.001, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_values = []
    for bi in range(n_bins):
        mask = (distances >= bin_edges[bi]) & (distances < bin_edges[bi + 1])
        if mask.sum() > 0:
            bin_values.append(float(attn_2d[mask].mean()))
        else:
            bin_values.append(0.0)

    # Center: 3x3 central region
    cy, cx = int(round(center_y)), int(round(center_x))
    r = 1  # ±1 = 3x3
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    center_val = float(attn_2d[y0:y1, x0:x1].mean())

    # Edge: outermost ring (perimeter patches)
    edge_mask = np.zeros_like(attn_2d, dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    edge_val = float(attn_2d[edge_mask].mean())

    ratio = center_val / (edge_val + 1e-10)

    return {
        "bin_centers": bin_centers.tolist(),
        "bin_values": bin_values,
        "center_weight": center_val,
        "edge_weight": edge_val,
        "center_edge_ratio": ratio,
        "valid": True,
    }


def analyze_attention(model_name, model_repo, embedding_pooling, selected_paths,
                      selected_idx, labels, predictions, flickr_ids, scene_types,
                      sample_size=None):
    """Run attention extraction and radial profile analysis for one model."""
    print(f"\n{'─'*60}")
    print(f"Model: {model_name} ({model_repo})")
    print(f"Embedding pooling: {embedding_pooling}")

    if sample_size and sample_size < len(selected_paths):
        # Stratified subsample
        rng = np.random.RandomState(RANDOM_SEED)
        correct_mask = predictions[selected_idx] == labels[selected_idx]
        correct_sub = np.where(correct_mask)[0]
        incorrect_sub = np.where(~correct_mask)[0]
        n_each = sample_size // 2
        sub_correct = rng.choice(correct_sub, size=min(n_each, len(correct_sub)), replace=False)
        sub_incorrect = rng.choice(incorrect_sub, size=min(n_each, len(incorrect_sub)), replace=False)
        sub_indices = np.concatenate([sub_correct, sub_incorrect])
        paths = [selected_paths[i] for i in sub_indices]
        local_idx = selected_idx[sub_indices]
    else:
        paths = selected_paths
        local_idx = selected_idx

    # Load model
    print(f"  Loading model...")
    model = get_model(model_repo)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params/1e6:.0f}M")

    # Extract attention maps
    t0 = time.time()
    attn_maps = extract_attention_maps(model, paths, batch_size=BATCH_SIZE)
    elapsed = time.time() - t0
    print(f"  Extracted {len(attn_maps)} attention maps in {elapsed:.1f}s "
          f"({elapsed/len(attn_maps):.2f}s/img)")

    # Determine grid
    valid_mask = np.array([m.shape != (1,) or m[0] != 0 for m in attn_maps])
    n_valid = valid_mask.sum()
    if n_valid == 0:
        print("  ❌ No valid attention maps extracted!")
        del model
        torch.cuda.empty_cache()
        return None

    n_patches = attn_maps[valid_mask][0].shape[0]
    grid_size = int(np.sqrt(n_patches))
    print(f"  Grid: {grid_size}×{grid_size} ({n_patches} patches)")

    # Compute radial profiles
    radial_profiles = []
    for i in range(len(attn_maps)):
        profile = compute_radial_profile(attn_maps[i], grid_size)
        profile["index"] = int(local_idx[i])
        profile["flickr_id"] = flickr_ids[local_idx[i]]
        profile["true_label"] = int(labels[local_idx[i]])
        profile["pred_label"] = int(predictions[local_idx[i]])
        profile["correct"] = bool(predictions[local_idx[i]] == labels[local_idx[i]])
        profile["scene_type"] = scene_types[local_idx[i]]
        radial_profiles.append(profile)

    del model
    torch.cuda.empty_cache()

    # Separate correct vs incorrect
    correct_profiles = [p for p in radial_profiles if p["correct"] and p.get("valid", True)]
    incorrect_profiles = [p for p in radial_profiles if not p["correct"] and p.get("valid", True)]

    n_correct = len(correct_profiles)
    n_incorrect = len(incorrect_profiles)

    # Aggregate radial profiles
    n_bins = 10
    correct_bin_means = np.zeros(n_bins)
    incorrect_bin_means = np.zeros(n_bins)

    for p in correct_profiles:
        correct_bin_means += np.array(p["bin_values"])
    if n_correct > 0:
        correct_bin_means /= n_correct

    for p in incorrect_profiles:
        incorrect_bin_means += np.array(p["bin_values"])
    if n_incorrect > 0:
        incorrect_bin_means /= n_incorrect

    # Center/edge stats
    center_correct = [p["center_weight"] for p in correct_profiles]
    edge_correct = [p["edge_weight"] for p in correct_profiles]
    center_incorrect = [p["center_weight"] for p in incorrect_profiles]
    edge_incorrect = [p["edge_weight"] for p in incorrect_profiles]

    ratios_correct = [p["center_edge_ratio"] for p in correct_profiles]
    ratios_incorrect = [p["center_edge_ratio"] for p in incorrect_profiles]

    # Also: overall mean attention map (spatial)
    if n_correct > 0:
        overall_correct = np.zeros(n_patches)
        for p in correct_profiles:
            if p.get("valid", True):
                overall_correct += attn_maps[radial_profiles.index(p)]
        overall_correct /= n_correct
    else:
        overall_correct = np.zeros(n_patches)

    if n_incorrect > 0:
        overall_incorrect = np.zeros(n_patches)
        for p in incorrect_profiles:
            if p.get("valid", True):
                overall_incorrect += attn_maps[radial_profiles.index(p)]
        overall_incorrect /= n_incorrect
    else:
        overall_incorrect = np.zeros(n_patches)

    # Difference map
    diff_map = overall_correct - overall_incorrect

    return {
        "model_name": model_name,
        "model_repo": model_repo,
        "n_params": int(n_params),
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "grid_size": grid_size,
        "n_patches": n_patches,
        "radial": {
            "bin_centers": correct_bin_means.tolist() if n_correct > 0 else [],
            "correct_mean_bin_values": correct_bin_means.tolist(),
            "incorrect_mean_bin_values": incorrect_bin_means.tolist(),
        },
        "center_edge_stats": {
            "correct": {
                "mean_center": float(np.mean(center_correct)) if center_correct else 0,
                "mean_edge": float(np.mean(edge_correct)) if edge_correct else 0,
                "mean_ratio": float(np.mean(ratios_correct)) if ratios_correct else 0,
                "std_ratio": float(np.std(ratios_correct)) if ratios_correct else 0,
            },
            "incorrect": {
                "mean_center": float(np.mean(center_incorrect)) if center_incorrect else 0,
                "mean_edge": float(np.mean(edge_incorrect)) if edge_incorrect else 0,
                "mean_ratio": float(np.mean(ratios_incorrect)) if ratios_incorrect else 0,
                "std_ratio": float(np.std(ratios_incorrect)) if ratios_incorrect else 0,
            },
        },
        "overall_maps": {
            "correct_mean": overall_correct.tolist(),
            "incorrect_mean": overall_incorrect.tolist(),
            "difference": diff_map.tolist(),
        },
        "per_image": radial_profiles,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Attention-Map Analysis — leica-look Issue #12")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Primary model: {MODEL_NAME} ({MODEL_REPO})")

    # 1. Load metadata
    flickr_ids, scene_types, bodies, labels, file_paths = load_metadata()
    n_total = len(labels)
    n_pos = int(labels.sum())
    n_neg = int((labels == 0).sum())
    print(f"\nDataset: {n_total} images ({n_pos} Leica, {n_neg} non-Leica)")

    # 2. Load embeddings and run CV classification
    emb_pooling = "cls_patch"
    emb_path = EMBEDDINGS_DIR / MODEL_NAME / f"{emb_pooling}.npy"
    if not emb_path.exists():
        print(f"❌ Embeddings not found: {emb_path}")
        sys.exit(1)

    X = np.load(emb_path).astype(np.float32)
    print(f"Embeddings: {X.shape} ({emb_pooling})")

    print(f"\nRunning {N_CV_FOLDS}-fold CV (LogisticRegression, C={C_VALUE})...")
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    all_preds = np.zeros(n_total)
    fold_aucs = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, labels)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(
            C=C_VALUE, penalty="l2", solver="lbfgs",
            max_iter=5000, random_state=RANDOM_SEED, class_weight="balanced"
        )
        clf.fit(X_train, y_train)
        all_preds[test_idx] = clf.predict_proba(X_test)[:, 1]

        fold_auc = roc_auc_score(y_test, all_preds[test_idx])
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold+1}: AUC={fold_auc:.4f}, n_test={len(test_idx)}")

    full_auc = roc_auc_score(labels, all_preds)
    print(f"Full CV AUC: {full_auc:.4f} (mean fold AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f})")

    # 3. Classify predictions
    predictions = (all_preds >= 0.5).astype(np.int64)
    correct = predictions == labels
    incorrect = ~correct
    correct_confidences = np.abs(all_preds - 0.5)  # distance from decision boundary

    print(f"\nClassification results:")
    print(f"  Correct: {correct.sum()} / {n_total} ({100*correct.sum()/n_total:.1f}%)")
    print(f"  Incorrect: {incorrect.sum()} / {n_total} ({100*incorrect.sum()/n_total:.1f}%)")

    # 4. Select images for attention analysis
    correct_idx = np.where(correct)[0]
    incorrect_idx = np.where(incorrect)[0]

    # Sort by confidence (most confident first)
    correct_sorted = correct_idx[np.argsort(-correct_confidences[correct_idx])]
    incorrect_sorted = incorrect_idx[np.argsort(-correct_confidences[incorrect_idx])]

    n_correct = min(N_CORRECT_TARGET, len(correct_sorted))
    n_incorrect = min(N_INCORRECT_TARGET, len(incorrect_sorted))

    selected_correct = correct_sorted[:n_correct]
    selected_incorrect = incorrect_sorted[:n_incorrect]
    selected_idx = np.concatenate([selected_correct, selected_incorrect])
    selected_paths = [file_paths[i] for i in selected_idx]

    print(f"\nSelected for attention analysis:")
    print(f"  Correct: {n_correct} (most confident)")
    print(f"  Incorrect: {n_incorrect} (most confident)")
    print(f"  Total: {len(selected_idx)}")

    # 5. Primary analysis: DINOv2-S
    print(f"\n{'='*60}")
    print("PRIMARY ANALYSIS: DINOv2-S/14")
    print(f"{'='*60}")

    primary_results = analyze_attention(
        MODEL_NAME, MODEL_REPO, emb_pooling,
        selected_paths, selected_idx, labels,
        predictions, flickr_ids, scene_types
    )

    if primary_results is None:
        print("❌ Primary analysis failed!")
        sys.exit(1)

    # 6. Validation sweep: DINOv2-g/14 (subset)
    print(f"\n{'='*60}")
    print(f"VALIDATION SWEEP: DINOv2-g/14 ({VALIDATION_SAMPLE} images)")
    print(f"{'='*60}")

    validation_results = analyze_attention(
        VALIDATION_MODEL_NAME, VALIDATION_MODEL_REPO, emb_pooling,
        selected_paths, selected_idx, labels,
        predictions, flickr_ids, scene_types,
        sample_size=VALIDATION_SAMPLE
    )

    # 7. Assemble final results
    final_results = {
        "config": {
            "primary_model": MODEL_NAME,
            "primary_repo": MODEL_REPO,
            "validation_model": VALIDATION_MODEL_NAME,
            "validation_repo": VALIDATION_MODEL_REPO,
            "embedding_pooling": emb_pooling,
            "n_cv_folds": N_CV_FOLDS,
            "C": C_VALUE,
            "random_seed": RANDOM_SEED,
            "attn_layer": ATTN_LAYER,
            "n_total_images": n_total,
            "n_pos": n_pos,
            "n_neg": n_neg,
        },
        "classification": {
            "cv_auc": float(full_auc),
            "cv_auc_mean": float(np.mean(fold_aucs)),
            "cv_auc_std": float(np.std(fold_aucs)),
            "n_correct": int(correct.sum()),
            "n_incorrect": int(incorrect.sum()),
            "accuracy": float(correct.sum() / n_total),
        },
        "primary_analysis": primary_results,
        "validation_analysis": validation_results,
    }

    # Save JSON
    json_path = OUTPUT_DIR / "attention_analysis.json"
    with open(json_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\n✅ Results saved to: {json_path}")

    # 8. Write analysis.md (done separately)
    print("\nDone. Run write_analysis.py to generate the markdown report.")


if __name__ == "__main__":
    main()
