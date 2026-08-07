#!/usr/bin/env python3
"""
MLP Probes — Issue #6
=====================
Run 2-layer MLP classifiers across all 7 models × 6 pooling strategies × 3 dataset sizes.
Grid: 7 × 6 × 3 = 126 evaluations.

Architecture: hidden=[256, 64], dropout=0.3, batch_size=64, max_epochs=200, early_stopping=20
Comparison with #5 (LR): uses identical stratification and seed to enable valid linear-vs-MLP comparison.

Output: experiments/discriminator-mlp/results.csv
"""

import csv
import os
import time
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
VERIFIED_CSV = REPO_ROOT / "data/registry/verified.csv"
EMBEDDINGS_DIR = Path("/mnt/nas-ai-models/training-data/leica-look/embeddings")
OUTPUT_DIR = REPO_ROOT / "experiments/discriminator-mlp"
OUTPUT_CSV = OUTPUT_DIR / "results.csv"

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
MODELS = [
    "dinov2-small",
    "dinov2-base",
    "dinov2-large",
    "dinov2-giant",
    "dinov3-vitl16",
    "siglip-so400m",
    "clip-vitl14",
]

POOLING = [
    "cls",
    "patch_mean",
    "patch_max",
    "patch_gem",
    "cls_patch",
    "multicrop",
]

DATASET_SIZES = [50, 100, 250]
RANDOM_SEED_BASE = 42
TEST_SIZE = 0.30

# MLP hyperparams
HIDDEN_LAYERS = [256, 64]
DROPOUT = 0.3
BATCH_SIZE = 64
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# MLP model
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_metadata() -> tuple[list[str], list[str], list[str], np.ndarray]:
    flickr_ids, scene_types, bodies, labels = [], [], [], []
    with open(VERIFIED_CSV, "r") as f:
        for row in csv.DictReader(f):
            flickr_ids.append(row["flickr_id"])
            scene_types.append(row["scene_type"] or "other")
            bodies.append(row["body"] or "unknown")
            labels.append(1 if row["class"] == "positive" else 0)
    return flickr_ids, scene_types, bodies, np.array(labels, dtype=np.int64)


def stratify_groups(scene_types: list[str], bodies: list[str]) -> np.ndarray:
    major_scenes = {"portrait", "landscape", "street", "night", "architecture", "macro"}
    collapsed_scenes = [s if s in major_scenes else "other" for s in scene_types]
    leica_bodies = {
        "LEICA M10", "LEICA M10-R", "LEICA M10-P", "LEICA M11",
        "LEICA SL (Typ 601)", "LEICA Q2", "LEICA Q3", "LEICA SL2", "LEICA SL3",
    }
    collapsed_bodies = []
    for b in bodies:
        if b in leica_bodies:
            collapsed_bodies.append("leica")
        elif b.startswith("Canon"):
            collapsed_bodies.append("canon")
        elif b.startswith("ILCE-"):
            collapsed_bodies.append("sony")
        elif b.startswith("NIKON"):
            collapsed_bodies.append("nikon")
        else:
            collapsed_bodies.append("other")

    groups = [f"{s}|{b}" for s, b in zip(collapsed_scenes, collapsed_bodies)]
    group_counts = Counter(groups)
    rare_groups = {g for g, c in group_counts.items() if c < 3}
    group_ids = []
    mapping = {}
    next_id = 0
    for g in groups:
        if g in rare_groups:
            g_key = "rare"
        else:
            g_key = g
        if g_key not in mapping:
            mapping[g_key] = next_id
            next_id += 1
        group_ids.append(mapping[g_key])
    return np.array(group_ids, dtype=np.int64)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_mlp(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              input_dim: int) -> tuple[MLP, float]:
    """Train MLP with early stopping. Returns (model, best_val_loss)."""
    model = MLP(input_dim, HIDDEN_LAYERS, DROPOUT).to(DEVICE)

    # Compute class weights for balanced loss
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(bx)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_loss


def predict_proba(model: MLP, X: np.ndarray) -> np.ndarray:
    """Return probability of class 1."""
    model.eval()
    ds = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    probs = []
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(DEVICE)
            logits = model(bx)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def run_split(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
              n_per_class: int, seed: int) -> dict:
    """Sample, split, train MLP, evaluate AUC."""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.RandomState(seed)

    # Stratified proportional sampling (same as LR script)
    unique_groups = np.unique(groups)
    pos_sampled, neg_sampled = [], []
    for g in unique_groups:
        pos_in_g = pos_idx[groups[pos_idx] == g]
        neg_in_g = neg_idx[groups[neg_idx] == g]
        prop_pos = len(pos_in_g) / len(pos_idx) if len(pos_idx) > 0 else 0
        prop_neg = len(neg_in_g) / len(neg_idx) if len(neg_idx) > 0 else 0
        n_pos_sample = max(0, min(len(pos_in_g), int(round(n_per_class * prop_pos))))
        n_neg_sample = max(0, min(len(neg_in_g), int(round(n_per_class * prop_neg))))
        if n_pos_sample > 0:
            pos_sampled.extend(rng.choice(pos_in_g, size=n_pos_sample, replace=False))
        if n_neg_sample > 0:
            neg_sampled.extend(rng.choice(neg_in_g, size=n_neg_sample, replace=False))

    if len(pos_sampled) < n_per_class:
        remaining = list(set(pos_idx) - set(pos_sampled))
        if remaining:
            extra = rng.choice(remaining, size=min(n_per_class - len(pos_sampled), len(remaining)), replace=False)
            pos_sampled.extend(extra)
    pos_sampled = pos_sampled[:n_per_class]

    if len(neg_sampled) < n_per_class:
        remaining = list(set(neg_idx) - set(neg_sampled))
        if remaining:
            extra = rng.choice(remaining, size=min(n_per_class - len(neg_sampled), len(remaining)), replace=False)
            neg_sampled.extend(extra)
    neg_sampled = neg_sampled[:n_per_class]

    sampled_idx = np.concatenate([pos_sampled, neg_sampled])
    X_sub = X[sampled_idx]
    y_sub = y[sampled_idx]
    g_sub = groups[sampled_idx]

    # Stratified split
    unique_in_sample = np.unique(g_sub)
    min_group_count = min(np.sum(g_sub == ug) for ug in unique_in_sample) if len(unique_in_sample) > 0 else 0
    if len(unique_in_sample) >= 2 and min_group_count >= 2:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_sub, y_sub, test_size=TEST_SIZE, stratify=g_sub, random_state=seed)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_sub, y_sub, test_size=TEST_SIZE, stratify=y_sub, random_state=seed)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=TEST_SIZE, stratify=y_sub, random_state=seed)

    # Further split train into train/val (85/15) for early stopping
    X_train2, X_val, y_train2, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=seed)

    # Scale
    scaler = StandardScaler()
    X_train2 = scaler.fit_transform(X_train2)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Train MLP
    model, best_val_loss = train_mlp(
        X_train2.astype(np.float32), y_train2.astype(np.float32),
        X_val.astype(np.float32), y_val.astype(np.float32),
        input_dim=X_train2.shape[1])

    # Predict
    y_prob = predict_proba(model, X_test.astype(np.float32))
    auc = float(roc_auc_score(y_test, y_prob))

    return {
        "auc": round(auc, 6),
        "test_pos": int(np.sum(y_test == 1)),
        "test_neg": int(np.sum(y_test == 0)),
        "val_loss": round(float(best_val_loss), 6),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_CSV}")
    print(f"Device: {DEVICE}")

    flickr_ids, scene_types, bodies, labels = load_metadata()
    groups = stratify_groups(scene_types, bodies)
    print(f"Loaded {len(labels)} verified images ({labels.sum()} pos, {(labels==0).sum()} neg)")

    fieldnames = [
        "model", "pooling", "dataset_size", "auc",
        "test_pos", "test_neg", "embedding_dim", "val_loss", "runtime_sec"
    ]

    existing = set()
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, "r") as f:
            for row in csv.DictReader(f):
                existing.add((row["model"], row["pooling"], row["dataset_size"]))
        print(f"Found {len(existing)} existing results — will skip")

    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not existing:
            writer.writeheader()

        total_configs = len(MODELS) * len(POOLING) * len(DATASET_SIZES)
        completed = len(existing)
        print(f"\nStarting: {total_configs} total, {completed} done")

        for model_name in MODELS:
            model_dir = EMBEDDINGS_DIR / model_name

            for pool_name in POOLING:
                emb_path = model_dir / f"{pool_name}.npy"
                if not emb_path.exists():
                    print(f"  ⚠️  Missing: {emb_path}")
                    continue

                t0 = time.time()
                X = np.load(emb_path).astype(np.float32)
                embedding_dim = X.shape[1]

                for n_per_class in DATASET_SIZES:
                    key = (model_name, pool_name, str(n_per_class))
                    if key in existing:
                        continue

                    n_pos = int(labels.sum())
                    n_neg = int((labels == 0).sum())
                    if n_pos < n_per_class or n_neg < n_per_class:
                        continue

                    seed_offset = DATASET_SIZES.index(n_per_class) * 1000
                    seed = RANDOM_SEED_BASE + seed_offset

                    split_info = run_split(X, labels, groups, n_per_class, seed)

                    row = {
                        "model": model_name,
                        "pooling": pool_name,
                        "dataset_size": str(n_per_class),
                        "auc": split_info["auc"],
                        "test_pos": split_info["test_pos"],
                        "test_neg": split_info["test_neg"],
                        "embedding_dim": embedding_dim,
                        "val_loss": split_info["val_loss"],
                        "runtime_sec": round(time.time() - t0, 2),
                    }
                    writer.writerow(row)
                    completed += 1

                elapsed = time.time() - t0
                print(f"  {model_name}/{pool_name}: dim={embedding_dim}, "
                      f"{len(DATASET_SIZES)} sizes in {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print(f"Done. {completed}/{total_configs} evaluations.")
    print(f"Results: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
