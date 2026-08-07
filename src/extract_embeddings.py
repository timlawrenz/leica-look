#!/usr/bin/env python3
"""
Extract frozen embeddings from all 7 vision models × 6 pooling strategies.
Caches to /mnt/nas-ai-models/training-data/leica-look/embeddings/

Resumable: skips models whose embedding files already exist.
Usage:
    HF_HOME=/mnt/nas-ai-models/huggingface-cache \
    .venv/bin/python src/extract_embeddings.py [--model dinov2-base] [--gpu 0]
"""

import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

os.environ["HF_HOME"] = "/mnt/nas-ai-models/huggingface-cache"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VERIFIED_CSV = "data/registry/verified.csv"
EMBEDDINGS_DIR = Path("/mnt/nas-ai-models/training-data/leica-look/embeddings")
BATCH_SIZE = 16  # will auto-reduce on OOM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Note: dim is the actual hidden_size from model config, verified via forward pass.
# SigLIP so400m outputs 1152 (not 1024 as in original spec).
# CLIP ViT-L vision outputs 1024 (not 768 — that's the projection dim).
MODELS = [
    {"name": "dinov2-small",  "repo": "facebook/dinov2-small",
     "model_class": "auto",   "dim": 384,  "img_size": 224, "has_patches": True},
    {"name": "dinov2-base",   "repo": "facebook/dinov2-base",
     "model_class": "auto",   "dim": 768,  "img_size": 224, "has_patches": True},
    {"name": "dinov2-large",  "repo": "facebook/dinov2-large",
     "model_class": "auto",   "dim": 1024, "img_size": 224, "has_patches": True},
    {"name": "dinov2-giant",  "repo": "facebook/dinov2-giant",
     "model_class": "auto",   "dim": 1536, "img_size": 224, "has_patches": True},
    {"name": "dinov3-vitl16", "repo": "facebook/dinov3-vitl16-pretrain-lvd1689m",
     "model_class": "auto",   "dim": 1024, "img_size": 224, "has_patches": True},
    {"name": "siglip-so400m", "repo": "google/siglip-so400m-patch14-384",
     "model_class": "siglip", "dim": 1152, "img_size": 384, "has_patches": True},
    {"name": "clip-vitl14",   "repo": "openai/clip-vit-large-patch14",
     "model_class": "clip",   "dim": 1024, "img_size": 224, "has_patches": True},
]

POOLING_STRATEGIES = ["cls", "patch_mean", "patch_max", "patch_gem", "cls_patch", "multicrop"]

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ImageDataset(Dataset):
    """Load images from verified.csv."""
    def __init__(self, image_paths: list, labels: list, image_ids: list,
                 img_size: int, is_multicrop: bool = False):
        self.image_paths = image_paths
        self.labels = labels
        self.image_ids = image_ids
        self.img_size = img_size
        self.is_multicrop = is_multicrop

    def __len__(self):
        return len(self.image_paths)

    def _load_and_preprocess(self, path: str) -> torch.Tensor:
        """Load image, convert to RGB, resize and normalize."""
        img = Image.open(path).convert("RGB")
        # Resize so shortest edge = img_size, center crop
        w, h = img.size
        scale = self.img_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)

        # Center crop to img_size × img_size
        left = (new_w - self.img_size) // 2
        top = (new_h - self.img_size) // 2
        img = img.crop((left, top, left + self.img_size, top + self.img_size))

        # Convert to tensor and normalize with ImageNet stats
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        return torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W)

    def _multicrop(self, path: str) -> list:
        """Return 5 crops: full image + 4 corners at 0.875 scale."""
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = self.img_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)

        # Full center crop
        left_c = (new_w - self.img_size) // 2
        top_c  = (new_h - self.img_size) // 2
        full = img.crop((left_c, top_c, left_c + self.img_size, top_c + self.img_size))

        # Corner crops at ~0.875 scale
        crop_sz = int(self.img_size * 0.875)
        corner_crops = []
        for x, y in [(0, 0), (new_w - crop_sz, 0),
                     (0, new_h - crop_sz), (new_w - crop_sz, new_h - crop_sz)]:
            corner = img.crop((x, y, x + crop_sz, y + crop_sz)).resize(
                (self.img_size, self.img_size), Image.BICUBIC)
            corner_crops.append(corner)

        crops = [full] + corner_crops
        tensors = []
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        for crop in crops:
            arr = np.array(crop, dtype=np.float32) / 255.0
            arr = (arr - mean) / std
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
        return tensors

    def __getitem__(self, idx):
        if self.is_multicrop:
            crops = self._multicrop(self.image_paths[idx])
            return torch.stack(crops, dim=0), idx  # (5, C, H, W)
        else:
            x = self._load_and_preprocess(self.image_paths[idx])
            return x, idx

# ---------------------------------------------------------------------------
# Pooling functions
# ---------------------------------------------------------------------------
def extract_poolings(last_hidden_state: torch.Tensor) -> dict:
    """
    last_hidden_state: (B, N+1, D) — CLS at position 0, patches at 1..N
    Returns dict of pooling_name -> (B, D) tensor.
    """
    cls_token = last_hidden_state[:, 0, :]          # (B, D)
    patches   = last_hidden_state[:, 1:, :]          # (B, N, D)

    results = {}
    results["cls"] = cls_token

    # Patch mean
    results["patch_mean"] = patches.mean(dim=1)

    # Patch max
    results["patch_max"] = patches.max(dim=1).values

    # Patch GeM (p=3)
    eps = 1e-6
    gem = patches.clamp(min=eps).pow(3).mean(dim=1).pow(1.0 / 3)
    results["patch_gem"] = gem

    # CLS + patch mean concat
    results["cls_patch"] = torch.cat([cls_token, results["patch_mean"]], dim=1)

    # Multicrop handled separately
    return results

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_info: dict):
    """Load model and return (model, forward_fn)."""
    from transformers import AutoModel, CLIPVisionModel, SiglipVisionModel, AutoImageProcessor

    repo = model_info["repo"]
    mc = model_info["model_class"]
    has_patches = model_info["has_patches"]

    if mc == "auto":
        model = AutoModel.from_pretrained(
            repo, cache_dir="/mnt/nas-ai-models/huggingface-cache", trust_remote_code=True)
    elif mc == "clip":
        model = CLIPVisionModel.from_pretrained(
            repo, cache_dir="/mnt/nas-ai-models/huggingface-cache")
    elif mc == "siglip":
        model = SiglipVisionModel.from_pretrained(
            repo, cache_dir="/mnt/nas-ai-models/huggingface-cache")
    else:
        raise ValueError(f"Unknown model_class: {mc}")

    model.eval()
    model.to(DEVICE)

    def forward_fn(x):
        """x: (B, C, H, W) on device. Returns last_hidden_state."""
        with torch.no_grad():
            out = model(x)
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state
        elif isinstance(out, (tuple, list)):
            return out[0]
        elif hasattr(out, "pooler_output"):
            # Fallback: some models only return pooled output (no patches)
            # Return as (B, 1, D) so pooling degenerates gracefully
            return out.pooler_output.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected output type: {type(out)}")

    return model, forward_fn

# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------
def load_data(csv_path: str) -> Tuple[list, list, list]:
    """Read verified.csv, return (paths, labels, ids)."""
    paths, labels, ids_ = [], [], []
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            paths.append(row["file_path"])
            labels.append(1 if row["class"] == "positive" else 0)
            ids_.append(row["flickr_id"])
    return paths, labels, ids_


def check_embeddings_exist(model_name: str) -> bool:
    """Return True if all 6 pooling files already exist for this model."""
    model_dir = EMBEDDINGS_DIR / model_name
    if not model_dir.exists():
        return False
    for pooling in POOLING_STRATEGIES:
        if not (model_dir / f"{pooling}.npy").exists():
            return False
    return True


def save_shared_files(labels: list, image_ids: list):
    """Save labels.npy and image_ids.txt (shared across all models)."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_DIR / "labels.npy", np.array(labels, dtype=np.int8).ravel())
    with open(EMBEDDINGS_DIR / "image_ids.txt", 'w') as f:
        for iid in image_ids:
            f.write(f"{iid}\n")
    print(f"  Saved labels.npy ({len(labels)} items) and image_ids.txt")


def extract_model(model_info: dict, image_paths: list) -> dict:
    """Extract all embedding types for one model. Returns {pooling: np.array}."""
    model_name = model_info["name"]
    model_dir = EMBEDDINGS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    if check_embeddings_exist(model_name):
        print(f"  ✅ All embeddings already exist — skipping")
        # Load existing to return dim info
        existing = {}
        for pooling in POOLING_STRATEGIES:
            arr = np.load(model_dir / f"{pooling}.npy")
            existing[pooling] = arr
        return existing

    print(f"  Loading model {model_info['repo']}...")
    model, forward_fn = load_model(model_info)
    img_size = model_info["img_size"]
    n_total = len(image_paths)

    # ---- Single-crop pass (for cls, patch_*, cls_patch) ----
    accumulators = {p: [] for p in POOLING_STRATEGIES if p != "multicrop"}

    dataset = ImageDataset(image_paths, [0]*n_total, [""]*n_total, img_size)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=4, pin_memory=True, drop_last=False)

    pbar = tqdm(total=n_total, desc=f"  {model_name} (single-crop)", unit="img")
    for batch_x, batch_idx in loader:
        batch_x = batch_x.to(DEVICE)
        try:
            hs = forward_fn(batch_x)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n  ⚠️ OOM — reducing batch and retrying")
                torch.cuda.empty_cache()
                # Process one by one for this batch
                hs_list = []
                for i in range(len(batch_x)):
                    xi = batch_x[i:i+1]
                    hs_list.append(forward_fn(xi))
                hs = torch.cat(hs_list, dim=0)
            else:
                raise

        poolings = extract_poolings(hs)
        for k, v in poolings.items():
            accumulators[k].append(v.cpu().numpy())

        pbar.update(len(batch_x))
    pbar.close()

    # Concatenate accumulators
    results = {}
    for k in POOLING_STRATEGIES:
        if k == "multicrop":
            continue
        results[k] = np.concatenate(accumulators[k], axis=0)

    # ---- Multicrop pass ----
    print(f"  {model_name} (multicrop — slower, 5× forward passes per image)...")
    mc_dataset = ImageDataset(image_paths, [0]*n_total, [""]*n_total, img_size,
                              is_multicrop=True)
    mc_loader = DataLoader(mc_dataset, batch_size=1, shuffle=False,
                          num_workers=2, pin_memory=True, drop_last=False)

    mc_cls_list = []
    pbar = tqdm(total=n_total, desc=f"  {model_name} (multicrop)", unit="img")
    for batch_x, batch_idx in mc_loader:
        # batch_x: (1, 5, C, H, W)
        batch_x = batch_x.squeeze(0).to(DEVICE)  # (5, C, H, W)
        # Process 5 crops together
        hs = forward_fn(batch_x)  # (5, N+1, D)
        cls_crops = hs[:, 0, :]   # (5, D)
        mc_avg = cls_crops.mean(dim=0, keepdim=True)  # (1, D)
        mc_cls_list.append(mc_avg.cpu().numpy())
        pbar.update(1)
    pbar.close()

    results["multicrop"] = np.concatenate(mc_cls_list, axis=0)

    # ---- Save ----
    for pooling in POOLING_STRATEGIES:
        path = model_dir / f"{pooling}.npy"
        arr = results[pooling].astype(np.float16)
        np.save(path, arr)
        print(f"    Saved {pooling}: {arr.shape} -> {path}")

    # ---- Validate ----
    for pooling in POOLING_STRATEGIES:
        arr = results[pooling]
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        if n_nan > 0 or n_inf > 0:
            print(f"    ⚠️  {pooling}: {n_nan} NaN, {n_inf} Inf!")
        else:
            print(f"    ✅ {pooling}: shape={arr.shape}, no NaN/Inf")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Extract vision model embeddings")
    parser.add_argument("--model", type=str, default=None,
                       help="Extract only one model (e.g., dinov2-base)")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                       help="Skip models with existing embeddings")
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    print(f"Using device: {DEVICE} (GPU {args.gpu})")
    print(f"Embeddings dir: {EMBEDDINGS_DIR}")

    # Load data
    image_paths, labels, image_ids = load_data(VERIFIED_CSV)
    print(f"Loaded {len(image_paths)} verified images ({sum(labels)} pos, {len(labels)-sum(labels)} neg)")

    # Save shared files (labels, image_ids)
    save_shared_files(labels, image_ids)

    # Filter models
    model_list = MODELS
    if args.model:
        model_list = [m for m in MODELS if m["name"] == args.model]
        if not model_list:
            print(f"Unknown model: {args.model}")
            print(f"Available: {[m['name'] for m in MODELS]}")
            sys.exit(1)

    # Extract each model
    results_summary = {}
    for model_info in model_list:
        name = model_info["name"]
        print(f"\n{'='*60}")
        print(f"Model: {name} ({model_info['repo']})")
        print(f"{'='*60}")

        if args.skip_existing and check_embeddings_exist(name):
            print(f"  ✅ Skipping — all embeddings already exist")
            results_summary[name] = "SKIPPED"
            continue

        try:
            extract_model(model_info, image_paths)
            results_summary[name] = "OK"
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            results_summary[name] = f"FAILED: {e}"

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for name, status in results_summary.items():
        emoji = "✅" if status in ("OK", "SKIPPED") else "❌"
        print(f"  {emoji} {name}: {status}")
        if status not in ("OK", "SKIPPED"):
            all_ok = False

    if all_ok:
        print("\n🎉 All models extracted successfully!")
    else:
        print("\n⚠️  Some models failed — re-run to retry (resumable)")
        sys.exit(1)


if __name__ == "__main__":
    main()
