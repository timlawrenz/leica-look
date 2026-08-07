#!/usr/bin/env python3
"""
Download missing vision models to NAS and verify them.
Usage: HF_HOME=/mnt/nas-ai-models/huggingface-cache python3 src/download_models.py
"""

import os
import sys
import traceback
import torch

os.environ["HF_HOME"] = "/mnt/nas-ai-models/huggingface-cache"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # faster downloads

from transformers import AutoModel, AutoImageProcessor
from huggingface_hub import snapshot_download

MODELS = [
    {
        "name": "facebook/dinov2-small",
        "repo": "facebook/dinov2-small",
        "model_type": "dinov2",
    },
    {
        "name": "facebook/dinov2-giant",
        "repo": "facebook/dinov2-giant",
        "model_type": "dinov2",
    },
    {
        "name": "google/siglip-so400m-patch14-384",
        "repo": "google/siglip-so400m-patch14-384",
        "model_type": "siglip",
    },
    {
        "name": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        "repo": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        "model_type": "clip",
    },
]

def download_model(repo: str) -> str:
    """Download model from HF Hub using snapshot_download."""
    print(f"  Downloading {repo}...")
    local_path = snapshot_download(
        repo_id=repo,
        cache_dir="/mnt/nas-ai-models/huggingface-cache",
        resume_download=True,
    )
    print(f"  -> {local_path}")
    return local_path

def verify_model(repo: str, model_type: str) -> bool:
    """Load model and run a forward pass on a random tensor to verify."""
    print(f"  Loading {repo}...")
    
    if model_type in ("dinov2", "siglip", "clip"):
        model = AutoModel.from_pretrained(
            repo,
            cache_dir="/mnt/nas-ai-models/huggingface-cache",
            trust_remote_code=True,
        )
    else:
        model = AutoModel.from_pretrained(
            repo,
            cache_dir="/mnt/nas-ai-models/huggingface-cache",
            trust_remote_code=True,
        )
    
    model.eval()
    
    # Determine input shape
    if "dinov2" in model_type:
        # DINOv2 uses 224x224 for small, 224x224 for giant
        input_tensor = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(input_tensor)
            # DINOv2 returns BaseModelOutput with last_hidden_state
            if hasattr(output, "last_hidden_state"):
                emb = output.last_hidden_state
            elif hasattr(output, "pooler_output") and output.pooler_output is not None:
                emb = output.pooler_output
            else:
                emb = output[0]
    
    elif model_type == "siglip":
        # SigLIP uses 384x384
        input_tensor = torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            output = model.get_image_features(input_tensor)
            # get_image_features returns BaseModelOutputWithPooling
            emb = output.pooler_output if hasattr(output, 'pooler_output') else output
    
    elif model_type == "clip":
        # CLIP ViT-bigG uses 224x224
        input_tensor = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model.get_image_features(input_tensor)
            # get_image_features returns BaseModelOutputWithPooling
            emb = output.pooler_output if hasattr(output, 'pooler_output') else output
    
    print(f"  Output shape: {emb.shape}")
    print(f"  Output dtype: {emb.dtype}")
    print(f"  Output mean:  {emb.mean().item():.6f}")
    return True

def main():
    results = {}
    failed = []
    
    for model_info in MODELS:
        name = model_info["name"]
        print(f"\n{'='*60}")
        print(f"Model: {name}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Download
            local_path = download_model(model_info["repo"])
            
            # Step 2: Verify
            verify_model(model_info["repo"], model_info["model_type"])
            
            results[name] = "OK"
            print(f"\n✅ {name}: DOWNLOADED + VERIFIED")
            
        except Exception as e:
            print(f"\n❌ {name}: FAILED")
            print(f"   Error: {e}")
            traceback.print_exc()
            results[name] = f"FAILED: {e}"
            failed.append(name)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, status in results.items():
        emoji = "✅" if status == "OK" else "❌"
        print(f"  {emoji} {name}: {status}")
    
    if failed:
        print(f"\n{len(failed)} model(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\nAll models downloaded and verified successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
