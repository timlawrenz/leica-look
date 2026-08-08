#!/usr/bin/env python3
"""
Evaluate an existing Phase 2a LoRA checkpoint: generate img2img transfer
samples using the correct diffusers >=0.35 FluxImg2ImgPipeline.

Loads the PEFT adapter from a checkpoint dir, runs the same 20 eval images
the training script would have used (non-Leica, seed=42), and saves
side-by-side input|transfer samples.

Usage (on strix):
    python eval_checkpoint.py --ckpt experiments/lora-transfer/runs/<run>/checkpoints/step_00500
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lora import DEFAULT_PROMPT, HF_CACHE, PROJECT_ROOT


def load_pipeline_with_adapter(ckpt_dir, dtype=torch.float16, device="cuda"):
    from diffusers.pipelines.flux.pipeline_flux_img2img import FluxImg2ImgPipeline
    from peft import PeftModel

    print("Loading FluxImg2ImgPipeline base...")
    pipe = FluxImg2ImgPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=dtype,
        cache_dir=str(HF_CACHE),
        local_files_only=False,
    )
    pipe.to(device)

    print(f"Loading LoRA adapter from {ckpt_dir}...")
    pipe.transformer = PeftModel.from_pretrained(
        pipe.transformer, str(ckpt_dir), is_trainable=False,
    )
    pipe.transformer.eval()
    return pipe


def load_eval_images(registry, n=20, seed=42, target=1024):
    rows = []
    with open(registry, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("class") == "negative":
                p = row.get("file_path", "")
                if p and Path(p).exists():
                    rows.append(p)
    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(rows), min(n, len(rows)), replace=False)
    tfm = T.Compose([T.CenterCrop(target), T.Resize((target, target),
                    interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(),
                    T.Normalize([0.5], [0.5])])
    return [tfm(Image.open(rows[i]).convert("RGB")) for i in idxs], [rows[i] for i in idxs]


@torch.no_grad()
def generate(pipe, img, steps=20, strength=0.30, guidance=3.5, seed=42):
    pil = T.ToPILImage()((img.cpu() * 0.5 + 0.5).clamp(0, 1))
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    out = pipe(
        prompt=DEFAULT_PROMPT, image=pil, num_inference_steps=steps,
        strength=strength, guidance_scale=guidance, generator=gen,
        output_type="pt",
    )
    return out.images[0] * 2.0 - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--registry", default=str(PROJECT_ROOT / "data" / "registry" / "verified.csv"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    outdir = Path(args.out) if args.out else ckpt.parent.parent / "evaluation" / "manual_step500"
    outdir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline_with_adapter(ckpt)
    imgs, paths = load_eval_images(args.registry, n=20, seed=42)

    print(f"Generating {len(imgs)} transfer samples -> {outdir}")
    saved = 0
    for i, (img, src) in enumerate(zip(imgs, paths)):
        try:
            transfer = generate(pipe, img, seed=42 + i)
            inp = (img.cpu() * 0.5 + 0.5).clamp(0, 1)
            trs = (transfer.cpu() * 0.5 + 0.5).clamp(0, 1)
            comp = torch.cat([inp, trs], dim=-1)
            T.ToPILImage()(comp).save(outdir / f"sample_{i:03d}.png")
            saved += 1
        except Exception as e:
            print(f"  WARN sample {i} failed: {type(e).__name__}: {e}")

    # Also write a mapping of src paths
    with open(outdir / "sources.txt", "w") as f:
        for p in paths:
            f.write(p + "\n")
    print(f"Saved {saved}/{len(imgs)} samples to {outdir}")
    # Exit non-zero only if nothing was produced
    sys.exit(0 if saved > 0 else 1)


if __name__ == "__main__":
    main()