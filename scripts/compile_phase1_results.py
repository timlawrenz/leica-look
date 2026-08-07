#!/usr/bin/env python3
"""Compile Phase 1 discriminator results into summary tables for the ledger."""
import csv
import json
from collections import defaultdict
from pathlib import Path

EXPERIMENTS = Path("/home/tim/source/activity/leica-look/experiments")

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def best_per_model_classifier(rows, key="auc", classifier="LR"):
    """Find best AUC per model per classifier."""
    best = defaultdict(lambda: {"auc": 0.0, "config": ""})
    for r in rows:
        auc = float(r[key])
        model = r["model"]
        if auc > best[model]["auc"]:
            pooling = r.get("pooling", "N/A")
            size = r.get("dataset_size", "N/A")
            extra = ""
            if "C" in r:
                extra = f", C={r['C']}"
            elif "k" in r:
                extra = f", k={r['k']}"
            best[model] = {
                "auc": auc,
                "config": f"{pooling}, n={size}{extra}",
                "pooling": pooling,
                "size": size,
                "classifier": classifier,
                "embedding_dim": r.get("embedding_dim", "N/A"),
            }
    return best

def best_overall_per_model(best_lr, best_mlp, best_knn):
    """Find best classifier per model."""
    models = set(best_lr.keys()) | set(best_mlp.keys()) | set(best_knn.keys())
    result = {}
    for m in models:
        candidates = []
        for d, name in [(best_lr, "LR"), (best_mlp, "MLP"), (best_knn, "k-NN")]:
            if m in d:
                candidates.append((d[m]["auc"], name, d[m]))
        candidates.sort(reverse=True)
        best_auc, best_name, best_info = candidates[0]
        result[m] = {
            "best_auc": best_auc,
            "best_classifier": best_name,
            "best_pooling": best_info["pooling"],
            "best_size": best_info["size"],
        }
    return result

def pooling_comparison(rows, key="auc"):
    """Average AUC by pooling across all models."""
    by_pooling = defaultdict(list)
    for r in rows:
        by_pooling[r["pooling"]].append(float(r[key]))
    
    result = {}
    for p, aucs in sorted(by_pooling.items()):
        result[p] = {
            "mean": sum(aucs) / len(aucs),
            "max": max(aucs),
            "min": min(aucs),
            "n": len(aucs),
        }
    return result

def size_scaling(rows, key="auc"):
    """Average AUC by dataset size."""
    by_size = defaultdict(list)
    for r in rows:
        by_size[r["dataset_size"]].append(float(r[key]))
    
    result = {}
    for s in sorted(by_size.keys(), key=int):
        aucs = by_size[s]
        result[s] = {
            "mean": sum(aucs) / len(aucs),
            "max": max(aucs),
            "n": len(aucs),
        }
    return result

def model_family_comparison(best_overall):
    """Compare model families."""
    dino_models = [m for m in best_overall if m.startswith("dinov2") or m.startswith("dinov3")]
    dino_aucs = [best_overall[m]["best_auc"] for m in dino_models]
    
    result = {
        "DINOv2-S": best_overall.get("dinov2-small", {}).get("best_auc", 0),
        "DINOv2-B": best_overall.get("dinov2-base", {}).get("best_auc", 0),
        "DINOv2-L": best_overall.get("dinov2-large", {}).get("best_auc", 0),
        "DINOv2-g": best_overall.get("dinov2-giant", {}).get("best_auc", 0),
        "DINOv3-L": best_overall.get("dinov3-vitl16", {}).get("best_auc", 0),
        "SigLIP": best_overall.get("siglip-so400m", {}).get("best_auc", 0),
        "CLIP-L": best_overall.get("clip-vitl14", {}).get("best_auc", 0),
    }
    result["DINOv2_mean"] = sum(a for m, a in result.items() if m.startswith("DINOv2-")) / 4
    result["CLIP_vs_DINOv2_delta"] = result["CLIP-L"] - result["DINOv2_mean"]
    return result

def main():
    # Load data
    lr_rows = load_csv(EXPERIMENTS / "discriminator-lr" / "results.csv")
    mlp_rows = load_csv(EXPERIMENTS / "discriminator-mlp" / "results.csv")
    knn_rows = load_csv(EXPERIMENTS / "discriminator-knn" / "results.csv")
    
    print(f"Loaded: {len(lr_rows)} LR, {len(mlp_rows)} MLP, {len(knn_rows)} k-NN rows")
    
    # Best per model
    best_lr = best_per_model_classifier(lr_rows, "auc", "LR")
    best_mlp = best_per_model_classifier(mlp_rows, "auc", "MLP")
    best_knn = best_per_model_classifier(knn_rows, "auc", "k-NN")
    
    best = best_overall_per_model(best_lr, best_mlp, best_knn)
    
    # Model ordering
    model_order = ["dinov2-small", "dinov2-base", "dinov2-large", "dinov2-giant", 
                   "dinov3-vitl16", "siglip-so400m", "clip-vitl14"]
    model_display = {
        "dinov2-small": "DINOv2 ViT-S/14",
        "dinov2-base": "DINOv2 ViT-B/14",
        "dinov2-large": "DINOv2 ViT-L/14",
        "dinov2-giant": "DINOv2 ViT-g/14",
        "dinov3-vitl16": "DINOv3 ViT-L/16",
        "siglip-so400m": "SigLIP SO400M",
        "clip-vitl14": "CLIP ViT-L/14",
    }
    
    # === Summary Table: Best per model per classifier ===
    print("\n" + "="*100)
    print("BEST AUC PER MODEL × CLASSIFIER")
    print("="*100)
    print(f"{'Model':<25} {'Best LR':>10} {'Best MLP':>10} {'Best k-NN':>10} {'Best Overall':>12} {'Classifier':>10} {'Pooling':>15} {'Size':>6}")
    print("-"*100)
    for m in model_order:
        lr_auc = best_lr.get(m, {}).get("auc", 0)
        mlp_auc = best_mlp.get(m, {}).get("auc", 0)
        knn_auc = best_knn.get(m, {}).get("auc", 0)
        b = best.get(m, {})
        print(f"{model_display[m]:<25} {lr_auc:>10.4f} {mlp_auc:>10.4f} {knn_auc:>10.4f} {b.get('best_auc',0):>12.4f} {b.get('best_classifier',''):>10} {b.get('best_pooling',''):>15} {b.get('best_size',''):>6}")
    
    # === LR Per-Config Detail ===
    print("\n" + "="*100)
    print("LR BEST CONFIG PER MODEL")
    print("="*100)
    for m in model_order:
        if m in best_lr:
            info = best_lr[m]
            print(f"  {model_display[m]:<25} AUC={info['auc']:.4f}  config={info['config']}  dim={info['embedding_dim']}")
    
    # === MLP Per-Config Detail ===
    print("\n" + "="*100)
    print("MLP BEST CONFIG PER MODEL")
    print("="*100)
    for m in model_order:
        if m in best_mlp:
            info = best_mlp[m]
            print(f"  {model_display[m]:<25} AUC={info['auc']:.4f}  config={info['config']}")
    
    # === k-NN Per-Config Detail ===
    print("\n" + "="*100)
    print("k-NN BEST CONFIG PER MODEL")
    print("="*100)
    for m in model_order:
        if m in best_knn:
            info = best_knn[m]
            print(f"  {model_display[m]:<25} AUC={info['auc']:.4f}  config={info['config']}")
    
    # === Pooling Comparison (LR only, most comprehensive) ===
    print("\n" + "="*100)
    print("POOLING STRATEGY COMPARISON (LR, all models, all sizes)")
    print("="*100)
    pool_lr = pooling_comparison(lr_rows)
    for p, stats in pool_lr.items():
        print(f"  {p:<15} mean={stats['mean']:.4f}  max={stats['max']:.4f}  min={stats['min']:.4f}  n={stats['n']}")
    
    # === Size Scaling ===
    print("\n" + "="*100)
    print("DATASET SIZE SCALING (LR, all models, all pooling)")
    print("="*100)
    size_lr = size_scaling(lr_rows)
    for s, stats in size_lr.items():
        print(f"  n={s:<6} mean={stats['mean']:.4f}  max={stats['max']:.4f}  n_configs={stats['n']}")
    
    # === Model family comparison ===
    print("\n" + "="*100)
    print("MODEL FAMILY COMPARISON (best overall per model)")
    print("="*100)
    family = model_family_comparison(best)
    for k, v in family.items():
        print(f"  {k:<25} {v:.4f}")
    
    # === Key finding: CLIP vs DINO ===
    print("\n" + "="*100)
    print("CLIP VS DINO ANALYSIS")
    print("="*100)
    clip_lr_aucs = [float(r["auc"]) for r in lr_rows if r["model"] == "clip-vitl14"]
    dino_lr_aucs = [float(r["auc"]) for r in lr_rows if r["model"].startswith("dinov")]
    print(f"  CLIP LR mean AUC:  {sum(clip_lr_aucs)/len(clip_lr_aucs):.4f} (n={len(clip_lr_aucs)})")
    print(f"  DINO LR mean AUC:  {sum(dino_lr_aucs)/len(dino_lr_aucs):.4f} (n={len(dino_lr_aucs)})")
    print(f"  CLIP - DINO delta: {(sum(clip_lr_aucs)/len(clip_lr_aucs) - sum(dino_lr_aucs)/len(dino_lr_aucs)):.4f}")
    
    # LR top 10
    lr_sorted = sorted(lr_rows, key=lambda r: float(r["auc"]), reverse=True)
    print("\n  Top 10 LR results:")
    for i, r in enumerate(lr_sorted[:10]):
        print(f"  {i+1:2}. {model_display.get(r['model'], r['model']):<25} AUC={float(r['auc']):.4f}  {r['pooling']}, n={r['dataset_size']}, C={r['C']}")
    
    # === k-NN k comparison ===
    print("\n" + "="*100)
    print("k-NN: k VALUE COMPARISON")
    print("="*100)
    for k_val in ["1", "5", "11"]:
        k_aucs = [float(r["auc"]) for r in knn_rows if r["k"] == k_val]
        print(f"  k={k_val:<3} mean={sum(k_aucs)/len(k_aucs):.4f}  max={max(k_aucs):.4f}  n={len(k_aucs)}")
    
    # === MLP vs LR delta ===
    print("\n" + "="*100)
    print("MLP vs LR DELTA PER MODEL")
    print("="*100)
    for m in model_order:
        lr_auc = best_lr.get(m, {}).get("auc", 0)
        mlp_auc = best_mlp.get(m, {}).get("auc", 0)
        delta = mlp_auc - lr_auc
        print(f"  {model_display[m]:<25} Δ={delta:+.4f}  (MLP {mlp_auc:.4f} - LR {lr_auc:.4f})")

if __name__ == "__main__":
    main()
