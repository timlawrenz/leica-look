#!/usr/bin/env python3
"""
Sample reviewer for the leica-look seed dataset (issue #2 curation).

Reads a manifest CSV and outputs a curated set of photo URLs grouped into:
  - AVERAGE: representative photos per lens type (for a visual sanity check)
  - OUTLIERS: photos likely to be rejected during curation, with a reason

The reviewer also downloads thumbnails into .flickr_cache/thumbs/<id>.jpg so the
human can eyeball them locally (the WebUI can embed them via MEDIA:).

Usage:
    python src/data/sample_for_review.py --manifest data/registry/positive_manifest.csv \
        --out /tmp/leica_review --per-lens 6 --outliers all
"""

import argparse
import csv
import os
import sys
import requests

# Lens labels we expect per class — used for grouping average samples
EXPECTED_POSITIVE = {"Summilux 50/1.4", "Summilux 35/1.4", "Summilux 28/1.4",
                     "APO-Summicron 50"}

# Bodies that indicate MONOCHROME (excluded by experiment design)
MONO_BODIES = ("MONOCHROM", "Q2 MONO", "M10 MONO", "M11 MONO")


def load_manifest(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def classify_outliers(rows):
    """Return list of (photo, reason) for photos that likely need rejection."""
    outliers = []
    for p in rows:
        reasons = []
        body = (p.get("body") or "").upper()
        lens = (p.get("lens_exif") or "").upper()
        # Monochrome body -> different signal (excluded by design)
        if any(m in body for m in ("MONOCHROM", "MONO")):
            reasons.append("monochrome body (excluded by design)")
        # Body not a real Leica target (search text leaked into EXIF body)
        if "LEICA" not in body and body:
            reasons.append(f"non-Leica body: {p.get('body')}")
        # Missing/blank lens EXIF is a red flag (shouldn't happen after match)
        if not lens:
            reasons.append("blank lens EXIF")
        if reasons:
            outliers.append((p, "; ".join(reasons)))
    return outliers


def pick_average(rows, per_lens):
    """Pick representative (average) photos per lens label. Rows are ordered by
    interestingness-desc already, so we sample a spread: first, middle, near-end."""
    by_lens = {}
    for p in rows:
        by_lens.setdefault(p["lens_label"], []).append(p)
    picks = []
    for lens, group in by_lens.items():
        n = len(group)
        k = min(per_lens, n)
        idxs = set()
        if n == 1:
            idxs = {0}
        else:
            for i in range(k):
                idxs.add(int(i * (n - 1) / max(k - 1, 1)))
        for i in sorted(idxs):
            picks.append(group[i])
    return picks


def download_thumb(photo, outdir):
    url = photo.get("thumb") or photo.get("url_q") or ""
    pid = photo["flickr_id"]
    dest = os.path.join(outdir, f"{pid}.jpg")
    if os.path.exists(dest):
        return dest
    if not url:
        return None
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            os.makedirs(outdir, exist_ok=True)
            with open(dest, "wb") as f:
                f.write(r.content)
            return dest
    except requests.RequestException:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description="Sample reviewer for curation")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=".flickr_cache/thumbs")
    ap.add_argument("--per-lens", type=int, default=6)
    ap.add_argument("--outliers", choices=["all", "none"], default="all")
    ap.add_argument("--no-download", action="store_true", help="print URLs only")
    ap.add_argument("--max-outliers", type=int, default=20)
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    print(f"Manifest: {args.manifest} — {len(rows)} photos\n")

    # ---- AVERAGE ----
    avg = pick_average(rows, args.per_lens)
    print(f"=== AVERAGE ({len(avg)} photos, up to {args.per_lens} per lens) ===")
    for p in avg:
        url = p.get("url", "?")
        print(f"  [{p['lens_label']:16} {p['body']:16} {p['scene_type']:12}] {url}")
        if not args.no_download:
            dest = download_thumb(p, args.out)
            if dest:
                print(f"    THUMB: {dest}")

    # ---- OUTLIERS ----
    if args.outliers == "all":
        bad = classify_outliers(rows)
        bad = bad[:args.max_outliers]
        print(f"\n=== OUTLIERS / likely-reject ({len(bad)} shown of total) ===")
        for p, reason in bad:
            print(f"  ! [{p['lens_label']:16} {p['body']:20}] {reason}\n    {p.get('url','')}")
            if not args.no_download:
                dest = download_thumb(p, args.out)
                if dest:
                    print(f"    THUMB: {dest}")

        # Summary of rejection categories
        print("\n=== REJECTION CATEGORY COUNTS ===")
        monochrome = sum(1 for _ in classify_outliers(rows) if "monochrome" in _[1])
        nonleica = sum(1 for _ in classify_outliers(rows) if "non-Leica" in _[1])
        print(f"  monochrome body:    {monochrome}")
        print(f"  non-Leica body:     {nonleica}")
        print(f"  total flagged:      {len(classify_outliers(rows))} / {len(rows)}")


if __name__ == "__main__":
    main()
