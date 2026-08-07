#!/usr/bin/env python3
"""
Prune the positive manifest per curation rules. Writes a cleaned manifest
(positive_manifest_clean.csv) and a rejected log with reason codes.

Curation rules (from Tim / experiment design):
  - monochrome body (M10 Monochrom, Q2 Monochrom, ...)  -> DROP (design excludes)
    Also catches photos OF a camera rather than BY it when they surface as
    monochrome body + catalog-like tags (manual flag support via --drop-ids).

Usage:
    python src/data/prune_manifest.py \
        --manifest data/registry/positive_manifest.csv \
        --drop-ids FILE        # optional: newline-separated flickr_ids to force-drop
"""

import argparse
import csv
import os
import sys
from collections import Counter

MONO_MARKERS = ("MONOCHROM",)


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def prune(rows, drop_ids):
    accepted, rejected = [], []
    for r in rows:
        body = (r.get("body") or "").upper()
        reasons = []
        if r["flickr_id"] in drop_ids:
            reasons.append("manual-drop")
        if any(m in body for m in MONO_MARKERS):
            reasons.append("monochrome-body")
        if not r.get("lens_exif"):
            reasons.append("blank-lens-exif")
        if reasons:
            r["reject_reason"] = ";".join(reasons)
            rejected.append(r)
        else:
            r["reject_reason"] = ""
            accepted.append(r)
    return accepted, rejected


def write(path, rows):
    # drop reject_reason from schema if empty for clean set
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--drop-ids", default=None, help="file of flickr_ids to force-drop")
    args = ap.parse_args()

    rows = load(args.manifest)
    drop_ids = set()
    if args.drop_ids and os.path.exists(args.drop_ids):
        drop_ids = {l.strip() for l in open(args.drop_ids) if l.strip()}
        print(f"loaded {len(drop_ids)} manual drop ids")

    accepted, rejected = prune(rows, drop_ids)

    base = args.manifest.replace("_manifest.csv", "")
    clean_path = f"{base}_manifest_clean.csv"
    rej_path = f"{base}_rejected.csv"
    write(clean_path, accepted)
    write(rej_path, rejected)

    print(f"ACCEPTED: {len(accepted)}  REJECTED: {len(rejected)}")
    print("reject reasons:", dict(Counter(r["reject_reason"] for r in rejected)))
    by_lens = Counter(r["lens_label"] for r in accepted)
    print("accepted by lens:", dict(by_lens))
    print(f"\nwrote {clean_path}")
    print(f"wrote {rej_path}")

    if len(accepted) < 250:
        print("\nWARNING: accepted < 250 — design floor for positive class")


if __name__ == "__main__":
    main()
