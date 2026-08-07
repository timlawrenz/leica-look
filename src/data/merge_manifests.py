#!/usr/bin/env python3
"""
Merge the clean positive manifest with the backfill manifest.

CRITICAL: re-validates EVERY row's lens EXIF against the CURRENT (corrected)
lens signatures in flickr_scrape.py, so any rows that slipped through the old
over-broad rules (e.g. Summilux 28mm f/1.7 matched as f/1.4) are dropped
regardless of which manifest they came from. Dedupes on flickr_id.

Usage:
    python src/data/merge_manifests.py \
        --base data/registry/positive_manifest_clean.csv \
        --add data/registry/positive_backfill.csv \
        --out data/registry/positive_merged.csv
"""

import argparse
import csv
import importlib.util
import os
import sys
from collections import Counter


def load_flickr_match():
    spec = importlib.util.spec_from_file_location(
        "flickr_scrape", "src/data/flickr_scrape.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.POSITIVE["lenses"], m.lens_matches


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--add", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lens_sigs, lens_matches = load_flickr_match()
    rows_by_id = {}
    dropped = Counter()
    src_of = {}

    for path, tag in [(args.base, "base"), (args.add, "add")]:
        for r in load(path):
            pid = r["flickr_id"]
            # Re-validate against current rules
            matched = None
            for label, req, forbid in lens_sigs:
                if lens_matches(r.get("lens_exif", ""), (label, req, forbid)):
                    matched = label
                    break
            if matched is None:
                dropped[f"unmatched({tag}):{r.get('lens_label')}"] += 1
                continue
            r["lens_label"] = matched  # canonicalize label
            if pid not in rows_by_id:
                rows_by_id[pid] = r
                src_of[pid] = tag
            # if duplicate, keep existing

    final = list(rows_by_id.values())
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=final[0].keys())
        w.writeheader()
        w.writerows(final)

    print(f"MERGED total: {len(final)}")
    print("dropped rows:", dict(dropped))
    print("by lens:", dict(Counter(r["lens_label"] for r in final)))
    print("from base vs add:", dict(Counter(src_of.values())))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
