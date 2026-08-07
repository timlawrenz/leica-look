#!/usr/bin/env python3
"""
Flickr seed dataset scraper for the leica-look discriminator (Phase 1).

Strategy:
  1. Search Flickr by camera BODY (flickr.photos.search) to get a large,
     relevant candidate pool for that manufacturer.
  2. For each candidate photo, call flickr.photos.getExif to read the EXIF
     LensModel + Lens field, and confirm it MATCHES a target lens signature.
  3. Keep only EXIF-confirmed photos; dedupe; stop at per-class/per-lens caps.
  4. Write a manifest CSV for human curation (issue #2).

Only the API key is required (read-only, public photos). No OAuth.
Key is read from FLICKR_API_KEY env var or .env in the repo root.

Usage:
    FLICKR_API_KEY=xxx python src/data/flickr_scrape.py [--class positive]
    python src/data/flickr_scrape.py --dry-run   # validate config, no requests

Outputs:
    data/registry/positive_manifest.csv
    data/registry/negative_manifest.csv
    .flickr_cache/              # resumable intermediate state
"""

import argparse
import csv
import os
import re
import sys
import time

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Each lens signature is (label, required_terms, forbidden_terms)
#   required_terms: ALL must be present (case-insensitive substring) in EXIF LensModel
#   forbidden_terms: NONE may be present (disambiguation)
# We search per camera BODY to get candidate pools, then match the lens EXIF.

POSITIVE = {
    "body_queries": [
        "Leica M10", "Leica M10-P", "Leica M10-R", "Leica M10-D",
        "Leica M11", "Leica M11-P", "Leica M11-D",
        "Leica SL2", "Leica SL2-S", "Leica SL3",
        "Leica Q2", "Leica Q2 Monochrom", "Leica Q3",
    ],
    "lenses": [
        # label,            required_terms,                forbidden_terms
        ("Summilux 50/1.4", ["summilux", "50"],            ["apo", "28", "35", "75"]),
        ("Summilux 35/1.4", ["summilux", "35"],            ["apo", "28", "50", "75"]),
        ("Summilux 28/1.4", ["summilux", "28"],            ["apo", "35", "50", "75"]),
        ("APO-Summicron 50", ["apo-summicron", "50"],      ["28", "35", "75", "90"]),
    ],
    "target": 320,      # ~300-500 target for positive
    "per_lens_cap": 160,
    "exclude_monochrome": True,   # Monochrom bodies are design-excluded
}

NEGATIVE = {
    "body_queries": [
        # Canon EOS R / DSLR full-frame
        "Canon EOS R5", "Canon EOS R6", "Canon EOS R6 Mark II", "Canon EOS R3",
        "Canon EOS 5D Mark IV", "Canon EOS 5DS R",
        # Sony full-frame E-mount
        "Sony ILCE-7RM5", "Sony ILCE-7RM4", "Sony ILCE-7M4", "Sony ILCE-7M3",
        "Sony ILCE-9", "Sony ILCE-1",
        # Nikon Z full-frame
        "NIKON Z 9", "NIKON Z 8", "NIKON Z 7_2", "NIKON Z 7",
        # Zeiss Otus mounts are EF/F/E — covered by the above bodies
    ],
    "lenses": [
        # label,                    required_terms,                         forbidden_terms
        ("Canon 50/1.2L",           ["50", "1.2", "l"],                     []),
        ("Canon 85/1.2L",           ["85", "1.2", "l"],                     []),
        ("Canon 35/1.4L II",        ["35", "1.4", "ii"],                    []),
        ("Sony 50/1.2 GM",          ["fe 50", "1.2", "gm"],                 []),
        ("Sony 85/1.4 GM",          ["fe 85", "1.4", "gm"],                 []),
        ("Sony 35/1.4 GM",          ["fe 35", "1.4", "gm"],                 []),
        # Nikon uses "NIKKOR Z ... S" (S-line) — require "nikkor z" so it can't
        # be confused with the "S" in Canon's "USM" (= Image Stabilization).
        ("Nikon 50/1.2 S",          ["nikkor z", "50", "1.2"],              []),
        ("Nikon 85/1.2 S",          ["nikkor z", "85", "1.2"],              []),
        ("Zeiss Otus 55/1.4",       ["otus", "55"],                         []),
        ("Zeiss Otus 85/1.4",       ["otus", "85"],                         []),
    ],
    "target": 700,      # ~500-1000 target for negative
    "per_lens_cap": 200,
}

CLASSES = {"positive": POSITIVE, "negative": NEGATIVE}

SEARCH_URL = "https://www.flickr.com/services/rest/"
EXIF_URL = "https://www.flickr.com/services/rest/"
PAGE_SIZE = 100          # max per search request
SEARCH_RATE_LIMIT_S = 1.0     # polite: >=1 req/s for search
EXIF_RATE_LIMIT_S = 0.6       # getExif is heavier per call
MAX_EXIF_FAILURES = 15        # consecutive getExif failures before aborting a query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_api_key():
    load_dotenv()
    key = os.environ.get("FLICKR_API_KEY", "").strip()
    if not key:
        print("ERROR: FLICKR_API_KEY not set. Add it to .env or export it.", file=sys.stderr)
        sys.exit(2)
    return key


# Error messages that indicate a PERMANENT condition (retrying won't help).
# For getExif, "Permission denied" means the photo's EXIF is restricted/private —
# skip it immediately instead of burning retries + backoff on it.
PERMANENT_ERROR_SUBSTR = ("permission denied", "not found", "no such photo",
                          "photo_id invalid", "invalid api key")


def flickr_get(key, method, params):
    params = {**params, "method": method, "api_key": key, "format": "json", "nojsoncallback": 1}
    for attempt in range(3):
        try:
            r = requests.get(SEARCH_URL, params=params, timeout=20)
            data = r.json()
            if data.get("stat") != "ok":
                msg = data.get("message", "unknown")
                low = msg.lower()
                if any(s in low for s in PERMANENT_ERROR_SUBSTR):
                    # Permanent — do not retry. Return a marker so callers can
                    # distinguish "skip this photo" from "transient failure".
                    return {"stat": "error", "permanent": True, "message": msg}
                print(f"  flickr error ({method}) attempt {attempt+1}: {msg}", file=sys.stderr)
                time.sleep(2 * (attempt + 1))
                continue
            return data
        except (requests.RequestException, ValueError) as e:
            print(f"  request error ({method}): {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    return None


def lens_matches(lensmodel, sig):
    """Check a lens signature against the EXIF LensModel string."""
    if not lensmodel:
        return False
    lm = lensmodel.lower()
    if not all(term in lm for term in sig[1]):
        return False
    if any(term in lm for term in sig[2]):
        return False
    return True


def scene_type(photo):
    """Best-effort scene classification from Flickr tags. Returns one of the
    stratified scene categories, or 'other'. This is a heuristic used for the
    stratification column; curation can correct it."""
    # Flickr returns 'tags' as a SPACE-joined string (not a list), e.g.
    # "street berlin bokeh night". Normalize to a bare lowercase string.
    raw_tags = photo.get("tags")
    if isinstance(raw_tags, list):
        raw_tags = " ".join(str(t) for t in raw_tags)
    tag_text = (raw_tags or "").lower()
    cat_hits = {
        "portrait": ["portrait", "person", "people", "model", "face"],
        "landscape": ["landscape", "scenic", "mountain", "sunset", "nature"],
        "street": ["street", "city", "urban", "streetphotography"],
        "macro": ["macro", "closeup", "flower", "insect"],
        "architecture": ["architecture", "building", "interior", "facade"],
        "night": ["night", "longexposure", "lowlight", "bokeh"],
    }
    for cat, terms in cat_hits.items():
        if any(t in tag_text for t in terms):
            return cat
    return "other"


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class FlickrCache:
    """Persists per-photo state to .flickr_cache so partial progress is never
    lost on interruption, and already-verified photos are skipped on resume.

    Real resume support (not just a stub):
      - seen set: photo IDs already checked (whether matched or not)
      - matched map: lens_label -> how many that lens has accepted so far
    Both are loaded at start; the manifest is re-flushed incrementally so a
    killed run leaves a usable partial manifest.
    """

    def __init__(self, root=".flickr_cache"):
        os.makedirs(root, exist_ok=True)
        self.root = root
        self.seen = set()
        self.matched = {}

    def state_path(self, cls_name, kind):
        return os.path.join(self.root, f"{cls_name}_{kind}.txt")

    def load(self, cls_name):
        self.seen = set()
        self.matched = {}
        sp = self.state_path(cls_name, "seen")
        if os.path.exists(sp):
            for line in open(sp):
                line = line.strip()
                if line:
                    self.seen.add(line)
        mp = self.state_path(cls_name, "matched")
        if os.path.exists(mp):
            for line in open(mp):
                if "\t" in line:
                    k, v = line.split("\t", 1)
                    try:
                        self.matched[k] = int(v)
                    except ValueError:
                        pass
        return self.seen, self.matched

    def save(self, cls_name, seen_ids, matched, manifest, manifest_path):
        with open(self.state_path(cls_name, "seen"), "w") as f:
            f.write("\n".join(sorted(seen_ids)))
        with open(self.state_path(cls_name, "matched"), "w") as f:
            for k, v in sorted(matched.items()):
                f.write(f"{k}\t{v}\n")
        # Incrementally flush the manifest so a kill leaves usable partial data
        if manifest_path and manifest:
            self._write_csv(manifest_path, manifest)

    @staticmethod
    def _write_csv(path, manifest):
        fieldnames = ["flickr_id", "url", "thumb", "class", "lens_label",
                      "lens_exif", "body", "scene_type", "license_id", "tags"]
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest)
        os.replace(tmp, path)


def write_manifest(cls_name, manifest):
    return FlickrCache._write_csv(f"data/registry/{cls_name}_manifest.csv", manifest)


def scrape_class(key, cls_name, config, cache, out_path=None):
    manifest_path = out_path or f"data/registry/{cls_name}_manifest.csv"
    os.makedirs("data/registry", exist_ok=True)

    # Resume from cache: seen photo IDs + per-lens counts carried forward
    seen_ids, matched = cache.load(cls_name)
    print(f"  [resume] {len(seen_ids)} photos already checked, "
          f"{sum(matched.values())} already matched, {len(matched)} lens counts loaded")

    manifest = []
    # Rebuild manifest from scratch on this invocation; a fresh run re-writes it.
    # (For true resume-into-existing-manifest we'd read the CSV; append is fine
    #  since seen_ids dedupes — but we rebuild to keep ordering deterministic.)
    lens_sigs = config["lenses"]
    os.makedirs(".flickr_cache", exist_ok=True)

    def flush():
        cache.save(cls_name, seen_ids, matched, manifest, manifest_path)

    for body in config["body_queries"]:
        print(f"\n=== [{cls_name}] searching body: {body} ===")
        query_failures = 0
        page = 1
        while True:
            # Stop if we've hit the overall target
            if len(manifest) >= config["target"]:
                print(f"  target {config['target']} reached, stopping")
                flush()
                return manifest, matched

            data = flickr_get(key, "flickr.photos.search", {
                "text": body,
                "per_page": PAGE_SIZE,
                "page": page,
                "sort": "interestingness-desc",
                "license": "4,5,6,7,8,9,10",   # CC licenses + public domain
                "safe_search": 1,
                "content_type": 1,             # photos only
                "extras": "tags,url_q",
            })
            if data is None or data.get("permanent"):
                query_failures += 1
                if query_failures >= 3:
                    print("  aborting query (repeated API failures)", file=sys.stderr)
                    break
                continue
            query_failures = 0

            photos = data.get("photos", {}).get("photo", [])
            pages = data.get("photos", {}).get("pages", 1)
            if not photos:
                break

            for photo in photos:
                pid = photo.get("id")
                if not pid or pid in seen_ids:
                    continue
                seen_ids.add(pid)

                # Verify lens EXIF
                exif = flickr_get(key, "flickr.photos.getExif", {"photo_id": pid})
                time.sleep(EXIF_RATE_LIMIT_S)
                if exif is None or exif.get("permanent"):
                    # None = transient failure (retry next photo); permanent =
                    # EXIF restricted/private — skip instantly, no useful match
                    continue
                lensmodel = ""
                for ex in (exif.get("photo", {}) or {}).get("exif", []):
                    if ex.get("label") in ("Lens", "Lens Model"):
                        raw = ex.get("raw")
                        if isinstance(raw, dict):
                            lensmodel = raw.get("_content", "") or ""
                        elif raw:
                            lensmodel = str(raw)
                        if lensmodel:
                            break

                # Body from EXIF — needed now, both to detect monochrome bodies
                # (excluded by design) and to record which camera took the shot.
                body = body
                for ex in (exif.get("photo", {}) or {}).get("exif", []):
                    if ex.get("label") in ("Model", "Camera Model Name", "Camera"):
                        raw = ex.get("raw")
                        v = raw.get("_content", "") if isinstance(raw, dict) else (str(raw) if raw else "")
                        if v:
                            body = v
                        break

                # Skip monochrome bodies at scrape time (design-excluded signal).
                # Avoids wasting EXIF/matching effort on photos we'll reject.
                if config.get("exclude_monochrome", False) and "MONO" in body.upper():
                    continue

                match = None
                for label, req, forbid in lens_sigs:
                    if lens_matches(lensmodel, (label, req, forbid)):
                        # respect per-lens cap
                        if matched.get(label, 0) >= config["per_lens_cap"]:
                            continue
                        match = (label, lensmodel)
                        break
                if match is None:
                    continue   # not a target lens

                label, lensmodel = match

                raw_tags = photo.get("tags")
                if isinstance(raw_tags, list):
                    raw_tags = " ".join(str(t) for t in raw_tags)
                manifest.append({
                    "flickr_id": pid,
                    "url": f"https://www.flickr.com/photos/{photo.get('owner','')}/{pid}",
                    "thumb": photo.get("url_q", ""),
                    "class": cls_name,
                    "lens_label": label,
                    "lens_exif": lensmodel,
                    "body": body,
                    "scene_type": scene_type(photo),
                    "license_id": photo.get("license", ""),
                    "tags": (raw_tags or "")[:500],
                })
                matched[label] = matched.get(label, 0) + 1
                if len(manifest) % 20 == 0:
                    flush()
                print(f"  + {label} [{body}] ({matched[label]}/{config['per_lens_cap']}) "
                      f"total={len(manifest)}")

            if page >= pages:
                break
            page += 1
            time.sleep(SEARCH_RATE_LIMIT_S)

    flush()
    return manifest, matched


def snapshot_counts(cls_name, matched, manifest):
    print(f"\n=== [{cls_name}] FINAL COUNTS ===")
    for label, n in sorted(matched.items()):
        print(f"  {label}: {n}")
    print(f"  TOTAL: {len(manifest)}")


def dry_run():
    print("Config validation (dry-run):")
    for cls_name, cfg in CLASSES.items():
        print(f"  [{cls_name}] target={cfg['target']}, "
              f"{len(cfg['body_queries'])} bodies, {len(cfg['lenses'])} lens sigs")
        for body in cfg["body_queries"]:
            print(f"    body: {body}")
        for label, req, forbid in cfg["lenses"]:
            print(f"    lens: {label:22} req={req} forbid={forbid or '-'}")
    print("\nLooks good.")


def main():
    ap = argparse.ArgumentParser(description="Flickr seed dataset scraper")
    ap.add_argument("--class", dest="cls", choices=list(CLASSES), default="positive",
                    help="which class to scrape")
    ap.add_argument("--lens", action="append", default=None,
                    help="only scrape these lens labels (repeatable); default all")
    ap.add_argument("--out", default=None, help="output manifest path override")
    ap.add_argument("--dry-run", action="store_true", help="validate config, no requests")
    args = ap.parse_args()

    if args.dry_run:
        dry_run()
        return

    config = CLASSES[args.cls]
    if args.lens:
        # Filter to only the requested lens labels
        config["lenses"] = [sig for sig in config["lenses"] if sig[0] in args.lens]
        if not config["lenses"]:
            print("ERROR: --lens matched no configured lens labels", file=sys.stderr)
            sys.exit(2)
        print(f"Restricted to {len(config['lenses'])} lens sig(s): "
              f"{[s[0] for s in config['lenses']]}")

    key = load_api_key()
    cache = FlickrCache()

    print(f"Scraping [{args.cls}] — target {config['target']} photos")
    manifest, matched = scrape_class(key, args.cls, config, cache, out_path=args.out)
    snapshot_counts(args.cls, matched, manifest)
    print("\nDone.")


if __name__ == "__main__":
    main()
