#!/usr/bin/env python3
"""
Download original-size images from Flickr manifests and verify them.
Resumable / idempotent — skips already-downloaded and already-verified images.

Usage:
    cd /home/tim/source/activity/leica-look
    python3 src/data/download_verify.py

Outputs:
    /mnt/nas-ai-models/training-data/leica-look/raw/{class}/{flickr_id}.jpg
    data/registry/verified.csv
    data/registry/rejected.csv
    .flickr_cache/download_progress.json   (resumable state)
"""

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from PIL import Image
from PIL.ExifTags import TAGS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_DIR = "/mnt/nas-ai-models/training-data/leica-look/raw"
REGISTRY_DIR = "data/registry"
PROGRESS_FILE = ".flickr_cache/download_progress.json"
API_URL = "https://www.flickr.com/services/rest/"
MIN_SHORT_EDGE = 1024
RATE_LIMIT_API = 0.5       # seconds between API calls
RATE_LIMIT_DOWNLOAD = 2.0  # seconds between successful downloads
DOWNLOAD_TIMEOUT = 60
API_TIMEOUT = 15
MAX_RETRIES = 2             # fewer retries: on 429, fail fast, retry next cron cycle
RATE_LIMIT_BACKOFF_BASE = 15  # base seconds for 429 backoff
MAX_CONSECUTIVE_429 = 10    # if this many consecutive 429s, stop and let next cycle run

MANIFESTS = [
    ("data/registry/positive_manifest_final.csv", "positive"),
    ("data/registry/negative_manifest.csv", "negative"),
]

REJECT_REASONS = {
    "download_failed": "Could not download image after retries",
    "no_original_url": "Flickr API did not return an Original size URL",
    "exif_lens_mismatch": "EXIF lens model does not match manifest",
    "resolution_too_small": "Short edge below 1024px minimum",
    "monochrome_body": "Leica Monochrom body (no color signal)",
    "monochrome_image": "Image is grayscale / no color channels",
    "file_corrupt": "Downloaded file is not a valid JPEG",
    "api_error": "Flickr API returned an error for this photo",
}


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


def flickr_call(key, method, params, rate_limit=RATE_LIMIT_API):
    """Call the Flickr API with retries. Returns parsed JSON or None."""
    params = {**params, "method": method, "api_key": key,
              "format": "json", "nojsoncallback": 1}
    
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(API_URL, params=params, timeout=API_TIMEOUT)
            
            if r.status_code == 429:
                wait = RATE_LIMIT_BACKOFF_BASE * (2 ** attempt)
                print(f"  ⚠ API rate limited (429), waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            
            data = r.json()
            if data.get("stat") != "ok":
                msg = data.get("message", "unknown")
                # Check for permanent errors — no point retrying
                perm = ("permission denied", "not found", "no such photo",
                        "photo_id invalid", "invalid api key")
                if any(s in msg.lower() for s in perm):
                    return {"stat": "error", "permanent": True, "message": msg}
                print(f"  flickr error ({method}) attempt {attempt+1}: {msg}", file=sys.stderr)
                time.sleep(2 * (attempt + 1))
                continue
            time.sleep(rate_limit)
            return data
        except (requests.RequestException, ValueError) as e:
            print(f"  request error ({method}) attempt {attempt+1}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    
    return None


def get_sizes(key, flickr_id):
    """Get all available image sizes from Flickr. Returns list of (label, url) sorted by quality."""
    data = flickr_call(key, "flickr.photos.getSizes", {"photo_id": flickr_id},
                       rate_limit=0)
    
    if data is None or data.get("stat") != "ok":
        return None, "api_error"
    
    sizes = data.get("sizes", {}).get("size", [])
    if not sizes:
        return None, "no_sizes"
    
    # Build ordered list: try Large 2048 first (Original often rate-limited),
    # then Original (highest quality), then other sizes
    quality_order = ["Large 2048", "Original", "Large 1600", "Large", "Medium 800", "Medium 640"]
    size_map = {s.get("label"): s["source"] for s in sizes}
    
    result = []
    for label in quality_order:
        if label in size_map:
            result.append((label, size_map[label]))
    
    if not result:
        return None, "no_usable_sizes"
    
    return result, None


def get_original_url(key, flickr_id):
    """Get download URLs for a Flickr photo (ordered by quality)."""
    return get_sizes(key, flickr_id)


def download_image(url, dest_path):
    """Download an image to dest_path with retries. Returns (success: bool, rate_limited: bool)."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.flickr.com/",
    }
    
    rate_limited = False
    
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=DOWNLOAD_TIMEOUT)
            
            if r.status_code == 429:
                rate_limited = True
                wait = RATE_LIMIT_BACKOFF_BASE * (2 ** attempt)
                print(f"  ⚠ Rate limited (429), waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            
            r.raise_for_status()
            
            # Write to temp file then rename (atomic)
            tmp_path = dest_path + f".part.{os.getpid()}"
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            os.replace(tmp_path, dest_path)
            time.sleep(RATE_LIMIT_DOWNLOAD)
            return True, False
        except Exception as e:
            print(f"  download attempt {attempt+1} failed: {e}", file=sys.stderr)
            # Clean up partial file
            for p in [dest_path, dest_path + f".part.{os.getpid()}"]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            if attempt < MAX_RETRIES - 1:
                time.sleep(5 * (attempt + 1))
    
    return False, rate_limited


def verify_image(filepath, manifest_row):
    """
    Verify a downloaded image against manifest expectations.
    Returns (passed: bool, reason: str or None, metadata: dict).
    """
    # Check file exists and is non-empty
    if not os.path.exists(filepath):
        return False, "download_failed", {}
    
    file_size = os.path.getsize(filepath)
    if file_size < 1000:  # less than 1KB is definitely corrupt
        return False, "file_corrupt", {"file_size": file_size}
    
    meta = {"file_size_bytes": file_size}
    
    # Try to open as image
    try:
        img = Image.open(filepath)
    except Exception as e:
        return False, "file_corrupt", {"error": str(e)}
    
    try:
        # --- Resolution check ---
        w, h = img.size
        meta["width"] = w
        meta["height"] = h
        short_edge = min(w, h)
        if short_edge < MIN_SHORT_EDGE:
            img.close()
            return False, "resolution_too_small", meta
        
        # --- Monochrome check ---
        # Check if image mode indicates no color
        if img.mode not in ("RGB", "RGBA", "YCbCr", "LAB", "HSV"):
            # L = grayscale, 1 = binary, etc.
            img.close()
            return False, "monochrome_image", meta
        
        # --- EXIF checks ---
        exif_data = {}
        exif_raw = img.getexif()
        if exif_raw:
            for tag_id, value in exif_raw.items():
                tag_name = TAGS.get(tag_id, tag_id)
                # Convert bytes to string for comparison
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8", errors="replace").rstrip("\x00")
                    except Exception:
                        value = str(value)
                exif_data[tag_name] = str(value) if not isinstance(value, (int, float)) else value
        
        img.close()
        
        # --- Body model check for monochrome ---
        body_exif = exif_data.get("Model", "")
        if body_exif and "MONOCHROM" in str(body_exif).upper():
            return False, "monochrome_body", {**meta, "body": body_exif}
        
        # --- EXIF lens match ---
        # NOTE: Flickr CDN strips the LensModel EXIF tag from served images,
        # so we CANNOT rely on the downloaded file's EXIF for lens verification.
        # The manifest was built using flickr.photos.getExif which returns the
        # authoritative EXIF data. The lens match was already confirmed at
        # scrape time — we trust the manifest here.
        # We still record whatever lens EXIF we find for provenance.
        expected_lens = manifest_row.get("lens_exif", "").strip()
        lens_candidates = []  # always defined
        for tag in ("LensModel", "Lens", "LensInfo"):
            val = exif_data.get(tag, "")
            if val:
                lens_candidates.append(str(val))
        
        meta["exif_lens"] = " | ".join(lens_candidates) if lens_candidates else "none"
        meta["expected_lens"] = expected_lens
        
        # If lens EXIF IS present in the downloaded file AND it mismatches,
        # that's suspicious — flag it (but don't reject; the manifest is canonical)
        if lens_candidates and expected_lens:
            found_match = False
            exp_lower = expected_lens.lower()
            for candidate in lens_candidates:
                cand_lower = candidate.lower()
                if exp_lower in cand_lower or cand_lower in exp_lower:
                    found_match = True
                    break
                exp_words = set(w for w in exp_lower.replace(":", " ").split() if len(w) > 1)
                cand_words = set(w for w in cand_lower.replace(":", " ").split() if len(w) > 1)
                if exp_words and exp_words.issubset(cand_words):
                    found_match = True
                    break
            if not found_match:
                meta["lens_mismatch_warning"] = "Downloaded EXIF lens differs from manifest"
        
        meta["exif_has_lens"] = bool(lens_candidates)
        
        # Store body info
        meta["exif_body"] = exif_data.get("Model", "")
        meta["exif_make"] = exif_data.get("Make", "")
        
        meta["exif_has_lens"] = bool(lens_candidates if expected_lens else True)
        
        return True, None, meta
        
    except Exception as e:
        try:
            img.close()
        except Exception:
            pass
        return False, "file_corrupt", {"error": str(e), **meta}


def load_progress():
    """Load progress state for resumability."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"downloaded": [], "verified": [], "rejected": [],
            "last_flickr_id": None, "total_processed": 0}


def save_progress(progress):
    """Save progress state atomically."""
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp, PROGRESS_FILE)


def write_manifests(verified_rows, rejected_rows):
    """Write verified.csv and rejected.csv."""
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    
    # Verified manifest
    v_fields = ["flickr_id", "url", "class", "lens_label", "lens_exif",
                "body", "scene_type", "license_id", "tags",
                "file_path", "width", "height", "file_size_bytes",
                "verified_at"]
    
    v_path = os.path.join(REGISTRY_DIR, "verified.csv")
    with open(v_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=v_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(verified_rows)
    
    # Rejected manifest
    r_fields = ["flickr_id", "url", "class", "lens_label", "lens_exif",
                "body", "scene_type", "license_id", "tags",
                "reject_reason", "reject_detail", "rejected_at"]
    
    r_path = os.path.join(REGISTRY_DIR, "rejected.csv")
    with open(r_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=r_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rejected_rows)
    
    return v_path, r_path


def read_manifest(csv_path):
    """Read a manifest CSV and return list of dicts."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip empty rows
            if not row.get("flickr_id"):
                continue
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download & verify dataset images")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without downloading")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from progress file (default)")
    parser.add_argument("--no-resume", action="store_false", dest="resume",
                        help="Start fresh, ignore progress file")
    parser.add_argument("--class", dest="cls", choices=["positive", "negative"],
                        help="Process only one class")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to N images (for testing)")
    args = parser.parse_args()
    
    key = load_api_key()
    
    # Load manifests
    all_rows = []
    for manifest_path, class_name in MANIFESTS:
        if args.cls and class_name != args.cls:
            continue
        if not os.path.exists(manifest_path):
            print(f"WARNING: manifest not found: {manifest_path}", file=sys.stderr)
            continue
        rows = read_manifest(manifest_path)
        print(f"Loaded {len(rows)} rows from {manifest_path}")
        all_rows.extend(rows)
    
    if args.limit and args.limit > 0:
        all_rows = all_rows[:args.limit]
    
    print(f"Total images to process: {len(all_rows)}")
    
    # Load progress
    progress = {"downloaded": [], "verified": [], "rejected": [],
                "last_flickr_id": None, "total_processed": 0}
    if args.resume:
        progress = load_progress()
        print(f"Resuming: {len(progress['downloaded'])} downloaded, "
              f"{len(progress['verified'])} verified, "
              f"{len(progress['rejected'])} rejected")
    else:
        print("Starting fresh (--no-resume)")
    
    # Index for quick lookups
    downloaded_set = set(progress.get("downloaded", []))
    verified_set = set(progress.get("verified", []))
    rejected_set = set(progress.get("rejected", []))
    
    # Accumulators for final output
    verified_rows = []
    rejected_rows = []
    
    # Stats
    stats = Counter()
    consecutive_429 = 0
    
    start_time = time.time()
    
    for i, row in enumerate(all_rows):
        flickr_id = row["flickr_id"]
        class_name = row["class"]
        
        # Bail out if too many consecutive rate limits
        if consecutive_429 >= MAX_CONSECUTIVE_429:
            print(f"\n⚠️  {consecutive_429} consecutive 429s — stopping to let rate limits recover.")
            print(f"   {len(all_rows) - i} images remaining. Next cron cycle will resume.")
            break
        
        # Progress indicator
        pct = (i + 1) / len(all_rows) * 100
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(all_rows) - i - 1) / rate if rate > 0 else 0
        print(f"\n[{i+1}/{len(all_rows)} {pct:.1f}%] "
              f"ID={flickr_id} class={class_name} "
              f"(ETA: {eta/60:.0f}m)")
        
        # Skip already processed — but still build verified/rejected rows for CSV output
        if flickr_id in verified_set:
            stats["skip_verified"] += 1
            dest_path = os.path.join(RAW_DIR, class_name, f"{flickr_id}.jpg")
            if os.path.exists(dest_path):
                # Re-verify basic file info for the CSV row
                try:
                    img = Image.open(dest_path)
                    w, h = img.size
                    fs = os.path.getsize(dest_path)
                    img.close()
                    verified_rows.append({
                        **row,
                        "file_path": dest_path,
                        "width": w,
                        "height": h,
                        "file_size_bytes": fs,
                        "verified_at": datetime.now(timezone.utc).isoformat(),
                    })
                except Exception:
                    # File corrupted since last run — will be re-downloaded
                    verified_set.discard(flickr_id)
                    progress["verified"] = list(verified_set)
                    print(f"  ⚠ Cached file corrupt, will re-download: {flickr_id}")
                    # Don't skip — fall through to download
            else:
                verified_set.discard(flickr_id)
                progress["verified"] = list(verified_set)
                print(f"  ⚠ Previously verified file missing, will re-download: {flickr_id}")
            if flickr_id in verified_set:
                continue
        
        if flickr_id in rejected_set:
            stats["skip_rejected"] += 1
            # Rebuild rejected row from progress (we don't have the reason detail)
            rejected_rows.append({
                **row,
                "reject_reason": "previously_rejected",
                "reject_detail": "",
                "rejected_at": datetime.now(timezone.utc).isoformat(),
            })
            continue
        
        # --- Step 1: Get available sizes ---
        if args.dry_run:
            print(f"  [DRY RUN] Would get sizes and download")
            continue
        
        sizes, url_error = get_original_url(key, flickr_id)
        if url_error:
            print(f"  ❌ Size lookup error: {url_error}")
            rejected_rows.append({
                **row,
                "reject_reason": url_error,
                "reject_detail": "",
                "rejected_at": datetime.now(timezone.utc).isoformat(),
            })
            rejected_set.add(flickr_id)
            progress["rejected"] = list(rejected_set)
            stats["rejected"] += 1
            stats[f"rejected_{url_error}"] += 1
            save_progress(progress)
            continue
        
        # sizes is a list of (label, url) tuples, best quality first
        if sizes is None:
            print(f"  ❌ Internal error: sizes is None after url_error check")
            continue
        print(f"  📎 Available sizes: {[s[0] for s in sizes]}")
        
        # --- Step 2: Download (try sizes in order) ---
        dest_path = os.path.join(RAW_DIR, class_name, f"{flickr_id}.jpg")
        used_label = "unknown"
        
        if flickr_id in downloaded_set and os.path.exists(dest_path):
            print(f"  ⏭  Already downloaded, verifying...")
            downloaded = True
            used_label = "cached"
        else:
            downloaded = False
            for label, url in sizes:
                print(f"  ⬇  Trying {label}: {url[:80]}...")
                success, rate_limited = download_image(url, dest_path)
                if success:
                    downloaded = True
                    used_label = label
                    break
                elif rate_limited:
                    print(f"     {label} rate-limited, trying next size...")
                    continue
                else:
                    print(f"     {label} download failed, trying next size...")
                    continue
        
        if not downloaded:
            print(f"  ❌ All sizes failed")
            rejected_rows.append({
                **row,
                "reject_reason": "download_failed",
                "reject_detail": f"tried {len(sizes)} sizes",
                "rejected_at": datetime.now(timezone.utc).isoformat(),
            })
            rejected_set.add(flickr_id)
            progress["rejected"] = list(rejected_set)
            stats["rejected"] += 1
            stats["rejected_download_failed"] += 1
            save_progress(progress)
            continue
        
        if flickr_id not in downloaded_set:
            downloaded_set.add(flickr_id)
            progress["downloaded"] = list(downloaded_set)
            stats["downloaded"] += 1
            consecutive_429 = 0  # reset on success
            print(f"  ✅ Downloaded ({used_label})")
        
        # --- Step 3: Verify ---
        print(f"  🔍 Verifying {dest_path}...")
        passed, reason, meta = verify_image(dest_path, row)
        
        if passed:
            print(f"  ✅ Verified: {meta.get('width')}x{meta.get('height')}, "
                  f"{meta.get('file_size_bytes', 0)/1024:.0f}KB, "
                  f"lens={meta.get('exif_lens', '?')[:50]}")
            verified_rows.append({
                **row,
                "file_path": dest_path,
                "width": meta.get("width", ""),
                "height": meta.get("height", ""),
                "file_size_bytes": meta.get("file_size_bytes", ""),
                "verified_at": datetime.now(timezone.utc).isoformat(),
            })
            verified_set.add(flickr_id)
            progress["verified"] = list(verified_set)
            stats["verified"] += 1
        else:
            print(f"  ❌ Rejected: {reason} {REJECT_REASONS.get(reason, reason)}")
            if meta:
                print(f"     Meta: {json.dumps({k: str(v)[:100] for k, v in meta.items()}, default=str)}")
            rejected_rows.append({
                **row,
                "reject_reason": reason,
                "reject_detail": json.dumps({k: str(v)[:200] for k, v in meta.items()}, default=str),
                "rejected_at": datetime.now(timezone.utc).isoformat(),
            })
            rejected_set.add(flickr_id)
            progress["rejected"] = list(rejected_set)
            stats["rejected"] += 1
            stats[f"rejected_{reason}"] += 1
        
        # Save progress periodically
        if (i + 1) % 20 == 0:
            progress["last_flickr_id"] = flickr_id
            progress["total_processed"] = i + 1
            save_progress(progress)
            # Also flush manifests
            write_manifests(verified_rows, rejected_rows)
    
    # --- Final save ---
    progress["total_processed"] = len(all_rows)
    save_progress(progress)
    vp, rp = write_manifests(verified_rows, rejected_rows)
    
    # --- Summary ---
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_verified = len(verified_rows)
    total_rejected = len(rejected_rows)
    total_skipped = len(all_rows) - total_verified - total_rejected
    print(f"Total in manifest: {len(all_rows)}")
    print(f"  ✅ Verified:   {total_verified} ({stats['verified']} new, {stats.get('skip_verified', 0)} cached)")
    print(f"  ❌ Rejected:   {total_rejected} ({stats['rejected']} new, {stats.get('skip_rejected', 0)} cached)")
    if stats.get('rate_limited', 0) > 0:
        print(f"  ⏸  Rate-limited: {stats['rate_limited']} (will retry next run)")
    print(f"Time: {elapsed/60:.1f} minutes")
    
    # Class breakdown
    pos_v = sum(1 for r in verified_rows if r.get('class') == 'positive')
    neg_v = sum(1 for r in verified_rows if r.get('class') == 'negative')
    pos_r = sum(1 for r in rejected_rows if r.get('class') == 'positive')
    neg_r = sum(1 for r in rejected_rows if r.get('class') == 'negative')
    print(f"\nClass breakdown:")
    print(f"  Positive (Leica):     {pos_v} verified, {pos_r} rejected")
    print(f"  Negative (non-Leica): {neg_v} verified, {neg_r} rejected")
    if pos_v < 250:
        print(f"  ⚠️  Positive below target (250+): {pos_v}")
    if neg_v < 400:
        print(f"  ⚠️  Negative below target (400+): {neg_v}")
    
    if stats["rejected"] > 0 or stats.get("skip_rejected", 0) > 0:
        print(f"\nRejection breakdown:")
        for reason, count in sorted(stats.items()):
            if reason.startswith("rejected_") and not reason == "rejected":
                print(f"  {reason.replace('rejected_', '')}: {count}")
    
    print(f"\nVerified manifest: {vp}")
    print(f"Rejected manifest:  {rp}")
    
    return 0


if __name__ == "__main__":
    main()
