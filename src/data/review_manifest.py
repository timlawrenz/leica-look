#!/usr/bin/env python3
"""
Generate a visual contact-sheet review page for manifest curation.

For each photo it downloads (or reuses) the Flickr thumbnail, then renders an
HTML page grouped by lens so the human can eyeball all photos at once and mark
any for rejection. Output: <out_dir>/review.html

Rejected selection: the HTML uses checkboxes with value=flickr_id. The human can
run `python sample_for_review.py` variant or just tell the agent the IDs; also
a tiny JS collects selected IDs into a textarea for easy copy-paste.

Usage:
    python src/data/review_manifest.py \
        --manifest data/registry/positive_manifest_final.csv \
        --out /home/tim/source/activity/leica-look/review
"""

import argparse
import csv
import os
import re
import requests
from collections import defaultdict

THUMB_DIR = "thumbs"


def load_manifest(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s)


def download_thumb(photo, outdir):
    url = photo.get("thumb") or photo.get("url_q") or ""
    if not url:
        return None
    dest = os.path.join(outdir, f"{photo['flickr_id']}.jpg")
    if os.path.exists(dest):
        return dest
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            os.makedirs(outdir, exist_ok=True)
            with open(dest, "wb") as f:
                f.write(r.content)
            return dest
    except requests.RequestException as e:
        print(f"  warn: download fail {photo['flickr_id']}: {e}")
    return None


def build_html(rows, thumbs, outpath):
    by_lens = defaultdict(list)
    for r in rows:
        by_lens[r["lens_label"]].append(r)

    parts = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<title>leica-look positive review</title>")
    parts.append("<style>"
                 "body{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}"
                 "h2{color:#ccc;margin-top:32px;border-bottom:1px solid #333;padding-bottom:6px}"
                 ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}"
                 ".card{background:#1c1c1c;border:1px solid #333;border-radius:8px;padding:8px;position:relative}"
                 ".card img{width:100%;height:auto;border-radius:4px;object-fit:cover}"
                 ".card .meta{font-size:11px;color:#999;margin-top:6px}"
                 ".card.reject{outline:3px solid #e0234a;background:#3a1520}"
                 ".card input{position:absolute;top:10px;right:10px;transform:scale(1.6)}"
                 ".count{color:#7aa}"
                 "#sel{width:100%;height:80px;font-family:monospace;background:#000;color:#7f7;border:1px solid #444}"
                 "</style></head><body>")
    parts.append("<h1>leica-look positive review</h1>")
    parts.append(f"<p class='count'>Total: {len(rows)} · group by lens. "
                 "Check the box on any photo to reject. Copy the IDs below.</p>")

    total = 0
    for lens in sorted(by_lens):
        group = by_lens[lens]
        total += len(group)
        parts.append(f"<h2>{lens} <span class='count'>({len(group)})</span></h2>")
        parts.append("<div class='grid'>")
        for r in group:
            img = thumbs.get(r["flickr_id"])
            url = r.get("url", "#")
            pid = r["flickr_id"]
            body = r.get("body", "")
            scene = r.get("scene_type", "")
            img_html = f"<img src='thumbs/{pid}.jpg'>" if img else "<div>no thumb</div>"
            parts.append(
                f"<div class='card' id='c-{pid}'>"
                f"<input type='checkbox' class='rej' value='{pid}'/>"
                f"<a href='{url}' target='_blank'>{img_html}</a>"
                f"<div class='meta'>{body} · {scene}<br>{pid}</div></div>")
        parts.append("</div>")

    parts.append("<h2>Reject these IDs</h2>")
    parts.append("<textarea id='sel' readonly placeholder='checked IDs appear here'></textarea>")
    parts.append("<script>"
                 "function collect(){"
                 "var ids=[];"
                 "document.querySelectorAll('.rej:checked').forEach(function(c){ids.push(c.value)});"
                 "document.getElementById('sel').value = ids.join('\\n');"
                 "}"
                 "document.querySelectorAll('.rej').forEach(function(c){"
                 "  c.addEventListener('change', function(){"
                 "    var card=document.getElementById('c-'+c.value);"
                 "    if(c.checked){card.classList.add('reject')}else{card.classList.remove('reject')}"
                 "    collect();"
                 "  });"
                 "});"
                 "</script>")
    parts.append("</body></html>")
    with open(outpath, "w") as f:
        f.write("".join(parts))
    print(f"wrote {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="review")
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    os.makedirs(args.out, exist_ok=True)
    thumbdir = os.path.join(args.out, THUMB_DIR)
    thumbs = {}
    for i, r in enumerate(rows):
        t = download_thumb(r, thumbdir)
        thumbs[r["flickr_id"]] = t
        if (i + 1) % 50 == 0:
            print(f"  downloaded {i+1}/{len(rows)} thumbs")
    print(f"downloaded {len([t for t in thumbs.values() if t])}/{len(rows)} thumbs")
    build_html(rows, thumbs, os.path.join(args.out, "review.html"))


if __name__ == "__main__":
    main()
