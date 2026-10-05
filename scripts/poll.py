#!/usr/bin/env python3
"""Poll @senatorhawley for new Instagram posts, translate them via the
hawley-translator Cloudflare Worker, and queue drafts for HUMAN REVIEW.

Scope: post caption + text visible in images (no video transcription).

OCR engine: PaddleOCR PP-OCRv5 — the SAME engine realufo.org's crawler uses
(crawler/ingest/ocr.py: PP-OCRv5_mobile_det + en_PP-OCRv5_mobile_rec @ score
threshold 0.5). It runs here on the VM (PaddleOCR can't run inside a
Cloudflare Worker); the Worker only translates. Needs the .venv-ocr venv
(see README); without it the script still runs, caption-only.

Image URLs are resolved via Instagram's public oEmbed endpoint
(/api/v1/oembed/) which returns a thumbnail_url (scontent CDN) for any post
type — incl. plain image posts that instagram-cli exposes no media URL for.
(Carousels only expose the first image's thumbnail.)

This script NEVER posts to Instagram. Drafts land in review_queue.json;
a human approves them before anything is published.

State lives under STATE_DIR (default: the goal's hidden_files dir):
  watermark.json      - seen post ids + failure counts (dedupe)
  review_queue.json   - list of draft dicts awaiting review
  images/<shortcode>/ - downloaded media for each queued draft

Requires: instagram-cli (linked account, read-only use), yt-dlp (fallback
for reel poster images). Env: HAWLEY_TRANSLATOR_URL=https://<worker>.workers.dev
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

DEFAULT_ACCOUNT_ID = "17841426633545671"  # linked @realufo_org — used read-only
DEFAULT_USERNAME = "senatorhawley"
DEFAULT_STATE_DIR = os.path.expanduser(
    "~/workspace/goals/hawley-ig-auto-translate-bot/hidden_files"
)
SHORTCODE_RE = re.compile(r"instagram\.com/(?:p|reel|reels)/([^/?#]+)")
SEEN_CAP = 500
MAX_FAILURES = 3

# OCR: same engine + settings as realufo.org's crawler (crawler/ingest/ocr.py).
OCR_DET, OCR_REC, OCR_SCORE_MIN = "PP-OCRv5_mobile_det", "en_PP-OCRv5_mobile_rec", 0.5


def lines_from_boxes(items):
    """[(text, (x1, y1, x2, y2))] -> lines, top-to-bottom then left-to-right."""
    lines = []  # [y_center, height, [(x1, text)]]
    for text, (x1, y1, x2, y2) in sorted(items, key=lambda it: (it[1][1] + it[1][3]) / 2):
        yc, h = (y1 + y2) / 2, y2 - y1
        if lines and abs(yc - lines[-1][0]) <= max(h, lines[-1][1]) / 2:
            lines[-1][2].append((x1, text))
        else:
            lines.append([yc, h, [(x1, text)]])
    return "\n".join(" ".join(t for _, t in sorted(words)) for _, _, words in lines)


_ocr_engine = None


def paddle_ocr():
    """Lazily init PaddleOCR (heavy import; needs the .venv-ocr venv).

    Raises RuntimeError if PaddleOCR isn't installed — callers treat that as
    'OCR unavailable' and continue caption-only.
    """
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise RuntimeError(f"PaddleOCR not installed: {e}")
        model = PaddleOCR(text_detection_model_name=OCR_DET,
                          text_recognition_model_name=OCR_REC,
                          use_doc_orientation_classify=True, use_doc_unwarping=False,
                          use_textline_orientation=False,
                          text_rec_score_thresh=OCR_SCORE_MIN)

        def run(png_path):
            r = model.predict(png_path)[0]
            scores = [float(s) for s in r["rec_scores"]]
            items = [(t, tuple(float(v) for v in b))
                     for t, b in zip(r["rec_texts"], r["rec_boxes"])]
            conf = sum(scores) / len(scores) if scores else 0.0
            return lines_from_boxes(items), conf

        _ocr_engine = run
    return _ocr_engine


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, timeout=60):
    """Run a command, return stdout. Raises RuntimeError on failure."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"timeout: {' '.join(cmd)} ({e})")
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()[:500]
        if "429" in err:
            raise RuntimeError("RATE_LIMITED: " + err)
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd[:4])}… {err}")
    return p.stdout


def cli(*args, timeout=60):
    return json.loads(run(["instagram-cli", *args], timeout=timeout))


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def shortcode_from_url(url):
    m = SHORTCODE_RE.search(url or "")
    return m.group(1) if m else None


def scan_image_urls(obj, found):
    """Best-effort scan of instagram-cli detail for any direct media URLs."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = k.lower()
            if isinstance(v, str) and v.startswith("http") and kl != "profile_picture_url":
                if any(s in kl for s in ("image", "display_url", "thumbnail", "media_url")):
                    found.append(v)
            else:
                scan_image_urls(v, found)
    elif isinstance(obj, list):
        for v in obj:
            scan_image_urls(v, found)


def oembed_thumbnail(post_url):
    """Instagram's public oEmbed endpoint returns a thumbnail_url (scontent CDN)
    for any post type — this is what unlocks plain image posts, for which
    instagram-cli exposes no media URL at all. (Carousels: first image only.)"""
    if not post_url:
        return None
    api = ("https://www.instagram.com/api/v1/oembed/?url="
           + urllib.parse.quote(post_url, safe=""))
    req = urllib.request.Request(api, headers={"User-Agent": "Mozilla/5.0 (Linux; Android 14)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            thumb = json.loads(r.read().decode()).get("thumbnail_url")
        return thumb if thumb and thumb.startswith("http") else None
    except Exception as e:
        log(f"  oEmbed lookup failed: {e}")
        return None


def resolve_media_urls(detail, post_url):
    """Return publicly-fetchable image URLs for OCR. Best effort.

    Order: (1) any image-ish URL fields instagram-cli may expose one day;
    (2) oEmbed thumbnail_url — works for every post type incl. image posts;
    (3) for reels/videos, the poster thumbnail via yt-dlp as a fallback.
    """
    found = []
    scan_image_urls(detail, found)
    if found:
        return list(dict.fromkeys(found))[:4]

    thumb = oembed_thumbnail(post_url)
    if thumb:
        return [thumb]

    media_type = str(detail.get("media_type") or "").lower()
    if media_type in ("video", "reel") and post_url:
        try:
            out = run(["yt-dlp", "--print", "thumbnail", "--no-warnings", post_url], timeout=90)
            yt_thumb = out.strip().splitlines()[0].strip()
            if yt_thumb.startswith("http"):
                return [yt_thumb]
        except RuntimeError as e:
            log(f"  yt-dlp thumbnail failed: {e}")
    return []


def download(url, dest_path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest_path, "wb") as f:
        f.write(r.read())


def translate(worker_url, caption, image_texts, image_urls=()):
    """POST pre-OCR'd texts to the Worker; it returns the translations.

    image_urls is a fallback: for any image that arrived without OCR text
    (PaddleOCR unavailable on this runtime), the Worker OCRs the URL itself
    with its vision model.
    """
    payload = json.dumps({
        "caption": caption,
        "image_texts": image_texts,
        "image_urls": list(image_urls),
    }).encode()
    req = urllib.request.Request(
        worker_url.rstrip("/") + "/translate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser(description="Poll Hawley IG posts and queue translation drafts.")
    ap.add_argument("--account-id", default=DEFAULT_ACCOUNT_ID)
    ap.add_argument("--username", default=DEFAULT_USERNAME)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--state-dir", default=DEFAULT_STATE_DIR)
    ap.add_argument("--worker-url", default=os.environ.get("HAWLEY_TRANSLATOR_URL", ""))
    ap.add_argument("--dry-run", action="store_true", help="list new posts without translating")
    args = ap.parse_args()

    os.makedirs(args.state_dir, exist_ok=True)
    os.makedirs(os.path.join(args.state_dir, "images"), exist_ok=True)
    wm_path = os.path.join(args.state_dir, "watermark.json")
    q_path = os.path.join(args.state_dir, "review_queue.json")
    watermark = load_json(wm_path, {"seen_ids": [], "failures": {}})
    queue = load_json(q_path, [])
    seen = set(watermark.get("seen_ids", []))
    queued_ids = {d.get("post_id") for d in queue}

    log(f"fetching latest {args.limit} posts from @{args.username}…")
    try:
        data = cli("posts", "--account-id", args.account_id,
                   "--username", args.username, "--limit", str(args.limit),
                   "--post-types", "POST,REEL")
    except RuntimeError as e:
        log(f"ABORT: {e}")
        return 2
    posts = data.get("posts", [])
    log(f"got {len(posts)} posts")

    fresh = [p for p in posts if p.get("post_id") not in seen]
    # process oldest first so the queue reads chronologically
    fresh.reverse()
    log(f"{len(fresh)} new since watermark")

    if args.dry_run:
        for p in fresh:
            cap = (p.get("post_caption") or "")[:80].replace("\n", " ")
            log(f"  NEW {p.get('post_id')} {p.get('url')} :: {cap}")
        return 0

    if not args.worker_url:
        log("ABORT: set --worker-url or HAWLEY_TRANSLATOR_URL")
        return 2

    processed = 0
    for p in fresh:
        post_id = p.get("post_id")
        try:
            detail = cli("post", "--account-id", args.account_id, "--id", post_id)
            d = (detail.get("posts") or [{}])[0]
            post_url = d.get("url") or p.get("url") or ""
            shortcode = shortcode_from_url(post_url) or post_id.replace("/", "_")
            caption_en = d.get("post_caption") or p.get("post_caption") or ""
            media_type = d.get("media_type") or ""
            created = d.get("created_at") or p.get("created_at")

            log(f"processing {shortcode} ({media_type})…")
            media_urls = resolve_media_urls(d, post_url)
            if not media_urls:
                log("  no fetchable media URL — caption-only draft")

            # download media for the local review archive
            img_dir = os.path.join(args.state_dir, "images", shortcode)
            os.makedirs(img_dir, exist_ok=True)
            local_paths = []
            for i, mu in enumerate(media_urls):
                ext = ".png" if ".png" in mu.split("?")[0] else ".jpg"
                dest = os.path.join(img_dir, f"img{i}{ext}")
                try:
                    download(mu, dest)
                    local_paths.append(os.path.relpath(dest, args.state_dir))
                except Exception as e:
                    log(f"  download failed for image {i}: {e}")

            # OCR with PaddleOCR (same engine as realufo.org's crawler)
            image_texts = []
            if local_paths:
                try:
                    ocr = paddle_ocr()
                except RuntimeError as e:
                    log(f"  {e} — caption-only draft")
                    ocr = None
                for lp in local_paths:
                    if not ocr:
                        break
                    try:
                        text, conf = ocr(os.path.join(args.state_dir, lp))
                        image_texts.append({"url": lp, "text_en": text})
                        log(f"  OCR {lp}: {len(text)} chars (conf {conf:.2f})")
                    except Exception as e:
                        log(f"  OCR failed for {lp}: {e}")
                        image_texts.append({"url": lp, "text_en": ""})

            # Fallback: if PaddleOCR produced no text (unavailable on this
            # runtime), hand the URLs to the Worker so IT can OCR them.
            result = translate(args.worker_url, caption_en, image_texts,
                               image_urls=media_urls)

            draft = {
                "post_id": post_id,
                "shortcode": shortcode,
                "url": post_url,
                "media_type": media_type,
                "created_at": created,
                "caption_en": caption_en,
                "caption_zh": result.get("caption_zh", ""),
                "image_texts": result.get("image_texts", []),
                "image_local_paths": local_paths,
                "warnings": result.get("warnings", []),
                "queued_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            if post_id not in queued_ids:
                queue.append(draft)
                queued_ids.add(post_id)
                save_json(q_path, queue)
            seen.add(post_id)
            watermark["failures"].pop(post_id, None)
            processed += 1
            log(f"  queued draft ({len(caption_en)} en chars -> {len(draft['caption_zh'])} zh chars)")
            time.sleep(2)  # be gentle with the APIs
        except RuntimeError as e:
            if "RATE_LIMITED" in str(e):
                log(f"ABORT on rate limit: {e}")
                break
            fails = watermark.get("failures", {}).get(post_id, 0) + 1
            watermark.setdefault("failures", {})[post_id] = fails
            log(f"  FAILED ({fails}/{MAX_FAILURES}): {e}")
            if fails >= MAX_FAILURES:
                log("  giving up on this post; marking seen to avoid a poison loop")
                seen.add(post_id)
        except Exception as e:  # noqa: BLE001 — never let one post kill the run
            log(f"  UNEXPECTED: {e}")

    watermark["seen_ids"] = sorted(seen)[-SEEN_CAP:]
    watermark["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save_json(wm_path, watermark)
    log(f"done: {processed} drafts queued, {len(queue)} total awaiting review")
    return 0


if __name__ == "__main__":
    sys.exit(main())
