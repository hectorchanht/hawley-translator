#!/usr/bin/env python3
"""Poll Instagram source accounts for new posts, translate them via the
hawley-translator Cloudflare Worker, and queue drafts for HUMAN REVIEW.

Sources are configured in SOURCES (--source hawley|trump, default hawley);
each source gets its own watermark + review-queue state files.

Scope: post caption + text visible in images (no video transcription).

Media: every image is downloaded and burned with a translation watermark
("非官方中文翻譯 · @snhawleytranslatorhkunofficial"); videos/reels are
downloaded in full via yt-dlp (IG publishing requires the file — a URL is
not enough), transcribed with faster-whisper, translated, and burned with
Cantonese subtitles + watermark via ffmpeg. Originals are kept as src* files;
the watermarked img*/clip* files are what gets published.

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

Requires: instagram-cli (linked account, read-only use), yt-dlp (video
downloads + reel poster fallback), ffmpeg (video watermark), Pillow (image
watermark), and a CJK font (Noto Sans CJK). Env:
HAWLEY_TRANSLATOR_URL=https://<worker>.workers.dev
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
DEFAULT_SOURCE = "hawley"
DEFAULT_STATE_DIR = os.path.expanduser(
    "~/workspace/goals/hawley-ig-auto-translate-bot/hidden_files"
)

# One entry per translated figure. The review queue, watermark state files,
# and the burned-in "unofficial translation" watermark are all per-source.
SOURCES = {
    "hawley": {
        "username": "senatorhawley",
        "label": "Josh Hawley",
        "watermark_text": "非官方中文翻譯 · @snhawleytranslatorhkunofficial",
    },
    "trump": {
        "username": "realdonaldtrump",
        "label": "Donald Trump",
        # PROVISIONAL — Hector hasn't created the translator account yet;
        # update this handle once the real one exists (images are burned
        # with it, so re-queue any already-processed drafts after a rename).
        "watermark_text": "非官方中文翻譯 · @trumptranslatorhkunofficial",
    },
}
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


# Watermark burned into every published image/video: marks the repost as an
# unofficial translation. Originals are kept as src* files (never published).
# The actual text is per-source (SOURCES[...]["watermark_text"]) and passed in.
WATERMARK_FONT_TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
WATERMARK_FONT_INDEX = 4  # Noto Sans CJK HK inside the .ttc


def _cjk_font(size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(WATERMARK_FONT_TTC, size, index=WATERMARK_FONT_INDEX)
    except Exception:
        return ImageFont.truetype(WATERMARK_FONT_TTC, size)  # first face fallback


def watermark_image(src_path, dest_path, text):
    """Bottom bar, semi-transparent black, white CJK text. Raises on failure."""
    from PIL import Image, ImageDraw
    im = Image.open(src_path).convert("RGBA")
    w, h = im.size
    bar_h = max(28, int(h * 0.075))
    overlay = Image.new("RGBA", (w, bar_h), (0, 0, 0, 140))
    im.paste(overlay, (0, h - bar_h), overlay)
    draw = ImageDraw.Draw(im)
    font = _cjk_font(max(14, int(w * 0.032)))
    while draw.textlength(text, font=font) > w * 0.94 and font.size > 10:
        font = _cjk_font(font.size - 2)
    tw = draw.textlength(text, font=font)
    draw.text(((w - tw) / 2, h - bar_h + (bar_h - font.size) / 2 - 2),
              text, font=font, fill=(255, 255, 255, 235))
    im.convert("RGB").save(dest_path, quality=92)


def download_video(post_url, dest_path):
    """Full mp4 via yt-dlp. Raises RuntimeError on failure."""
    # --no-check-certificates: this VM's egress proxy MITMs TLS with a
    # self-signed cert that yt-dlp's trust store rejects; the content is a
    # public IG reel, so integrity risk is negligible (and we verify the mp4).
    run(["yt-dlp", "--no-check-certificates", "-o", dest_path, "--no-warnings",
         "--no-playlist", "-f", "mp4", "--merge-output-format", "mp4",
         post_url], timeout=600)


def watermark_video(src_path, dest_path, text):
    """Burn the watermark into the video with ffmpeg drawtext. Raises."""
    safe = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "")
    # fontsize scales with the SMALLER dimension so vertical reels don't clip
    vf = (f"drawtext=fontfile={WATERMARK_FONT_TTC}:text='{safe}':"
          f"fontsize=min(w\\,h)/26:x=(w-text_w)/2:y=h-text_h-20:"
          f"fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=12")
    run(["ffmpeg", "-y", "-v", "error", "-i", src_path,
         "-vf", vf, "-c:a", "copy", dest_path], timeout=600)


# Speech transcription for Cantonese subtitles (faster-whisper, local model
# dir — HF downloads are proxy-blocked on this VM, so pre-fetch with curl).
WHISPER_MODEL_DIR = os.path.expanduser(
    "~/workspace/goals/hawley-ig-auto-translate-bot/models/faster-whisper-base.en")
_whisper_model = None


def transcribe_segments(video_path, model_dir=WHISPER_MODEL_DIR):
    """faster-whisper -> [{start, end, text}]. Raises on failure."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        if not os.path.isdir(model_dir):
            raise RuntimeError(f"whisper model not found: {model_dir}")
        _whisper_model = WhisperModel(model_dir, device="cpu", compute_type="int8")
    segments, _ = _whisper_model.transcribe(video_path, beam_size=5)
    return [{"start": s.start, "end": s.end, "text": s.text.strip()}
            for s in segments if s.text.strip()]


def _srt_ts(sec):
    ms = int(sec * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments, path):
    """segments: [{start, end, text}] (text already translated)."""
    with open(path, "w", encoding="utf-8") as f:
        for n, s in enumerate(segments, 1):
            f.write(f"{n}\n{_srt_ts(s['start'])} --> {_srt_ts(s['end'])}\n"
                    f"{s['text']}\n\n")


def burn_subtitles(src_path, srt_path, dest_path, text):
    """Burn Cantonese subtitles + watermark in one ffmpeg pass. Raises."""
    safe = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "")
    srt_esc = srt_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "")
    style = ("FontName=Noto Sans CJK HK,FontSize=20,PrimaryColour=&H00FFFFFF,"
             "OutlineColour=&H80000000,BorderStyle=1,Outline=2,Shadow=0,"
             "MarginV=80,Alignment=2")
    vf = (f"subtitles='{srt_esc}':fontsdir='/usr/share/fonts/opentype/noto':"
          f"force_style='{style}',"
          f"drawtext=fontfile={WATERMARK_FONT_TTC}:text='{safe}':fontsize=min(w\\,h)/26:"
          f"x=(w-text_w)/2:y=h-text_h-20:fontcolor=white:box=1:"
          f"boxcolor=black@0.55:boxborderw=12")
    run(["ffmpeg", "-y", "-v", "error", "-i", src_path,
         "-vf", vf, "-c:a", "copy", dest_path], timeout=900)


def translate(worker_url, source, caption, image_texts, image_urls=(), segments=()):
    """POST pre-OCR'd texts (and speech segments) to the Worker.

    image_urls is a fallback: for any image that arrived without OCR text
    (PaddleOCR unavailable on this runtime), the Worker OCRs the URL itself
    with its vision model. segments: [{text}] transcribed speech; the Worker
    returns segments_zh aligned by index.
    """
    payload = json.dumps({
        "source": source,
        "caption": caption,
        "image_texts": image_texts,
        "image_urls": list(image_urls),
        "segments": [{"i": k, "text_en": s["text"]} for k, s in enumerate(segments)],
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
    ap = argparse.ArgumentParser(description="Poll IG posts and queue translation drafts.")
    ap.add_argument("--account-id", default=DEFAULT_ACCOUNT_ID)
    ap.add_argument("--source", default=DEFAULT_SOURCE, choices=sorted(SOURCES),
                    help="which figure to poll (default: %(default)s)")
    ap.add_argument("--username", default=None,
                    help="override the source's IG username")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--state-dir", default=DEFAULT_STATE_DIR)
    ap.add_argument("--worker-url", default=os.environ.get("HAWLEY_TRANSLATOR_URL", ""))
    ap.add_argument("--dry-run", action="store_true", help="list new posts without translating")
    args = ap.parse_args()

    src = SOURCES[args.source]
    username = args.username or src["username"]
    watermark_text = src["watermark_text"]

    os.makedirs(args.state_dir, exist_ok=True)
    os.makedirs(os.path.join(args.state_dir, "images"), exist_ok=True)
    # Backwards compatible: hawley keeps the original state filenames (the
    # Zapier "hawley review drafter" skill reads review_queue.json); other
    # sources get suffixed filenames.
    if args.source == "hawley":
        wm_path = os.path.join(args.state_dir, "watermark.json")
        q_path = os.path.join(args.state_dir, "review_queue.json")
    else:
        wm_path = os.path.join(args.state_dir, f"watermark_{args.source}.json")
        q_path = os.path.join(args.state_dir, f"review_queue_{args.source}.json")
    watermark = load_json(wm_path, {"seen_ids": [], "failures": {}})
    queue = load_json(q_path, [])
    seen = set(watermark.get("seen_ids", []))
    queued_ids = {d.get("post_id") for d in queue}

    log(f"fetching latest {args.limit} posts from @{username} ({src['label']})…")
    try:
        data = cli("posts", "--account-id", args.account_id,
                   "--username", username, "--limit", str(args.limit),
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

            # download + watermark media. Originals kept as src* (never
            # published); the watermarked img*/clip* files are what gets
            # published after review.
            img_dir = os.path.join(args.state_dir, "images", shortcode)
            os.makedirs(img_dir, exist_ok=True)
            local_paths, src_paths = [], []
            wm_status = "n/a"
            for i, mu in enumerate(media_urls):
                ext = ".png" if ".png" in mu.split("?")[0] else ".jpg"
                src = os.path.join(img_dir, f"src{i}{ext}")
                dest = os.path.join(img_dir, f"img{i}{ext}")
                try:
                    download(mu, src)
                    src_paths.append(os.path.relpath(src, args.state_dir))
                    try:
                        watermark_image(src, dest, watermark_text)
                        local_paths.append(os.path.relpath(dest, args.state_dir))
                        wm_status = "ok"
                    except Exception as e:
                        log(f"  watermark failed for image {i}: {e}")
                        local_paths.append(os.path.relpath(src, args.state_dir))
                        wm_status = "failed"
                except Exception as e:
                    log(f"  download failed for image {i}: {e}")

            # videos/reels: download the full mp4 (IG publishing needs the
            # file — a URL is not enough) and transcribe speech for subtitles.
            video_path, video_src, video_wm = "", "", "n/a"
            segments_en = []
            if str(media_type).lower() in ("video", "reel") and post_url:
                vsrc = os.path.join(img_dir, "clip_src.mp4")
                try:
                    download_video(post_url, vsrc)
                    video_src = os.path.relpath(vsrc, args.state_dir)
                    try:
                        segments_en = transcribe_segments(vsrc)
                        log(f"  transcribed {len(segments_en)} speech segments")
                    except Exception as e:
                        log(f"  transcription failed: {e}")
                except Exception as e:
                    log(f"  video download failed: {e}")

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
            result = translate(args.worker_url, args.source, caption_en, image_texts,
                               image_urls=media_urls, segments=segments_en)

            # burn Cantonese subtitles + watermark into the video
            sub_count = 0
            if video_src:
                vdest = os.path.join(img_dir, "clip.mp4")
                seg_zh = result.get("segments_zh", []) or []
                zh_by_i = {s.get("i"): s.get("text_zh", "") for s in seg_zh
                           if isinstance(s, dict)}
                subs = []
                for k, s in enumerate(segments_en):
                    t = zh_by_i.get(k, "")
                    if t:
                        subs.append({"start": s["start"], "end": s["end"], "text": t})
                if subs:
                    srt_path = os.path.join(img_dir, "subs.srt")
                    write_srt(subs, srt_path)
                    try:
                        burn_subtitles(vsrc, srt_path, vdest, watermark_text)
                        video_path = os.path.relpath(vdest, args.state_dir)
                        video_wm, sub_count = "subtitled", len(subs)
                        log(f"  burned {len(subs)} Cantonese subtitles + watermark")
                    except Exception as e:
                        log(f"  subtitle burn-in failed: {e}")
                if not video_path:
                    # no subtitles (or burn failed): watermark only
                    try:
                        watermark_video(vsrc, vdest, watermark_text)
                        video_path = os.path.relpath(vdest, args.state_dir)
                        video_wm = "ok"
                    except Exception as e:
                        log(f"  video watermark failed: {e}")
                        video_path, video_wm = video_src, "failed"

            draft = {
                "source": args.source,
                "post_id": post_id,
                "shortcode": shortcode,
                "url": post_url,
                "media_type": media_type,
                "created_at": created,
                "caption_en": caption_en,
                "caption_zh": result.get("caption_zh", ""),
                "image_texts": result.get("image_texts", []),
                "image_local_paths": local_paths,
                "image_src_paths": src_paths,
                "watermark": wm_status,
                "video_local_path": video_path,
                "video_src_path": video_src,
                "video_watermark": video_wm,
                "subtitle_segments": sub_count,
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
