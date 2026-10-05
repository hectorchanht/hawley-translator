# hawley-translator

Auto-translation pipeline for the Instagram fan account
[@snhawleytranslatorhkunofficial](https://www.instagram.com/snhawleytranslatorhkunofficial),
which publishes Traditional Chinese (Hong Kong style) translations of
US Senator Josh Hawley's posts ([@senatorhawley](https://www.instagram.com/senatorhawley)).

Scope: **post caption + text visible in images**. No video transcription (yet).

> ⛔ **Review gate is mandatory.** This pipeline drafts translations only.
> Nothing is ever posted to Instagram automatically — a human reviews and
> approves every draft before publishing. See "Still needed" below.

## Architecture (in words)

```
┌──────────────────┐      ┌─────────────────────────┐      ┌──────────────────────┐
│  cron on Linux   │      │  Cloudflare Worker      │      │  human reviewer      │
│  VM (this repo:  │      │  `hawley-translator`    │      │  (Hector)            │
│  scripts/poll.py)│      │  src/index.js           │      │                      │
│                  │      │                         │      │  reads               │
│  1. instagram-cli│ POST │  4. translate caption   │ JSON │  review_queue.json   │
│     reads latest │─────▶│     + OCR text via      │─────▶│  + downloaded media  │
│     @senatorhawley│/trans│     Workers AI LLM into │      │                      │
│     posts (newer │ late │     Traditional Chinese │      │  approves → publish  │
│     than watermark│      │     (HK 書面語，霍利)    │      │  manually (IG app    │
│                  │      │                         │      │  or future poster)  │
│  2. resolve image│      │  No DB, no secrets,     │      │                      │
│     via oEmbed   │      │  no posting. OCR only   │      │                      │
│  3. OCR: Paddle- │      │  as fallback when the   │      │                      │
│     OCR on VM    │      │  VM can't run PaddleOCR │      │                      │
│                  │      │                         │      │                      │
│  5. append draft │      │                         │      │                      │
│     to review    │      │                         │      │                      │
│     queue, save  │      │                         │      │                      │
│     watermark    │      │                         │      │                      │
└──────────────────┘      └─────────────────────────┘      └──────────────────────┘
```

Data flow per new post:

1. **Poll** — `scripts/poll.py` asks `instagram-cli` for the latest posts from
   `@senatorhawley`, filters out ids already in `watermark.json`.
2. **Resolve media** — image URLs come from Instagram's public oEmbed endpoint
   (`/api/v1/oembed/` → `thumbnail_url`, a scontent CDN URL), which works for
   every post type — this is what unlocks plain image posts, for which
   `instagram-cli` exposes no media URL at all. (Carousels: first image only.)
   Reels fall back to `yt-dlp --print thumbnail`.
3. **OCR (two tiers)** — Tier 1: PaddleOCR **PP-OCRv5** (`PP-OCRv5_mobile_det` +
   `en_PP-OCRv5_mobile_rec`, score threshold 0.5) — the *same engine and
   settings* realufo.org's crawler uses (`realufo/crawler/ingest/ocr.py`).
   PaddleOCR needs Python + native libs so it can't run inside a Cloudflare
   Worker; it runs here on the VM via the `.venv-ocr` venv (see Cron setup).
   Tier 2 (fallback): if PaddleOCR can't run on the VM, the poller also sends
   the image URLs and the Worker OCRs them with a Workers AI vision model —
   so image text is still extracted either way. The draft records which tier
   produced the text (via `warnings`).
4. **Translate** — caption + OCR'd texts are POSTed to the Worker's
   `/translate`, which returns Traditional Chinese (HK 書面語，霍利 for
   Hawley, @mentions/#hashtags kept, line breaks preserved).
5. **Queue** — media is downloaded locally, the draft (EN + ZH + OCR results +
   warnings) is appended to `review_queue.json`, the watermark advances.

## Deploy the Worker

Cloudflare API tokens do **not** work from the operator VM (all tokens 401
there) — deploys go through the dashboard, never `wrangler deploy` from here.

1. Cloudflare dashboard → **Workers & Pages** → **Create** → **Connect to Git**.
2. Select the `hectorchanht/hawley-translator` repo, branch `main`.
3. Build settings: no build command needed (plain JS worker). The git
   integration auto-deploys on every push to `main`.
4. Under **Bindings**, confirm the **AI** binding exists (Workers AI).
   `wrangler.toml` already declares `[ai] binding = "AI"` — no secrets needed.
5. Note the `*.workers.dev` URL → set it as `HAWLEY_TRANSLATOR_URL` on the VM.
6. Smoke test: `GET /health` should return `{"ok":true}`.

## Cron setup (VM)

```bash
# one-time: dedicated venv for PaddleOCR (heavy: ~2 GB with paddlepaddle)
python3 -m venv ~/workspace/goals/hawley-ig-auto-translate-bot/.venv-ocr
~/workspace/goals/hawley-ig-auto-translate-bot/.venv-ocr/bin/pip install -r scripts/requirements.txt

# every 4 hours: poll, OCR, translate, queue drafts for review
0 */4 * * * HAWLEY_TRANSLATOR_URL=https://hawley-translator.<acct>.workers.dev \
  ~/workspace/goals/hawley-ig-auto-translate-bot/.venv-ocr/bin/python \
  /path/to/hawley-translator/scripts/poll.py \
  --state-dir ~/workspace/goals/hawley-ig-auto-translate-bot/hidden_files \
  >> ~/workspace/goals/hawley-ig-auto-translate-bot/hidden_files/poll.log 2>&1
```

Dry run first (lists new posts, translates nothing):

```bash
~/workspace/goals/hawley-ig-auto-translate-bot/.venv-ocr/bin/python scripts/poll.py --dry-run
```

## Still needed (not done by this scaffold)

- [ ] **Hector links `@snhawleytranslatorhkunofficial` to Muse** in Meta
      Accounts Center (today only `@realufo_org` is linked, so nothing can be
      published to the translator account yet).
- [ ] **Explicit approval before ANY live auto-posting.** Publishing = sending
      as Hector; per his standing rules this needs his explicit go-ahead.
      The pipeline is intentionally review-gated until then.
- [ ] **Review-queue approval flow** — decide where drafts get approved
      (e.g. Telegram gate like the RealUFO Shorts flow, or manual IG posting).
- [ ] Optional later: video transcription, carousel multi-image resolution,
      scheduled poster once auto-posting is approved.

## Known limitations

- PaddleOCR's models can't be fetched with `huggingface_hub` from the
  operator VM (the egress proxy mangles HF URLs) — download them with
  `curl -L` into `~/.paddlex/official_models/<model>/` instead, and
  `pip install paddlepaddle` separately (paddleocr doesn't pull it in).
  Note (2026-10-05): PaddlePaddle 3.3.1's oneDNN executor fails on
  `PP-OCRv5_mobile_det` on this VM (`ConvertPirAttribute2RuntimeAttribute`
  unimplemented) — the same pinned versions run fine in GitHub Actions.
  The Worker's vision-model OCR fallback covers this VM until the runtime
  is fixed.
- oEmbed `thumbnail_url` is 640px and carousels expose only the first image —
  small text and later carousel slides aren't OCR'd (caption still translated).
- `scontent` CDN URLs expire — the poller downloads local copies immediately
  for the review archive; don't store the URLs.
- Workers AI free tier: 10,000 neurons/day — a few translated posts/day is
  comfortably inside it.
