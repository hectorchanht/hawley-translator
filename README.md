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
│  1. instagram-cli│ POST │  2. OCR each image via  │ JSON │  review_queue.json   │
│     reads latest │─────▶│     Workers AI vision   │─────▶│  + downloaded media  │
│     @senatorhawley│/trans│     model               │      │                      │
│     posts (newer │ late │  3. translate caption   │      │  approves → publish  │
│     than watermark│      │     + OCR text via      │      │  manually (IG app    │
│                  │      │     Workers AI LLM into │      │  or future poster)  │
│  4. append draft │      │     Traditional Chinese │      │                      │
│     to review    │      │     (HK 書面語，霍利)    │      │                      │
│     queue, save  │      │                         │      │                      │
│     watermark    │      │  No DB, no secrets,     │      │                      │
└──────────────────┘      │  no posting.            │      └──────────────────────┘
                          └──────────────────────┘
```

Data flow per new post:

1. **Poll** — `scripts/poll.py` asks `instagram-cli` for the latest posts from
   `@senatorhawley`, filters out ids already in `watermark.json`.
2. **Resolve media** — for each new post it tries to get publicly-fetchable
   image URLs: any image fields `instagram-cli` exposes, else (for reels/videos)
   the poster thumbnail via `yt-dlp --print thumbnail`. Plain image/carousel
   posts currently can't be resolved from a datacenter IP (Instagram
   login-walls them) — those drafts are caption-only with a warning.
3. **Translate** — the caption + image URLs are POSTed to the Worker's
   `/translate`. The Worker OCRs the images (vision model), then translates
   everything in one LLM call (`霍利` for Hawley, @mentions/#hashtags kept,
   line breaks preserved).
4. **Queue** — media is downloaded locally, the draft (EN + ZH + OCR results +
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
# requirements on the VM: instagram-cli (linked account), python3, yt-dlp
pip install -r scripts/requirements.txt   # yt-dlp

# every 4 hours: poll, translate, queue drafts for review
0 */4 * * * HAWLEY_TRANSLATOR_URL=https://hawley-translator.<acct>.workers.dev \
  /usr/bin/python3 /path/to/hawley-translator/scripts/poll.py \
  --state-dir ~/workspace/goals/hawley-ig-auto-translate-bot/hidden_files \
  >> ~/workspace/goals/hawley-ig-auto-translate-bot/hidden_files/poll.log 2>&1
```

Dry run first (lists new posts, translates nothing):

```bash
python3 scripts/poll.py --dry-run
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

- Instagram serves its post pages / `og:image` only to logged-in browsers;
  datacenter IPs get a login wall. Hence image/carousel posts currently yield
  caption-only drafts. Reels/videos work via `yt-dlp` thumbnails (public CDN).
- `scontent` CDN URLs expire — the poller sends them to the Worker immediately
  and downloads local copies for the review archive; don't store the URLs.
- Workers AI free tier: 10,000 neurons/day — a few translated posts/day is
  comfortably inside it.
