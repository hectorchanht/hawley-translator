/**
 * translate-america — Cloudflare Worker (translation engine for the 譯美國 Translate America network)
 *
 * Translates Instagram posts (caption + text visible in images) into
 * Traditional Chinese, Hong Kong written style, for the translator fan
 * accounts (currently @snhawleytranslatorhkunofficial for US Senator Josh
 * Hawley; multi-source — the poller sends `source` and the Worker applies
 * per-source name rules).
 *
 * OCR is done in TWO tiers:
 *   1. Preferred: PaddleOCR PP-OCRv5 on the VM (scripts/poll.py) — the same
 *      engine realufo.org's crawler uses. The poller sends pre-extracted text
 *      as `image_texts`. (PaddleOCR needs Python + native libs, so it can't
 *      run inside a Worker.)
 *   2. Fallback: if the poller couldn't OCR (e.g. PaddleOCR unavailable on
 *      that runtime), it also sends `image_urls` and the Worker OCRs them
 *      with a Workers AI vision model.
 *
 * Endpoints:
 *   GET  /health     -> { ok: true }
 *   POST /translate  -> { caption_zh, image_texts, warnings }
 *
 * POST /translate body:
 *   {
 *     "caption": "English caption text",
 *     "source": "hawley" | "trump",        // per-source name rules
 *     "target": "zh-Hant" | "en",           // en = no translation, English SEO only
 *     "image_texts": [                       // preferred: pre-OCR'd
 *       { "url": "images/<sc>/img0.jpg", "text_en": "OCR'd English text" }
 *     ],
 *     "image_urls": ["https://…jpg"],         // fallback: Worker OCRs these
 *     "segments": [                           // video speech segments
 *       { "i": 0, "text_en": "If you break it, you pay for it." }
 *     ]
 *   }
 *
 * Response:
 *   {
 *     "caption_zh": "繁體中文翻譯…",          // "" when target=en
 *     "image_texts": [
 *       { "url": "…", "text_en": "…", "text_zh": "翻譯…" }
 *     ],
 *     "segments_zh": [ { "i": 0, "text_zh": "翻譯…" } ],
 *     "seo_hook": "IG opening hook (target language)…",
 *     "seo_hashtags": ["#霍利", "#JoshHawley", …],
 *     "warnings": ["…"]   // empty when everything worked
 *   }
 *
 * Notes:
 * - No database, no secrets, no posting — this Worker only translates.
 * - Review/publishing happens outside (see scripts/poll.py).
 */

const VISION_MODEL = "@cf/meta/llama-3.2-11b-vision-instruct";
// NOTE 2026-10-05: @cf/meta/llama-3.1-8b-instruct was deprecated by Cloudflare
// on 2026-05-30 (Worker returned 5028). Llama 3.3 70B FP8 is the current
// text model. Account is on Workers Paid, so neuron overage is billed tiny,
// not hard-capped.
const TEXT_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
const MAX_IMAGES = 4;
const MAX_IMAGE_BYTES = 2 * 1024 * 1024; // guard for vision-model input

// Per-source person-name rules. The poller sends `source` in the POST body;
// unknown sources get the generic rules (mentions/hashtags kept).
const SOURCE_NAME_RULES = {
  hawley: '- Render "Josh Hawley" / "Hawley" / "Senator Hawley" as 霍利 / 霍利參議員.',
  trump:
    '- Render "Donald Trump" / "Trump" / "President Trump" as 特朗普 / 特朗普總統.',
};

// The one hashtag every post for that source must carry (SEO anchor tag).
const SOURCE_MAIN_TAG = {
  hawley: "#霍利",
  trump: "#特朗普",
};

// English anchor tags for the "en" target (original clips, English SEO).
const SOURCE_MAIN_TAG_EN = {
  hawley: "#JoshHawley",
  trump: "#DonaldTrump",
};

function extractJson(raw) {
  const cleaned = String(raw)
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```\s*$/, "");
  // The model sometimes emits the JSON twice (plain + fenced copy) or trails
  // garbage. Grab the FIRST complete top-level object via brace matching
  // instead of first-"{" to last-"}" (2026-10-05: that broke on double emission).
  const start = cleaned.indexOf("{");
  if (start === -1) throw new Error("no JSON object in model output");
  let depth = 0, inStr = false, esc = false;
  for (let i = start; i < cleaned.length; i++) {
    const c = cleaned[i];
    if (inStr) {
      if (esc) esc = false;
      else if (c === "\\") esc = true;
      else if (c === '"') inStr = false;
    } else if (c === '"') {
      inStr = true;
    } else if (c === "{") {
      depth++;
    } else if (c === "}") {
      depth--;
      if (depth === 0) return JSON.parse(cleaned.slice(start, i + 1));
    }
  }
  throw new Error("no complete JSON object in model output");
}

async function ocrImage(env, url) {
  try {
    const res = await fetch(url, {
      headers: { "User-Agent": "Mozilla/5.0 (compatible; hawley-translator/1.0)" },
    });
    if (!res.ok) return { ok: false, error: `fetch failed: HTTP ${res.status}` };
    const buf = await res.arrayBuffer();
    if (buf.byteLength === 0 || buf.byteLength > MAX_IMAGE_BYTES) {
      return { ok: false, error: `unusable image size: ${buf.byteLength} bytes` };
    }
    const out = await env.AI.run(VISION_MODEL, {
      prompt:
        "Transcribe ALL text visible in this image, exactly as written, " +
        "preserving line breaks. Return ONLY the transcribed text and nothing else. " +
        "If there is no readable text, return exactly: NONE",
      image: [...new Uint8Array(buf)],
    });
    const text = String(out?.response ?? "").trim();
    if (!text || /^none\.?$/i.test(text)) return { ok: true, text: "" };
    return { ok: true, text };
  } catch (e) {
    return { ok: false, error: String(e?.message || e).slice(0, 200) };
  }
}

// Caption/images/SEO prompt — no segments. Segments are always translated
// in separate small batches (below) so no single Workers AI call gets big
// enough to hit the 3046 timeout.
function buildCaptionPrompt(caption, imageTexts, source) {
  const nameRule = SOURCE_NAME_RULES[source] || "";
  const mainTag = SOURCE_MAIN_TAG[source] || "#翻譯";
  const rules = [
    "You are a professional translator writing for a Hong Kong audience.",
    "Translate the English content in INPUT below into Traditional Chinese, Hong Kong written style (書面語，繁體中文).",
    "STRICT RULES:",
    "- Translate ONLY what is in INPUT. Do NOT add commentary, explanations, background, people, events, or any content not present in the source.",
    "- Keep @mentions (e.g. @senatorhawley) and #hashtags unchanged.",
    "- Preserve line breaks and paragraph structure.",
    "- Be faithful and concise.",
    "- Also produce:",
    '  - "seo_hook": one punchy Instagram opening line (under 50 Chinese characters, ' +
      "Cantonese-flavoured, may start with ONE emoji), capturing the most newsworthy point " +
      "from the source material only.",
    '  - "seo_hashtags": 5-8 hashtags as JSON strings, each starting with #. Mix Traditional ' +
      `Chinese tags (e.g. ${mainTag}) and English tags (e.g. #JoshHawley). Always include ${mainTag}. ` +
      "Topical and HK-audience relevant; never generic spam like #love or #instagood.",
  ];
  if (nameRule) rules.push("- " + nameRule);
  const input = {
    caption: caption || "",
    image_texts: imageTexts.map((t) => t || ""),
  };
  return rules.join("\n") +
    "\n\nINPUT (JSON):\n" + JSON.stringify(input) +
    '\n\nReturn ONLY a JSON object with EXACTLY these keys — no markdown fences, ' +
    "no commentary before or after:\n" +
    '{"caption_zh": "...", "image_texts_zh": ["...", "..."], ' +
    '"seo_hook": "...", "seo_hashtags": ["#..."]}' +
    "\n\nYOUR JSON RESPONSE:";
}

// Segments-only prompt for batches after the first (the first batch uses the
// full prompt with caption + SEO). Keeps each Workers AI call small enough to
// avoid 3046 timeouts on dense videos.
function buildSegmentsPrompt(segments) {
  const input = segments.map((s) => ({ i: s.i, text_en: s.text_en || "" }));
  return [
    "You are a professional translator writing for a Hong Kong audience.",
    "Translate each English segment below into Traditional Chinese, Hong Kong written style (書面語，繁體中文).",
    "STRICT RULES:",
    "- Translate ONLY. Do not add commentary, explanations, or content not in the source.",
    "- These become burned-in video subtitles: keep each one short " +
      "(one line, under 20 Chinese characters when possible), Cantonese-flavoured " +
      "where natural (嘅, 咁, 係, 唔), faithful to the spoken meaning.",
    "- If a segment is empty, return an empty string for it.",
    "",
    "INPUT (JSON):",
    JSON.stringify(input),
    "",
    "Return ONLY a JSON object with EXACTLY this key — no markdown fences, no commentary:",
    '{"segments_zh": [{"i": 0, "text_zh": "..."}]}',
    "",
    "YOUR JSON RESPONSE:",
  ].join("\n");
}

// English-target prompt: no translation — just an SEO hook + hashtags for
// the English-speaking audience account (original clips, watermarked).
function buildEnglishSeoPrompt(caption, imageTexts, source) {
  const mainTag = SOURCE_MAIN_TAG_EN[source] || "#Politics";
  const parts = [
    "You are a social media editor writing for an English-speaking Instagram audience.",
    "Given the Instagram caption (and any text visible in its images) below, produce:",
    '- "seo_hook": one punchy opening line (under 120 characters, may start with ONE emoji), ' +
      "capturing the most newsworthy point. No hashtags inside the hook.",
    `- "seo_hashtags": 5 to 8 English hashtags as JSON strings, each starting with #. ` +
      `Always include ${mainTag}. Topical and specific (US politics, AI regulation, ` +
      "the people named); never generic spam like #love or #instagood.",
    "",
    "Return ONLY valid JSON, no markdown fences, with this exact shape:",
    '{"seo_hook": "...", "seo_hashtags": ["#..."]}',
    "",
    "CAPTION:",
    caption || "",
  ];
  imageTexts.forEach((t, i) => {
    parts.push("", `IMAGE ${i + 1} TEXT:`, t || "");
  });
  return parts.join("\n");
}

async function handleTranslate(request, env) {
  const warnings = [];
  let body;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: "invalid JSON body" }, { status: 400 });
  }

  const caption = typeof body.caption === "string" ? body.caption : "";
  const source = typeof body.source === "string" ? body.source : "hawley";
  const target = body.target === "en" ? "en" : "zh-Hant";
  const inTexts = Array.isArray(body.image_texts) ? body.image_texts : [];
  const inUrls = (Array.isArray(body.image_urls) ? body.image_urls : [])
    .filter((u) => typeof u === "string" && u.startsWith("http"));
  const inSegs = (Array.isArray(body.segments) ? body.segments : [])
    .filter((s) => s && typeof s.text_en === "string")
    .map((s, k) => ({ i: typeof s.i === "number" ? s.i : k, text_en: s.text_en }))
    .slice(0, 200);
  const n = Math.min(MAX_IMAGES, Math.max(inTexts.length, inUrls.length));

  const slots = [];
  for (let i = 0; i < n; i++) {
    slots.push({
      url: typeof inTexts[i]?.url === "string" && inTexts[i].url
        ? inTexts[i].url
        : (inUrls[i] || ""),
      text_en: typeof inTexts[i]?.text_en === "string" ? inTexts[i].text_en : "",
      text_zh: "",
    });
  }

  // Fallback tier: Worker-side vision OCR for slots that arrived without text.
  await Promise.all(
    slots.map(async (s, i) => {
      if (s.text_en || !s.url.startsWith("http")) return;
      const r = await ocrImage(env, s.url);
      if (r.ok) {
        s.text_en = r.text;
      } else {
        warnings.push(`image ${i + 1}: Worker OCR skipped (${r.error})`);
      }
    })
  );

  const mainTag = SOURCE_MAIN_TAG[source] || "#翻譯";

  // English target: no translation — original clips for English speakers.
  // One cheap model call for the SEO hook + hashtags only.
  if (target === "en") {
    let seoHook = "";
    let seoTags = [];
    try {
      const out = await env.AI.run(TEXT_MODEL, {
        prompt: buildEnglishSeoPrompt(caption, slots.map((s) => s.text_en), source),
        max_tokens: 512,
      });
      const parsed = extractJson(String(out?.response ?? ""));
      seoHook = typeof parsed.seo_hook === "string" ? parsed.seo_hook : "";
      const tagList = Array.isArray(parsed.seo_hashtags) ? parsed.seo_hashtags : [];
      seoTags = tagList.filter((t) => typeof t === "string" && t.startsWith("#")).slice(0, 10);
    } catch (e) {
      warnings.push(`english SEO failed: ${String(e?.message || e).slice(0, 200)}`);
    }
    const mainTagEn = SOURCE_MAIN_TAG_EN[source] || "#Politics";
    if (seoTags.length && !seoTags.includes(mainTagEn)) seoTags.unshift(mainTagEn);
    slots.forEach((s) => { s.text_zh = s.text_en; });
    return Response.json({
      caption_zh: "",
      image_texts: slots,
      segments_zh: [],
      seo_hook: seoHook,
      seo_hashtags: seoTags,
      warnings,
    });
  }

  let captionZh = "";
  let segmentsZh = [];
  let seoHook = "";
  let seoTags = [];
  // Two-phase translation: (1) caption/images/SEO in one small call;
  // (2) segments in tiny char-batched calls. A single Workers AI call over
  // too much text times out (3046), so nothing here is allowed to get big.
  // Partial results survive per call.
  const segByI = new Map();
  // Phase 1: caption + images + SEO (no segments).
  try {
    const out = await env.AI.run(TEXT_MODEL, {
      prompt: buildCaptionPrompt(caption, slots.map((s) => s.text_en), source),
      max_tokens: 2048,
    });
    const parsed = extractJson(String(out?.response ?? ""));
    captionZh = typeof parsed.caption_zh === "string" ? parsed.caption_zh : "";
    const zhList = Array.isArray(parsed.image_texts_zh) ? parsed.image_texts_zh : [];
    slots.forEach((s, i) => {
      s.text_zh = typeof zhList[i] === "string" ? zhList[i] : "";
    });
    seoHook = typeof parsed.seo_hook === "string" ? parsed.seo_hook : "";
    const tagList = Array.isArray(parsed.seo_hashtags) ? parsed.seo_hashtags : [];
    seoTags = tagList.filter((t) => typeof t === "string" && t.startsWith("#")).slice(0, 10);
    if (seoTags.length && !seoTags.includes(mainTag)) seoTags.unshift(mainTag);
  } catch (e) {
    warnings.push(`caption/SEO failed: ${String(e?.message || e).slice(0, 120)}`);
  }
  try {
    // Phase 2: segments in small batches. Each batch is independent —
    // one bad batch doesn't kill the others.
    const SEG_BATCH_CHARS = 200;
    const segBatches = [];
    let cur = [], curLen = 0;
    for (const s of inSegs) {
      const l = (s.text_en || "").length;
      if (cur.length && curLen + l > SEG_BATCH_CHARS) { segBatches.push(cur); cur = []; curLen = 0; }
      cur.push(s); curLen += l;
    }
    if (cur.length) segBatches.push(cur);
    for (let b = 0; b < segBatches.length; b++) {
      try {
        const out = await env.AI.run(TEXT_MODEL, {
          prompt: buildSegmentsPrompt(segBatches[b]),
          max_tokens: 2048,
        });
        const parsed = extractJson(String(out?.response ?? ""));
        const segList = Array.isArray(parsed.segments_zh) ? parsed.segments_zh : [];
        for (const s of segList) {
          if (s && typeof s.i === "number" && typeof s.text_zh === "string") segByI.set(s.i, s.text_zh);
        }
      } catch (e) {
        warnings.push(`segments batch ${b + 1}/${segBatches.length} failed: ${String(e?.message || e).slice(0, 120)}`);
      }
    }
    segmentsZh = inSegs.map((s) => ({ i: s.i, text_zh: segByI.get(s.i) ?? "" }));
  } catch (e) {
    warnings.push(`translation failed: ${String(e?.message || e).slice(0, 200)}`);
  }

  return Response.json({
    caption_zh: captionZh,
    image_texts: slots,
    segments_zh: segmentsZh,
    seo_hook: seoHook,
    seo_hashtags: seoTags,
    warnings,
  });
}

const SITE_HTML = `<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>譯美國 · Translate America — 美國政壇,譯返廣東話</title>
<meta name="description" content="美國政壇人物嘅 IG posts,每日繁體中文翻譯,加廣東話字幕。旗下帳號:霍利翻譯、特朗普翻譯、Global News Shorts。">
<style>
:root{
  --bg:#0b1020; --bg2:#111832; --card:#151d3d; --line:#26305e;
  --text:#eef1ff; --muted:#9aa3c7; --accent:#f5b301; --accent2:#e63946;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Noto Sans TC","PingFang TC","Microsoft JhengHei",sans-serif;line-height:1.7;-webkit-font-smoothing:antialiased}
a{color:inherit}
.wrap{max-width:960px;margin:0 auto;padding:0 20px}
header{position:sticky;top:0;z-index:10;background:rgba(11,16,32,.9);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
header .wrap{display:flex;align-items:center;justify-content:space-between;height:60px}
.logo{font-weight:800;font-size:18px;letter-spacing:.5px}
.logo span{color:var(--accent)}
nav{display:flex;gap:18px;font-size:14px}
nav a{text-decoration:none;color:var(--muted)}
nav a:hover{color:var(--text)}
@media(max-width:640px){nav a:nth-child(3){display:none}}
.hero{padding:72px 0 56px;text-align:center}
.badge{display:inline-block;font-size:12px;letter-spacing:2px;color:var(--accent);border:1px solid var(--accent);border-radius:999px;padding:4px 14px;margin-bottom:20px}
.hero h1{font-size:clamp(32px,6vw,54px);line-height:1.3;font-weight:800;margin-bottom:16px}
.hero h1 em{font-style:normal;color:var(--accent)}
.hero p{color:var(--muted);max-width:640px;margin:0 auto 28px;font-size:16px}
.cta{display:flex;gap:12px;justify-content:center;flex-wrap:wrap}
.btn{display:inline-block;padding:12px 26px;border-radius:10px;font-weight:700;font-size:15px;text-decoration:none;cursor:pointer;border:0}
.btn-gold{background:var(--accent);color:#1a1405}
.btn-ghost{background:transparent;color:var(--text);border:1px solid var(--line)}
.btn-ghost:hover{border-color:var(--muted)}
section{padding:48px 0}
h2{font-size:24px;font-weight:800;margin-bottom:8px}
.sec-sub{color:var(--muted);font-size:14px;margin-bottom:24px}
.demo{background:var(--bg2);border:1px solid var(--line);border-radius:16px;padding:24px}
.demo label{font-size:13px;color:var(--muted);display:block;margin-bottom:6px}
.demo select{background:var(--card);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:8px 12px;font-size:14px;margin-bottom:14px}
.demo textarea{width:100%;min-height:110px;background:var(--card);color:var(--text);border:1px solid var(--line);border-radius:10px;padding:12px;font-size:15px;font-family:inherit;resize:vertical}
.demo textarea:focus{outline:none;border-color:var(--accent)}
.demo .row{display:flex;gap:10px;align-items:center;margin:14px 0;flex-wrap:wrap}
#out{display:none;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-top:6px}
#out.show{display:block}
#out .zh{font-size:16px;margin-bottom:12px;white-space:pre-wrap}
#out .tags{font-size:13px;color:var(--accent)}
#out .warn{font-size:12px;color:var(--accent2);margin-top:8px}
.loading{color:var(--muted);font-size:14px}
.note{font-size:12px;color:var(--muted);margin-top:12px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;text-decoration:none;display:block;transition:transform .15s,border-color .15s}
.card:hover{transform:translateY(-2px);border-color:var(--accent)}
.card .handle{font-weight:800;font-size:16px;margin-bottom:6px;color:var(--accent)}
.card .desc{font-size:13px;color:var(--muted)}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}
.step{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px}
.step .n{display:inline-flex;width:32px;height:32px;border-radius:50%;background:var(--accent);color:#1a1405;font-weight:800;align-items:center;justify-content:center;margin-bottom:10px}
.step h3{font-size:15px;margin-bottom:6px}
.step p{font-size:13px;color:var(--muted)}
.disclaimer{border:1px dashed var(--line);border-radius:12px;padding:18px 20px;font-size:13px;color:var(--muted);text-align:center}
footer{border-top:1px solid var(--line);padding:24px 0;color:var(--muted);font-size:12px;text-align:center}
footer a{color:var(--muted)}
</style>
</head>
<body>
<header>
  <div class="wrap">
    <div class="logo">譯美國<span> · </span>Translate America</div>
    <nav>
      <a href="#demo">即時翻譯</a>
      <a href="#accounts">旗下帳號</a>
      <a href="#how">點運作</a>
    </nav>
  </div>
</header>

<div class="wrap">
  <div class="hero">
    <div class="badge">翻譯網絡 · TRANSLATION NETWORK</div>
    <h1>美國發生咩事,<em>譯返俾香港人睇。</em></h1>
    <p>每日監察美國政壇人物嘅 Instagram — 霍利參議員、特朗普 — 翻譯成香港人睇得明嘅繁體中文,重要片段加埋廣東話字幕;英文原片就直出 Global News Shorts。真人覆核,先至出街。</p>
    <div class="cta">
      <a class="btn btn-gold" href="#demo">即刻試下翻譯</a>
      <a class="btn btn-ghost" href="#accounts">睇旗下帳號</a>
    </div>
  </div>

  <section id="demo">
    <h2>即時翻譯試玩</h2>
    <p class="sec-sub">貼一段英文 caption 入嚟,同我哋部機用同一套 AI 即時譯做繁體中文。</p>
    <div class="demo">
      <label for="src">邊個講嘅?</label>
      <select id="src">
        <option value="hawley">Josh Hawley 霍利參議員</option>
        <option value="trump">Donald Trump 特朗普</option>
      </select>
      <label for="cap">英文原文</label>
      <textarea id="cap" placeholder="Paste an English caption here…&#10;例如: Big Tech has too much power. It's time to break them up."></textarea>
      <div class="row">
        <button class="btn btn-gold" id="go" type="button">翻譯成繁體中文</button>
      </div>
      <div id="out"></div>
      <p class="note">由 Cloudflare Workers AI 即時翻譯 · 同正式 pipeline 同一粒 model · 機器翻譯僅供參考</p>
    </div>
  </section>

  <section id="accounts">
    <h2>旗下帳號</h2>
    <p class="sec-sub">一個網絡,三種口味。</p>
    <div class="cards">
      <a class="card" href="https://www.instagram.com/snhawleytranslatorhkunofficial" target="_blank" rel="noopener">
        <div class="handle">@snhawleytranslatorhkunofficial</div>
        <div class="desc">霍利參議員 · 繁體中文翻譯<br>Sen. Josh Hawley, in Traditional Chinese</div>
      </a>
      <a class="card" href="https://www.instagram.com/trumptranslatorhkunofficial" target="_blank" rel="noopener">
        <div class="handle">@trumptranslatorhkunofficial</div>
        <div class="desc">特朗普 · 繁體中文翻譯<br>Donald Trump, in Traditional Chinese</div>
      </a>
      <a class="card" href="https://www.instagram.com/globalnewsshorts" target="_blank" rel="noopener">
        <div class="handle">@globalnewsshorts</div>
        <div class="desc">環球新聞 shorts · 英文原片直出<br>USA &amp; world, in shorts</div>
      </a>
    </div>
  </section>

  <section id="how">
    <h2>點樣運作</h2>
    <p class="sec-sub">全自動流水線,最後一關係人。</p>
    <div class="steps">
      <div class="step"><div class="n">1</div><h3>監察</h3><p>每 4 小時掃一次目標 IG 帳號,有新 post 即刻發現。</p></div>
      <div class="step"><div class="n">2</div><h3>讀圖</h3><p>OCR 抽出圖片入面嘅文字,影片就轉做廣東話字幕。</p></div>
      <div class="step"><div class="n">3</div><h3>翻譯</h3><p>Workers AI 譯成香港人寫法嘅繁體中文,人名有固定譯法。</p></div>
      <div class="step"><div class="n">4</div><h3>覆核</h3><p>真人睇過、改過,先至出 post。唔會自動亂出街。</p></div>
    </div>
  </section>

  <section>
    <div class="disclaimer">
      非官方 fans 翻譯網絡,同任何被翻譯嘅人物及其團隊無關。<br>
      Unofficial fan translation network. Not affiliated with or endorsed by any officeholder.
    </div>
  </section>
</div>

<footer>
  <div class="wrap">
    © 2026 譯美國 Translate America · Built on Cloudflare Workers + Workers AI<br>
    <a href="/health">API status</a>
  </div>
</footer>

<script>
(function(){
  var go = document.getElementById('go');
  var out = document.getElementById('out');
  go.addEventListener('click', function(){
    var caption = document.getElementById('cap').value.trim();
    var source = document.getElementById('src').value;
    if(!caption){ out.className='show'; out.innerHTML='<div class="loading">請貼一段英文入嚟先。</div>'; return; }
    out.className='show';
    out.innerHTML='<div class="loading">翻譯緊…</div>';
    go.disabled = true;
    fetch('/translate', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ caption: caption, source: source, target: 'zh-Hant' })
    }).then(function(r){ return r.json(); }).then(function(d){
      var html = '<div class="zh"></div><div class="tags"></div><div class="warn"></div>';
      out.innerHTML = html;
      out.querySelector('.zh').textContent = d.caption_zh || '(翻譯唔到,試過另一段)';
      out.querySelector('.tags').textContent = (d.seo_hashtags||[]).join(' ');
      if(d.warnings && d.warnings.length){ out.querySelector('.warn').textContent = '注意: ' + d.warnings.join('; '); }
      go.disabled = false;
    }).catch(function(){
      out.innerHTML='<div class="loading">出錯,稍後再試。</div>';
      go.disabled = false;
    });
  });
})();
</script>
</body>
</html>
`;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/health") {
      return Response.json({ ok: true, time: new Date().toISOString() });
    }
    if (request.method === "GET" && url.pathname === "/") {
      return new Response(SITE_HTML, {
        headers: { "Content-Type": "text/html; charset=utf-8" },
      });
    }
    if (request.method === "POST" && url.pathname === "/translate") {
      return handleTranslate(request, env);
    }
    return Response.json({ error: "not found" }, { status: 404 });
  },
};
