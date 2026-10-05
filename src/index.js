/**
 * hawley-translator — Cloudflare Worker
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
  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");
  if (start === -1 || end === -1) throw new Error("no JSON object in model output");
  return JSON.parse(cleaned.slice(start, end + 1));
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

function buildTranslationPrompt(caption, imageTexts, segments, source) {
  const nameRule = SOURCE_NAME_RULES[source] || "";
  const mainTag = SOURCE_MAIN_TAG[source] || "#翻譯";
  // NOTE 2026-10-05: the old prompt listed segments as "SPEECH N:" blocks and
  // the model pattern-completed them (emitting "SPEECH 17: <english>" instead
  // of JSON, in English instead of Chinese). Input is now JSON and the
  // response-format instruction sits at the very END, right before output.
  const rules = [
    "You are a professional translator writing for a Hong Kong audience.",
    "Translate the English content in INPUT below into Traditional Chinese, Hong Kong written style (書面語，繁體中文).",
    "STRICT RULES:",
    "- Translate ONLY what is in INPUT. Do NOT add commentary, explanations, background, people, events, or any content not present in the source.",
    "- Keep @mentions (e.g. @senatorhawley) and #hashtags unchanged.",
    "- Preserve line breaks and paragraph structure.",
    "- Be faithful and concise.",
    "- If a segment is empty, return an empty string for it.",
    "- \"segments\" become burned-in video subtitles: keep each one short " +
      "(one line, under 20 Chinese characters when possible), Cantonese-flavoured " +
      "where natural (嘅, 咁, 係, 唔), faithful to the spoken meaning.",
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
    segments: segments.map((s) => ({ i: s.i, text_en: s.text_en || "" })),
  };
  return rules.join("\n") +
    "\n\nINPUT (JSON):\n" + JSON.stringify(input) +
    '\n\nReturn ONLY a JSON object with EXACTLY these keys — no markdown fences, ' +
    "no commentary before or after:\n" +
    '{"caption_zh": "...", "image_texts_zh": ["...", "..."], ' +
    '"segments_zh": [{"i": 0, "text_zh": "..."}], ' +
    '"seo_hook": "...", "seo_hashtags": ["#..."]}' +
    "\n\nYOUR JSON RESPONSE:";
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

  const prompt = buildTranslationPrompt(caption, slots.map((s) => s.text_en), inSegs, source);
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
  try {
    const out = await env.AI.run(TEXT_MODEL, {
      prompt,
      max_tokens: 4096,
    });
    const raw = String(out?.response ?? "");
    try {
      const parsed = extractJson(raw);
      captionZh = typeof parsed.caption_zh === "string" ? parsed.caption_zh : "";
      const zhList = Array.isArray(parsed.image_texts_zh) ? parsed.image_texts_zh : [];
      slots.forEach((s, i) => {
        s.text_zh = typeof zhList[i] === "string" ? zhList[i] : "";
      });
      const segList = Array.isArray(parsed.segments_zh) ? parsed.segments_zh : [];
      const segByI = new Map(segList.filter((s) => s && typeof s.i === "number")
        .map((s) => [s.i, typeof s.text_zh === "string" ? s.text_zh : ""]));
      segmentsZh = inSegs.map((s) => ({ i: s.i, text_zh: segByI.get(s.i) ?? "" }));
      seoHook = typeof parsed.seo_hook === "string" ? parsed.seo_hook : "";
      const tagList = Array.isArray(parsed.seo_hashtags) ? parsed.seo_hashtags : [];
      seoTags = tagList.filter((t) => typeof t === "string" && t.startsWith("#")).slice(0, 10);
      if (seoTags.length && !seoTags.includes(mainTag)) seoTags.unshift(mainTag);
    } catch {
      // Best-effort fallback: surface raw model output for the human reviewer.
      warnings.push("translation output was not valid JSON; returning raw model text for review");
      captionZh = raw.trim();
    }
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

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/health") {
      return Response.json({ ok: true, time: new Date().toISOString() });
    }
    if (request.method === "GET" && url.pathname === "/") {
      return Response.json({
        name: "hawley-translator",
        usage: "POST /translate with { caption, image_texts: [{url, text_en}], image_urls? }",
      });
    }
    if (request.method === "POST" && url.pathname === "/translate") {
      return handleTranslate(request, env);
    }
    return Response.json({ error: "not found" }, { status: 404 });
  },
};
