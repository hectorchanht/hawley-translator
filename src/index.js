/**
 * hawley-translator — Cloudflare Worker
 *
 * Translates Instagram posts (caption + text visible in images) into
 * Traditional Chinese, Hong Kong written style, for the fan account
 * @snhawleytranslatorhkunofficial (translations of US Senator Josh Hawley).
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
 *     "image_texts": [                       // preferred: pre-OCR'd
 *       { "url": "images/<sc>/img0.jpg", "text_en": "OCR'd English text" }
 *     ],
 *     "image_urls": ["https://…jpg"]          // fallback: Worker OCRs these
 *   }
 *
 * Response:
 *   {
 *     "caption_zh": "繁體中文翻譯…",
 *     "image_texts": [
 *       { "url": "…", "text_en": "…", "text_zh": "翻譯…" }
 *     ],
 *     "warnings": ["…"]   // empty when everything worked
 *   }
 *
 * Notes:
 * - No database, no secrets, no posting — this Worker only translates.
 * - Review/publishing happens outside (see scripts/poll.py).
 */

const VISION_MODEL = "@cf/meta/llama-3.2-11b-vision-instruct";
const TEXT_MODEL = "@cf/meta/llama-3.1-8b-instruct";
const MAX_IMAGES = 4;
const MAX_IMAGE_BYTES = 2 * 1024 * 1024; // guard for vision-model input

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

function buildTranslationPrompt(caption, imageTexts) {
  const parts = [
    "You are a professional translator writing for a Hong Kong audience.",
    "Translate the English content below into Traditional Chinese, Hong Kong written style (書面語，繁體中文).",
    "Rules:",
    '- Render "Josh Hawley" / "Hawley" / "Senator Hawley" as 霍利 / 霍利參議員.',
    "- Keep @mentions (e.g. @senatorhawley) and #hashtags unchanged.",
    "- Preserve line breaks and paragraph structure.",
    "- Be faithful and concise. Do not add commentary, explanations, or extra content.",
    "- If a segment is empty, return an empty string for it.",
    "",
    "Return ONLY valid JSON, no markdown fences, with this exact shape:",
    '{"caption_zh": "...", "image_texts_zh": ["...", "..."]}',
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
  const inTexts = Array.isArray(body.image_texts) ? body.image_texts : [];
  const inUrls = (Array.isArray(body.image_urls) ? body.image_urls : [])
    .filter((u) => typeof u === "string" && u.startsWith("http"));
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

  const prompt = buildTranslationPrompt(caption, slots.map((s) => s.text_en));
  let captionZh = "";
  try {
    const out = await env.AI.run(TEXT_MODEL, {
      prompt,
      max_tokens: 2048,
    });
    const raw = String(out?.response ?? "");
    try {
      const parsed = extractJson(raw);
      captionZh = typeof parsed.caption_zh === "string" ? parsed.caption_zh : "";
      const zhList = Array.isArray(parsed.image_texts_zh) ? parsed.image_texts_zh : [];
      slots.forEach((s, i) => {
        s.text_zh = typeof zhList[i] === "string" ? zhList[i] : "";
      });
    } catch {
      // Best-effort fallback: surface raw model output for the human reviewer.
      warnings.push("translation output was not valid JSON; returning raw model text for review");
      captionZh = raw.trim();
    }
  } catch (e) {
    warnings.push(`translation failed: ${String(e?.message || e).slice(0, 200)}`);
  }

  return Response.json({ caption_zh: captionZh, image_texts: slots, warnings });
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
