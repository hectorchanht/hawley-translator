/**
 * hawley-translator — Cloudflare Worker
 *
 * Translates Instagram posts (caption + text visible in images) into
 * Traditional Chinese, Hong Kong written style, for the fan account
 * @snhawleytranslatorhkunofficial (translations of US Senator Josh Hawley).
 *
 * Endpoints:
 *   GET  /health     -> { ok: true }
 *   POST /translate  -> { caption_zh, image_texts, warnings }
 *
 * POST /translate body:
 *   {
 *     "caption": "English caption text",
 *     "image_urls": ["https://...jpg", ...]   // up to 4, publicly fetchable
 *   }
 *
 * Response:
 *   {
 *     "caption_zh": "繁體中文翻譯…",
 *     "image_texts": [
 *       { "url": "https://…", "text_en": "OCR text or ''", "text_zh": "翻譯…" }
 *     ],
 *     "warnings": ["…"]   // empty when everything worked
 *   }
 *
 * Notes:
 * - OCR via Workers AI vision model; translation via Workers AI LLM.
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
  let imageUrls = Array.isArray(body.image_urls) ? body.image_urls : [];
  imageUrls = imageUrls.filter((u) => typeof u === "string" && u.startsWith("http")).slice(0, MAX_IMAGES);

  // 1) OCR every image in parallel.
  const ocrResults = await Promise.all(imageUrls.map((url) => ocrImage(env, url)));
  const imageTexts = ocrResults.map((r, i) => {
    if (!r.ok) warnings.push(`image ${i + 1}: OCR skipped (${r.error})`);
    return { url: imageUrls[i], text_en: r.ok ? r.text : "", text_zh: "" };
  });

  // 2) Translate caption + OCR'd texts in one LLM call.
  const prompt = buildTranslationPrompt(caption, imageTexts.map((t) => t.text_en));
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
      imageTexts.forEach((t, i) => {
        t.text_zh = typeof zhList[i] === "string" ? zhList[i] : "";
      });
    } catch {
      // Best-effort fallback: surface raw model output for the human reviewer.
      warnings.push("translation output was not valid JSON; returning raw model text for review");
      captionZh = raw.trim();
    }
  } catch (e) {
    warnings.push(`translation failed: ${String(e?.message || e).slice(0, 200)}`);
  }

  return Response.json({ caption_zh: captionZh, image_texts: imageTexts, warnings });
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
        usage: "POST /translate with { caption, image_urls[] }",
      });
    }
    if (request.method === "POST" && url.pathname === "/translate") {
      return handleTranslate(request, env);
    }
    return Response.json({ error: "not found" }, { status: 404 });
  },
};
