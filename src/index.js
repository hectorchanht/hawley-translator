/**
 * hawley-translator — Cloudflare Worker
 *
 * Translates Instagram posts (caption + text visible in images) into
 * Traditional Chinese, Hong Kong written style, for the fan account
 * @snhawleytranslatorhkunofficial (translations of US Senator Josh Hawley).
 *
 * OCR does NOT happen here: PaddleOCR (PP-OCRv5 — the same engine realufo.org's
 * crawler uses) runs on the VM inside scripts/poll.py, because PaddleOCR
 * needs Python + native libs and can't run in a Worker. This Worker only
 * translates pre-extracted text via Workers AI.
 *
 * Endpoints:
 *   GET  /health     -> { ok: true }
 *   POST /translate  -> { caption_zh, image_texts, warnings }
 *
 * POST /translate body:
 *   {
 *     "caption": "English caption text",
 *     "image_texts": [
 *       { "url": "images/<shortcode>/img0.jpg", "text_en": "OCR'd English text" }
 *     ]
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

const TEXT_MODEL = "@cf/meta/llama-3.1-8b-instruct";
const MAX_IMAGES = 4;

function extractJson(raw) {
  const cleaned = String(raw)
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```\s*$/, "");
  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");
  if (start === -1 || end === -1) throw new Error("no JSON object in model output");
  return JSON.parse(cleaned.slice(start, end + 1));
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
  const imageTexts = inTexts.slice(0, MAX_IMAGES).map((t) => ({
    url: typeof t?.url === "string" ? t.url : "",
    text_en: typeof t?.text_en === "string" ? t.text_en : "",
    text_zh: "",
  }));

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
        usage: "POST /translate with { caption, image_texts: [{url, text_en}] }",
      });
    }
    if (request.method === "POST" && url.pathname === "/translate") {
      return handleTranslate(request, env);
    }
    return Response.json({ error: "not found" }, { status: 404 });
  },
};
