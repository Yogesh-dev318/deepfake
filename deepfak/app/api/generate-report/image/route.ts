// app/api/generate-report/image/route.ts
import { NextRequest, NextResponse } from "next/server";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { PDFDocument, StandardFonts, rgb } from "pdf-lib";

/* ---------------- Helpers ---------------- */

function dataUrlToUint8Array(dataUrl?: string | null) {
  if (!dataUrl) return null;
  const m = dataUrl.match(/^data:(.+);base64,(.*)$/);
  if (!m) return null;
  return new Uint8Array(Buffer.from(m[2], "base64"));
}
function dataUrlMime(dataUrl?: string | null) {
  if (!dataUrl) return null;
  const m = dataUrl.match(/^data:(.+);base64,(.*)$/);
  return m ? m[1] : null;
}

function buildImagePrompt(result: { isFake: boolean; timestamp?: string; image_id?: string | null }) {
  const verdict = result.isFake ? "Likely AI-Generated" : "Appears Authentic";
  return `You are an expert image forensic analyst.
Write a concise professional report with these sections:
1) Title: "Image Analysis Report"
2) Result: one line stating the system's verdict: "${verdict}"
3) Summary: 2–3 sentences explaining the verdict (no numeric confidence).
4) Key Observations: 3–6 short bullets about what visual cues or model attention imply.
5) Suggested Next Steps: 3–6 short bullets for human verification.

Important: DO NOT include any numeric confidence values. Use "-" for bullets. End with a short disclaimer that the analysis is automated and advisory only.

Metadata:
- Timestamp: ${result.timestamp || "N/A"}
- Image ID: ${result.image_id || "N/A"}

Return plain text only.`;
}

function localImageFallback(result: { isFake: boolean; timestamp?: string; image_id?: string | null }) {
  const verdict = result.isFake ? "Likely AI-Generated" : "Appears Authentic";
  return `Image Analysis Report

Result: ${verdict}

Summary:
This automated analysis indicates the media is ${verdict.toLowerCase()}. Use this as an advisory starting point for verification.

Key Observations:
- ${result.isFake ? "Detected texture/edge artifacts and blending cues consistent with generated images." : "No clear generative artifacts detected at the tested resolution."}
- ${result.isFake ? "Localized irregularities in shading or compositing were observed." : "Visual features appear consistent with photographic sources."}
- ${result.isFake ? "Model attention heatmap shows focus on areas with synthetic cues." : "Model attention aligns with plausible subject regions."}

Suggested Next Steps:
- Check original metadata (EXIF) and source provenance.
- Perform reverse-image search for potential originals.
- Obtain higher-resolution or original camera files for detailed forensic review.

Timestamp: ${result.timestamp || "N/A"}
${result.image_id ? `Image ID: ${result.image_id}\n` : ""}
Disclaimer:
This analysis is automated and advisory only.`;
}

/* Exponential backoff retry and helpers */

/**
 * Minimal structural type for the generative model used.
 * We only call generateContent(prompt) and don't rely on other SDK internals here.
 */
type GenerativeModelLike = {
  generateContent: (prompt: string) => Promise<unknown>;
};

function getErrorMessage(err: unknown): string {
  if (!err) return "Unknown error";
  if (typeof err === "string") return err;
  if (err instanceof Error) return err.message;
  try {
    const maybe = err as { message?: unknown };
    if (typeof maybe.message === "string") return maybe.message;
  } catch {
    // ignore
  }
  return "Unknown error";
}

async function generateWithRetry(
  model: GenerativeModelLike,
  prompt: string,
  maxRetries = 3,
  baseDelay = 1000
): Promise<{ ok: true; response: unknown } | { ok: false; error: unknown }> {
  let attempt = 0;
  while (attempt <= maxRetries) {
    try {
      const resp = await model.generateContent(prompt);
      return { ok: true, response: resp };
    } catch (err: unknown) {
      attempt++;
      if (attempt > maxRetries) return { ok: false, error: err };
      const jitter = Math.floor(Math.random() * 300);
      const delay = baseDelay * Math.pow(2, attempt - 1) + jitter;
      console.warn(`Gemini attempt ${attempt} failed; retrying in ${delay}ms`, getErrorMessage(err));
      await new Promise((r) => setTimeout(r, delay));
    }
  }
  return { ok: false, error: new Error("Retries exhausted") };
}

async function extractTextFromSdkResponse(sdkResp: unknown): Promise<string> {
  try {
    if (!sdkResp) return "";

    // Common SDK shapes:
    // 1) sdkResp.response.text() is a function
    // 2) sdkResp.candidates[0].content.parts[0].text is a string
    // 3) sdkResp is a string

    const respObj = sdkResp as { response?: unknown; candidates?: unknown[] };

    // Case: response.text() function
    const responseCandidate = respObj.response as unknown;
    if (responseCandidate && typeof responseCandidate === "object") {
      const maybeText = (responseCandidate as { text?: unknown }).text;
      if (typeof maybeText === "function") {
        const textFn = maybeText as () => Promise<string>;
        return await textFn();
      }
    }

    // Case: nested candidates/content/parts[0].text
    const cand = respObj.candidates as unknown;
    if (Array.isArray(cand) && cand[0] && typeof cand[0] === "object") {
      const first = cand[0] as Record<string, unknown>;
      const content = first.content as Record<string, unknown> | undefined;
      const parts = content?.parts as unknown;
      if (Array.isArray(parts) && parts[0] && typeof parts[0] === "object") {
        const part0 = parts[0] as Record<string, unknown>;
        const text = part0.text;
        if (typeof text === "string") return text;
      }
    }

    if (typeof sdkResp === "string") return sdkResp;
    return JSON.stringify(sdkResp);
  } catch (e: unknown) {
    console.error("extractTextFromSdkResponse error", getErrorMessage(e));
    return "";
  }
}

/* Utility: split sections */
function splitIntoSections(text: string) {
  const sections: Record<string, string> = {};
  if (!text) return sections;
  const lines = text.split(/\r?\n/);
  let cur: string | null = null;
  let buf: string[] = [];
  for (const raw of lines) {
    const line = raw.trim();
    const match = line.match(/^(Title|Result|Summary|Key Observations|Suggested Next Steps|Disclaimer|Image ID)[:\s]*$/i);
    if (match) {
      if (cur) sections[cur] = buf.join("\n").trim();
      cur = match[1];
      buf = [];
    } else {
      buf.push(raw);
    }
  }
  if (cur) sections[cur] = buf.join("\n").trim();
  if (Object.keys(sections).length === 0) sections["Summary"] = text.trim();
  return sections;
}

/* ---------------- Route ---------------- */

export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(() => null);
    if (!body) return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });

    const { result, preview, heatmap } = body as {
      result?: { isFake?: boolean; timestamp?: string; image_id?: string | null };
      preview?: string | null;
      heatmap?: string | null;
    };

    if (!result || typeof result.isFake !== "boolean") {
      return NextResponse.json({ error: "Missing or invalid 'result' object." }, { status: 400 });
    }

    const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
    if (!GEMINI_API_KEY) return NextResponse.json({ error: "GEMINI_API_KEY not configured." }, { status: 500 });

    const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" }) as unknown as GenerativeModelLike;

    const prompt = buildImagePrompt(result as { isFake: boolean; timestamp?: string; image_id?: string | null });
    const genResult = await generateWithRetry(model, prompt, 3, 1000);

    let reportText = "";
    let usedFallback = false;

    if (genResult.ok && genResult.response) {
      reportText = await extractTextFromSdkResponse(genResult.response);
      if (!reportText || !reportText.trim()) {
        usedFallback = true;
        reportText = localImageFallback(result as { isFake: boolean; timestamp?: string; image_id?: string | null });
      }
    } else {
      usedFallback = true;
      console.error("Gemini error:", getErrorMessage((genResult as { ok: false; error: unknown }).error));
      reportText = localImageFallback(result as { isFake: boolean; timestamp?: string; image_id?: string | null });
    }

    // Build PDF with pdf-lib (formatted)
    const pdfDoc = await PDFDocument.create();
    const font = await pdfDoc.embedFont(StandardFonts.Helvetica);
    const page = pdfDoc.addPage();
    const { width, height } = page.getSize();
    const margin = 50;
    const maxWidth = width - margin * 2;
    let y = height - margin;

    // Title
    page.drawText("Image Analysis Report", { x: margin, y: y - 20, size: 20, font, color: rgb(0, 0, 0) });
    y -= 20 + 14;

    // Metadata: Result, Timestamp, Image ID
    const verdictText = result.isFake ? "Likely AI-Generated" : "Appears Authentic";
    page.drawText(`Result: ${verdictText}`, { x: margin, y: y - 11, size: 11, font });
    y -= 11 + 6;
    if (result.timestamp) {
      page.drawText(`Timestamp: ${result.timestamp}`, { x: margin, y: y - 11, size: 11, font });
      y -= 11 + 6;
    }
    if (result.image_id) {
      page.drawText(`Image ID: ${result.image_id}`, { x: margin, y: y - 11, size: 11, font });
      y -= 11 + 10;
    }
    if (usedFallback) {
      page.drawText("Note: Gemini unavailable; fallback used.", {
        x: margin,
        y: y - 9,
        size: 9,
        font,
        color: rgb(1, 0, 0),
      });
      y -= 16;
    }

    // Helpers for wrapped text and bullets
    const drawWrapped = (text: string, fontSize = 11) => {
      const words = text.split(/\s+/);
      let line = "";
      for (const w of words) {
        const test = line ? `${line} ${w}` : w;
        if (font.widthOfTextAtSize(test, fontSize) > maxWidth) {
          page.drawText(line, { x: margin, y: y - fontSize, size: fontSize, font });
          y -= fontSize + 5;
          line = w;
        } else line = test;
      }
      if (line) {
        page.drawText(line, { x: margin, y: y - fontSize, size: fontSize, font });
        y -= fontSize + 6;
      }
    };

    const renderHeading = (text: string) => {
      page.drawText(text, { x: margin, y: y - 13, size: 13, font });
      y -= 19;
    };
    const renderBullets = (text: string) => {
      const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
      for (const ln of lines) {
        const content = ln.startsWith("-") ? ln.replace(/^-+\s*/, "") : ln;
        page.drawText("•", { x: margin, y: y - 11, size: 11, font });
        const startX = margin + 14;
        let l = "";
        const words = content.split(/\s+/);
        for (const w of words) {
          const test = l ? `${l} ${w}` : w;
          if (font.widthOfTextAtSize(test, 11) > maxWidth - 14) {
            page.drawText(l, { x: startX, y: y - 11, size: 11, font });
            y -= 16;
            l = w;
          } else l = test;
        }
        if (l) {
          page.drawText(l, { x: startX, y: y - 11, size: 11, font });
          y -= 16;
        }
      }
      y -= 6;
    };

    const sections = splitIntoSections(reportText);
    if (sections["Summary"]) {
      renderHeading("Summary");
      drawWrapped(sections["Summary"]);
    } else {
      renderHeading("Summary");
      drawWrapped(reportText);
    }
    if (sections["Key Observations"]) {
      renderHeading("Key Observations");
      renderBullets(sections["Key Observations"]);
    }
    if (sections["Suggested Next Steps"]) {
      renderHeading("Suggested Next Steps");
      renderBullets(sections["Suggested Next Steps"]);
    }
    if (sections["Disclaimer"]) {
      renderHeading("Disclaimer");
      drawWrapped(sections["Disclaimer"]);
    }

    // Images page (preview and heatmap if present)
    const previewBytes = dataUrlToUint8Array(preview ?? null);
    const heatmapBytes = dataUrlToUint8Array(heatmap ?? null);
    if (previewBytes || heatmapBytes) {
      const imgPage = pdfDoc.addPage();
      const top = imgPage.getSize().height - 50;
      let x = 50;
      const drawImage = async (bytes: Uint8Array, mime: string | null, offsetX: number) => {
        try {
          const embedded = mime?.includes("png") ? await pdfDoc.embedPng(bytes) : await pdfDoc.embedJpg(bytes);
          const scale = Math.min(260 / embedded.width, 380 / embedded.height, 1);
          imgPage.drawImage(embedded, {
            x: offsetX,
            y: top - embedded.height * scale,
            width: embedded.width * scale,
            height: embedded.height * scale,
          });
        } catch (e: unknown) {
          console.warn("embed image error:", getErrorMessage(e));
        }
      };
      if (previewBytes) {
        await drawImage(previewBytes, dataUrlMime(preview), x);
        x += 280;
      }
      if (heatmapBytes) {
        await drawImage(heatmapBytes, dataUrlMime(heatmap), x);
      }
    }

    const pdfBytes = await pdfDoc.save();
    const headers: Record<string, string> = {
      "Content-Type": "application/pdf",
      "Content-Disposition": `attachment; filename="image-report-${Date.now()}.pdf"`,
    };
    if (usedFallback) headers["x-used-fallback"] = "1";
    return new NextResponse(Buffer.from(pdfBytes), { status: 200, headers });
  } catch (err: unknown) {
    console.error("image report error:", getErrorMessage(err));
    return NextResponse.json({ error: getErrorMessage(err) || "Internal server error" }, { status: 500 });
  }
}
