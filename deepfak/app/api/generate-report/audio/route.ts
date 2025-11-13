// app/api/generate-report/audio/route.ts
import { NextRequest, NextResponse } from "next/server";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { PDFDocument, StandardFonts, rgb } from "pdf-lib";

/* ---------------- Helpers (same as image) ---------------- */

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



function buildAudioPrompt(result: { isFake: boolean; timestamp?: string; image_id?: string | null }) {
  const verdict = result.isFake ? "Likely AI-Generated" : "Appears Authentic";
  return `You are an expert audio forensic analyst.
Write a concise professional report with these sections:
1) Title: "Audio Analysis Report"
2) Result: one line stating the system's verdict: "${verdict}"
3) Summary: 2–3 sentences explaining the verdict (no numeric confidence).
4) Key Observations: 3–6 short bullets about what audio cues or model focus imply.
5) Suggested Next Steps: 3–6 short bullets for human verification.

Important: DO NOT include any numeric confidence values. Use "-" for bullets. End with a short disclaimer that the analysis is automated and advisory only.

Metadata:
- Timestamp: ${result.timestamp || "N/A"}
- Audio ID: ${result.image_id || "N/A"}

Return plain text only.`;
}

function localAudioFallback(result: { isFake: boolean; timestamp?: string; image_id?: string | null }) {
  const verdict = result.isFake ? "Likely AI-Generated" : "Appears Authentic";
  return `Audio Analysis Report

Result: ${verdict}

Summary:
This automated analysis indicates the audio is ${verdict.toLowerCase()}. Use as an advisory starting point for manual verification.

Key Observations:
- ${result.isFake ? "Detected spectral patterns and artifacts consistent with synthetic voice generation." : "No clear synthetic artifacts detected at the tested resolution."}
- ${result.isFake ? "Unnatural frequency-band anomalies and temporal smoothing were noted." : "Frequency and temporal patterns are consistent with natural speech."}
- ${result.isFake ? "Spectrogram highlights regions of synthetic alteration." : "Model attention aligns with expected phonetic features."}

Suggested Next Steps:
- Compare with original recordings or known authentic samples.
- Examine metadata and provenance if available.
- Obtain higher-quality sources or raw audio for detailed forensic analysis.

Timestamp: ${result.timestamp || "N/A"}
${result.image_id ? `Audio ID: ${result.image_id}\n` : ""}
Disclaimer:
This analysis is automated and advisory only.`;
}

/* Exponential backoff and helpers are same as image file */
async function generateWithRetry(model: any, prompt: string, maxRetries = 3, baseDelay = 1000) {
  let attempt = 0;
  while (attempt <= maxRetries) {
    try {
      const resp = await model.generateContent(prompt);
      return { ok: true, response: resp };
    } catch (err: any) {
      attempt++;
      if (attempt > maxRetries) return { ok: false, error: err };
      const jitter = Math.floor(Math.random() * 300);
      const delay = baseDelay * Math.pow(2, attempt - 1) + jitter;
      console.warn(`Gemini attempt ${attempt} failed; retrying in ${delay}ms`, err?.message || err);
      await new Promise((r) => setTimeout(r, delay));
    }
  }
  return { ok: false, error: new Error("Retries exhausted") };
}
async function extractTextFromSdkResponse(sdkResp: any): Promise<string> {
  try {
    if (!sdkResp) return "";
    if (typeof sdkResp.response?.text === "function") return await sdkResp.response.text();
    if (sdkResp?.candidates?.[0]?.content?.parts?.[0]?.text) return sdkResp.candidates[0].content.parts[0].text;
    if (typeof sdkResp === "string") return sdkResp;
    return JSON.stringify(sdkResp);
  } catch (e) {
    console.error("extractTextFromSdkResponse error", e);
    return "";
  }
}
function splitIntoSections(text: string) {
  const sections: Record<string, string> = {};
  if (!text) return sections;
  const lines = text.split(/\r?\n/);
  let cur: string | null = null;
  let buf: string[] = [];
  for (const raw of lines) {
    const line = raw.trim();
    const match = line.match(/^(Title|Result|Summary|Key Observations|Suggested Next Steps|Disclaimer|Audio ID)[:\s]*$/i);
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
    const { result, preview, heatmap } = body;
    if (!result || typeof result.isFake !== "boolean") return NextResponse.json({ error: "Missing or invalid 'result'." }, { status: 400 });

    const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
    if (!GEMINI_API_KEY) return NextResponse.json({ error: "GEMINI_API_KEY not configured." }, { status: 500 });

    const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

    const prompt = buildAudioPrompt(result);
    const genResult = await generateWithRetry(model, prompt, 3, 1000);

    let reportText = "";
    let usedFallback = false;

    if (genResult.ok && genResult.response) {
      reportText = await extractTextFromSdkResponse(genResult.response);
      if (!reportText || !reportText.trim()) { usedFallback = true; reportText = localAudioFallback(result); }
    } else {
      usedFallback = true;
      console.error("Gemini error:", genResult.error);
      reportText = localAudioFallback(result);
    }

    // Build PDF (audio-specific title, rest similar)
    const pdfDoc = await PDFDocument.create();
    const font = await pdfDoc.embedFont(StandardFonts.Helvetica);
    const page = pdfDoc.addPage();
    const { width, height } = page.getSize();
    const margin = 50;
    const maxWidth = width - margin * 2;
    let y = height - margin;

    page.drawText("Audio Analysis Report", { x: margin, y: y - 20, size: 20, font, color: rgb(0, 0, 0) });
    y -= 20 + 14;

    const verdictText = result.isFake ? "Likely AI-Generated" : "Appears Authentic";
    page.drawText(`Result: ${verdictText}`, { x: margin, y: y - 11, size: 11, font });
    y -= 11 + 6;
    if (result.timestamp) { page.drawText(`Timestamp: ${result.timestamp}`, { x: margin, y: y - 11, size: 11, font }); y -= 11 + 6; }
    if (result.image_id) { page.drawText(`Audio ID: ${result.image_id}`, { x: margin, y: y - 11, size: 11, font }); y -= 11 + 10; }
    if (usedFallback) { page.drawText("Note: Gemini unavailable; fallback used.", { x: margin, y: y - 9, size: 9, font, color: rgb(1, 0, 0) }); y -= 16; }

    const drawWrapped = (text: string, fontSize = 11) => {
      const words = text.split(/\s+/);
      let line = "";
      for (const w of words) {
        const test = line ? `${line} ${w}` : w;
        if (font.widthOfTextAtSize(test, fontSize) > maxWidth) { page.drawText(line, { x: margin, y: y - fontSize, size: fontSize, font }); y -= fontSize + 5; line = w; } else line = test;
      }
      if (line) { page.drawText(line, { x: margin, y: y - fontSize, size: fontSize, font }); y -= fontSize + 6; }
    };

    const renderHeading = (t: string) => { page.drawText(t, { x: margin, y: y - 13, size: 13, font }); y -= 19; };
    const renderBullets = (text: string) => {
      const lines = text.split("\n").map(l => l.trim()).filter(Boolean);
      for (const ln of lines) {
        const content = ln.startsWith("-") ? ln.replace(/^-+\s*/, "") : ln;
        page.drawText("•", { x: margin, y: y - 11, size: 11, font });
        const startX = margin + 14;
        let l = "";
        const words = content.split(/\s+/);
        for (const w of words) {
          const test = l ? `${l} ${w}` : w;
          if (font.widthOfTextAtSize(test, 11) > maxWidth - 14) { page.drawText(l, { x: startX, y: y - 11, size: 11, font }); y -= 16; l = w; } else l = test;
        }
        if (l) { page.drawText(l, { x: startX, y: y - 11, size: 11, font }); y -= 16; }
      }
      y -= 6;
    };

    const sections = splitIntoSections(reportText);
    if (sections["Summary"]) { renderHeading("Summary"); drawWrapped(sections["Summary"]); } else { renderHeading("Summary"); drawWrapped(reportText); }
    if (sections["Key Observations"]) { renderHeading("Key Observations"); renderBullets(sections["Key Observations"]); }
    if (sections["Suggested Next Steps"]) { renderHeading("Suggested Next Steps"); renderBullets(sections["Suggested Next Steps"]); }
    if (sections["Disclaimer"]) { renderHeading("Disclaimer"); drawWrapped(sections["Disclaimer"]); }

    // If a spectrogram (heatmap) was provided, add a page and embed it
    const heatmapBytes = dataUrlToUint8Array(heatmap);
    const previewBytes = dataUrlToUint8Array(preview);
    if (heatmapBytes || previewBytes) {
      const imgPage = pdfDoc.addPage();
      const top = imgPage.getSize().height - 50;
      let x = 50;
      const drawImage = async (bytes: Uint8Array, mime: string | null, offsetX: number) => {
        try {
          const embedded = mime?.includes("png") ? await pdfDoc.embedPng(bytes) : await pdfDoc.embedJpg(bytes);
          const scale = Math.min(260 / embedded.width, 380 / embedded.height, 1);
          imgPage.drawImage(embedded, { x: offsetX, y: top - embedded.height * scale, width: embedded.width * scale, height: embedded.height * scale });
        } catch (e) { console.warn("embed image error:", e); }
      };
      if (previewBytes) { await drawImage(previewBytes, dataUrlMime(preview), x); x += 280; }
      if (heatmapBytes) { await drawImage(heatmapBytes, dataUrlMime(heatmap), x); }
    }

    const pdfBytes = await pdfDoc.save();
    const headers: Record<string, string> = {
      "Content-Type": "application/pdf",
      "Content-Disposition": `attachment; filename="audio-report-${Date.now()}.pdf"`,
    };
    if (usedFallback) headers["x-used-fallback"] = "1";
    return new NextResponse(Buffer.from(pdfBytes), { status: 200, headers });
  } catch (err: any) {
    console.error("audio report error:", err);
    return NextResponse.json({ error: err?.message || "Internal server error" }, { status: 500 });
  }
}
