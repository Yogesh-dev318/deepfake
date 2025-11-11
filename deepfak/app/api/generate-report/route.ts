import { NextRequest, NextResponse } from "next/server";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { PDFDocument, StandardFonts, rgb } from "pdf-lib";

/* ---------- Helper: Decode image data URLs ---------- */
function dataUrlToUint8Array(dataUrl?: string | null) {
  if (!dataUrl) return null;
  const match = dataUrl.match(/^data:(.+);base64,(.*)$/);
  if (!match) return null;
  const base64 = match[2];
  const binary = Buffer.from(base64, "base64");
  return new Uint8Array(binary);
}

function dataUrlMime(dataUrl?: string | null) {
  if (!dataUrl) return null;
  const match = dataUrl.match(/^data:(.+);base64,(.*)$/);
  return match ? match[1] : null;
}

/* ---------- Prompt for Gemini (NO confidence) ---------- */
function buildPrompt(result: {
  isFake: boolean;
  timestamp?: string;
  image_id?: string | null;
}) {
  const verdict = result.isFake ? "Likely AI-Generated" : "Appears Authentic";
  return `You are a forensic AI image analyst.
Write a structured professional report that contains the following sections:

1) Title: "Image Analysis Report"
2) Result: (one line stating the system's verdict: "${verdict}")
3) Summary: 2–3 sentences explaining what the result means.
4) Key Observations: 3–6 bullet points summarizing what visual cues or evidence the model might have relied on.
5) Suggested Next Steps: 3–6 bullet points advising how a human investigator could verify the result.

Important:
- Do NOT include any numeric confidence values.
- Keep the language formal, neutral, and concise.
- Use "-" as bullet characters.
- End with a short disclaimer stating that the analysis is automated and advisory only.

Additional metadata:
- Timestamp: ${result.timestamp || "N/A"}
- Image ID: ${result.image_id || "N/A"}

Return plain text only.`;
}

/* ---------- Local fallback text generator ---------- */
function localGenerateReportText(result: {
  isFake: boolean;
  timestamp?: string;
  image_id?: string | null;
}) {
  const verdict = result.isFake ? "Likely AI-Generated" : "Appears Authentic";
  return `Image Analysis Report

Result: ${verdict}

Summary:
This automated analysis examined the provided image and determined it is ${verdict.toLowerCase()}. The system evaluated multiple visual cues to form this conclusion.

Key Observations:
- ${result.isFake ? "Detected texture and lighting patterns consistent with AI-generated imagery." : "No visual inconsistencies typical of AI generation were detected."}
- ${result.isFake ? "Localized shading irregularities and blending seams were noted." : "Edges, shading, and noise distribution appear natural."}
- ${result.isFake ? "Attention heatmap highlighted areas with synthetic blending characteristics." : "Model attention concentrated on realistic subject features."}

Suggested Next Steps:
- Examine metadata (EXIF) for editing traces or missing camera info.
- Perform reverse image search to check originality.
- Verify image origin with known authentic references.
- Consider manual expert review for confirmation.

Timestamp: ${result.timestamp || "N/A"}
${result.image_id ? `Image ID: ${result.image_id}\n` : ""}
Disclaimer:
This report was generated automatically. Interpret results as advisory guidance, not absolute proof.`;
}

/* ---------- Gemini Retry with exponential backoff ---------- */
async function generateWithRetry(model: any, prompt: string, maxRetries = 3, baseDelay = 1000) {
  let attempt = 0;
  while (attempt <= maxRetries) {
    try {
      const response = await model.generateContent(prompt);
      return { ok: true, response };
    } catch (err: any) {
      attempt++;
      if (attempt > maxRetries) {
        return { ok: false, error: err };
      }
      const jitter = Math.floor(Math.random() * 300);
      const delay = baseDelay * Math.pow(2, attempt - 1) + jitter;
      console.warn(`Gemini attempt ${attempt} failed; retrying in ${delay}ms:`, err?.message);
      await new Promise((r) => setTimeout(r, delay));
    }
  }
  return { ok: false, error: new Error("Retries exhausted") };
}

/* ---------- Parse report sections for clean layout ---------- */
function splitIntoSections(text: string) {
  const sections: Record<string, string> = {};
  if (!text) return sections;
  const lines = text.split(/\r?\n/);
  let current: string | null = null;
  let buffer: string[] = [];
  for (const line of lines) {
    const headingMatch = line.match(/^(Title|Result|Summary|Key Observations|Suggested Next Steps|Disclaimer)[:\s]?$/i);
    if (headingMatch) {
      if (current) sections[current] = buffer.join("\n").trim();
      current = headingMatch[1];
      buffer = [];
    } else {
      buffer.push(line);
    }
  }
  if (current) sections[current] = buffer.join("\n").trim();
  if (Object.keys(sections).length === 0) sections["Summary"] = text.trim();
  return sections;
}

/* ---------- Main API Route ---------- */
export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(() => null);
    if (!body) return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
    const { result, preview, heatmap } = body;
    if (!result) return NextResponse.json({ error: "Missing result." }, { status: 400 });

    const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
    if (!GEMINI_API_KEY) return NextResponse.json({ error: "Missing GEMINI_API_KEY." }, { status: 500 });

    // Gemini setup
    const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

    const prompt = buildPrompt(result);
    const genResult = await generateWithRetry(model, prompt, 3, 1000);

    let reportText = "";
    let usedFallback = false;

    if (genResult.ok && genResult.response) {
      try {
        const resp = genResult.response;
        if (typeof resp.response?.text === "function") {
          reportText = await resp.response.text();
        } else if (resp?.candidates?.[0]?.content?.parts?.[0]?.text) {
          reportText = resp.candidates[0].content.parts[0].text;
        } else if (typeof resp === "string") {
          reportText = resp;
        } else {
          reportText = JSON.stringify(resp);
        }
      } catch (e) {
        console.error("Failed to parse Gemini output:", e);
        usedFallback = true;
        reportText = localGenerateReportText(result);
      }
    } else {
      usedFallback = true;
      console.error("Gemini generation failed:", genResult.error);
      reportText = localGenerateReportText(result);
    }

    /* ---------- Create formatted PDF with pdf-lib ---------- */
    const pdfDoc = await PDFDocument.create();
    const font = await pdfDoc.embedFont(StandardFonts.Helvetica);
    const page = pdfDoc.addPage();
    const { width, height } = page.getSize();
    const margin = 50;
    const maxWidth = width - margin * 2;
    let y = height - margin;

    // Title
    const titleSize = 20;
    page.drawText("Image Analysis Report", {
      x: margin,
      y: y - titleSize,
      size: titleSize,
      font,
      color: rgb(0, 0, 0),
    });
    y -= titleSize + 14;

    // Metadata
    const metaSize = 11;
    const verdict = result.isFake ? "Likely AI-Generated" : "Appears Authentic";
    page.drawText(`Result: ${verdict}`, { x: margin, y: y - metaSize, size: metaSize, font });
    y -= metaSize + 6;
    if (result.timestamp) {
      page.drawText(`Timestamp: ${result.timestamp}`, { x: margin, y: y - metaSize, size: metaSize, font });
      y -= metaSize + 6;
    }
    if (result.image_id) {
      page.drawText(`Image ID: ${result.image_id}`, { x: margin, y: y - metaSize, size: metaSize, font });
      y -= metaSize + 10;
    }
    if (usedFallback) {
      page.drawText("Note: Gemini service unavailable; fallback report generated.", {
        x: margin,
        y: y - 9,
        size: 9,
        font,
        color: rgb(1, 0, 0),
      });
      y -= 18;
    }

    // Helper to draw wrapped paragraphs
    const drawWrapped = (text: string, fontSize = 11) => {
      const words = text.split(/\s+/);
      let line = "";
      for (const word of words) {
        const test = line ? `${line} ${word}` : word;
        const widthUsed = font.widthOfTextAtSize(test, fontSize);
        if (widthUsed > maxWidth) {
          page.drawText(line, { x: margin, y: y - fontSize, size: fontSize, font });
          y -= fontSize + 5;
          line = word;
        } else {
          line = test;
        }
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
      for (const line of lines) {
        const content = line.startsWith("-") ? line.replace(/^-+\s*/, "") : line;
        page.drawText("•", { x: margin, y: y - 11, size: 11, font });
        const startX = margin + 14;
        const words = content.split(/\s+/);
        let l = "";
        for (const w of words) {
          const test = l ? `${l} ${w}` : w;
          if (font.widthOfTextAtSize(test, 11) > maxWidth - 14) {
            page.drawText(l, { x: startX, y: y - 11, size: 11, font });
            y -= 16;
            l = w;
          } else {
            l = test;
          }
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

    // Add images if available
    const previewBytes = dataUrlToUint8Array(preview);
    const heatmapBytes = dataUrlToUint8Array(heatmap);
    if (previewBytes || heatmapBytes) {
      const imgPage = pdfDoc.addPage();
      const { width: iw, height: ih } = imgPage.getSize();
      const top = ih - 50;
      let x = 50;

      const drawImg = async (bytes: Uint8Array, mime: string | null, offsetX: number) => {
        try {
          const embedded = mime && mime.includes("png")
            ? await pdfDoc.embedPng(bytes)
            : await pdfDoc.embedJpg(bytes);
          const scale = Math.min(250 / embedded.width, 350 / embedded.height, 1);
          imgPage.drawImage(embedded, {
            x: offsetX,
            y: top - embedded.height * scale,
            width: embedded.width * scale,
            height: embedded.height * scale,
          });
        } catch (err) {
          console.warn("Failed embedding image:", err);
        }
      };
      if (previewBytes) {
        const mime = dataUrlMime(preview);
        await drawImg(previewBytes, mime, x);
        x += 280;
      }
      if (heatmapBytes) {
        const mime = dataUrlMime(heatmap);
        await drawImg(heatmapBytes, mime, x);
      }
    }

    const pdfBytes = await pdfDoc.save();
    return new NextResponse(Buffer.from(pdfBytes), {
      status: 200,
      headers: {
        "Content-Type": "application/pdf",
        "Content-Disposition": `attachment; filename="image-report-${Date.now()}.pdf"`,
      },
    });
  } catch (err: any) {
    console.error("generate-report fatal error:", err);
    return NextResponse.json({ error: err?.message || "Internal server error" }, { status: 500 });
  }
}
