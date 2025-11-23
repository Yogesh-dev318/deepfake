"use client";

import { useState, FC, useEffect } from "react";
import Image from "next/image";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Loader2, CheckCircle, XCircle, Wand2, FileAudio, Music, Download, FilePlus } from "lucide-react";
import axios from "axios";
import { FileUpload } from "@/components/ui/file-upload";
import { ShootingStars } from "@/components/ui/shooting-stars";
import { StarsBackground } from "@/components/ui/stars-background";

/* =========================
   Types
   ========================= */
interface AnalysisResultType {
  isFake: boolean;
  confidence: number;
  timestamp: string;
  audio_id?: string; // uuid returned by /predict_audio
}

/* =========================
   Helpers
   ========================= */

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

/* =========================
   Hook: useAudioAnalysis
   ========================= */
const useAudioAnalysis = () => {
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResultType | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  const [isGeneratingSpectrogram, setIsGeneratingSpectrogram] = useState<boolean>(false);
  const [spectrogramImage, setSpectrogramImage] = useState<string | null>(null); // data URL
  const [spectrogramError, setSpectrogramError] = useState<string | null>(null);

  // Helper to convert ArrayBuffer (PNG bytes) to data URL
  const arrayBufferToDataUrl = async (buffer: ArrayBuffer) => {
    const blob = new Blob([buffer], { type: "image/png" });
    return await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = () => reject(new Error("Failed to read blob as data URL"));
      reader.readAsDataURL(blob);
    });
  };

  const analyzeAudio = async (file: File | null) => {
    if (!file) return;

    setIsAnalyzing(true);
    setAnalysisResult(null);
    setApiError(null);
    setSpectrogramImage(null);
    setSpectrogramError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/predict_audio`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      const data = response.data;

      // Expect predict_audio to return: { prediction: "fake"|"real", confidence: 0.xx, audio_id: "<uuid>" }
      const isFake = data.prediction === "fake";
      const confidence = Math.round((typeof data.confidence === "number" ? data.confidence : 0) * 100);
      const audio_id = typeof data.audio_id === "string" ? data.audio_id : undefined;

      setAnalysisResult({
        isFake,
        confidence,
        timestamp: new Date().toLocaleString(),
        audio_id,
      });
    } catch (err: unknown) {
      if (axios.isAxiosError(err)) {
        // safely extract server message from possible shapes without using `any`
        const axiosErr = err as { response?: { data?: unknown } };
        const serverData = axiosErr.response?.data;
        let serverError = "Analysis failed due to a server error. Please try again.";

        if (serverData) {
          if (typeof serverData === "string") {
            serverError = serverData;
          } else if (typeof serverData === "object" && serverData !== null) {
            const sd = serverData as Record<string, unknown>;
            if (typeof sd.message === "string") serverError = sd.message;
            else if (typeof sd.error === "string") serverError = sd.error;
          }
        }

        setApiError(serverError);
      } else if (err instanceof Error) {
        setApiError(err.message);
      } else {
        setApiError("An unknown error occurred during analysis.");
      }
      console.error("analyzeAudio error:", getErrorMessage(err));
    } finally {
      setIsAnalyzing(false);
    }
  };

  // NOTE: This API expects JSON { "audio_id": "<uuid>" } and returns raw PNG bytes.
  const generateSpectrogram = async (file: File | null) => {
    // We'll rely on audio_id stored in analysisResult
    if (!file) {
      setSpectrogramError("No file available. Please upload and analyze an audio file first.");
      return;
    }

    const audio_id = analysisResult?.audio_id;
    if (!audio_id) {
      setSpectrogramError("Missing audio_id. Please run analysis first so the server can cache the audio and return an audio_id.");
      return;
    }

    setIsGeneratingSpectrogram(true);
    setSpectrogramImage(null);
    setSpectrogramError(null);

    try {
      const response = await axios.post(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/audio_saliency`,
        { audio_id },
        {
          headers: { "Content-Type": "application/json" },
          responseType: "arraybuffer", // important: receive binary PNG
        }
      );

      const dataUrl = await arrayBufferToDataUrl(response.data);
      setSpectrogramImage(dataUrl);
    } catch (err: unknown) {
      if (axios.isAxiosError(err)) {
        // Try to parse JSON error message if present, without using `any`
        let serverMsg = "Spectrogram generation failed. Please try again.";
        try {
          const axiosErr = err as { response?: { headers?: Record<string, string>; data?: unknown } };
          const contentType = axiosErr.response?.headers?.["content-type"] || "";
          const respData = axiosErr.response?.data;

          if (respData && typeof contentType === "string" && contentType.includes("application/json")) {
            let parsed: unknown = undefined;
            if (typeof respData === "string") {
              try {
                parsed = JSON.parse(respData);
              } catch {
                parsed = undefined;
              }
            } else {
              parsed = respData;
            }

            if (parsed && typeof parsed === "object") {
              const p = parsed as Record<string, unknown>;
              if (typeof p.message === "string") serverMsg = p.message;
              else if (typeof p.error === "string") serverMsg = p.error;
            }
          } else if (respData && typeof respData === "string") {
            serverMsg = respData;
          }
        } catch {
          // ignore parse errors
        }
        setSpectrogramError(serverMsg);
      } else if (err instanceof Error) {
        setSpectrogramError(err.message);
      } else {
        setSpectrogramError("An unknown error occurred during spectrogram generation.");
      }
      console.error("generateSpectrogram error:", getErrorMessage(err));
    } finally {
      setIsGeneratingSpectrogram(false);
    }
  };

  const clearSpectrogram = () => {
    setSpectrogramImage(null);
    setSpectrogramError(null);
  };

  return {
    isAnalyzing,
    analysisResult,
    apiError,
    analyzeAudio,
    setAnalysisResult,
    setApiError,
    isGeneratingSpectrogram,
    spectrogramImage,
    spectrogramError,
    generateSpectrogram,
    clearSpectrogram,
  };
};

/* =========================
   Main Page Component
   ========================= */
export default function AudioDetectPage() {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);

  const {
    isAnalyzing,
    analysisResult,
    apiError,
    analyzeAudio,
    setAnalysisResult,
    setApiError,
    isGeneratingSpectrogram,
    spectrogramImage,
    spectrogramError,
    generateSpectrogram,
    clearSpectrogram,
  } = useAudioAnalysis();

  useEffect(() => {
    // cleanup not strictly necessary since spectrogramImage is a data URL,
    // but keep hook available for future blob URL usage.
    return () => {};
  }, []);

  // Convert a blob: URL (or any URL) to a Data URL (base64). If already data URL, return as-is.
  const urlToDataURL = async (url: string | null) => {
    if (!url) return null;
    if (url.startsWith("data:")) return url;
    try {
      const resp = await fetch(url);
      const blob = await resp.blob();
      return await new Promise<string | null>((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result as string | null);
        reader.onerror = () => reject(new Error("Failed converting url to dataURL"));
        reader.readAsDataURL(blob);
      });
    } catch (e) {
      console.error("Failed converting url to dataURL", e);
      return null;
    }
  };

  // Generate report by calling server-side Next.js route which uses Gemini + pdf-lib
  const generateReportSecure = async () => {
    if (!analysisResult) return;
    setIsGeneratingReport(true);
    try {
      // For audio, we send preview: null, heatmap: spectrogramImage (data URL)
      const spectrogramDataUrl = await urlToDataURL(spectrogramImage);

      const payload = {
        result: {
          // Keep only fields server expects for "result" (server ignores confidence in PDF)
          isFake: analysisResult.isFake,
          timestamp: analysisResult.timestamp,
          image_id: analysisResult.audio_id || null, // map audio_id to image_id field so server can include it
        },
        preview: null,
        heatmap: spectrogramDataUrl,
      };

      const res = await fetch("/api/generate-report/audio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(`Report generation failed: ${res.status} ${txt}`);
      }

      const pdfBlob = await res.blob();
      const url = URL.createObjectURL(pdfBlob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `audio-report-${Date.now()}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err: unknown) {
      console.error("generateReportSecure error", getErrorMessage(err));
      setApiError(getErrorMessage(err) || "Failed to generate report.");
    } finally {
      setIsGeneratingReport(false);
    }
  };

  const handleFileSelect = (selectedFile: File) => {
    setAnalysisResult(null);
    setError(null);
    setApiError(null);
    clearSpectrogram();

    const acceptedTypes = ["audio/mpeg", "audio/wav", "audio/ogg", "audio/flac"];
    if (!acceptedTypes.includes(selectedFile.type)) {
      setError("Please upload a valid audio file (mp3, wav, ogg, flac).");
      setFile(null);
      return;
    }

    const MAX_FILE_BYTES = 15 * 1024 * 1024; // 15 MB for audio
    if (selectedFile.size > MAX_FILE_BYTES) {
      setError("File is too large. Please upload an audio file under 15 MB.");
      setFile(null);
      return;
    }

    setFile(selectedFile);
  };

  const handleReset = () => {
    setFile(null);
    setError(null);
    setApiError(null);
    setAnalysisResult(null);
    clearSpectrogram();
  };

  const downloadSpectrogram = () => {
    if (!spectrogramImage) return;
    const a = document.createElement("a");
    a.href = spectrogramImage;
    a.download = `spectrogram-${Date.now()}.png`;
    a.click();
  };

  const displayedError = error || apiError;

  return (
    <div className="relative w-full min-h-screen bg-black text-white py-24 px-4 sm:px-6 lg:px-8">
      <ShootingStars />
      <StarsBackground />
      <div className="relative z-10 container mx-auto max-w-3xl flex flex-col items-center">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400">
            Audio Authenticity Check
          </h1>
          <p className="text-lg text-neutral-400 mt-4">
            Upload an audio file to scan for signs of AI voice cloning or manipulation.
          </p>
        </motion.div>

        <div className="w-full">
          <AnimatePresence mode="wait">
            {analysisResult ? (
              <AnalysisResult
                result={analysisResult}
                onReset={handleReset}
                fileName={file?.name}
                onGenerateSpectrogram={() => generateSpectrogram(file)}
                isGeneratingSpectrogram={isGeneratingSpectrogram}
                spectrogramImage={spectrogramImage}
                spectrogramError={spectrogramError}
                onGenerateReportSecure={generateReportSecure}
                isGeneratingReport={isGeneratingReport}
                onDownloadSpectrogram={downloadSpectrogram}
              />
            ) : (
              <motion.div
                key="upload"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.3 }}
                className="w-full"
              >
                {!file ? (
                  <FileUpload
                    onChange={(files) => {
                      const firstFile = files?.[0];
                      if (firstFile) handleFileSelect(firstFile);
                    }}
                  />
                ) : (
                  <div className="w-full flex flex-col items-center gap-6">
                    <div className="w-full max-w-md p-8 rounded-lg border border-neutral-800 bg-black flex flex-col items-center justify-center text-center">
                      <FileAudio size={48} className="text-neutral-500" />
                      <p className="mt-4 font-semibold text-lg truncate" title={file.name}>{file.name}</p>
                      <p className="text-neutral-400 text-sm">({(file.size / 1024 / 1024).toFixed(2)} MB)</p>
                    </div>

                    <div className="flex items-center gap-4">
                      <Button onClick={handleReset} variant="outline" className="px-6 py-3">
                        Change File
                      </Button>

                      <Button onClick={() => analyzeAudio(file)} size="lg" className="px-8 py-6 text-lg font-semibold bg-gradient-to-r from-blue-500 to-purple-500 hover:opacity-90 group" disabled={isAnalyzing}>
                        {isAnalyzing ? (
                          <>
                            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                            Analyzing...
                          </>
                        ) : (
                          <>
                            Analyze Audio
                            <Wand2 className="ml-2 h-5 w-5 transition-transform duration-300 group-hover:rotate-12" />
                          </>
                        )}
                      </Button>
                    </div>
                  </div>
                )}

                {displayedError && <p className="text-red-500 text-center mt-4">{displayedError}</p>}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}

/* =========================
   Sub-components
   ========================= */

interface AnalysisResultProps {
  result: AnalysisResultType;
  onReset: () => void;
  fileName?: string;
  onGenerateSpectrogram: () => void;
  isGeneratingSpectrogram: boolean;
  spectrogramImage: string | null;
  spectrogramError: string | null;
  onGenerateReportSecure: () => Promise<void>;
  isGeneratingReport: boolean;
  onDownloadSpectrogram: () => void;
}

const AnalysisResult: FC<AnalysisResultProps> = ({
  result,
  onReset,
  fileName,
  onGenerateSpectrogram,
  isGeneratingSpectrogram,
  spectrogramImage,
  spectrogramError,
  onGenerateReportSecure,
  isGeneratingReport,
  onDownloadSpectrogram,
}) => {
  const { isFake } = result;
  const resultColor = isFake ? "text-red-400" : "text-green-400";
  const resultBorder = isFake ? "border-red-500/30" : "border-green-500/30";
  const resultIcon = isFake ? <XCircle size={48} className={resultColor} /> : <CheckCircle size={48} className={resultColor} />;

  return (
    <motion.div
      key="result"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.5, ease: "easeInOut" }}
      className={`w-full p-8 border ${resultBorder} bg-neutral-900/50 rounded-lg flex flex-col items-center text-center`}
    >
      <FileAudio size={40} className="text-neutral-600 mb-4" />
      {fileName && <p className="text-neutral-400 mb-6 truncate max-w-xs">{fileName}</p>}

      {resultIcon}

      <h2 className={`text-3xl font-bold mt-4 ${resultColor}`}>
        {isFake ? "Likely AI-Generated" : "Appears Authentic"}
      </h2>

      <p className="text-neutral-500 text-sm max-w-md">
        Disclaimer: This analysis is based on our AI model and is not a definitive guarantee. Please use this information responsibly.
      </p>

      <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mt-8">
        <Button onClick={onReset} size="lg" variant="outline" className="px-8 py-6 text-lg">
          Analyze Another File
        </Button>

        <Button
          onClick={onGenerateSpectrogram}
          size="lg"
          variant="default"
          className="px-8 py-6 text-lg bg-gradient-to-r from-teal-400 to-blue-500 hover:opacity-90 group"
          disabled={isGeneratingSpectrogram}
        >
          {isGeneratingSpectrogram ? (
            <>
              <Loader2 className="mr-2 h-5 w-5 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              Show Spectrogram
              <Music className="ml-2 h-5 w-5 transition-transform duration-300 group-hover:rotate-12" />
            </>
          )}
        </Button>

        <Button
          onClick={onGenerateReportSecure}
          size="lg"
          variant="default"
          className="px-8 py-6 text-lg bg-gradient-to-r from-indigo-500 to-violet-600 hover:opacity-90 group"
          disabled={isGeneratingReport || !spectrogramImage}
        >
          {isGeneratingReport ? (
            <>
              <Loader2 className="mr-2 h-5 w-5 animate-spin" />
              Generating Report...
            </>
          ) : (
            <>
              Generate Report
              <FilePlus className="ml-2 h-5 w-5" />
            </>
          )}
        </Button>
      </div>

      <AnimatePresence>
        {spectrogramImage && (
          <motion.div
            initial={{ opacity: 0, height: 0, y: 20 }}
            animate={{ opacity: 1, height: "auto", y: 0 }}
            exit={{ opacity: 0, height: 0, y: 20 }}
            transition={{ duration: 0.4, ease: "easeInOut" }}
            className="w-full max-w-lg mt-8"
          >
            <h3 className="text-xl font-semibold text-center mb-3">
              Audio Spectrogram
            </h3>
            <p className="text-neutral-400 text-center text-sm mb-4">
              This spectrogram visualizes the frequency content of the audio file over time.
            </p>
            <div className="w-full rounded-lg overflow-hidden border border-neutral-700 bg-black flex items-center justify-center">
              {/* Next.js Image requires width and height; using reasonable defaults for display */}
              <div className="relative w-full" style={{ height: 400 }}>
                <Image
                  src={spectrogramImage}
                  alt="Audio spectrogram"
                  fill
                  style={{ objectFit: "contain" }}
                  sizes="(max-width: 1024px) 100vw, 1024px"
                  priority={false}
                />
              </div>
            </div>

            <div className="flex items-center justify-center gap-3 mt-4">
              <Button onClick={onDownloadSpectrogram} size="sm" variant="outline" className="px-4 py-2">
                <Download className="mr-2 h-4 w-4" /> Download Spectrogram
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {spectrogramError && <p className="text-red-500 text-center mt-4">{spectrogramError}</p>}
    </motion.div>
  );
};
