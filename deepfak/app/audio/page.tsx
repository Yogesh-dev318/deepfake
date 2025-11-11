"use client";

import { useState, FC, useEffect } from "react";
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
      reader.onerror = (e) => reject(e);
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
    } catch (err) {
      if (axios.isAxiosError(err)) {
        const serverError = err.response?.data?.message || "Analysis failed due to a server error. Please try again.";
        setApiError(serverError);
      } else if (err instanceof Error) {
        setApiError(err.message);
      } else {
        setApiError("An unknown error occurred during analysis.");
      }
      console.error(err);
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
    } catch (err: any) {
      if (axios.isAxiosError(err)) {
        // Try to parse JSON error message if present
        let serverMsg = "Spectrogram generation failed. Please try again.";
        try {
          const contentType = err.response?.headers?.["content-type"] || "";
          if (err.response?.data && contentType.includes("application/json")) {
            const parsed = typeof err.response.data === "string" ? JSON.parse(err.response.data) : err.response.data;
            serverMsg = parsed?.message || parsed?.error || serverMsg;
          } else if (err.response?.data && typeof err.response.data === "string") {
            serverMsg = err.response.data;
          }
        } catch (e) {
          // ignore parse errors
        }
        setSpectrogramError(serverMsg);
      } else if (err instanceof Error) {
        setSpectrogramError(err.message);
      } else {
        setSpectrogramError("An unknown error occurred during spectrogram generation.");
      }
      console.error(err);
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
        reader.onerror = reject;
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
    } catch (err: any) {
      console.error("generateReportSecure error", err);
      setApiError(err?.message || "Failed to generate report.");
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
  const { isFake, confidence } = result;
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

      <p className="text-neutral-300 text-lg mt-2">
        Our model is <span className="font-bold text-white">{confidence}%</span> confident in this result.
      </p>

      <div className="w-full bg-neutral-800 rounded-full h-2.5 my-6">
        <motion.div
          className={`h-2.5 rounded-full ${isFake ? 'bg-red-500' : 'bg-green-500'}`}
          initial={{ width: 0 }}
          animate={{ width: `${confidence}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        ></motion.div>
      </div>

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
            animate={{ opacity: 1, height: 'auto', y: 0 }}
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
              <img src={spectrogramImage} alt="Audio spectrogram" className="w-full h-auto object-contain" />
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
