"use client";

import { useState, FC, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Loader2, CheckCircle, XCircle, Wand2, FilePlus, Download } from "lucide-react";
import axios from "axios";
import { FileUpload } from "@/components/ui/file-upload";
import { ShootingStars } from "@/components/ui/shooting-stars";
import { StarsBackground } from "@/components/ui/stars-background";

// --- Types ---
interface AnalysisResultType {
  isFake: boolean;
  confidence: number;
  timestamp: string;
  image_id: string | null;
}

// ------------------ useImageAnalysis (same as your hook, lightly adapted) ------------------
const useImageAnalysis = () => {
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResultType | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  const [isGeneratingHeatmap, setIsGeneratingHeatmap] = useState<boolean>(false);
  const [heatmapImage, setHeatmapImage] = useState<string | null>(null); // blob: URL or null
  const [heatmapError, setHeatmapError] = useState<string | null>(null);

  const clearHeatmap = useCallback(() => {
    setHeatmapImage(prevUrl => {
      if (prevUrl && prevUrl.startsWith("blob:")) {
        try { URL.revokeObjectURL(prevUrl); } catch {}
      }
      return null;
    });
    setHeatmapError(null);
  }, []);

  const analyzeImage = async (file: File | null) => {
    if (!file) return;
    setIsAnalyzing(true);
    setAnalysisResult(null);
    setApiError(null);
    clearHeatmap();

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(`${process.env.NEXT_PUBLIC_BACKEND_URL}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const data = response.data;
      const isFake = data.label === "Fake";
      const confidence = Math.round(data.probability * 100);
      setAnalysisResult({
        isFake,
        confidence,
        timestamp: new Date().toLocaleString(),
        image_id: data.image_id || null,
      });
    } catch (err: any) {
      if (axios.isAxiosError(err)) {
        const serverError =
          err.response?.data?.error || err.response?.data?.message ||
          "Analysis failed due to a server error. Please try again.";
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

  const generateHeatmap = async (file: File | null) => {
    if (!file) return;
    setIsGeneratingHeatmap(true);
    clearHeatmap();

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(`${process.env.NEXT_PUBLIC_BACKEND_URL}/gradcam`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        responseType: "blob",
      });

      const blobUrl = URL.createObjectURL(response.data);
      setHeatmapImage(blobUrl);
    } catch (err: any) {
      let errorMessage = "Heatmap generation failed. Please try again.";
      if (axios.isAxiosError(err)) {
        const res = err.response;
        // If server returned JSON error as blob, try to parse it
        if (res && res.data instanceof Blob) {
          const contentType = res.headers?.["content-type"] || res.headers?.["Content-Type"];
          if (contentType && contentType.includes("application/json")) {
            try {
              const text = await res.data.text();
              const json = JSON.parse(text);
              errorMessage = json.message || json.error || errorMessage;
            } catch (parseError) {
              console.error("Failed to parse error blob:", parseError);
            }
          }
        } else if (res?.data && typeof res.data === "object") {
          errorMessage = res.data.error || res.data.message || errorMessage;
        }
      } else if (err instanceof Error) {
        errorMessage = err.message;
      }
      setHeatmapError(errorMessage);
      console.error(err);
    } finally {
      setIsGeneratingHeatmap(false);
    }
  };

  return {
    isAnalyzing,
    analysisResult,
    apiError,
    analyzeImage,
    setAnalysisResult,
    setApiError,
    isGeneratingHeatmap,
    heatmapImage,
    heatmapError,
    generateHeatmap,
    clearHeatmap,
  };
};

// ------------------ Main Page Component ------------------
export default function DetectPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null); // blob: URL
  const [error, setError] = useState<string | null>(null);

  const {
    isAnalyzing,
    analysisResult,
    apiError,
    analyzeImage,
    setAnalysisResult,
    setApiError,
    isGeneratingHeatmap,
    heatmapImage,
    heatmapError,
    generateHeatmap,
    clearHeatmap,
  } = useImageAnalysis();

  const [isGeneratingReport, setIsGeneratingReport] = useState(false);

  // Clean up blob URLs on unmount
  useEffect(() => {
    return () => {
      if (preview && preview.startsWith("blob:")) {
        try { URL.revokeObjectURL(preview); } catch {}
      }
      if (heatmapImage && heatmapImage.startsWith("blob:")) {
        try { URL.revokeObjectURL(heatmapImage); } catch {}
      }
    };
    // We intentionally omit heatmapImage from deps to avoid re-running; cleanup on unmount only
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Convert a blob: URL (or any URL) to a Data URL (base64)
  const urlToDataURL = async (url: string | null) => {
    if (!url) return null;
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

  // Generate report by calling server-side Next.js route which uses Gemini + pdfkit
  const generateReportSecure = async () => {
    if (!analysisResult) return;
    setIsGeneratingReport(true);
    try {
      const previewDataUrl = await urlToDataURL(preview);
      const heatmapDataUrl = await urlToDataURL(heatmapImage);

      // Minimal payload; server can reject if too large. Consider multipart if images are huge.
      const payload = {
        result: analysisResult,
        preview: previewDataUrl,
        heatmap: heatmapDataUrl,
      };

      const res = await fetch("/api/generate-report/image", {
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
      a.download = `image-report-${Date.now()}.pdf`;
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
    // Reset prior state
    setAnalysisResult(null);
    setError(null);
    setApiError(null);
    clearHeatmap();

    const acceptedTypes = ["image/jpeg", "image/png", "image/webp", "image/gif"];
    if (!acceptedTypes.includes(selectedFile.type)) {
      setError("Please upload a valid image file (png, jpg, webp).");
      setFile(null);
      setPreview(null);
      return;
    }

    const MAX_FILE_BYTES = 10 * 1024 * 1024; // 10 MB
    if (selectedFile.size > MAX_FILE_BYTES) {
      setError("File is too large. Please upload an image under 10 MB.");
      setFile(null);
      setPreview(null);
      return;
    }

    setFile(selectedFile);

    // Revoke old preview URL if it exists
    if (preview) {
      try { URL.revokeObjectURL(preview); } catch {}
    }
    const previewUrl = URL.createObjectURL(selectedFile);
    setPreview(previewUrl);
  };

  const handleReset = () => {
    setFile(null);
    if (preview) {
      try { URL.revokeObjectURL(preview); } catch {}
    }
    setPreview(null);
    setError(null);
    setApiError(null);
    setAnalysisResult(null);
    clearHeatmap();
  };

  const displayedError = error || apiError;

  // Download heatmap action
  const downloadHeatmap = () => {
    if (!heatmapImage) return;
    const a = document.createElement("a");
    a.href = heatmapImage;
    a.download = `heatmap-${Date.now()}.png`;
    a.click();
  };

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
            Image Authenticity Check
          </h1>
          <p className="text-lg text-neutral-400 mt-4">
            Upload an image to scan for signs of AI generation or manipulation.
          </p>
        </motion.div>

        <div className="w-full">
          <AnimatePresence mode="wait">
            {analysisResult ? (
              <AnalysisResult
                result={analysisResult}
                onReset={handleReset}
                preview={preview}
                onGenerateHeatmap={() => generateHeatmap(file)}
                isGeneratingHeatmap={isGeneratingHeatmap}
                heatmapImage={heatmapImage}
                heatmapError={heatmapError}
                onGenerateReportSecure={generateReportSecure}
                isGeneratingReport={isGeneratingReport}
                onDownloadHeatmap={downloadHeatmap}
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
                {!preview ? (
                  <FileUpload
                    onChange={(files) => {
                      const firstFile = files?.[0];
                      if (firstFile) handleFileSelect(firstFile);
                    }}
                  />
                ) : (
                  <div className="w-full flex flex-col items-center gap-6">
                    <div className="w-full max-w-md aspect-square rounded-lg overflow-hidden border border-neutral-800 bg-black flex items-center justify-center">
                      <img src={preview} alt="Image preview" className="max-w-full max-h-full object-contain" />
                    </div>
                    <div className="flex items-center gap-4">
                      <Button onClick={handleReset} variant="outline" className="px-6 py-3">
                        Change Image
                      </Button>
                      <Button
                        onClick={() => analyzeImage(file)}
                        size="lg"
                        className="px-8 py-6 text-lg font-semibold bg-gradient-to-r from-blue-500 to-purple-500 hover:opacity-90 group"
                        disabled={isAnalyzing}
                      >
                        {isAnalyzing ? (
                          <>
                            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                            Analyzing...
                          </>
                        ) : (
                          <>
                            Analyze Image
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

// ------------------ AnalysisResult component (updated with report button) ------------------
interface AnalysisResultProps {
  result: AnalysisResultType;
  onReset: () => void;
  preview: string | null;
  onGenerateHeatmap: () => void;
  isGeneratingHeatmap: boolean;
  heatmapImage: string | null;
  heatmapError: string | null;
  onGenerateReportSecure: () => Promise<void>;
  isGeneratingReport: boolean;
  onDownloadHeatmap: () => void;
}

const AnalysisResult: FC<AnalysisResultProps> = ({
  result,
  onReset,
  preview,
  onGenerateHeatmap,
  isGeneratingHeatmap,
  heatmapImage,
  heatmapError,
  onGenerateReportSecure,
  isGeneratingReport,
  onDownloadHeatmap,
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
      {preview && (
        <div className="w-48 h-48 rounded-lg overflow-hidden border-2 border-neutral-700 mb-6">
          <img src={preview} alt="Analyzed image" className="w-full h-full object-cover"/>
        </div>
      )}
      
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
          Analyze Another Image
        </Button>
        
        <Button
          onClick={onGenerateHeatmap}
          size="lg"
          variant="default"
          className="px-8 py-6 text-lg bg-gradient-to-r from-teal-400 to-blue-500 hover:opacity-90 group"
          disabled={isGeneratingHeatmap}
        >
          {isGeneratingHeatmap ? (
            <>
              <Loader2 className="mr-2 h-5 w-5 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              Show Heatmap
              <Wand2 className="ml-2 h-5 w-5 transition-transform duration-300 group-hover:rotate-12" />
            </>
          )}
        </Button>

        <Button
          onClick={onGenerateReportSecure}
          size="lg"
          variant="default"
          className="px-8 py-6 text-lg bg-gradient-to-r from-indigo-500 to-violet-600 hover:opacity-90 group"
          disabled={isGeneratingReport || !heatmapImage}
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
        {heatmapImage && (
          <motion.div
            initial={{ opacity: 0, height: 0, y: 20 }}
            animate={{ opacity: 1, height: 'auto', y: 0 }}
            exit={{ opacity: 0, height: 0, y: 20 }}
            transition={{ duration: 0.4, ease: "easeInOut" }}
            className="w-full max-w-md mt-8"
          >
            <h3 className="text-xl font-semibold text-center mb-3">
              Analysis Heatmap
            </h3>
            <p className="text-neutral-400 text-center text-sm mb-4">
              This heatmap highlights the regions our model focused on.
            </p>
            <div className="w-full aspect-square rounded-lg overflow-hidden border border-neutral-700 bg-black flex items-center justify-center">
              <img src={heatmapImage} alt="Analysis heatmap" className="max-w-full max-h-full object-contain" />
            </div>

            <div className="flex items-center justify-center gap-3 mt-4">
              <Button onClick={onDownloadHeatmap} size="sm" variant="outline" className="px-4 py-2">
                <Download className="mr-2 h-4 w-4" /> Download Heatmap
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {heatmapError && <p className="text-red-500 text-center mt-4">{heatmapError}</p>}
      
    </motion.div>
  );
};
