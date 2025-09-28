"use client";

import { useState, FC } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Loader2, CheckCircle, XCircle, Wand2 } from "lucide-react";
import axios from 'axios';
import { FileUpload } from "@/components/ui/file-upload";
import { ShootingStars } from "@/components/ui/shooting-stars";
import { StarsBackground } from "@/components/ui/stars-background";

// --- Type Definitions ---
interface AnalysisResultType {
  isFake: boolean;
  confidence: number;
  timestamp: string;
}

// --- Custom Hook ---
const useImageAnalysis = () => {
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResultType | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  const analyzeImage = async (file: File | null) => {
    if (!file) return;

    setIsAnalyzing(true);
    setAnalysisResult(null);
    setApiError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Call Flask backend
      const response = await axios.post(process.env.NEXT_PUBLIC_BACKEND_URL + "/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const data = response.data;

      // Map Flask response to expected type
      const isFake = data.label === "Fake";
      const confidence = Math.round(data.probability * 100);

      setAnalysisResult({
        isFake,
        confidence,
        timestamp: new Date().toLocaleString(),
      });

    } catch (err) {
      if (axios.isAxiosError(err)) {
        const serverError =
          err.response?.data?.error ||
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

  return { isAnalyzing, analysisResult, apiError, analyzeImage, setAnalysisResult, setApiError };
};

// --- Main Page Component ---
export default function DetectPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { isAnalyzing, analysisResult, apiError, analyzeImage, setAnalysisResult, setApiError } = useImageAnalysis();

  const handleFileSelect = (selectedFile: File) => {
    setAnalysisResult(null);
    setError(null);
    setApiError(null);

    const acceptedTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
    if (!acceptedTypes.includes(selectedFile.type)) {
      setError("Please upload a valid image file (png, jpg, webp).");
      setFile(null);
      setPreview(null);
      return;
    }

    setFile(selectedFile);
    const previewUrl = URL.createObjectURL(selectedFile);
    setPreview(previewUrl);
  };

  const handleReset = () => {
    setFile(null);
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    setPreview(null);
    setError(null);
    setApiError(null);
    setAnalysisResult(null);
  }

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
            Image Authenticity Check
          </h1>
          <p className="text-lg text-neutral-400 mt-4">
            Upload an image to scan for signs of AI generation or manipulation.
          </p>
        </motion.div>

        <div className="w-full">
          <AnimatePresence mode="wait">
            {analysisResult ? (
              <AnalysisResult result={analysisResult} onReset={handleReset} preview={preview} />
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
                      <Button onClick={() => analyzeImage(file)} size="lg" className="px-8 py-6 text-lg font-semibold bg-gradient-to-r from-blue-500 to-purple-500 hover:opacity-90 group" disabled={isAnalyzing}>
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

// --- Sub-components for clarity ---
interface AnalysisResultProps {
  result: AnalysisResultType;
  onReset: () => void;
  preview: string | null;
}

const AnalysisResult: FC<AnalysisResultProps> = ({ result, onReset, preview }) => {
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

      <Button onClick={onReset} size="lg" variant="outline" className="mt-8 px-8 py-6 text-lg">
        Analyze Another Image
      </Button>
    </motion.div>
  );
};
