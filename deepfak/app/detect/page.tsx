"use client";

import { useState, useCallback, FC } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { UploadCloud, Loader2, CheckCircle, XCircle, Wand2 } from "lucide-react";
import { useDropzone, DropzoneRootProps, DropzoneInputProps, FileRejection } from "react-dropzone";
import axios, { AxiosError } from 'axios';

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
      // This endpoint should point to your Flask API.
      // You may need to configure a proxy in your Next.js config for local development.
      const response = await axios.post("/api/predict", formData);

      const data = response.data;

      // Assumes the Flask API returns a JSON object like:
      // { "prediction": "fake" or "real", "confidence": 0.98 }
      const isFake = data.prediction === 'fake';
      const confidence = Math.round(data.confidence * 100);

      setAnalysisResult({
        isFake,
        confidence,
        timestamp: new Date().toLocaleString(),
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

  return { isAnalyzing, analysisResult, apiError, analyzeImage, setAnalysisResult, setApiError };
};


// --- Main Page Component ---

export default function DetectPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dropzoneError, setDropzoneError] = useState<string | null>(null);
  const { isAnalyzing, analysisResult, apiError, analyzeImage, setAnalysisResult, setApiError } = useImageAnalysis();

  const onDrop = useCallback((acceptedFiles: File[], fileRejections: FileRejection[]) => {
    setAnalysisResult(null);
    setDropzoneError(null);
    setApiError(null);
    if (fileRejections.length > 0) {
        setDropzoneError("Please upload an image file (png, jpg, etc.).");
        return;
    }

    const selectedFile = acceptedFiles[0];
    setFile(selectedFile);
    const previewUrl = URL.createObjectURL(selectedFile);
    setPreview(previewUrl);
  }, [setAnalysisResult, setApiError]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.png', '.jpg', '.webp'] },
    maxFiles: 1,
  });
  
  const handleReset = () => {
      setFile(null);
      if (preview) {
        URL.revokeObjectURL(preview);
      }
      setPreview(null);
      setDropzoneError(null);
      setApiError(null);
      setAnalysisResult(null);
  }

  const error = dropzoneError || apiError;

  return (
    <div className="w-full min-h-screen bg-black text-white py-24 px-4 sm:px-6 lg:px-8">
      <div className="container mx-auto max-w-3xl flex flex-col items-center">
        
        {/* Header */}
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

        {/* Main Content Area */}
        <div className="w-full">
            <AnimatePresence mode="wait">
                {analysisResult ? (
                    <AnalysisResult result={analysisResult} onReset={handleReset} preview={preview} />
                ) : (
                    <UploadArea 
                        getRootProps={getRootProps}
                        getInputProps={getInputProps}
                        isDragActive={isDragActive}
                        preview={preview}
                        error={error}
                        isAnalyzing={isAnalyzing}
                        onAnalyze={() => analyzeImage(file)}
                        onReset={handleReset}
                    />
                )}
            </AnimatePresence>
        </div>
      </div>
    </div>
  );
}


// --- Sub-components for clarity ---

interface UploadAreaProps {
    getRootProps: <T extends DropzoneRootProps>(props?: T) => T;
    getInputProps: <T extends DropzoneInputProps>(props?: T) => T;
    isDragActive: boolean;
    preview: string | null;
    error: string | null;
    isAnalyzing: boolean;
    onAnalyze: () => void;
    onReset: () => void;
}

const UploadArea: FC<UploadAreaProps> = ({ getRootProps, getInputProps, isDragActive, preview, error, isAnalyzing, onAnalyze, onReset }) => (
    <motion.div
      key="upload"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.3 }}
      className="w-full"
    >
      {!preview ? (
        <div
          {...getRootProps()}
          className={`w-full p-12 border-2 border-dashed rounded-lg cursor-pointer transition-colors
          ${isDragActive ? "border-blue-500 bg-blue-500/10" : "border-neutral-700 hover:border-neutral-500"}
          flex flex-col items-center justify-center text-center`}
        >
          <input {...getInputProps()} />
          <UploadCloud size={48} className="text-neutral-500 mb-4" />
          <p className="text-lg font-semibold">
            {isDragActive ? "Drop the image here..." : "Drag & drop an image, or click to select"}
          </p>
          <p className="text-neutral-400 text-sm mt-1">PNG, JPG, WEBP accepted</p>
        </div>
      ) : (
        <div className="w-full flex flex-col items-center gap-6">
          <div className="w-full max-w-md aspect-square rounded-lg overflow-hidden border border-neutral-800 bg-black flex items-center justify-center">
            <img src={preview} alt="Image preview" className="max-w-full max-h-full object-contain" />
          </div>
          <div className="flex items-center gap-4">
            <Button onClick={onReset} variant="outline" className="px-6 py-3">
              Change Image
            </Button>
            <Button onClick={onAnalyze} size="lg" className="px-8 py-6 text-lg font-semibold bg-gradient-to-r from-blue-500 to-purple-500 hover:opacity-90 group" disabled={isAnalyzing}>
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
      
      {error && <p className="text-red-500 text-center mt-4">{error}</p>}
    </motion.div>
);

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

