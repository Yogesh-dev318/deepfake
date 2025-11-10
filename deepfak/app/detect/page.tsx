"use client";

import { useState, FC } from "react";
import { motion, AnimatePresence } from "framer-motion";
// Assuming these are local components, imports are fine.
// Note: In a real single-file environment, these would be defined in this file.
// For this example, we'll assume they exist as you've imported them.
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
  // --- NEW ---
  // We add image_id here to potentially use the "preferred path"
  // of the Grad-CAM endpoint, although the fallback is implemented below.
  image_id: string | null; 
}

// --- Custom Hook (MODIFIED for new /gradcam endpoint) ---
const useImageAnalysis = () => {
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResultType | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  const [isGeneratingHeatmap, setIsGeneratingHeatmap] = useState<boolean>(false);
  // heatmapImage will now store a blob: URL
  const [heatmapImage, setHeatmapImage] = useState<string | null>(null);
  const [heatmapError, setHeatmapError] = useState<string | null>(null);

  // --- NEW FUNCTION to clear heatmap state (MODIFIED) ---
  const clearHeatmap = () => {
    // Revoke the old blob URL to prevent memory leaks
    setHeatmapImage(prevUrl => {
      if (prevUrl && prevUrl.startsWith('blob:')) {
        URL.revokeObjectURL(prevUrl);
      }
      return null;
    });
    setHeatmapError(null);
  };
  // -------------------------------------------

  const analyzeImage = async (file: File | null) => {
    if (!file) return;

    setIsAnalyzing(true);
    setAnalysisResult(null);
    setApiError(null);
    clearHeatmap(); // Clear previous heatmap

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(process.env.NEXT_PUBLIC_BACKEND_URL + "/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const data = response.data;

      const isFake = data.label === "Fake";
      const confidence = Math.round(data.probability * 100);

      setAnalysisResult({
        isFake,
        confidence,
        timestamp: new Date().toLocaleString(),
        image_id: data.image_id || null, // Store the image_id if provided
      });

    } catch (err) {
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

  // --- MODIFIED FUNCTION for Grad-CAM Heatmap ---
  const generateHeatmap = async (file: File | null) => {
    // We use the file as a fallback, matching the backend logic.
    // If analysisResult.image_id exists, that would be the "preferred" path.
    // For simplicity, we stick to the "fallback" file upload path which your
    // backend supports and the frontend was already doing.
    
    if (!file) return; 

    setIsGeneratingHeatmap(true);
    // Clear previous heatmap *before* new request
    clearHeatmap();

    const formData = new FormData();
    formData.append("file", file);

    try {
      // *** KEY CHANGE: ***
      // We expect a 'blob' (raw image data), not JSON.
      const response = await axios.post(process.env.NEXT_PUBLIC_BACKEND_URL + "/gradcam", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        responseType: 'blob', // <-- Tell axios to expect binary data
      });

      // *** KEY CHANGE: ***
      // Create a temporary URL from the image blob
      const blobUrl = URL.createObjectURL(response.data);
      setHeatmapImage(blobUrl);

    } catch (err) {
      let errorMessage = "Heatmap generation failed. Please try again.";
      if (axios.isAxiosError(err)) {
        // Error responses *are* JSON (as per your Flask code),
        // but axios will read it as a blob. We must convert it back.
        if (err.response && err.response.data && err.response.data.type === 'application/json') {
          try {
            // Read the error blob as text, then parse as JSON
            const errorJsonText = await err.response.data.text();
            const errorData = JSON.parse(errorJsonText);
            errorMessage = errorData.message || errorMessage;
          } catch (parseError) {
            console.error("Failed to parse error blob:", parseError);
          }
        } else if (err.response?.data?.error) {
           errorMessage = err.response.data.error;
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

// --- Main Page Component (No changes needed) ---
export default function DetectPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
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

  const handleFileSelect = (selectedFile: File) => {
    setAnalysisResult(null);
    setError(null);
    setApiError(null);
    clearHeatmap(); // Clear heatmap on new file select

    const acceptedTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
    if (!acceptedTypes.includes(selectedFile.type)) {
      setError("Please upload a valid image file (png, jpg, webp).");
      setFile(null);
      setPreview(null);
      return;
    }

    setFile(selectedFile);
    
    // Revoke old preview URL if it exists
    if (preview) {
      URL.revokeObjectURL(preview);
    }
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
    clearHeatmap(); // Clear heatmap on reset
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
              <AnalysisResult
                result={analysisResult}
                onReset={handleReset}
                preview={preview}
                onGenerateHeatmap={() => generateHeatmap(file)} // Pass the handler
                isGeneratingHeatmap={isGeneratingHeatmap}
                heatmapImage={heatmapImage} // This is now a blob: URL
                heatmapError={heatmapError}
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

// --- Sub-components (No changes needed) ---

interface AnalysisResultProps {
  result: AnalysisResultType;
  onReset: () => void;
  preview: string | null;
  onGenerateHeatmap: () => void;
  isGeneratingHeatmap: boolean;
  heatmapImage: string | null;
  heatmapError: string | null;
}

const AnalysisResult: FC<AnalysisResultProps> = ({
  result,
  onReset,
  preview,
  onGenerateHeatmap,
  isGeneratingHeatmap,
  heatmapImage,
  heatmapError
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
              {/* This img tag now correctly displays the blob: URL */}
              <img src={heatmapImage} alt="Analysis heatmap" className="max-w-full max-h-full object-contain" />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {heatmapError && <p className="text-red-500 text-center mt-4">{heatmapError}</p>}
      
    </motion.div>
  );
};