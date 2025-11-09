"use client";

import { useState, FC } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Loader2, CheckCircle, XCircle, Wand2, FileAudio, Music } from "lucide-react";
import axios, { AxiosError } from 'axios';
import { FileUpload } from "@/components/ui/file-upload";
import { ShootingStars } from "@/components/ui/shooting-stars";
import { StarsBackground } from "@/components/ui/stars-background";

// --- Type Definitions ---

interface AnalysisResultType {
  isFake: boolean;
  confidence: number;
  timestamp: string;
}

// --- Custom Hook for Audio Analysis (MODIFIED) ---

const useAudioAnalysis = () => {
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResultType | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  // --- NEW STATE for Spectrogram ---
  const [isGeneratingSpectrogram, setIsGeneratingSpectrogram] = useState<boolean>(false);
  const [spectrogramImage, setSpectrogramImage] = useState<string | null>(null);
  const [spectrogramError, setSpectrogramError] = useState<string | null>(null);
  // ---------------------------------

  const analyzeAudio = async (file: File | null) => {
    if (!file) return;

    setIsAnalyzing(true);
    setAnalysisResult(null);
    setApiError(null);
    setSpectrogramImage(null); // Clear previous spectrogram
    setSpectrogramError(null); // Clear previous spectrogram error

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Assuming a different endpoint for audio
      const response = await axios.post(process.env.NEXT_PUBLIC_BACKEND_URL + "/predict_audio", formData); 
      const data = response.data;
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

  // --- NEW FUNCTION for Spectrogram ---
  const generateSpectrogram = async (file: File | null) => {
    if (!file) return;

    setIsGeneratingSpectrogram(true);
    setSpectrogramImage(null);
    setSpectrogramError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      // *** ASSUMPTION: New endpoint is named '/spectrogram' ***
      // This endpoint is expected to return a JSON with a base64 image string
      // e.g., { "spectrogram_image": "data:image/jpeg;base64,..." }
      const response = await axios.post(process.env.NEXT_PUBLIC_BACKEND_URL + "/spectrogram", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (response.data && response.data.spectrogram_image) {
        setSpectrogramImage(response.data.spectrogram_image);
      } else {
        throw new Error("Invalid response format from spectrogram server.");
      }

    } catch (err) {
      if (axios.isAxiosError(err)) {
        const serverError =
          err.response?.data?.error ||
          "Spectrogram generation failed. Please try again.";
        setSpectrogramError(serverError);
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
  
  // --- NEW FUNCTION to clear spectrogram state ---
  const clearSpectrogram = () => {
    setSpectrogramImage(null);
    setSpectrogramError(null);
  };
  // -------------------------------------------


  return {
    isAnalyzing,
    analysisResult,
    apiError,
    analyzeAudio,
    setAnalysisResult,
    setApiError,
    // --- NEW exports ---
    isGeneratingSpectrogram,
    spectrogramImage,
    spectrogramError,
    generateSpectrogram,
    clearSpectrogram,
    // -------------------
  };
};


// --- Main Page Component (MODIFIED) ---

export default function AudioDetectPage() {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  // --- MODIFIED: Destructure new items from hook ---
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
  // --------------------------------------------------

  const handleFileSelect = (selectedFile: File) => {
    setAnalysisResult(null);
    setError(null);
    setApiError(null);
    clearSpectrogram(); // Clear spectrogram on new file select

    const acceptedTypes = ['audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/flac'];
    if (!acceptedTypes.includes(selectedFile.type)) {
      setError("Please upload a valid audio file (mp3, wav, ogg, flac).");
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
      clearSpectrogram(); // --- MODIFIED: Clear spectrogram on reset ---
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
            Audio Authenticity Check
          </h1>
          <p className="text-lg text-neutral-400 mt-4">
            Upload an audio file to scan for signs of AI voice cloning or manipulation.
          </p>
        </motion.div>

        <div className="w-full">
            <AnimatePresence mode="wait">
                {analysisResult ? (
                    // --- MODIFIED: Pass new props to AnalysisResult ---
                    <AnalysisResult
                      result={analysisResult}
                      onReset={handleReset}
                      fileName={file?.name}
                      onGenerateSpectrogram={() => generateSpectrogram(file)} // Pass the handler
                      isGeneratingSpectrogram={isGeneratingSpectrogram}
                      spectrogramImage={spectrogramImage}
                      spectrogramError={spectrogramError}
                    />
                    // ----------------------------------------------------
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

// --- Sub-components (MODIFIED) ---

// --- MODIFIED: Update props interface ---
interface AnalysisResultProps {
    result: AnalysisResultType;
    onReset: () => void;
    fileName?: string;
    onGenerateSpectrogram: () => void;
    isGeneratingSpectrogram: boolean;
    spectrogramImage: string | null;
    spectrogramError: string | null;
}
// ------------------------------------

// --- MODIFIED: Update component signature and add new UI elements ---
const AnalysisResult: FC<AnalysisResultProps> = ({
  result,
  onReset,
  fileName,
  onGenerateSpectrogram,
  isGeneratingSpectrogram,
  spectrogramImage,
  spectrogramError
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

            {/* --- NEW: Button Container --- */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mt-8">
              <Button onClick={onReset} size="lg" variant="outline" className="px-8 py-6 text-lg">
                  Analyze Another File
              </Button>

              {/* --- NEW: Spectrogram Button --- */}
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
            </div>

            {/* --- NEW: Spectrogram Display Area --- */}
            <AnimatePresence>
              {spectrogramImage && (
                <motion.div
                  initial={{ opacity: 0, height: 0, y: 20 }}
                  animate={{ opacity: 1, height: 'auto', y: 0 }}
                  exit={{ opacity: 0, height: 0, y: 20 }}
                  transition={{ duration: 0.4, ease: "easeInOut" }}
                  className="w-full max-w-lg mt-8" // Made this wider for spectrograms
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
                </motion.div>
              )}
            </AnimatePresence>
            
            {/* --- NEW: Spectrogram Error Display --- */}
            {spectrogramError && <p className="text-red-500 text-center mt-4">{spectrogramError}</p>}
            
        </motion.div>
    );
};