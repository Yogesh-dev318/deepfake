"use client";
import Link from "next/link";
import { motion } from "framer-motion";
import { BentoGrid, BentoGridItem } from "@/components/ui/bento-grid";
import {
  ShieldCheck,
  Zap,
  TestTubeDiagonal,
  GraduationCap,
  Eye,
  BadgeCheck,
  UploadCloud,
  Cpu,
  FileCheck,
  ArrowRight,
} from "lucide-react";
import { Vortex } from "@/components/ui/vortex";
import { Button } from "@/components/ui/button";
import Footer from "@/components/footer"; // Import the footer

export default function Home() {
  const features = [
    {
      title: "Advanced AI Detection",
      description:
        "Leveraging state-of-the-art models to identify sophisticated digital manipulations and artifacts with high precision.",
      icon: <GraduationCap className="h-4 w-4 text-neutral-500" />,
      className: "md:col-span-2",
    },
    {
      title: "Rapid Analysis",
      description:
        "Get results in seconds. Our optimized pipeline ensures swift processing of your images without compromising accuracy.",
      icon: <Zap className="h-4 w-4 text-neutral-500" />,
      className: "md:col-span-1",
    },
    {
      title: "In-Depth Reporting",
      description:
        "Receive a detailed breakdown of the analysis, including confidence scores and potential manipulation heatmaps.",
      icon: <TestTubeDiagonal className="h-4 w-4 text-neutral-500" />,
      className: "md:col-span-1",
    },
    {
      title: "Build Trust & Security",
      description:
        "Verify identities, protect your platform from malicious content, and ensure the authenticity of user-submitted media.",
      icon: <ShieldCheck className="h-4 w-4 text-neutral-500" />,
      className: "md:col-span-1",
    },
    {
      title: "Visual Anomaly Detection",
      description:
        "Our algorithms are trained to spot subtle visual inconsistencies that are often invisible to the naked eye.",
      icon: <Eye className="h-4 w-4 text-neutral-500" />,
      className: "md:col-span-1",
    },
  ];

  const steps = [
    {
      title: "1. Upload Image",
      description: "Securely upload the image or video you want to analyze.",
      icon: <UploadCloud size={40} />,
    },
    {
      title: "2. AI Processing",
      description:
        "Our model scans for manipulation traces and artifacts in seconds.",
      icon: <Cpu size={40} />,
    },
    {
      title: "3. Get Report",
      description:
        "Receive a clear, actionable report with a confidence score.",
      icon: <FileCheck size={40} />,
    },
  ];

  return (
    <main className="w-full bg-black/[0.96] text-white overflow-hidden">
      {/* Hero Section with Vortex Background */}
      <div className="w-full mx-auto rounded-md h-screen overflow-hidden">
        <Vortex
          backgroundColor="black"
          className="flex items-center flex-col justify-center px-2 md:px-10 py-4 w-full h-full"
        >
          <motion.div
            initial={{ opacity: 0.0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{
              delay: 0.3,
              duration: 0.8,
              ease: "easeInOut",
            }}
            className="flex flex-col items-center justify-center text-center"
          >
            <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-white to-neutral-400 py-4">
              Unmasking Digital Deception
            </h1>
            <p className="text-base md:text-xl text-neutral-300 max-w-3xl mx-auto mt-4">
              In an era of artificial reality, our platform provides the ground
              truth. Protect your integrity, verify authenticity, and build
              trust with the world's most advanced deepfake detection
              technology.
            </p>
            <div className="flex flex-col sm:flex-row items-center gap-6 mt-10">
              <Link href={"/detect"}>
                <Button
                  size="lg"
                  className="px-8 py-7 text-lg font-semibold bg-gradient-to-r from-blue-500 to-purple-500 hover:opacity-90 transition-opacity duration-300 rounded-full group"
                >
                  Start Analyzing Now
                  <ArrowRight className="ml-2 h-5 w-5 transition-transform duration-300 group-hover:translate-x-1" />
                </Button>
              </Link>
            </div>
          </motion.div>
        </Vortex>
      </div>

      {/* How It Works Section */}
      <div id="how-it-works" className="py-24 w-full bg-black relative z-10 px-4 scroll-mt-20">
        <div className="container mx-auto">
          <h2 className="text-4xl md:text-5xl font-bold text-center bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400 mb-16">
            Simple Steps to Certainty
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-12 max-w-5xl mx-auto">
            {steps.map((step, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 50 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: i * 0.2 }}
                className="flex flex-col items-center text-center p-6 rounded-lg border border-neutral-800 bg-neutral-900/50"
              >
                <div className="p-4 bg-blue-500/10 rounded-full mb-4 text-blue-400">
                  {step.icon}
                </div>
                <h3 className="text-2xl font-bold mb-2">{step.title}</h3>
                <p className="text-neutral-400">{step.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div id="features" className="py-24 w-full relative z-10 px-4 bg-black scroll-mt-20">
        <div className="container mx-auto">
          <h2 className="text-4xl md:text-5xl font-bold text-center bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400 mb-16">
            A Powerful, Intuitive & Trustworthy Tool
          </h2>
          <BentoGrid className="max-w-6xl mx-auto">
            {features.map((feature, i) => (
              <BentoGridItem
                key={i}
                title={feature.title}
                description={feature.description}
                icon={feature.icon}
                className={feature.className}
              />
            ))}
          </BentoGrid>
        </div>
      </div>
      <Footer />
    </main>
  );
}

