"use client";
import { ShootingStars } from "@/components/ui/shooting-stars";
import { StarsBackground } from "@/components/ui/stars-background";
import { SignIn } from "@clerk/nextjs";
import { useTheme } from "next-themes";

export default function Page() {
  const { theme } = useTheme();
  const Theme =
    theme === "dark"
      ? {
          variables: {
            colorPrimary: "#ffffff",
            colorBackground: "#1e1e1e",
            colorText: "#f5f5f5",
            colorInputBackground: "#333333",
            colorInputText: "#ffffff",
            colorButton: "#4caf50", 
          },
        }
      : {
          variables: {
            colorPrimary: "#ffffff",
            colorBackground: "#ffffff",
            colorText: "#333333",
            colorInputBackground: "#f5f5f5",
            colorInputText: "#1e1e1e",
            colorButton: "#007bff",
          },
        };

  return (
    <div className="relative w-full h-screen bg-black">
      {/* Animated Backgrounds */}
      <ShootingStars />
      <StarsBackground />

      {/* Centered Sign-In Component */}
      <div className="flex justify-center items-center w-full h-full">
        <SignIn  />
      </div>
    </div>
    
  );
}