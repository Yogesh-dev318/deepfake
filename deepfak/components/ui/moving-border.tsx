"use client";
import React, { useRef, useEffect, useState } from "react";
import { cn } from "@/lib/utils";

interface MovingBorderProps {
  children: React.ReactNode;
  borderRadius?: string;
  borderWidth?: number;
  duration?: number;
  color?: string;
  className?: string;
  as?: keyof JSX.IntrinsicElements;
  [key: string]: any;
}

export const MovingBorder: React.FC<MovingBorderProps> = ({
  children,
  borderRadius = "0.75rem",
  borderWidth = 1,
  duration = 2000,
  color = "rgb(59, 130, 246)",
  className,
  as: Component = "div",
  ...props
}) => {
  const ref = useRef<HTMLElement>(null);
  const [isHovered, setIsHovered] = useState(false);

  useEffect(() => {
    if (!ref.current) return;

    const element = ref.current;
    
    const updateGradient = () => {
      const rect = element.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;
      
      const gradient = `conic-gradient(from 0deg at ${centerX}px ${centerY}px, ${color}, transparent, ${color})`;
      element.style.background = gradient;
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (!ref.current) return;
      
      const rect = ref.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      const gradient = `conic-gradient(from 0deg at ${x}px ${y}px, ${color}, transparent, ${color})`;
      ref.current.style.background = gradient;
    };

    const handleMouseEnter = () => {
      setIsHovered(true);
      updateGradient();
    };

    const handleMouseLeave = () => {
      setIsHovered(false);
      updateGradient();
    };

    element.addEventListener("mousemove", handleMouseMove);
    element.addEventListener("mouseenter", handleMouseEnter);
    element.addEventListener("mouseleave", handleMouseLeave);

    // Initial gradient
    updateGradient();

    return () => {
      element.removeEventListener("mousemove", handleMouseMove);
      element.removeEventListener("mouseenter", handleMouseEnter);
      element.removeEventListener("mouseleave", handleMouseLeave);
    };
  }, [color, duration]);

  return (
    <Component
      ref={ref}
      className={cn(
        "relative overflow-hidden",
        "before:absolute before:inset-0 before:rounded-[inherit] before:p-[1px] before:bg-gradient-to-r before:from-transparent before:via-current before:to-transparent",
        "after:absolute after:inset-[1px] after:rounded-[inherit] after:bg-background",
        className
      )}
      style={{
        borderRadius,
        background: `conic-gradient(from 0deg, ${color}, transparent, ${color})`,
        padding: `${borderWidth}px`,
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      {...props}
    >
      <div className="relative z-10 h-full w-full rounded-[inherit] bg-background">
        {children}
      </div>
    </Component>
  );
};
