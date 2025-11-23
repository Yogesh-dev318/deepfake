"use client";
import React, { useRef, useEffect } from "react";
import { cn } from "@/lib/utils";

interface MovingBorderProps {
  children: React.ReactNode;
  borderRadius?: string;
  borderWidth?: number;
  duration?: number;
  color?: string;
  className?: string;
  as?: React.ElementType;
  // allow additional HTML attributes but keep them typed safely
  [key: string]: unknown;
}

export const MovingBorder: React.FC<MovingBorderProps> = ({
  children,
  borderRadius = "0.75rem",
  borderWidth = 1,
  duration = 2000,
  color = "rgb(59, 130, 246)",
  className,
  as: InnerTag = "div",
  ...props
}) => {
  // use a stable div ref to attach mouse listeners and style changes
  const outerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const element = outerRef.current;
    if (!element) return;

    const updateGradient = (clientX?: number, clientY?: number) => {
      const rect = element.getBoundingClientRect();
      const centerX = typeof clientX === "number" ? clientX : rect.width / 2;
      const centerY = typeof clientY === "number" ? clientY : rect.height / 2;

      // Use coordinates relative to the element for the gradient center
      const gradient = `conic-gradient(from 0deg at ${centerX}px ${centerY}px, ${color}, transparent, ${color})`;
      element.style.background = gradient;
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (!element) return;
      const rect = element.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      updateGradient(x, y);
    };

    const handleMouseEnter = () => {
      updateGradient();
    };

    const handleMouseLeave = () => {
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

  // We want to allow arbitrary HTML attributes to be passed in `props`,
  // but when spreading onto the outer div we must satisfy JSX typing.
  const safeProps = props as React.HTMLAttributes<HTMLDivElement>;

  const Inner = InnerTag as React.ElementType;

  return (
    <div
      ref={outerRef}
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
      {...safeProps}
    >
      <Inner className="relative z-10 h-full w-full rounded-[inherit] bg-background">
        {children}
      </Inner>
    </div>
  );
};
