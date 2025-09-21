"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { SignedIn, SignedOut, UserButton } from "@clerk/nextjs";
import { Button } from "./ui/button";
import { Menu, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const Navbar = () => {
  const [scrolled, setScrolled] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const pathname = usePathname();

  // Effect to handle scroll detection
  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 10);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Effect to disable body scroll when mobile menu is open
  useEffect(() => {
    if (isMenuOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "auto";
    }
  }, [isMenuOpen]);

  // Hide Navbar on authentication pages
  const isAuthPage =
    pathname.startsWith("/sign-in") || pathname.startsWith("/sign-up");
  if (isAuthPage) {
    return null;
  }

  const navLinks = (
    <>
      <SignedOut>
        <div className="flex items-center gap-2">
          <Link href="/sign-in" passHref>
            <Button variant="ghost" className="text-base md:text-lg">
              Log In
            </Button>
          </Link>
          <Link href="/sign-up" passHref>
            <Button className="text-base md:text-lg bg-blue-500 hover:bg-blue-600 text-white rounded-full px-6">
              Sign Up
            </Button>
          </Link>
        </div>
      </SignedOut>
      <SignedIn>
        <UserButton
          appearance={{
            elements: {
              avatarBox: "w-10 h-10",
            },
          }}
        />
      </SignedIn>
    </>
  );

  return (
    <>
      <header
        className={`fixed top-0 left-0 w-full z-50 transition-all duration-300 ease-in-out ${
          scrolled
            ? "bg-black/80 backdrop-blur-sm border-b border-neutral-800"
            : "bg-transparent border-b border-transparent"
        }`}
      >
        <div className="container mx-auto flex h-20 items-center justify-between px-4 sm:px-6 lg:px-8">
          {/* Logo */}
          <Link
            href="/"
            className="pointer"
            onClick={() => setIsMenuOpen(false)}
          >
            <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-blue-400 to-blue-600">
            Aletheia
            </h1>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-6">{navLinks}</nav>

          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="p-2 rounded-md transition-colors hover:bg-neutral-800"
              aria-label="Toggle menu"
            >
              {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>
      </header>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
            className="fixed top-20 left-0 w-full h-[calc(100vh-5rem)] bg-black/95 backdrop-blur-lg z-40 md:hidden"
          >
            <div className="container mx-auto flex flex-col items-center justify-center h-full gap-8">
              <SignedOut>
                <div className="flex flex-col items-center gap-6">
                  <Link href="/sign-in" passHref>
                    <Button
                      variant="outline"
                      className="w-48 py-6 text-lg border-neutral-700"
                      onClick={() => setIsMenuOpen(false)}
                    >
                      Log In
                    </Button>
                  </Link>
                  <Link href="/sign-up" passHref>
                    <Button
                      className="w-48 py-6 text-lg bg-blue-500 hover:bg-blue-600 text-white"
                      onClick={() => setIsMenuOpen(false)}
                    >
                      Sign Up
                    </Button>
                  </Link>
                </div>
              </SignedOut>
              <SignedIn>
                <UserButton
                  afterSignOutUrl="/"
                  appearance={{
                    elements: {
                      avatarBox: "w-16 h-16",
                    },
                  }}
                />
              </SignedIn>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default Navbar;

