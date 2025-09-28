"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { SignedIn, SignedOut, UserButton } from "@clerk/nextjs";
import { Button } from "./ui/button";
import { Menu, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";

const Navbar = () => {
  // --- HOOKS ---
  // All hooks must be called at the top level and unconditionally.
  const [scrolled, setScrolled] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 10);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  useEffect(() => {
    document.body.style.overflow = isMenuOpen ? "hidden" : "auto";
  }, [isMenuOpen]);

  // --- RENDER LOGIC ---
  // Conditionally render null *after* all hooks have been called.
  const authRoutes = ["/sign-in", "/sign-up"];
  if (authRoutes.some(route => pathname.startsWith(route))) {
    return null;
  }
  
  const navItems = [
    { href: "/detect", label: "Image Analysis" },
    { href: "/audio", label: "Audio Analysis" },
  ];

  const navLinks = (
    <div className="flex items-center gap-8">
      {navItems.map((item) => (
        <Link
          key={item.href}
          href={item.href}
          className={cn(
            "text-lg transition-colors hover:text-white",
            pathname === item.href ? "text-white font-semibold" : "text-neutral-400"
          )}
        >
          {item.label}
        </Link>
      ))}
    </div>
  );
  
  const authButtons = (
     <div className="flex items-center gap-4">
        <SignedOut>
           <Link href="/sign-in">
              <Button variant="ghost" className="text-lg text-neutral-400 hover:text-white">Log In</Button>
           </Link>
           <Link href="/sign-up">
             <Button className="px-5 py-2 bg-gradient-to-r from-blue-500 to-purple-500 text-white font-semibold rounded-lg hover:opacity-90 transition-opacity">
                Sign Up
             </Button>
           </Link>
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
      </div>
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
          <Link href="/" className="pointer" onClick={() => setIsMenuOpen(false)}>
            <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-blue-400 to-blue-600">
            Aletheia
            </h1>
          </Link>

          <nav className="hidden md:flex items-center gap-8">
            {navLinks}
            {authButtons}
          </nav>

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
                {navItems.map((item) => (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "text-2xl transition-colors hover:text-white",
                       pathname === item.href ? "text-white font-semibold" : "text-neutral-400"
                    )}
                    onClick={() => setIsMenuOpen(false)}
                  >
                    {item.label}
                  </Link>
                ))}
                <div className="absolute bottom-24 flex flex-col items-center gap-6">
                  <SignedOut>
                    <Link href="/sign-in">
                        <Button variant="ghost" className="w-48 py-6 text-xl text-neutral-400" onClick={() => setIsMenuOpen(false)}>
                          Log In
                        </Button>
                    </Link>
                    <Link href="/sign-up">
                        <Button className="w-48 py-6 text-xl bg-gradient-to-r from-blue-500 to-purple-500 text-white font-semibold rounded-lg hover:opacity-90 transition-opacity" onClick={() => setIsMenuOpen(false)}>
                          Sign Up
                        </Button>
                    </Link>
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
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};
export default Navbar;

