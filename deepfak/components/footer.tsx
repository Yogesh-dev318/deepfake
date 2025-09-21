import Link from "next/link";
import { Github } from "lucide-react";

const Footer = () => {
  return (
    <footer className="w-full bg-black border-t border-neutral-800 py-12 px-4 sm:px-6 lg:px-8 relative z-10">
      <div className="container mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Column 1: Logo and Mission */}
          <div className="flex flex-col items-start">
            <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-blue-400 to-blue-600 mb-2">
            Aletheia
            </h2>
            <p className="text-neutral-400 text-sm max-w-xs">
              Unmasking digital deception with advanced AI to build a more
              trustworthy online world.
            </p>
          </div>

          {/* Column 2: Quick Links */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">
              Quick Links
            </h3>
            <ul className="space-y-2">
              <li>
                <Link
                  href="/#features"
                  className="text-neutral-400 hover:text-white transition-colors"
                >
                  Features
                </Link>
              </li>
              <li>
                <Link
                  href="/#how-it-works"
                  className="text-neutral-400 hover:text-white transition-colors"
                >
                  How It Works
                </Link>
              </li>
               <li>
                <Link
                  href="/detect"
                  className="text-neutral-400 hover:text-white transition-colors"
                >
                  Start Analyzing
                </Link>
              </li>
            </ul>
          </div>

          {/* Column 3: Legal */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Legal</h3>
            <ul className="space-y-2">
              <li>
                <Link
                  href="/privacy-policy"
                  className="text-neutral-400 hover:text-white transition-colors"
                >
                  Privacy Policy
                </Link>
              </li>
              <li>
                <Link
                  href="/terms-of-service"
                  className="text-neutral-400 hover:text-white transition-colors"
                >
                  Terms of Service
                </Link>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-12 pt-8 border-t border-neutral-800 flex flex-col sm:flex-row justify-between items-center">
          <p className="text-neutral-500 text-sm mb-4 sm:mb-0">
            © {new Date().getFullYear()} DeepFake. All rights reserved.
          </p>
          <div className="flex items-center space-x-4">
            <a
              href="https://github.com/Yogesh-dev318"
              target="_blank"
              rel="noopener noreferrer"
              className="text-neutral-500 hover:text-white transition-colors"
              aria-label="GitHub"
            >
              <Github size={24} />
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;

