import type { Metadata } from "next";
import "./globals.css";

import { AuthProvider } from "@/lib/auth";
import { ToastProvider } from "@/components/Toast";

export const metadata: Metadata = {
  title: "DCASS - Dynamic Context-Aware Semantic Steganography",
  description: "Zero-modification semantic steganography system with AI-driven stealth",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        <AuthProvider>
          <ToastProvider>{children}</ToastProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
