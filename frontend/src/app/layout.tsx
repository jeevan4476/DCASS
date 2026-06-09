import type { Metadata } from "next";
import "./globals.css";

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
        {children}
      </body>
    </html>
  );
}
