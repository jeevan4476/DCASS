import type { Metadata } from 'next';
import './globals.css';

import { AuthProvider } from '@/lib/auth';
import { ToastProvider } from '@/components/Toast';

export const metadata: Metadata = {
  title: 'DCASS · Semantic Steganography & RS-ECC Workspace',
  description: 'Dynamic Context-Aware Semantic Steganography with live Reed-Solomon erasure recovery',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased bg-background text-foreground">
        <AuthProvider>
          <ToastProvider>{children}</ToastProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
