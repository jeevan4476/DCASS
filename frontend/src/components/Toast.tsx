'use client';

import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';

export interface Toast {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  duration?: number;
  action?: { label: string; onClick: () => void };
}

interface ToastContextValue {
  toast: (toast: Omit<Toast, 'id'>) => string;
  dismiss: (id: string) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = useCallback((toast: Omit<Toast, 'id'>) => {
    const id = Math.random().toString(36).slice(2, 9);
    const newToast = { ...toast, id };
    setToasts((prev) => [...prev, newToast]);
    return id;
  }, []);

  const dismiss = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ toast: addToast, dismiss }}>
      {children}
      <ToastContainer toasts={toasts} onDismiss={dismiss} />
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used within ToastProvider');
  return ctx;
}

function ToastContainer({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: string) => void }) {
  const icons = {
    success: (
      <svg className="w-5 h-5 text-success" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
      </svg>
    ),
    error: (
      <svg className="w-5 h-5 text-error" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
      </svg>
    ),
    warning: (
      <svg className="w-5 h-5 text-warning" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    ),
    info: (
      <svg className="w-5 h-5 text-info" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
  };

  const borders = {
    success: 'border-l-success',
    error: 'border-l-error',
    warning: 'border-l-warning',
    info: 'border-l-info',
  };

  return (
    <AnimatePresence>
      <div className="fixed bottom-6 right-6 z-toast flex flex-col gap-3 pointer-events-none max-w-sm w-full">
        {toasts.map((toast) => (
          <ToastItem
            key={toast.id}
            toast={toast}
            icon={icons[toast.type]}
            borderClass={borders[toast.type]}
            onDismiss={onDismiss}
          />
        ))}
      </div>
    </AnimatePresence>
  );
}

function ToastItem({
  toast,
  icon,
  borderClass,
  onDismiss,
}: {
  toast: Toast;
  icon: React.ReactNode;
  borderClass: string;
  onDismiss: (id: string) => void;
}) {
  const [hovered, setHovered] = useState(false);

  React.useEffect(() => {
    const duration = toast.duration ?? 5000;
    const timer = setTimeout(() => onDismiss(toast.id), duration);
    return () => clearTimeout(timer);
  }, [toast.id, toast.duration, onDismiss]);

  return (
    <motion.div
      className={`pointer-events-auto glass-strong border ${borderClass} rounded-xl p-4 shadow-modal flex items-start gap-3 min-w-0 ${hovered ? 'shadow-glow-primary' : ''}`}
      initial={{ opacity: 0, x: 100, scale: 0.95 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 100, scale: 0.95 }}
      transition={{ type: 'spring', stiffness: 400, damping: 30 }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="flex-shrink-0 mt-0.5">{icon}</div>
      <div className="flex-1 min-w-0">
        <p className="font-semibold text-foreground">{toast.title}</p>
        {toast.message && <p className="text-body-sm text-muted mt-0.5">{toast.message}</p>}
        {toast.action && (
          <button
            onClick={() => { toast.action?.onClick(); onDismiss(toast.id); }}
            className="mt-2 text-body-sm text-primary hover:underline font-medium"
          >
            {toast.action.label}
          </button>
        )}
      </div>
      <button
        onClick={() => onDismiss(toast.id)}
        className="flex-shrink-0 text-muted hover:text-foreground transition-colors p-1 -mr-1 -mt-1"
        aria-label="Dismiss"
      >
        <X className="w-4 h-4" />
      </button>
    </motion.div>
  );
}

// Convenience hooks
export function useToastHelpers() {
  const { toast, dismiss } = useToast();

  return {
    success: (title: string, message?: string, options?: Partial<Toast>) =>
      toast({ type: 'success', title, message, ...options }),
    error: (title: string, message?: string, options?: Partial<Toast>) =>
      toast({ type: 'error', title, message, ...options }),
    warning: (title: string, message?: string, options?: Partial<Toast>) =>
      toast({ type: 'warning', title, message, ...options }),
    info: (title: string, message?: string, options?: Partial<Toast>) =>
      toast({ type: 'info', title, message, ...options }),
    dismiss,
  };
}