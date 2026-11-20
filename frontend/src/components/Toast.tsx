'use client';

import React, { createContext, useContext, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle2, AlertCircle, Info, AlertTriangle, X } from 'lucide-react';

type ToastType = 'success' | 'error' | 'info' | 'warning';

interface Toast {
  id: string;
  type: ToastType;
  title: string;
  message?: string;
}

interface ToastContextValue {
  addToast: (type: ToastType, title: string, message?: string) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const addToast = useCallback(
    (type: ToastType, title: string, message?: string) => {
      const id = `${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
      setToasts((prev) => [...prev.slice(-3), { id, type, title, message }]);
      setTimeout(() => removeToast(id), 4200);
    },
    [removeToast],
  );

  const styles: Record<ToastType, { icon: React.ReactNode; border: string; accent: string }> = {
    success: {
      icon: <CheckCircle2 className="w-4 h-4 text-success shrink-0 mt-0.5" />,
      border: 'border-success/30',
      accent: 'text-success',
    },
    error: {
      icon: <AlertCircle className="w-4 h-4 text-error shrink-0 mt-0.5" />,
      border: 'border-error/30',
      accent: 'text-error',
    },
    warning: {
      icon: <AlertTriangle className="w-4 h-4 text-warning shrink-0 mt-0.5" />,
      border: 'border-warning/30',
      accent: 'text-warning',
    },
    info: {
      icon: <Info className="w-4 h-4 text-info shrink-0 mt-0.5" />,
      border: 'border-info/30',
      accent: 'text-info',
    },
  };

  return (
    <ToastContext.Provider value={{ addToast }}>
      {children}
      <div className="fixed bottom-4 right-4 z-toast flex flex-col gap-2 max-w-sm w-full pointer-events-none px-4 sm:px-0">
        <AnimatePresence>
          {toasts.map((toast) => {
            const st = styles[toast.type];
            return (
              <motion.div
                key={toast.id}
                initial={{ opacity: 0, y: 12, scale: 0.96 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 8, scale: 0.96 }}
                className={`pointer-events-auto bg-surface border ${st.border} rounded-xl p-3.5 shadow-elevated flex items-start gap-2.5`}
              >
                {st.icon}
                <div className="flex-1 min-w-0">
                  <p className={`text-body-xs font-semibold ${st.accent}`}>{toast.title}</p>
                  {toast.message && (
                    <p className="text-body-xs text-muted mt-0.5 break-words">{toast.message}</p>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => removeToast(toast.id)}
                  className="text-muted hover:text-foreground p-0.5 rounded"
                  aria-label="Dismiss"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
}

export function useToastHelpers() {
  const ctx = useContext(ToastContext);
  const noop = () => {};
  if (!ctx) {
    return { success: noop, error: noop, info: noop, warning: noop };
  }
  return {
    success: (title: string, message?: string) => ctx.addToast('success', title, message),
    error: (title: string, message?: string) => ctx.addToast('error', title, message),
    info: (title: string, message?: string) => ctx.addToast('info', title, message),
    warning: (title: string, message?: string) => ctx.addToast('warning', title, message),
  };
}
