'use client';

/**
 * Phase F — Minimal client-side auth context.
 *
 * Stores the JWT in localStorage (see api.ts). Exposes { user, login,
 * register, logout, loading } so pages can gate on auth state.
 *
 * See docs/GAN_RL_INTEGRATION_PLAN.md Phase F.6 (R6) for the XSS caveat:
 * localStorage is readable by any script on the page. Fine for a research
 * demo; switch to httpOnly cookies before any public deployment.
 */

import React, { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';

import {
  AuthUser,
  authLogin,
  authMe,
  authRegister,
  getAuthToken,
  setAuthToken,
} from './api';

interface AuthContextValue {
  user: AuthUser | null;
  loading: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  // On mount: if we have a token, validate it against /auth/me.
  useEffect(() => {
    let cancelled = false;
    const token = getAuthToken();
    if (!token) {
      setLoading(false);
      return;
    }
    authMe()
      .then((me) => {
        if (!cancelled) setUser(me);
      })
      .catch(() => {
        // Interceptor already cleared the bad token on 401.
        if (!cancelled) setUser(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    const resp = await authLogin(username, password);
    setAuthToken(resp.access_token);
    setUser(resp.user);
  }, []);

  const register = useCallback(async (username: string, password: string) => {
    const resp = await authRegister(username, password);
    setAuthToken(resp.access_token);
    setUser(resp.user);
  }, []);

  const logout = useCallback(() => {
    setAuthToken(null);
    setUser(null);
    router.push('/login');
  }, [router]);

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth() must be used inside <AuthProvider>');
  return ctx;
}

/**
 * Hook to redirect to /login if the user is not authenticated. Use at the
 * top of any page that calls authenticated endpoints.
 *
 *   const { user, loading } = useRequireAuth();
 *   if (!user) return null;  // redirect in flight
 */
export function useRequireAuth(): AuthContextValue {
  const auth = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (!auth.loading && !auth.user) {
      const next = encodeURIComponent(pathname || '/');
      router.replace(`/login?next=${next}`);
    }
  }, [auth.loading, auth.user, router, pathname]);

  return auth;
}
