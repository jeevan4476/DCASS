'use client';

import React, { useState, useEffect, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { Sparkles, User, Lock, AlertCircle, ArrowRight } from 'lucide-react';
import { PageWrapper, Card, Button, Input, LoadingSpinner, useToastHelpers } from '@/components';
import { useAuth } from '@/lib/auth';
import { getErrorMessage, getApiBase } from '@/lib/api';

function LoginForm() {
  const { login } = useAuth();
  const router = useRouter();
  const searchParams = useSearchParams();
  const { error: showError } = useToastHelpers();
  // Only allow internal relative paths as the post-login redirect target;
  // anything else (absolute URLs, protocol-relative //host, backslashes)
  // would be an open redirect.
  const rawNext = searchParams.get('next');
  const nextPath =
    rawNext && rawNext.startsWith('/') && !rawNext.startsWith('//') && !rawNext.includes('\\')
      ? rawNext
      : '/chat';

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [apiOrigin, setApiOrigin] = useState('');

  useEffect(() => {
    if (typeof window !== 'undefined') setApiOrigin(getApiBase());
  }, []);

  const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await login(username, password);
      router.replace(nextPath);
    } catch (err) {
      const msg = getErrorMessage(err, 'Invalid credentials');
      setError(msg);
      showError('Sign in failed', msg);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <PageWrapper hideNav className="flex items-center justify-center p-4 sm:p-8">
      <div className="w-full max-w-md space-y-6">
        <div className="text-center space-y-2">
          <div className="w-12 h-12 rounded-2xl bg-primary-muted border border-primary/25 text-primary flex items-center justify-center mx-auto">
            <Sparkles className="w-6 h-6" />
          </div>
          <h1 className="font-serif text-display-sm font-medium text-foreground">
            Sign in to DCASS
          </h1>
          <p className="text-body-sm text-muted">
            Access covert chat, on-disk RS-ECC erasure testing, and live wire inspection
          </p>
          {apiOrigin && (
            <p className="text-[11px] font-mono text-muted/80 truncate">{apiOrigin}</p>
          )}
        </div>

        <Card>
          <form onSubmit={onSubmit} className="space-y-4">
            {error && (
              <div className="p-3 rounded-xl bg-error-muted border border-error/30 text-error text-body-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            <Input
              id="username"
              type="text"
              autoComplete="username"
              label="Username"
              placeholder="alice"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              disabled={submitting}
              leftIcon={<User className="w-4 h-4" />}
            />

            <Input
              id="password"
              type="password"
              autoComplete="current-password"
              label="Password"
              placeholder="Enter password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              disabled={submitting}
              leftIcon={<Lock className="w-4 h-4" />}
            />

            <Button
              type="submit"
              variant="primary"
              size="lg"
              className="w-full"
              loading={submitting}
              rightIcon={<ArrowRight className="w-4 h-4" />}
            >
              Sign in
            </Button>
          </form>

          <div className="mt-5 pt-4 border-t border-border space-y-3">
            <p className="text-body-xs text-muted text-center">
              Quick demo accounts (click to fill):
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              {['alice/alice12345', 'bob/bob12345', 'charlie/charlie12345'].map((cred) => {
                const [u, p] = cred.split('/');
                return (
                  <button
                    key={cred}
                    type="button"
                    onClick={() => {
                      setUsername(u);
                      setPassword(p);
                    }}
                    className="px-2.5 py-1 rounded-lg bg-surface-muted hover:bg-surface-hover border border-border font-mono text-body-xs text-foreground transition-colors"
                  >
                    {cred}
                  </button>
                );
              })}
            </div>
            <p className="text-body-xs text-muted text-center pt-1">
              New here?{' '}
              <Link href="/register" className="text-primary font-medium hover:underline">
                Create an account
              </Link>
            </p>
          </div>
        </Card>
      </div>
    </PageWrapper>
  );
}

export default function LoginPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen flex items-center justify-center bg-background">
          <LoadingSpinner size="lg" />
        </div>
      }
    >
      <LoginForm />
    </Suspense>
  );
}
