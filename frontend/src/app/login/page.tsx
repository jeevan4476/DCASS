'use client';

import { useState, Suspense } from 'react';
import { motion } from 'framer-motion';
import { useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { Shield, Zap, Brain, Eye, User, Lock, AlertCircle, ChevronRight } from 'lucide-react';
import { PageWrapper, Card, Button, Input, useToastHelpers, LoadingSpinner } from '@/components';
import { useAuth } from '@/lib/auth';
import { getErrorMessage } from '@/lib/api';

function LoginForm() {
  const { login } = useAuth();
  const router = useRouter();
  const searchParams = useSearchParams();
  const { error: showError } = useToastHelpers();
  const nextPath = searchParams.get('next') || '/encode';

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await login(username, password);
      router.replace(nextPath);
    } catch (err: unknown) {
      const message = getErrorMessage(err, 'Login failed');
      setError(message);
      showError('Login failed', message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <PageWrapper fullWidth className="min-h-screen flex items-center justify-center py-12">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.19, 1, 0.22, 1] }}
        className="w-full max-w-md"
      >
        {/* Logo & Title */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mx-auto mb-4">
            <Shield className="w-8 h-8 text-primary" />
          </div>
          <h1 className="text-display-sm font-bold text-foreground mb-2">Welcome back</h1>
          <p className="text-body-lg text-muted">Sign in to access the DCASS stealth pipeline</p>
        </div>

        {/* Features */}
        <div className="grid grid-cols-3 gap-3 mb-8">
          {[
            { icon: Zap, label: 'RL/GAN Scheduling' },
            { icon: Brain, label: 'Semantic Encoding' },
            { icon: Eye, label: 'Zero-Modification' },
          ].map((item, i) => {
            const Icon = item.icon;
            return (
              <motion.div
                key={item.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 + i * 0.1 }}
                className="p-3 bg-surface/50 border border-border rounded-xl text-center"
              >
                <Icon className="w-5 h-5 text-primary mx-auto mb-1" />
                <span className="text-body-xs text-muted">{item.label}</span>
              </motion.div>
            );
          })}
        </div>

        {/* Login Form */}
        <Card>
          <form onSubmit={onSubmit} className="space-y-5">
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-start gap-3 p-4 bg-error/10 border border-error/20 rounded-xl text-error"
                role="alert"
              >
                <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-medium">Sign in failed</p>
                  <p className="text-body-sm mt-1">{error}</p>
                </div>
              </motion.div>
            )}

            <div>
              <Input
                id="username"
                type="text"
                autoComplete="username"
                value={username}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setUsername(e.target.value)}
                placeholder="Enter your username"
                required
                disabled={submitting}
                leftIcon={<User className="w-4 h-4" />}
                label="Username"
              />
            </div>

            <div>
              <Input
                id="password"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setPassword(e.target.value)}
                placeholder="Enter your password"
                required
                disabled={submitting}
                leftIcon={<Lock className="w-4 h-4" />}
                label="Password"
              />
            </div>

            <Button
              type="submit"
              className="w-full justify-center gap-2"
              size="lg"
              disabled={submitting}
            >
              {submitting ? (
                <>
                  <svg className="animate-spin -ml-1 mr-2 h-5 w-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Signing in…
                </>
              ) : (
                <>
                  Sign in
                  <ChevronRight className="w-4 h-4" />
                </>
              )}
            </Button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-body-sm text-muted">
              No account?{' '}
              <Link href="/register" className="text-primary hover:underline font-medium">
                Create one
              </Link>
            </p>
          </div>

          <div className="mt-6 p-3 bg-surface/50 border border-border rounded-lg">
            <p className="text-body-xs text-muted mb-2 font-medium">Demo accounts (local only)</p>
            <div className="flex flex-wrap gap-2 justify-center">
              {['alice/alice12345', 'bob/bob12345', 'charlie/charlie12345'].map((cred) => (
                <code key={cred} className="bg-surface px-2 py-1 rounded text-body-xs text-primary font-mono border border-border">
                  {cred}
                </code>
              ))}
            </div>
          </div>
        </Card>
      </motion.div>
    </PageWrapper>
  );
}

export default function LoginPage() {
  return (
    <Suspense fallback={<PageWrapper fullWidth className="min-h-screen flex items-center justify-center"><div className="w-full max-w-md"><div className="text-center"><div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mx-auto mb-4"><Shield className="w-8 h-8 text-primary" /></div><LoadingSpinner size="lg" className="mx-auto" /></div></div></PageWrapper>}>
      <LoginForm />
    </Suspense>
  );
}