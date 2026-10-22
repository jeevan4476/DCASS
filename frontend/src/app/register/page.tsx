'use client';

import { useState, Suspense } from 'react';
import { motion } from 'framer-motion';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Shield, Zap, Brain, Eye, User, Lock, AlertCircle, ChevronRight, UserPlus } from 'lucide-react';
import { PageWrapper, Card, Button, Input, useToastHelpers, LoadingSpinner } from '@/components';
import { useAuth } from '@/lib/auth';
import { getErrorMessage } from '@/lib/api';

function RegisterForm() {
  const { register } = useAuth();
  const router = useRouter();
  const { error: showError } = useToastHelpers();

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError(null);
    if (username.trim().length < 3) {
      setError('Username must be at least 3 characters');
      return;
    }
    if (password !== confirm) {
      setError('Passwords do not match');
      return;
    }
    if (password.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }
    setSubmitting(true);
    try {
      await register(username, password);
      router.replace('/encode');
    } catch (err: unknown) {
      const message = getErrorMessage(err, 'Registration failed');
      setError(message);
      showError('Registration failed', message);
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
            <UserPlus className="w-8 h-8 text-primary" />
          </div>
          <h1 className="text-display-sm font-bold text-foreground mb-2">Create account</h1>
          <p className="text-body-lg text-muted">Join the DCASS stealth communication network</p>
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

        {/* Register Form */}
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
                  <p className="font-medium">Registration failed</p>
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
                placeholder="Choose a username"
                required
                disabled={submitting}
                leftIcon={<User className="w-4 h-4" />}
                label="Username"
                error={username.length > 0 && username.length < 3 ? 'Must be at least 3 characters' : undefined}
              />
            </div>

            <div>
              <Input
                id="password"
                type="password"
                autoComplete="new-password"
                value={password}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setPassword(e.target.value)}
                placeholder="Create a password"
                required
                disabled={submitting}
                leftIcon={<Lock className="w-4 h-4" />}
                label="Password"
                error={password.length > 0 && password.length < 8 ? 'Must be at least 8 characters' : undefined}
              />
            </div>

            <div>
              <Input
                id="confirm"
                type="password"
                autoComplete="new-password"
                value={confirm}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setConfirm(e.target.value)}
                placeholder="Confirm your password"
                required
                disabled={submitting}
                leftIcon={<Lock className="w-4 h-4" />}
                label="Confirm Password"
                error={confirm.length > 0 && password !== confirm ? 'Passwords do not match' : undefined}
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
                  Creating account…
                </>
              ) : (
                <>
                  Create account
                  <ChevronRight className="w-4 h-4" />
                </>
              )}
            </Button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-body-sm text-muted">
              Already have an account?{' '}
              <Link href="/login" className="text-primary hover:underline font-medium">
                Sign in
              </Link>
            </p>
          </div>

          <div className="mt-6 p-3 bg-surface/50 border border-border rounded-lg">
            <p className="text-body-xs text-muted mb-2 font-medium">Requirements</p>
            <ul className="space-y-1 text-body-xs text-muted">
              <li className="flex items-center gap-2"><span className="text-primary">→</span> Username: 3-40 characters</li>
              <li className="flex items-center gap-2"><span className="text-primary">→</span> Password: minimum 8 characters</li>
              <li className="flex items-center gap-2"><span className="text-primary">→</span> Accounts are local-only (demo)</li>
            </ul>
          </div>
        </Card>
      </motion.div>
    </PageWrapper>
  );
}

export default function RegisterPage() {
  return (
    <Suspense fallback={<PageWrapper fullWidth className="min-h-screen flex items-center justify-center"><div className="w-full max-w-md"><div className="text-center"><div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mx-auto mb-4"><UserPlus className="w-8 h-8 text-primary" /></div><LoadingSpinner size="lg" className="mx-auto" /></div></div></PageWrapper>}>
      <RegisterForm />
    </Suspense>
  );
}