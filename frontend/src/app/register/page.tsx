'use client';

import React, { useState, Suspense } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { UserPlus, User, Lock, AlertCircle, ArrowRight } from 'lucide-react';
import { PageWrapper, Card, Button, Input, LoadingSpinner, useToastHelpers } from '@/components';
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
      setError('Username must be at least 3 characters.');
      return;
    }
    if (password.length < 8) {
      setError('Password must be at least 8 characters.');
      return;
    }
    if (password !== confirm) {
      setError('Passwords do not match.');
      return;
    }
    setSubmitting(true);
    try {
      await register(username, password);
      router.replace('/chat');
    } catch (err) {
      const msg = getErrorMessage(err, 'Registration failed');
      setError(msg);
      showError('Registration failed', msg);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <PageWrapper hideNav className="flex items-center justify-center p-4 sm:p-8">
      <div className="w-full max-w-md space-y-6">
        <div className="text-center space-y-2">
          <div className="w-12 h-12 rounded-2xl bg-primary-muted border border-primary/25 text-primary flex items-center justify-center mx-auto">
            <UserPlus className="w-6 h-6" />
          </div>
          <h1 className="font-serif text-display-sm font-medium text-foreground">
            Create an account
          </h1>
          <p className="text-body-sm text-muted">
            Join the local DCASS workspace to send and decode covert sessions
          </p>
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
              label="Username"
              placeholder="Choose a username (3+ chars)"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              disabled={submitting}
              leftIcon={<User className="w-4 h-4" />}
            />

            <Input
              id="password"
              type="password"
              label="Password"
              placeholder="Minimum 8 characters"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              disabled={submitting}
              leftIcon={<Lock className="w-4 h-4" />}
            />

            <Input
              id="confirm"
              type="password"
              label="Confirm password"
              placeholder="Re-enter password"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
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
              Create account
            </Button>
          </form>

          <p className="mt-5 pt-4 border-t border-border text-body-xs text-muted text-center">
            Already have an account?{' '}
            <Link href="/login" className="text-primary font-medium hover:underline">
              Sign in
            </Link>
          </p>
        </Card>
      </div>
    </PageWrapper>
  );
}

export default function RegisterPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen flex items-center justify-center bg-background">
          <LoadingSpinner size="lg" />
        </div>
      }
    >
      <RegisterForm />
    </Suspense>
  );
}
