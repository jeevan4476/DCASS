'use client';

/**
 * Phase F — Send page.
 *
 * Combines encode + transmit into one authenticated flow:
 *   1. Pick a recipient from the dropdown (populated via GET /api/users).
 *   2. Enter the plaintext message.
 *   3. Submit — the page calls /api/encode, then /api/transmit with
 *      the resulting media_ids and the chosen recipient_username.
 *   4. The response's session_id is shown so the sender can watch for it
 *      in the recipient's inbox (as a different logged-in user).
 */

import { useCallback, useEffect, useState } from 'react';
import Link from 'next/link';

import { useRequireAuth } from '@/lib/auth';
import {
  AuthUser,
  encodeMessage,
  listUsers,
  transmit,
} from '@/lib/api';

type Mode = 'static' | 'rl' | 'gan' | 'auto';

export default function SendPage() {
  const { user, loading: authLoading, logout } = useRequireAuth();

  const [recipients, setRecipients] = useState<AuthUser[]>([]);
  const [recipient, setRecipient] = useState<string>('');
  const [message, setMessage] = useState('');
  const [mode, setMode] = useState<Mode>('auto');
  const [speed, setSpeed] = useState<number>(100);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState<{ session_id?: string; mode_used: string; total_packets: number } | null>(null);

  const loadRecipients = useCallback(async () => {
    if (!user) return;
    try {
      const resp = await listUsers();
      setRecipients(resp.users);
      if (resp.users.length > 0 && !recipient) {
        setRecipient(resp.users[0].username);
      }
    } catch (err: unknown) {
      const detail =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        (err as Error)?.message ||
        'Failed to load recipients';
      setError(detail);
    }
  }, [user, recipient]);

  useEffect(() => {
    if (user) loadRecipients();
  }, [user, loadRecipients]);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSent(null);
    if (!recipient) {
      setError('select a recipient');
      return;
    }
    if (!message.trim()) {
      setError('message cannot be empty');
      return;
    }
    setBusy(true);
    try {
      // 1. Encode the plaintext into media_ids.
      const encoded = await encodeMessage({
        message,
        use_ecc: true,
        mode: 'exact_vcp',
      });
      // 2. Transmit through the stealth scheduler to the chosen recipient.
      const txResp = await transmit({
        media_ids: encoded.media_ids,
        recipient_username: recipient,
        mode,
        message,
        speed_multiplier: speed,
      });
      setSent({
        session_id: txResp.session_id,
        mode_used: txResp.mode_used,
        total_packets: txResp.total_packets,
      });
    } catch (err: unknown) {
      const detail =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        (err as Error)?.message ||
        'Send failed';
      setError(detail);
    } finally {
      setBusy(false);
    }
  };

  if (authLoading || !user) {
    return (
      <div className="min-h-screen flex items-center justify-center text-gray-500">
        Loading…
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-2xl mx-auto">
        <header className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Send a message</h1>
            <p className="text-sm text-gray-500">
              Signed in as <span className="font-medium">{user.username}</span>
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Link href="/inbox" className="text-sm text-blue-600 hover:underline">
              Inbox
            </Link>
            <button onClick={logout} className="text-sm text-gray-500 hover:text-gray-700">
              Sign out
            </button>
          </div>
        </header>

        <form onSubmit={onSubmit} className="bg-white border border-gray-200 rounded p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1" htmlFor="recipient">
              Recipient
            </label>
            <select
              id="recipient"
              value={recipient}
              onChange={(e) => setRecipient(e.target.value)}
              className="w-full border border-gray-300 rounded px-3 py-2"
              required
            >
              {recipients.length === 0 && <option value="">(no other users yet)</option>}
              {recipients.map((u) => (
                <option key={u.id} value={u.username}>
                  {u.username}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1" htmlFor="message">
              Message
            </label>
            <textarea
              id="message"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              rows={4}
              className="w-full border border-gray-300 rounded px-3 py-2 font-mono text-sm"
              placeholder="Meet at the cafe at noon"
              required
            />
            <p className="text-xs text-gray-500 mt-1">
              Encoded exact-byte-per-item with Reed-Solomon error correction.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1" htmlFor="mode">
                Stealth mode
              </label>
              <select
                id="mode"
                value={mode}
                onChange={(e) => setMode(e.target.value as Mode)}
                className="w-full border border-gray-300 rounded px-3 py-2"
              >
                <option value="auto">auto (RL → GAN → static)</option>
                <option value="rl">RL (PPO policy)</option>
                <option value="gan">GAN (WGAN-GP)</option>
                <option value="static">static (NoiseController)</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1" htmlFor="speed">
                Speed multiplier
              </label>
              <input
                id="speed"
                type="number"
                min={1}
                max={10000}
                step={1}
                value={speed}
                onChange={(e) => setSpeed(Number(e.target.value) || 1)}
                className="w-full border border-gray-300 rounded px-3 py-2"
              />
              <p className="text-xs text-gray-500 mt-1">
                1 = real wall clock; 100 = 100× faster (demo).
              </p>
            </div>
          </div>

          {error && (
            <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded px-3 py-2">
              {error}
            </div>
          )}

          {sent && (
            <div className="text-sm text-green-700 bg-green-50 border border-green-200 rounded px-3 py-2">
              Transmission started in background. mode_used={' '}
              <code>{sent.mode_used}</code> · {sent.total_packets} packets · session{' '}
              <code>{sent.session_id}</code>
              <div className="mt-1 text-xs text-green-700/80">
                Sign in as{' '}
                <strong>{recipient}</strong> in another window to see it arrive in their
                inbox.
              </div>
            </div>
          )}

          <button
            type="submit"
            disabled={busy || recipients.length === 0}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-2 rounded transition"
          >
            {busy ? 'Encoding + transmitting…' : 'Send'}
          </button>
        </form>
      </div>
    </div>
  );
}
