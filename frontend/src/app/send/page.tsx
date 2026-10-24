'use client';

import { useCallback, useEffect, useState } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { Send, Mail, Loader2, CheckCircle, AlertCircle, Users, Zap, Brain, Eye, ChevronDown, Shield } from 'lucide-react';

import { useRequireAuth } from '@/lib/auth';
import { AuthUser, encodeMessage, listUsers, transmit, getErrorMessage } from '@/lib/api';
import { PageWrapper, Card, Badge, LoadingSpinner, useToastHelpers } from '@/components';

type Mode = 'static' | 'rl' | 'gan' | 'auto';

const modeOptions = [
  { value: 'auto', label: 'Auto (RL → GAN → Static)', icon: Zap, description: 'Automatic scheduler fallback; the recorded GAN path is under review' },
  { value: 'rl', label: 'RL (PPO Policy)', icon: Brain, description: 'Reinforcement learning optimized scheduling' },
  { value: 'gan', label: 'GAN (WGAN-GP)', icon: Eye, description: 'Generative adversarial temporal patterns' },
  { value: 'static', label: 'Static (NoiseController)', icon: Shield, description: 'Mathematical noise injection fallback' },
];

export default function SendPage() {
  const { user, loading: authLoading, logout } = useRequireAuth();
  const { success, error: showError } = useToastHelpers();

  const [recipients, setRecipients] = useState<AuthUser[]>([]);
  const [recipient, setRecipient] = useState<string>('');
  const [message, setMessage] = useState('');
  const [mode, setMode] = useState<Mode>('auto');
  const [speed, setSpeed] = useState<number>(100);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState<{ session_id?: string; mode_used: string; total_packets: number } | null>(null);
  const [recipientMenuOpen, setRecipientMenuOpen] = useState(false);

  const loadRecipients = useCallback(async () => {
    if (!user) return;
    try {
      const resp = await listUsers(true);
      setRecipients(resp.users);
      if (resp.users.length > 0 && !recipient) {
        setRecipient(resp.users[0].username);
      }
    } catch (err: unknown) {
      const detail = getErrorMessage(err, 'Failed to load recipients');
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
      setError('Select a recipient');
      return;
    }
    if (!message.trim()) {
      setError('Message cannot be empty');
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
      success('Message sent!', `Transmission started with ${txResp.mode_used.toUpperCase()} mode (${txResp.total_packets} packets)`);
    } catch (err: unknown) {
      const detail = getErrorMessage(err, 'Send failed');
      setError(detail);
      showError('Send failed', detail);
    } finally {
      setBusy(false);
    }
  };

  if (authLoading || !user) {
    return (
      <PageWrapper fullWidth>
        <LoadingSpinner size="lg" />
      </PageWrapper>
    );
  }

  const selectedMode = modeOptions.find(m => m.value === mode);

  return (
    <PageWrapper
      title="Send Message"
      description="Encode and transmit a secret message through the stealth pipeline"
      action={
        <Link href="/inbox" className="btn-secondary btn-sm">
          <Mail className="w-4 h-4" />
          Inbox
        </Link>
      }
    >
      <div className="max-w-3xl mx-auto">
        <Card className="mb-6">
          <div className="flex items-center justify-between gap-4 flex-wrap sm:flex-nowrap">
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary shrink-0">
                <Send className="w-5 h-5" />
              </div>
              <div className="min-w-0">
                <h2 className="font-semibold text-foreground truncate">Signed in as {user.username}</h2>
                <p className="text-body-sm text-muted">Your messages are encoded with Reed-Solomon ECC and transmitted via stealth scheduler</p>
              </div>
            </div>
            <Link href="/inbox" className="btn-ghost btn-sm shrink-0">
              <Mail className="w-4 h-4" />
              View Inbox
            </Link>
          </div>
        </Card>

        <form onSubmit={onSubmit} className="space-y-6">
          {/* Recipient */}
          <Card>
            <h3 className="text-heading-sm font-semibold mb-4">Recipient</h3>
            <div className="relative">
              <select
                id="recipient"
                value={recipient}
                onChange={(e) => { setRecipient(e.target.value); setRecipientMenuOpen(false); }}
                className="input w-full appearance-none pr-10"
                required
                disabled={busy}
              >
                {recipients.length === 0 && <option value="">No other users registered</option>}
                {recipients.map((u) => (
                  <option key={u.id} value={u.username}>{u.username}</option>
                ))}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted pointer-events-none" />
            </div>
            {recipients.length === 0 && (
              <p className="text-body-sm text-warning mt-2 flex items-center gap-1">
                <Users className="w-4 h-4" />
                No recipients available. Ask another user to register, or create a test account.
              </p>
            )}
          </Card>

          {/* Message */}
          <Card>
            <h3 className="text-heading-sm font-semibold mb-4">Message</h3>
            <div className="relative">
              <textarea
                id="message"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                rows={5}
                className="input font-mono text-body-sm resize-none"
                placeholder="Meet at the cafe at noon..."
                required
                disabled={busy}
              />
              <div className="absolute bottom-2 right-2 text-body-xs text-muted">
                {message.length} / 1000 chars
              </div>
            </div>
            <p className="text-body-xs text-muted mt-2 flex items-center gap-1">
              <Shield className="w-3.5 h-3.5" />
              CRC-verified frame with Reed-Solomon error correction; zero false accepts observed after decoder hardening
            </p>
          </Card>

          {/* Options Grid */}
          <Card>
            <h3 className="text-heading-sm font-semibold mb-4">Transmission Settings</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Stealth Mode */}
              <div>
                <label className="label" htmlFor="mode">Stealth Mode</label>
                <div className="relative">
                  <select
                    id="mode"
                    value={mode}
                    onChange={(e) => setMode(e.target.value as Mode)}
                    className="input appearance-none pr-10"
                    disabled={busy}
                  >
                    {modeOptions.map((m) => (
                      <option key={m.value} value={m.value}>{m.label}</option>
                    ))}
                  </select>
                  <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted pointer-events-none" />
                </div>
                {selectedMode && (
                  <p className="text-body-xs text-muted mt-2 flex items-center gap-1">
                    <selectedMode.icon className="w-3.5 h-3.5" />
                    {selectedMode.description}
                  </p>
                )}
                <p className="text-body-xs text-muted mt-1">
                  This demo always encodes with the exact_vcp codec. The DSSC codec and the per-mode scheduler
                  selection are scheduled for a later release.
                </p>
              </div>

              {/* Speed Multiplier */}
              <div>
                <label className="label" htmlFor="speed">Speed Multiplier</label>
                <div className="relative">
                  <input
                    id="speed"
                    type="number"
                    min={1}
                    max={10000}
                    step={1}
                    value={speed}
                    onChange={(e) => setSpeed(Number(e.target.value) || 1)}
                    className="input pr-20"
                    disabled={busy}
                  />
                  <span className="absolute right-3 top-1/2 -translate-y-1/2 text-body-sm text-muted">×</span>
                </div>
                <p className="text-body-xs text-muted mt-2">
                  1 = real-time; 100 = 100× faster (demo). Higher values compress delays.
                </p>
              </div>
            </div>
          </Card>

          {/* Error / Success */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-start gap-3 p-4 bg-error/10 border border-error/20 rounded-xl text-error"
              role="alert"
            >
              <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-medium">Send failed</p>
                <p className="text-body-sm mt-1">{error}</p>
              </div>
            </motion.div>
          )}

          {sent && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-start gap-3 p-4 bg-success/10 border border-success/20 rounded-xl text-success"
            >
              <CheckCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-medium">Transmission started</p>
                <div className="text-body-sm mt-1 flex flex-wrap gap-4">
                  <span><strong>Mode:</strong> <code className="bg-surface px-1.5 py-0.5 rounded">{sent.mode_used}</code></span>
                  <span><strong>Packets:</strong> {sent.total_packets}</span>
                  <span><strong>Session:</strong> <code className="bg-surface px-1.5 py-0.5 rounded">{sent.session_id}</code></span>
                </div>
                <p className="text-body-xs mt-2">
                  Sign in as <strong>{recipient}</strong> in another window to see it arrive in their inbox.
                </p>
              </div>
            </motion.div>
          )}

          {/* Submit */}
          <button
            type="submit"
            disabled={busy || recipients.length === 0}
            className="btn-primary btn-lg w-full justify-center gap-2"
          >
            {busy ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Encoding + transmitting…
              </>
            ) : (
              <>
                <Send className="w-5 h-5" />
                Send Message
              </>
            )}
          </button>
        </form>

        {/* Info */}
        <Card className="bg-primary/5 border-primary/20 mt-8">
          <h4 className="font-semibold text-primary mb-3 flex items-center gap-2">
            <Zap className="w-4 h-4" />
            How it works
          </h4>
          <ol className="space-y-2 text-body-sm text-muted">
            <li className="flex items-start gap-2"><span className="text-primary font-mono">1.</span> Message is chunked and encoded via VCP (Voronoi Codebook Partitioning)</li>
            <li className="flex items-start gap-2"><span className="text-primary font-mono">2.</span> Reed-Solomon ECC adds parity bytes for error correction</li>
            <li className="flex items-start gap-2"><span className="text-primary font-mono">3.</span> Stealth scheduler (RL/GAN/Static) determines packet timing & channels</li>
            <li className="flex items-start gap-2"><span className="text-primary font-mono">4.</span> Packets written to recipient's shared_channel directory</li>
            <li className="flex items-start gap-2"><span className="text-primary font-mono">5.</span> Recipient decodes on-demand from their inbox</li>
          </ol>
        </Card>
      </div>
    </PageWrapper>
  );
}