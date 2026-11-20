'use client';

import React, { useCallback, useEffect, useState } from 'react';
import Link from 'next/link';
import {
  Mail,
  RefreshCw,
  Trash2,
  Eye,
  ShieldCheck,
  AlertCircle,
  Clock,
  User,
  MessageSquare,
  CheckCircle2,
  XCircle,
} from 'lucide-react';

import { useRequireAuth } from '@/lib/auth';
import {
  InboxItem,
  InboxDecodeResponse,
  decodeInboxSession,
  deleteInboxSession,
  getInbox,
  getErrorMessage,
} from '@/lib/api';
import {
  PageWrapper,
  Card,
  Badge,
  Button,
  LoadingSpinner,
  EmptyState,
  CarrierPreviewModal,
  CarrierPreviewTarget,
  useToastHelpers,
} from '@/components';

export default function InboxPage() {
  const { user, loading: authLoading } = useRequireAuth();
  const { success, error: showError } = useToastHelpers();

  const [items, setItems] = useState<InboxItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [decoded, setDecoded] = useState<Record<string, InboxDecodeResponse>>({});
  const [busySession, setBusySession] = useState<string | null>(null);
  const [previewTarget, setPreviewTarget] = useState<CarrierPreviewTarget | null>(null);

  const refresh = useCallback(async () => {
    if (!user) return;
    setLoading(true);
    setError(null);
    try {
      const resp = await getInbox();
      setItems(resp.items);
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to load inbox'));
    } finally {
      setLoading(false);
    }
  }, [user]);

  useEffect(() => {
    if (user) refresh();
  }, [user, refresh]);

  const onDecode = async (sessionId: string) => {
    setBusySession(sessionId);
    setError(null);
    try {
      const resp = await decodeInboxSession(sessionId);
      setDecoded((prev) => ({ ...prev, [sessionId]: resp }));
      setItems((prev) =>
        prev.map((it) => (it.session_id === sessionId ? { ...it, decoded: resp.ecc_success } : it)),
      );
      if (resp.ecc_success) {
        success(
          'Decoded from Disk',
          resp.ecc_errors_fixed?.length
            ? `Recovered 100% via RS-ECC (${resp.ecc_errors_fixed.length} byte(s) repaired)`
            : 'Message decoded and CRC-16 verified',
        );
      } else {
        showError('Fail-Closed Rejection', 'Missing packets exceeded Reed-Solomon parity budget');
      }
    } catch (err) {
      const msg = getErrorMessage(err, 'Decode failed');
      setError(msg);
      showError('Decode failed', msg);
    } finally {
      setBusySession(null);
    }
  };

  const onDelete = async (sessionId: string) => {
    if (!confirm('Delete this session and all its on-disk packets?')) return;
    setBusySession(sessionId);
    try {
      await deleteInboxSession(sessionId);
      setItems((prev) => prev.filter((it) => it.session_id !== sessionId));
      setDecoded((prev) => {
        const { [sessionId]: _, ...rest } = prev;
        return rest;
      });
      success('Deleted', 'Session removed from inbox');
    } catch (err) {
      showError('Delete failed', getErrorMessage(err));
    } finally {
      setBusySession(null);
    }
  };

  if (authLoading || !user) {
    return (
      <PageWrapper>
        <div className="py-24 flex justify-center">
          <LoadingSpinner size="lg" />
        </div>
      </PageWrapper>
    );
  }

  return (
    <PageWrapper
      title="Inbox"
      description={`Sessions addressed to @${user.username} — incomplete sessions decode directly via RS-ECC erasure recovery`}
      action={
        <div className="flex items-center gap-2">
          <Link href="/erasure" className="btn-secondary btn-sm">
            <ShieldCheck className="w-3.5 h-3.5" />
            RS-ECC Lab
          </Link>
          <Button
            variant="ghost"
            size="sm"
            onClick={refresh}
            disabled={loading}
            leftIcon={<RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />}
          >
            Refresh
          </Button>
        </div>
      }
    >
      <div className="max-w-4xl space-y-4 pb-12">
        {error && (
          <div className="p-4 rounded-xl bg-error-muted border border-error/30 text-error text-body-sm flex items-center gap-2.5">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {items.length === 0 && !loading ? (
          <Card>
            <EmptyState
              icon={<Mail className="w-10 h-10" />}
              title="Inbox is empty"
              description="Messages sent to you via Covert Chat or RS-ECC Lab appear here for on-demand decoding."
              action={
                <Link href="/chat" className="btn-primary">
                  <MessageSquare className="w-4 h-4" />
                  Open Covert Chat
                </Link>
              }
            />
          </Card>
        ) : (
          items.map((item) => {
            const missingCount = Math.max(0, item.total_items - item.received_packets);
            const detail = decoded[item.session_id];
            const busy = busySession === item.session_id;

            return (
              <Card key={item.session_id} className="p-0 overflow-hidden">
                <div className="p-5 flex flex-wrap items-start justify-between gap-4">
                  <div className="min-w-0 flex-1 space-y-1.5">
                    <div className="flex items-center gap-2.5 flex-wrap">
                      <span className="inline-flex items-center gap-1.5 font-semibold text-foreground capitalize">
                        <User className="w-4 h-4 text-primary" />
                        {item.sender_username || '(unknown)'}
                      </span>
                      <span className="font-mono text-body-xs px-2 py-0.5 rounded-md bg-surface-muted border border-border text-muted">
                        {item.session_id}
                      </span>
                      {item.mode_used && (
                        <Badge variant="info" size="sm">
                          {item.mode_used.toUpperCase()}
                        </Badge>
                      )}
                      {missingCount > 0 ? (
                        <Badge variant="warning" size="sm" dot>
                          {missingCount} packet(s) dropped on wire
                        </Badge>
                      ) : (
                        <Badge variant="success" size="sm" dot>
                          Complete
                        </Badge>
                      )}
                    </div>

                    <div className="flex items-center gap-4 text-body-xs text-muted flex-wrap font-mono">
                      <span className="inline-flex items-center gap-1">
                        <Clock className="w-3.5 h-3.5" />
                        {new Date(item.timestamp * 1000).toLocaleString()}
                      </span>
                      <span>
                        {item.received_packets}/{item.total_items} packet files on disk
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    <Link
                      href={`/erasure?session=${encodeURIComponent(item.session_id)}`}
                      className="btn-secondary btn-sm"
                    >
                      Inspect Wire
                    </Link>
                    <Button
                      variant="primary"
                      size="sm"
                      onClick={() => onDecode(item.session_id)}
                      disabled={item.received_packets === 0 || busy}
                      loading={busy}
                      leftIcon={detail ? <Eye className="w-3.5 h-3.5" /> : <ShieldCheck className="w-3.5 h-3.5" />}
                    >
                      {missingCount > 0 ? 'Recover with RS-ECC' : detail ? 'Re-read' : 'Decode'}
                    </Button>
                    <button
                      type="button"
                      onClick={() => onDelete(item.session_id)}
                      disabled={busy}
                      className="btn-ghost btn-sm text-muted hover:text-error"
                      aria-label="Delete session"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {detail && (
                  <div className="border-t border-border bg-surface-muted/45 p-5 space-y-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2 text-body-sm font-medium">
                        {detail.ecc_success ? (
                          <CheckCircle2 className="w-4 h-4 text-success" />
                        ) : (
                          <XCircle className="w-4 h-4 text-error" />
                        )}
                        <span className={detail.ecc_success ? 'text-success' : 'text-error'}>
                          {detail.ecc_success
                            ? 'Recovered Plaintext (CRC-16 Verified)'
                            : 'Fail-Closed Rejection (Beyond Parity Budget)'}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 flex-wrap">
                        <Badge variant="info" size="sm">
                          {(detail.verification_rate * 100).toFixed(1)}% packets on disk
                        </Badge>
                        {detail.ecc_errors_fixed?.length > 0 && (
                          <Badge variant="primary" size="sm">
                            RS-ECC repaired {detail.ecc_errors_fixed.length} byte(s)
                          </Badge>
                        )}
                        {typeof detail.elapsed_ms === 'number' && (
                          <Badge variant="neutral" size="sm">
                            {detail.elapsed_ms.toFixed(1)} ms
                          </Badge>
                        )}
                      </div>
                    </div>

                    <div className="p-4 rounded-xl bg-surface border border-border font-mono text-body-sm text-foreground whitespace-pre-wrap break-words">
                      {detail.ecc_success
                        ? detail.reconstructed_meaning
                        : '(Corrupted beyond RS-ECC capacity — cleanly rejected)'}
                    </div>

                    {Array.isArray(detail.items) && detail.items.length > 0 && (
                      <div className="space-y-1.5">
                        <div className="text-[11px] font-mono text-muted uppercase tracking-wider">
                          Carrier Files ({detail.items.length}) — click to view
                        </div>
                        <div className="flex flex-wrap gap-1.5 max-h-28 overflow-y-auto">
                          {detail.items.map((carrier, idx) => {
                            const dropped = !carrier.verified || carrier.media_id === '[DROPPED]';
                            return dropped ? (
                              <span
                                key={idx}
                                className="px-2 py-0.5 rounded-md bg-error-muted border border-error/30 font-mono text-[11px] text-error"
                              >
                                #{idx} DROPPED
                              </span>
                            ) : (
                              <button
                                key={idx}
                                type="button"
                                onClick={() => setPreviewTarget({ mediaId: carrier.media_id, seq: idx })}
                                className="px-2 py-0.5 rounded-md bg-surface hover:border-primary border border-border font-mono text-[11px] text-foreground hover:text-primary transition-colors"
                              >
                                #{idx} {carrier.media_id}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </Card>
            );
          })
        )}
      </div>

      <CarrierPreviewModal
        target={previewTarget}
        onClose={() => setPreviewTarget(null)}
      />
    </PageWrapper>
  );
}
