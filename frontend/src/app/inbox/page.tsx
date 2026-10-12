'use client';

import { useCallback, useEffect, useState } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { Mail, Send, Loader2, RefreshCw, LogOut, MessageSquare, Trash2, Eye, Shield, CheckCircle, AlertCircle, Clock, User, X } from 'lucide-react';

import { useRequireAuth } from '@/lib/auth';
import { InboxItem, InboxDecodeResponse, decodeInboxSession, deleteInboxSession, getInbox, getErrorMessage } from '@/lib/api';
import { PageWrapper, Card, Badge, LoadingSpinner, EmptyState, useToastHelpers, Tooltip } from '@/components';

export default function InboxPage() {
  const { user, loading: authLoading, logout } = useRequireAuth();
  const { success, error: showError } = useToastHelpers();

  const [items, setItems] = useState<InboxItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [decoded, setDecoded] = useState<Record<string, InboxDecodeResponse>>({});
  const [busySession, setBusySession] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!user) return;
    setLoading(true);
    setError(null);
    try {
      const resp = await getInbox();
      setItems(resp.items);
    } catch (err: unknown) {
      const message = getErrorMessage(err, 'Failed to load inbox');
      setError(message);
    } finally {
      setLoading(false);
    }
  }, [user]);

  useEffect(() => {
    if (user) refresh();
  }, [user, refresh]);

  const onDecode = async (sessionId: string) => {
    setBusySession(sessionId);
    try {
      const resp = await decodeInboxSession(sessionId);
      setDecoded((prev) => ({ ...prev, [sessionId]: resp }));
      setItems((prev) =>
        prev.map((it) => (it.session_id === sessionId ? { ...it, decoded: true } : it)),
      );
      const rate = typeof resp.verification_rate === 'number' ? (resp.verification_rate * 100).toFixed(1) : '100.0';
      success('Decoded', `Message recovered with ${rate}% verification`);
    } catch (err: unknown) {
      const message = getErrorMessage(err, 'Decode failed');
      setError(message);
      showError('Decode failed', message);
    } finally {
      setBusySession(null);
    }
  };

  const onDelete = async (sessionId: string) => {
    if (!confirm('Delete this message? This cannot be undone.')) return;
    setBusySession(sessionId);
    try {
      await deleteInboxSession(sessionId);
      setItems((prev) => prev.filter((it) => it.session_id !== sessionId));
      setDecoded((prev) => {
        const { [sessionId]: _, ...rest } = prev;
        return rest;
      });
      success('Deleted', 'Message removed from inbox');
    } catch (err: unknown) {
      const message = getErrorMessage(err, 'Delete failed');
      setError(message);
      showError('Delete failed', message);
    } finally {
      setBusySession(null);
    }
  };

  if (authLoading || !user) {
    return (
      <PageWrapper fullWidth>
        <LoadingSpinner size="lg" />
      </PageWrapper>
    );
  }

  const getModeColor = (mode?: string | null) => {
    switch (mode?.toLowerCase()) {
      case 'rl': return 'success';
      case 'gan': return 'info';
      case 'static': return 'warning';
      default: return 'neutral';
    }
  };

  return (
    <PageWrapper
      title="Inbox"
      description={`Messages addressed to ${user.username}`}
      action={
        <div className="flex items-center gap-2">
          <Link href="/send" className="btn-secondary btn-sm">
            <Send className="w-4 h-4" />
            Send
          </Link>
          <button onClick={refresh} disabled={loading} className="btn-ghost btn-sm">
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      }
    >
      <div className="max-w-4xl mx-auto">
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 flex items-start gap-3 p-4 bg-error/10 border border-error/20 rounded-xl text-error"
            role="alert"
          >
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-medium">Failed to load inbox</p>
              <p className="text-body-sm mt-1">{error}</p>
              <button onClick={refresh} className="mt-2 text-body-sm text-primary hover:underline">Retry</button>
            </div>
          </motion.div>
        )}

        {items.length === 0 && !loading ? (
          <Card className="text-center py-16">
            <EmptyState
              icon={<Mail className="w-12 h-12" />}
              title="Inbox is empty"
              description="No messages have been sent to you yet. Send a test message to yourself or ask another user to transmit."
              action={
                <Link href="/send" className="btn-primary">
                  <Send className="w-4 h-4" />
                  Send a Message
                </Link>
              }
            />
          </Card>
        ) : (
          <div className="space-y-3">
            {items.map((item) => {
              const complete = item.received_packets >= item.total_items;
              const detail = decoded[item.session_id];
              const isDecoding = busySession === item.session_id;

              return (
                <motion.article
                  key={item.session_id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <Card className="overflow-hidden">
                      <div className="flex items-start justify-between gap-4 p-4 flex-wrap sm:flex-nowrap">
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-3 flex-wrap">
                          <div className="flex items-center gap-2">
                            <User className="w-5 h-5 text-muted" />
                            <span className="font-medium text-foreground">
                              {item.sender_username || '(unknown)'}
                            </span>
                          </div>
                          {item.mode_used && (
                            <Tooltip content={`Transmission mode: ${item.mode_used.toUpperCase()}`}>
                              <Badge variant={getModeColor(item.mode_used)} dot size="sm">
                                {item.mode_used.toUpperCase()}
                              </Badge>
                            </Tooltip>
                          )}
                          <span className="text-body-xs text-muted px-2 py-0.5 bg-surface rounded">
                            #{item.session_id.slice(0, 8)}
                          </span>
                        </div>
                        <div className="flex items-center gap-4 mt-2 text-body-sm text-muted flex-wrap">
                          <span className="flex items-center gap-1">
                            <Clock className="w-3.5 h-3.5" />
                            {new Date(item.timestamp * 1000).toLocaleString()}
                          </span>
                          <span className="flex items-center gap-1 font-mono tabular-nums">
                            <MessageSquare className="w-3.5 h-3.5" />
                            {item.received_packets}/{item.total_items} packets
                          </span>
                          {!complete && (
                            <span className="flex items-center gap-1 text-warning">
                              <AlertCircle className="w-3.5 h-3.5" />
                              Incomplete
                            </span>
                          )}
                        </div>
                      </div>

                      <div className="flex items-center gap-2 shrink-0">
                        <Tooltip content="Decode message">
                          <button
                            onClick={() => onDecode(item.session_id)}
                            disabled={!complete || isDecoding}
                            className="btn-primary btn-sm gap-2"
                          >
                            {isDecoding ? (
                              <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                Decoding…
                              </>
                            ) : detail ? (
                              <>
                                <Eye className="w-4 h-4" />
                                View
                              </>
                            ) : (
                              <>
                                <Shield className="w-4 h-4" />
                                Decode
                              </>
                            )}
                          </button>
                        </Tooltip>
                        <Tooltip content="Delete message">
                          <button
                            onClick={() => onDelete(item.session_id)}
                            disabled={isDecoding}
                            className="btn-ghost btn-sm text-error hover:text-error hover:bg-error/10"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </Tooltip>
                      </div>
                    </div>

                    {detail && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="border-t border-border bg-surface/50"
                      >
                        <div className="p-4 space-y-3">
                          <div className="flex items-center justify-between gap-3 flex-wrap">
                            <h4 className="font-semibold text-foreground">Decoded Message</h4>
                            <div className="flex items-center gap-2 text-body-xs flex-wrap">
                              <Badge variant={detail.ecc_success ? 'success' : 'error'} dot size="sm">
                                {detail.ecc_success ? 'ECC OK' : 'ECC FAILED'}
                              </Badge>
                              <Badge variant="info" size="sm">
                                {(typeof detail.verification_rate === 'number' ? (detail.verification_rate * 100).toFixed(1) : '100.0')}% verified
                              </Badge>
                              {Array.isArray(detail.ecc_errors_fixed) && detail.ecc_errors_fixed.length > 0 && (
                                <Badge variant="warning" size="sm">
                                  Fixed {detail.ecc_errors_fixed.length} errors
                                </Badge>
                              )}
                              {detail.cached && (
                                <Badge variant="neutral" size="sm">Cached</Badge>
                              )}
                            </div>
                          </div>
                          <div className="bg-background border border-border rounded-lg p-4 font-mono text-body-sm text-foreground whitespace-pre-wrap max-h-60 overflow-y-auto">
                            {detail.reconstructed_meaning || '(empty)'}
                          </div>
                          <div className="flex items-center gap-4 text-body-xs text-muted flex-wrap">
                            {typeof detail.elapsed_ms === 'number' && (
                              <span>Decoded in {detail.elapsed_ms.toFixed(1)}ms</span>
                            )}
                            {Array.isArray(detail.ecc_errors_fixed) && detail.ecc_errors_fixed.length > 0 && (
                              <span>Corrected errors: {detail.ecc_errors_fixed.join(', ')}</span>
                            )}
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </Card>
                </motion.article>
              );
            })}
          </div>
        )}
      </div>
    </PageWrapper>
  );
}