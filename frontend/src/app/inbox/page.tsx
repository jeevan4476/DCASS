'use client';

/**
 * Phase F — Inbox page.
 *
 * Lists sessions addressed to the current user (GET /api/inbox) and lets
 * them decode or delete each one. Decode is on-demand because it costs a
 * SemanticDecoder call; the result is cached server-side after the first
 * decode so repeat clicks are fast.
 */

import { useCallback, useEffect, useState } from 'react';
import Link from 'next/link';

import { useRequireAuth } from '@/lib/auth';
import {
  InboxItem,
  InboxDecodeResponse,
  decodeInboxSession,
  deleteInboxSession,
  getInbox,
} from '@/lib/api';

export default function InboxPage() {
  const { user, loading: authLoading, logout } = useRequireAuth();

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
      const message =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        (err as Error)?.message ||
        'Failed to load inbox';
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
      // Reflect the new decoded status in the list without a full refresh.
      setItems((prev) =>
        prev.map((it) =>
          it.session_id === sessionId ? { ...it, decoded: true } : it,
        ),
      );
    } catch (err: unknown) {
      const message =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        (err as Error)?.message ||
        'decode failed';
      setError(message);
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
    } catch (err: unknown) {
      const message =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        (err as Error)?.message ||
        'delete failed';
      setError(message);
    } finally {
      setBusySession(null);
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
      <div className="max-w-4xl mx-auto">
        <header className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Inbox</h1>
            <p className="text-sm text-gray-500">
              Messages addressed to <span className="font-medium">{user.username}</span>
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Link
              href="/send"
              className="text-sm text-blue-600 hover:underline"
            >
              Send a message
            </Link>
            <button
              onClick={logout}
              className="text-sm text-gray-500 hover:text-gray-700"
            >
              Sign out
            </button>
            <button
              onClick={refresh}
              disabled={loading}
              className="text-sm bg-white border border-gray-300 rounded px-3 py-1 hover:bg-gray-50"
            >
              {loading ? 'Refreshing…' : 'Refresh'}
            </button>
          </div>
        </header>

        {error && (
          <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded px-3 py-2 mb-4">
            {error}
          </div>
        )}

        {items.length === 0 && !loading ? (
          <div className="bg-white border border-gray-200 rounded p-8 text-center text-gray-500">
            Nothing in your inbox yet.{' '}
            <Link href="/send" className="text-blue-600 hover:underline">
              Send a test message
            </Link>
            .
          </div>
        ) : (
          <ul className="space-y-3">
            {items.map((item) => {
              const complete = item.received_packets >= item.total_items;
              const detail = decoded[item.session_id];
              return (
                <li
                  key={item.session_id}
                  className="bg-white border border-gray-200 rounded p-4"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="text-sm font-medium text-gray-900">
                        From {item.sender_username || '(unknown)'}
                        <span className="text-gray-400 font-normal">
                          {' '}· mode: {item.mode_used || 'unknown'}
                        </span>
                      </div>
                      <div className="text-xs text-gray-500 mt-0.5">
                        session <code>{item.session_id}</code> ·{' '}
                        {new Date(item.timestamp * 1000).toLocaleString()} ·{' '}
                        {item.received_packets}/{item.total_items} packets
                        {!complete && (
                          <span className="text-amber-600"> (incomplete)</span>
                        )}
                      </div>
                    </div>
                    <div className="flex gap-2 shrink-0">
                      <button
                        onClick={() => onDecode(item.session_id)}
                        disabled={!complete || busySession === item.session_id}
                        className="text-sm bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded px-3 py-1"
                      >
                        {busySession === item.session_id
                          ? 'Decoding…'
                          : item.decoded
                          ? 'Show decoded'
                          : 'Decode'}
                      </button>
                      <button
                        onClick={() => onDelete(item.session_id)}
                        disabled={busySession === item.session_id}
                        className="text-sm bg-white border border-gray-300 text-gray-700 hover:bg-gray-50 rounded px-3 py-1"
                      >
                        Delete
                      </button>
                    </div>
                  </div>

                  {detail && (
                    <div className="mt-3 border-t border-gray-100 pt-3 text-sm">
                      <div className="font-medium text-gray-900">Decoded message</div>
                      <div className="mt-1 text-gray-800 whitespace-pre-wrap">
                        {detail.reconstructed_meaning}
                      </div>
                      <div className="mt-2 text-xs text-gray-500">
                        verification {(detail.verification_rate * 100).toFixed(1)}% ·
                        ECC {detail.ecc_success ? 'OK' : 'FAILED'}
                        {detail.ecc_errors_fixed.length > 0 && (
                          <> · fixed {detail.ecc_errors_fixed.length} errors</>
                        )}
                        {detail.cached && <> · (cached)</>}
                      </div>
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
