'use client';

import React, { useState, useEffect, useCallback, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import {
  Send,
  Trash2,
  Shuffle,
  Unlock,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  HardDrive,
  RefreshCw,
  Image as ImageIcon,
  FileText,
  Volume2,
  Activity,
  Inbox,
  Terminal,
  Eye,
  RotateCcw,
} from 'lucide-react';

import {
  PageWrapper,
  Card,
  Badge,
  Button,
  LoadingSpinner,
  StatGrid,
  CarrierPreviewModal,
  CarrierPreviewTarget,
  useToastHelpers,
} from '@/components';
import { useRequireAuth } from '@/lib/auth';
import {
  wireSendSession,
  getWireSessions,
  getWireSession,
  wireDropPacket,
  wireDropBatch,
  wireTamperPacket,
  wireRestorePackets,
  wireDecodeSession,
  wireDeleteSession,
  listUsers,
  WireSessionSnapshot,
  WireSessionDecodeResult,
  WireSlot,
  AuthUser,
  getErrorMessage,
} from '@/lib/api';

const PRESET_MESSAGES = [
  'Classified directive: Exfiltrate via Channel 3 at 0400Z.',
  'Meet at the north gate at 21:00 with the signed ledger.',
  'Reed-Solomon recovers 100% of dropped packets on the wire.',
];

const PARITY_CHOICES = [4, 8, 12, 16];

function ModalityLabel({ modality }: { modality: string }) {
  if (modality === 'image') {
    return (
      <span className="inline-flex items-center gap-1 text-[10px] font-mono text-info">
        <ImageIcon className="w-3 h-3" />
        IMG
      </span>
    );
  }
  if (modality === 'audio') {
    return (
      <span className="inline-flex items-center gap-1 text-[10px] font-mono text-purple">
        <Volume2 className="w-3 h-3" />
        AUD
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 text-[10px] font-mono text-success">
      <FileText className="w-3 h-3" />
      TXT
    </span>
  );
}

function ErasureLabContent() {
  const { user, loading: authLoading } = useRequireAuth();
  const searchParams = useSearchParams();
  const { success, error: showError, info } = useToastHelpers();

  const [message, setMessage] = useState(PRESET_MESSAGES[0]);
  const [engineMode, setEngineMode] = useState<'exact_vcp' | 'dssc'>('exact_vcp');
  const [parityBytes, setParityBytes] = useState<number>(8);
  const [diversityMode, setDiversityMode] = useState<'round_robin' | 'balanced' | 'best'>('round_robin');
  const [recipient, setRecipient] = useState<string>('');
  const [peers, setPeers] = useState<AuthUser[]>([]);
  const [sending, setSending] = useState(false);

  const [sessions, setSessions] = useState<WireSessionSnapshot[]>([]);
  const [activeSession, setActiveSession] = useState<WireSessionSnapshot | null>(null);
  const [loadingSessions, setLoadingSessions] = useState(true);

  const [clickAction, setClickAction] = useState<'inspect' | 'drop' | 'tamper'>('inspect');
  const [previewTarget, setPreviewTarget] = useState<CarrierPreviewTarget | null>(null);
  const [mutatingSeq, setMutatingSeq] = useState<number | null>(null);
  const [batchDropping, setBatchDropping] = useState(false);
  const [restoring, setRestoring] = useState(false);
  const [deletedLog, setDeletedLog] = useState<Array<{ action: 'rm' | 'restore'; path: string }>>([]);

  const [decoding, setDecoding] = useState(false);
  const [decodeResult, setDecodeResult] = useState<WireSessionDecodeResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadSessions = useCallback(
    async (preferSessionId?: string) => {
      try {
        const res = await getWireSessions();
        setSessions(res.sessions);
        const targetId = preferSessionId || searchParams.get('session');
        if (targetId) {
          const found = res.sessions.find((s) => s.session_id === targetId);
          if (found) {
            setActiveSession(found);
            return;
          }
        }
        setActiveSession((prev) => {
          if (prev) {
            const refreshed = res.sessions.find((s) => s.session_id === prev.session_id);
            if (refreshed) return refreshed;
          }
          return res.sessions[0] || null;
        });
      } catch (err) {
        setError(getErrorMessage(err, 'Failed to read wire sessions from disk'));
      } finally {
        setLoadingSessions(false);
      }
    },
    [searchParams],
  );

  useEffect(() => {
    if (!user) return;
    loadSessions();
    listUsers(false)
      .then((res) => {
        setPeers(res.users);
        const defaultPeer =
          res.users.find((u) => u.username !== user.username)?.username || user.username;
        setRecipient((prev) => prev || defaultPeer);
      })
      .catch(() => {});
  }, [user, loadSessions]);

  const handleTransmitToDisk = async () => {
    const clean = message.trim();
    if (!clean) return;
    setSending(true);
    setError(null);
    setDecodeResult(null);
    setDeletedLog([]);
    try {
      const snap = await wireSendSession({
        message: clean,
        recipient_username: recipient || undefined,
        engine_mode: engineMode,
        diversity_mode: diversityMode,
        ecc_parity_bytes: parityBytes,
        stealth_mode: 'auto',
      });
      setActiveSession(snap);
      await loadSessions(snap.session_id);
      success(
        'Transmitted to Wire',
        `Wrote ${snap.total_items} real packet JSON files to ${snap.channel_dir}/`,
      );
    } catch (err) {
      const msg = getErrorMessage(err, 'Transmission failed');
      setError(msg);
      showError('Transmission failed', msg);
    } finally {
      setSending(false);
    }
  };

  const handleSelectSession = async (sessionId: string) => {
    setError(null);
    setDecodeResult(null);
    setDeletedLog([]);
    try {
      const snap = await getWireSession(sessionId);
      setActiveSession(snap);
    } catch (err) {
      showError('Failed to load session', getErrorMessage(err));
    }
  };

  const executeDropSeq = async (seq: number) => {
    if (!activeSession) return;
    setMutatingSeq(seq);
    setError(null);
    try {
      const updated = await wireDropPacket(activeSession.session_id, seq);
      if (updated.deleted_files?.length) {
        setDeletedLog((prev) =>
          [...updated.deleted_files!.map((path) => ({ action: 'rm' as const, path })), ...prev].slice(0, 20),
        );
      }
      setActiveSession(updated);
      setSessions((prev) =>
        prev.map((s) => (s.session_id === updated.session_id ? updated : s)),
      );
      setDecodeResult(null);
      setPreviewTarget(null);
      info('Packet Deleted from Disk', `Unlinked slot #${seq} from ${updated.channel_dir}/`);
    } catch (err) {
      showError('Delete failed', getErrorMessage(err));
    } finally {
      setMutatingSeq(null);
    }
  };

  const executeTamperSeq = async (seq: number) => {
    if (!activeSession) return;
    setMutatingSeq(seq);
    setError(null);
    try {
      const updated = await wireTamperPacket(activeSession.session_id, seq);
      setActiveSession(updated);
      setSessions((prev) =>
        prev.map((s) => (s.session_id === updated.session_id ? updated : s)),
      );
      setDecodeResult(null);
      setPreviewTarget(null);
      info('Packet Tampered on Disk', `Overwrote slot #${seq} media_id with a wrong-cluster carrier`);
    } catch (err) {
      showError('Tamper failed', getErrorMessage(err));
    } finally {
      setMutatingSeq(null);
    }
  };

  const executeRestore = async (sequences?: number[]) => {
    if (!activeSession) return;
    if (sequences && sequences.length === 1) {
      setMutatingSeq(sequences[0]);
    } else {
      setRestoring(true);
    }
    setError(null);
    try {
      const updated = await wireRestorePackets(activeSession.session_id, sequences);
      if (updated.restored_files?.length) {
        setDeletedLog((prev) =>
          [...updated.restored_files!.map((path) => ({ action: 'restore' as const, path })), ...prev].slice(0, 20),
        );
      }
      setActiveSession(updated);
      setSessions((prev) =>
        prev.map((s) => (s.session_id === updated.session_id ? updated : s)),
      );
      setDecodeResult(null);
      success(
        sequences ? `Restored Packet #${sequences[0]}` : 'Restored All Packets',
        `Restored ${updated.restored_files?.length || 0} file(s) on disk in ${updated.channel_dir}/`,
      );
    } catch (err) {
      showError('Restore failed', getErrorMessage(err));
    } finally {
      setMutatingSeq(null);
      setRestoring(false);
    }
  };

  const handleSlotClick = async (pkt: WireSlot) => {
    if (!activeSession) return;
    if (pkt.status === 'deleted') {
      await executeRestore([pkt.seq]);
      return;
    }
    if (clickAction === 'inspect' && pkt.media_id) {
      setPreviewTarget({
        mediaId: pkt.media_id,
        packetPath: pkt.rel_path,
        seq: pkt.seq,
        channelId: pkt.channel_id,
        role: pkt.role,
        status: pkt.status,
      });
      return;
    }
    if (clickAction === 'drop') {
      await executeDropSeq(pkt.seq);
    } else if (clickAction === 'tamper') {
      await executeTamperSeq(pkt.seq);
    }
  };

  const handleQuickDeleteCount = async (count: number) => {
    if (!activeSession) return;
    const intactPayload = activeSession.packets
      .filter((p) => p.status === 'on_disk' && p.role === 'payload')
      .map((p) => p.seq);
    const intactAny = activeSession.packets
      .filter((p) => p.status === 'on_disk')
      .map((p) => p.seq);
    const pool = intactPayload.length >= count ? intactPayload : intactAny;
    if (pool.length === 0) {
      showError('No packets left', 'All packets in this session are already deleted.');
      return;
    }

    const toDelete: number[] = [];
    const n = Math.min(count, pool.length);
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(((i + 0.5) * pool.length) / n);
      const seq = pool[Math.min(idx, pool.length - 1)];
      if (!toDelete.includes(seq)) toDelete.push(seq);
    }
    for (const seq of pool) {
      if (toDelete.length >= n) break;
      if (!toDelete.includes(seq)) toDelete.push(seq);
    }

    setBatchDropping(true);
    setError(null);
    try {
      const updated = await wireDropBatch(activeSession.session_id, toDelete);
      if (updated.deleted_files?.length) {
        setDeletedLog((prev) =>
          [...updated.deleted_files!.map((path) => ({ action: 'rm' as const, path })), ...prev].slice(0, 20),
        );
      }
      setActiveSession(updated);
      setSessions((prev) =>
        prev.map((s) => (s.session_id === updated.session_id ? updated : s)),
      );
      setDecodeResult(null);
      info(
        `Deleted ${toDelete.length} File(s) from Wire`,
        `Unlinked ${toDelete.length} JSON packet file(s) from ${updated.channel_dir}/`,
      );
    } catch (err) {
      showError('Packet drop failed', getErrorMessage(err));
    } finally {
      setBatchDropping(false);
    }
  };

  const handleDecodeFromDisk = async () => {
    if (!activeSession) return;
    setDecoding(true);
    setError(null);
    try {
      const res = await wireDecodeSession(activeSession.session_id);
      setDecodeResult(res);
      await loadSessions(activeSession.session_id);
      if (res.ecc_success) {
        success(
          '100% Exact Recovery from Disk',
          `Recovered message from ${res.surviving_packets}/${res.total_packets} surviving files in ${res.elapsed_ms}ms`,
        );
      } else {
        showError(
          'Fail-Closed Rejection',
          `Loss exceeded R=${res.ecc_parity_bytes} parity budget; CRC-16 cleanly rejected.`,
        );
      }
    } catch (err) {
      const msg = getErrorMessage(err, 'Decode failed');
      setError(msg);
      showError('Decode failed', msg);
    } finally {
      setDecoding(false);
    }
  };

  const handleDeleteSession = async (sessionId: string) => {
    try {
      await wireDeleteSession(sessionId);
      if (activeSession?.session_id === sessionId) {
        setActiveSession(null);
        setDecodeResult(null);
        setDeletedLog([]);
      }
      await loadSessions();
      info('Session Removed', `Cleaned up ${sessionId} from disk`);
    } catch (err) {
      showError('Delete failed', getErrorMessage(err));
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

  const parity = activeSession ? activeSession.ecc_parity_bytes : parityBytes;
  const isExactVcp = activeSession ? activeSession.engine_mode === 'exact_vcp' : engineMode === 'exact_vcp';
  const hasExactWeight =
    activeSession != null && typeof activeSession.exact_fault_weight === 'number';
  const estFaultWeight = (() => {
    if (!activeSession) return 0;
    if (typeof activeSession.exact_fault_weight === 'number') {
      return activeSession.exact_fault_weight;
    }
    if (isExactVcp) {
      return activeSession.dropped_count + activeSession.tampered_count * 2;
    }
    const bpc = activeSession.bits_per_carrier || 13.5;
    const erasedBytes = new Set<number>();
    for (const seq of activeSession.dropped_sequences || []) {
      const bitStart = Math.floor(seq * bpc);
      const bitEnd = Math.max(bitStart, Math.ceil((seq + 1) * bpc) - 1);
      for (let b = Math.floor(bitStart / 8); b <= Math.floor(bitEnd / 8); b++) {
        erasedBytes.add(b);
      }
    }
    const tamperedBytes = new Set<number>();
    for (const seq of activeSession.tampered_sequences || []) {
      const bitStart = Math.floor(seq * bpc);
      const bitEnd = Math.max(bitStart, Math.ceil((seq + 1) * bpc) - 1);
      for (let b = Math.floor(bitStart / 8); b <= Math.floor(bitEnd / 8); b++) {
        if (!erasedBytes.has(b)) tamperedBytes.add(b);
      }
    }
    return erasedBytes.size + tamperedBytes.size * 2;
  })();
  const displayedFaultWeight = hasExactWeight
    ? estFaultWeight
    : decodeResult && decodeResult.ecc_success
    ? decodeResult.repaired_byte_indices.length
    : estFaultWeight;
  const dsscExceedTarget = activeSession
    ? Math.ceil((parity * 8) / (activeSession.bits_per_carrier || 13.5)) + 1
    : 6;
  const overBudget = activeSession
    ? decodeResult
      ? !decodeResult.ecc_success
      : estFaultWeight > parity
    : false;

  return (
    <PageWrapper
      title="RS-ECC Erasure Lab"
      description="Real on-disk packet inspection, physical file deletion, and Reed-Solomon recovery over storage/shared_channel"
      action={
        <div className="flex items-center gap-2">
          <Link href="/wire" className="btn-secondary btn-sm">
            <Activity className="w-3.5 h-3.5" />
            Open Live Wire
          </Link>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => loadSessions(activeSession?.session_id)}
            leftIcon={<RefreshCw className="w-3.5 h-3.5" />}
          >
            Sync Disk
          </Button>
        </div>
      }
    >
      <div className="space-y-6 pb-12">
        {error && (
          <div className="p-4 rounded-xl bg-error-muted border border-error/30 text-error text-body-sm flex items-center gap-2.5">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Step 1: Transmit a real session onto disk */}
        <Card
          title="1. Transmit Real Packet Files onto the Wire"
          subtitle="Encodes your message with CRC-16 + RS-ECC parity and writes individual packet JSON files into storage/shared_channel/"
        >
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
            <div className="lg:col-span-7 space-y-3">
              <label className="label">Message to Transmit</label>
              <textarea
                rows={3}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Type any secret message to transmit onto the shared channel..."
                className="input font-mono text-body-sm resize-none"
              />
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-body-xs text-muted">Presets:</span>
                {PRESET_MESSAGES.map((preset, i) => (
                  <button
                    key={i}
                    type="button"
                    onClick={() => setMessage(preset)}
                    className="px-2.5 py-1 rounded-lg bg-surface-muted hover:bg-surface-hover border border-border text-body-xs text-muted hover:text-foreground transition-colors"
                  >
                    Sample {i + 1}
                  </button>
                ))}
              </div>
            </div>

            <div className="lg:col-span-5 space-y-3.5">
              <div className="grid grid-cols-2 gap-2.5">
                <div>
                  <label className="label">Codec Mode</label>
                  <div className="grid grid-cols-2 gap-1.5">
                    {(['exact_vcp', 'dssc'] as const).map((m) => (
                      <button
                        key={m}
                        type="button"
                        onClick={() => setEngineMode(m)}
                        className={`px-2.5 py-2 rounded-xl border text-xs font-medium transition-all ${
                          engineMode === m
                            ? 'border-primary bg-primary-muted text-primary font-semibold'
                            : 'border-border bg-surface-muted/60 text-muted hover:text-foreground'
                        }`}
                      >
                        {m === 'exact_vcp' ? 'Exact VCP' : 'DSSC'}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="label">Recipient Inbox</label>
                  <select
                    value={recipient}
                    onChange={(e) => setRecipient(e.target.value)}
                    className="input py-2 text-body-xs font-mono"
                  >
                    {peers.map((p) => (
                      <option key={p.id} value={p.username}>
                        @{p.username} {p.username === user.username ? '(self)' : ''}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2.5">
                <div>
                  <label className="label">RS Parity Bytes (R)</label>
                  <div className="flex gap-1.5">
                    {PARITY_CHOICES.map((r) => (
                      <button
                        key={r}
                        type="button"
                        onClick={() => setParityBytes(r)}
                        className={`flex-1 py-1.5 rounded-lg border font-mono text-xs font-semibold transition-all ${
                          parityBytes === r
                            ? 'border-primary bg-primary-muted text-primary'
                            : 'border-border bg-surface-muted/60 text-muted hover:text-foreground'
                        }`}
                      >
                        {r}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="label">Modality Spread</label>
                  <select
                    value={diversityMode}
                    onChange={(e) =>
                      setDiversityMode(e.target.value as 'round_robin' | 'balanced' | 'best')
                    }
                    className="input py-1.5 text-body-xs font-mono"
                  >
                    <option value="round_robin">round_robin</option>
                    <option value="balanced">balanced</option>
                    <option value="best">best</option>
                  </select>
                </div>
              </div>

              <div className="p-3 rounded-xl bg-surface-muted/70 border border-border text-body-xs flex items-center justify-between">
                <span className="text-muted">
                  Singleton bound: <strong className="font-mono text-foreground">2e + v &le; {parityBytes}B</strong>
                </span>
                <Button
                  variant="primary"
                  size="sm"
                  onClick={handleTransmitToDisk}
                  loading={sending}
                  leftIcon={<Send className="w-3.5 h-3.5" />}
                >
                  Transmit to Disk
                </Button>
              </div>
            </div>
          </div>
        </Card>

        {/* Active On-Disk Sessions Strip */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-body-xs font-mono uppercase tracking-wider text-muted mr-1">
            Sessions on Disk ({sessions.length}):
          </span>
          {loadingSessions ? (
            <LoadingSpinner size="sm" />
          ) : sessions.length === 0 ? (
            <span className="text-body-xs text-muted">
              None yet. Transmit above or send a message in Chat.
            </span>
          ) : (
            sessions.slice(0, 8).map((s) => {
              const selected = activeSession?.session_id === s.session_id;
              return (
                <button
                  key={s.session_id}
                  type="button"
                  onClick={() => handleSelectSession(s.session_id)}
                  className={`px-3 py-1.5 rounded-xl border text-body-xs font-mono transition-all flex items-center gap-2 ${
                    selected
                      ? 'bg-surface border-primary text-foreground shadow-subtle font-semibold'
                      : 'bg-surface-muted/60 border-border text-muted hover:text-foreground'
                  }`}
                >
                  <span>{s.session_id}</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-surface-muted border border-border">
                    {s.surviving_count}/{s.total_items}
                  </span>
                  {s.dropped_count > 0 && (
                    <span className="text-[10px] text-error font-semibold">-{s.dropped_count}</span>
                  )}
                </button>
              );
            })
          )}
        </div>

        {/* Step 2 & 3: Active On-Disk Packet Inspector + Real Deletion + Real Recovery */}
        {activeSession && (
          <div className="space-y-6">
            <Card
              title="2. Inspect, Delete, or Tamper Real Packet Files on Disk"
              subtitle={`Channel directory: ${activeSession.channel_dir}/  ·  ${activeSession.sender_username} → ${activeSession.recipient_username}  ·  ${activeSession.engine_mode.toUpperCase()} (R=${activeSession.ecc_parity_bytes}B)`}
              action={
                <div className="flex items-center gap-2 flex-wrap">
                  <div className="inline-flex rounded-xl bg-surface-muted p-0.5 border border-border">
                    <button
                      type="button"
                      onClick={() => setClickAction('inspect')}
                      className={`px-2.5 py-1 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-all ${
                        clickAction === 'inspect'
                          ? 'bg-surface text-foreground shadow-subtle font-semibold'
                          : 'text-muted hover:text-foreground'
                      }`}
                    >
                      <Eye className="w-3.5 h-3.5 text-primary" />
                      Inspect / View
                    </button>
                    <button
                      type="button"
                      onClick={() => setClickAction('drop')}
                      className={`px-2.5 py-1 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-all ${
                        clickAction === 'drop'
                          ? 'bg-surface text-error shadow-subtle font-semibold'
                          : 'text-muted hover:text-foreground'
                      }`}
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                      Quick Delete
                    </button>
                    <button
                      type="button"
                      onClick={() => setClickAction('tamper')}
                      className={`px-2.5 py-1 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-all ${
                        clickAction === 'tamper'
                          ? 'bg-surface text-warning shadow-subtle font-semibold'
                          : 'text-muted hover:text-foreground'
                      }`}
                    >
                      <Shuffle className="w-3.5 h-3.5" />
                      Quick Tamper
                    </button>
                  </div>
                  <button
                    type="button"
                    onClick={() => handleDeleteSession(activeSession.session_id)}
                    className="btn-ghost btn-sm text-muted hover:text-error"
                    title="Delete entire session from disk"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              }
            >
              {/* Quick Physical Drop Toolbar */}
              <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                <div className="flex flex-wrap items-center gap-2">
                  <Button
                    variant="secondary"
                    size="sm"
                    disabled={batchDropping || activeSession.surviving_count === 0}
                    onClick={() => handleQuickDeleteCount(1)}
                  >
                    <Trash2 className="w-3.5 h-3.5 text-error" />
                    Delete 1 Packet
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    disabled={batchDropping || activeSession.surviving_count < 2}
                    onClick={() => handleQuickDeleteCount(2)}
                  >
                    <Trash2 className="w-3.5 h-3.5 text-error" />
                    Delete 2 Packets
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    disabled={batchDropping || activeSession.surviving_count < 3}
                    onClick={() => handleQuickDeleteCount(3)}
                  >
                    <Trash2 className="w-3.5 h-3.5 text-error" />
                    Delete 3 Packets
                  </Button>
                  {isExactVcp && activeSession.dropped_count < parity && (
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={batchDropping}
                      onClick={() => handleQuickDeleteCount(parity - activeSession.dropped_count)}
                    >
                      Drop up to Max (R={parity})
                    </Button>
                  )}
                  <Button
                    variant="secondary"
                    size="sm"
                    disabled={batchDropping || activeSession.surviving_count === 0}
                    onClick={() =>
                      handleQuickDeleteCount(
                        Math.max(1, (isExactVcp ? parity + 1 : dsscExceedTarget) - activeSession.dropped_count),
                      )
                    }
                  >
                    Exceed Limit (Fail-Closed)
                  </Button>
                  {(activeSession.dropped_count > 0 || activeSession.tampered_count > 0) && (
                    <Button
                      variant="success"
                      size="sm"
                      loading={restoring}
                      onClick={() => executeRestore()}
                      leftIcon={<RotateCcw className="w-3.5 h-3.5" />}
                    >
                      Restore All Packets
                    </Button>
                  )}
                </div>

                <div className="flex items-center gap-2 text-body-xs font-mono">
                  <Badge variant={activeSession.dropped_count > 0 ? 'warning' : 'success'} dot>
                    {activeSession.surviving_count} / {activeSession.total_items} files on disk
                  </Badge>
                  {activeSession.dropped_count > 0 && (
                    <Badge variant="error">{activeSession.dropped_count} deleted</Badge>
                  )}
                  {activeSession.tampered_count > 0 && (
                    <Badge variant="warning">{activeSession.tampered_count} tampered</Badge>
                  )}
                </div>
              </div>

              {/* Capacity Meter */}
              <div className="p-3.5 rounded-xl bg-surface-muted/70 border border-border mb-4 space-y-2">
                <div className="flex flex-wrap items-center justify-between gap-2 text-body-xs">
                  <span className="font-medium text-foreground flex items-center gap-1.5">
                    <HardDrive className="w-3.5 h-3.5 text-primary" />
                    {clickAction === 'inspect'
                      ? 'Click any carrier tile to view its asset, disk location, SHA-256, or delete its file'
                      : clickAction === 'drop'
                      ? 'Quick Delete active: clicking a carrier immediately unlinks its packet file from disk'
                      : 'Quick Tamper active: clicking a carrier overwrites its media_id on disk'}
                  </span>
                  <span className={`font-mono font-semibold ${overBudget ? 'text-error' : 'text-success'}`}>
                    {decodeResult
                      ? decodeResult.ecc_success
                        ? `Recovered (${displayedFaultWeight} / ${parity}B RS parity used)`
                        : `Exceeded R=${parity}B parity limit (clean fail-closed)`
                      : overBudget
                      ? `${hasExactWeight || isExactVcp ? '' : 'Est. ~'}${estFaultWeight}B > R=${parity}B limit (expect fail-closed)`
                      : `${hasExactWeight || isExactVcp ? '' : 'Est. ~'}${estFaultWeight} / ${parity}B RS-ECC budget`}
                  </span>
                </div>
                <div className="h-2 rounded-full bg-surface border border-border overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-300 ${
                      overBudget ? 'bg-error' : activeSession.dropped_count > 0 ? 'bg-primary' : 'bg-success'
                    }`}
                    style={{
                      width: `${Math.min(100, Math.round((displayedFaultWeight / Math.max(1, parity)) * 100))}%`,
                    }}
                  />
                </div>
              </div>

              {/* Real Packet Files Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2 max-h-[340px] overflow-y-auto pr-1">
                {activeSession.packets.map((pkt) => {
                  const isDeleted = pkt.status === 'deleted';
                  const isTampered = pkt.status === 'tampered';
                  const isBusy = mutatingSeq === pkt.seq;
                  const roleTint =
                    pkt.role === 'header'
                      ? 'border-info/40 bg-info-muted/40'
                      : pkt.role === 'parity'
                      ? 'border-purple/40 bg-purple-muted/40'
                      : 'border-border bg-surface';

                  const tileClass = isDeleted
                    ? 'border-error/60 border-dashed bg-error-muted/60 text-error hover:border-success hover:bg-success-muted/40 hover:text-success cursor-pointer'
                    : isTampered
                    ? 'border-warning/70 bg-warning-muted/70 text-warning'
                    : `${roleTint} hover:border-primary hover:shadow-subtle`;

                  return (
                    <button
                      key={pkt.seq}
                      type="button"
                      disabled={isBusy}
                      onClick={() => handleSlotClick(pkt)}
                      title={
                        isDeleted
                          ? `Slot #${pkt.seq} deleted from disk — click to restore file back to disk`
                          : `${pkt.media_id} (${pkt.rel_path}) — Click to ${clickAction}`
                      }
                      className={`p-2 rounded-xl border text-left transition-all ${tileClass}`}
                    >
                      <div className="flex items-center justify-between gap-1 mb-1">
                        <span className="font-mono text-[11px] font-semibold">#{pkt.seq}</span>
                        {isBusy ? (
                          <span className="text-[9px] font-mono">...</span>
                        ) : isDeleted ? (
                          <span className="text-[9px] font-mono font-bold uppercase px-1.5 py-0.5 rounded bg-error/15 text-error">
                            GONE
                          </span>
                        ) : isTampered ? (
                          <span className="text-[9px] font-mono font-bold uppercase px-1.5 py-0.5 rounded bg-warning/20 text-warning">
                            MOD
                          </span>
                        ) : (
                          <ModalityLabel modality={pkt.modality} />
                        )}
                      </div>
                      <div className="font-mono text-[10px] truncate text-muted">
                        {isDeleted ? 'click to restore' : pkt.media_id}
                      </div>
                      <div className="flex items-center justify-between mt-1 pt-1 border-t border-border/60 font-mono text-[10px]">
                        <span className="uppercase text-muted">
                          {pkt.role === 'header' ? 'HDR' : pkt.role === 'parity' ? 'ECC' : 'MSG'}
                        </span>
                        <span className="text-foreground font-medium">
                          {isDeleted
                            ? '∅'
                            : pkt.bytes_hex.length > 0
                            ? `0x${pkt.bytes_hex[0]}`
                            : `CH${pkt.channel_id + 1}`}
                        </span>
                      </div>
                    </button>
                  );
                })}
              </div>

              {/* Filesystem Mutation Audit Log */}
              {deletedLog.length > 0 && (
                <div className="mt-4 p-3 rounded-xl bg-surface-muted border border-border font-mono text-[11px] space-y-1">
                  <div className="flex items-center gap-1.5 text-muted font-semibold uppercase tracking-wider text-[10px] mb-1">
                    <Terminal className="w-3.5 h-3.5 text-primary" />
                    Filesystem packet log (real disk operations)
                  </div>
                  <div className="max-h-24 overflow-y-auto space-y-0.5">
                    {deletedLog.map((entry, i) => (
                      <div
                        key={`${entry.path}_${i}`}
                        className={`truncate ${entry.action === 'rm' ? 'text-error' : 'text-success'}`}
                      >
                        {entry.action} {entry.path}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Execute Real Decode Action */}
              <div className="mt-5 pt-4 border-t border-border flex flex-wrap items-center justify-between gap-3">
                <div className="text-body-xs text-muted">
                  Receiver will scan <code className="text-foreground">{activeSession.channel_dir}/</code>,
                  detect missing sequence gaps automatically, and run GF(2&sup8;) RS-ECC erasure decoding.
                </div>
                <Button
                  variant="primary"
                  size="lg"
                  onClick={handleDecodeFromDisk}
                  loading={decoding}
                  leftIcon={<Unlock className="w-4 h-4" />}
                >
                  Decode from {activeSession.surviving_count} Surviving Disk Files
                </Button>
              </div>
            </Card>

            {/* Step 3: Real Recovery Proof */}
            {decodeResult && (
              <Card
                title="3. Receiver Recovery Proof (Read Directly from Disk)"
                subtitle={`Reassembled ${decodeResult.surviving_packets}/${decodeResult.total_packets} surviving files in ${decodeResult.channel_dir}/`}
                action={
                  <Link href="/inbox" className="btn-secondary btn-sm">
                    <Inbox className="w-3.5 h-3.5" />
                    Verify in Inbox
                  </Link>
                }
              >
                <div className="space-y-5">
                  <div
                    className={`p-5 rounded-2xl border flex flex-wrap items-center justify-between gap-4 ${
                      decodeResult.ecc_success
                        ? 'bg-success-muted border-success/30'
                        : 'bg-error-muted border-error/30'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      {decodeResult.ecc_success ? (
                        <CheckCircle2 className="w-6 h-6 text-success shrink-0 mt-0.5" />
                      ) : (
                        <XCircle className="w-6 h-6 text-error shrink-0 mt-0.5" />
                      )}
                      <div>
                        <h3
                          className={`font-serif text-heading-md font-medium ${
                            decodeResult.ecc_success ? 'text-success' : 'text-error'
                          }`}
                        >
                          {decodeResult.ecc_success
                            ? '100% Bit-Exact Message Recovered from Incomplete Wire Files'
                            : 'Fail-Closed Rejection: Deleted Files Exceeded RS-ECC Parity'}
                        </h3>
                        <p className="text-body-xs text-muted mt-1">
                          {decodeResult.ecc_success
                            ? `${decodeResult.dropped_packets} packet file(s) missing on disk${
                                decodeResult.tampered_packets > 0
                                  ? ` and ${decodeResult.tampered_packets} tampered`
                                  : ''
                              }. RS-ECC reconstructed ${decodeResult.repaired_byte_indices.length} byte(s) and passed CRC-16 verification.`
                            : `${decodeResult.dropped_packets} missing file(s) and ${decodeResult.tampered_packets} tampered file(s) exceeded R=${decodeResult.ecc_parity_bytes} parity bytes. CRC-16 refused to emit corrupted output.`}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant={decodeResult.ecc_success ? 'success' : 'error'} size="lg">
                        {decodeResult.ecc_success ? 'CRC-16 VALID' : 'CRC REJECTED'}
                      </Badge>
                      <Badge variant="neutral" size="lg">
                        {decodeResult.elapsed_ms} ms
                      </Badge>
                    </div>
                  </div>

                  <StatGrid
                    columns={4}
                    stats={[
                      {
                        label: 'Files on Disk',
                        value: `${decodeResult.surviving_packets} / ${decodeResult.total_packets}`,
                        variant: decodeResult.dropped_packets > 0 ? 'warning' : 'success',
                      },
                      {
                        label: 'Deleted Packets',
                        value: decodeResult.dropped_packets,
                        variant: decodeResult.dropped_packets > 0 ? 'error' : 'neutral',
                      },
                      {
                        label: 'RS-ECC Bytes Repaired',
                        value: decodeResult.repaired_byte_indices.length,
                        variant: decodeResult.ecc_success ? 'primary' : 'error',
                      },
                      {
                        label: 'RS Parity Budget',
                        value: `R = ${decodeResult.ecc_parity_bytes}B`,
                        variant: 'info',
                      },
                    ]}
                  />

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <div
                      className={`p-4 rounded-xl border ${
                        decodeResult.ecc_success
                          ? 'bg-success-muted/40 border-success/30'
                          : 'bg-error-muted/40 border-error/30'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-body-xs font-semibold uppercase tracking-wider text-success">
                          Recovered with RS-ECC Erasure Decoding
                        </span>
                        <Badge variant={decodeResult.ecc_success ? 'success' : 'error'} size="sm">
                          {decodeResult.ecc_success ? '100% EXACT' : 'REJECTED'}
                        </Badge>
                      </div>
                      <div className="p-3.5 rounded-xl bg-surface border border-border font-mono text-body-sm text-foreground break-words min-h-[72px]">
                        {decodeResult.ecc_success
                          ? decodeResult.reconstructed_message
                          : '(Rejected cleanly by RS-ECC + CRC-16 fail-closed guard)'}
                      </div>
                    </div>

                    <div className="p-4 rounded-xl border border-border bg-surface-muted/50">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-body-xs font-semibold uppercase tracking-wider text-warning">
                          Without RS-ECC (Raw Surviving Bytes)
                        </span>
                        <Badge variant="warning" size="sm">
                          UNPROTECTED
                        </Badge>
                      </div>
                      <div className="p-3.5 rounded-xl bg-surface border border-border font-mono text-body-sm text-muted break-words min-h-[72px]">
                        {decodeResult.unprotected_preview || '(no readable payload)'}
                      </div>
                    </div>
                  </div>

                  {decodeResult.repaired_byte_indices.length > 0 && (
                    <div className="p-3.5 rounded-xl bg-surface-muted/70 border border-border text-body-xs font-mono">
                      <span className="text-muted">Missing sequence slots detected on disk: </span>
                      <span className="text-error font-semibold">
                        [{decodeResult.dropped_sequences.join(', ') || 'none'}]
                      </span>
                      <span className="text-muted mx-2">&middot;</span>
                      <span className="text-muted">Codeword byte indices repaired by RS-ECC: </span>
                      <span className="text-success font-semibold">
                        [{decodeResult.repaired_byte_indices.join(', ')}]
                      </span>
                    </div>
                  )}
                </div>
              </Card>
            )}
          </div>
        )}
      </div>

      {/* Unified Carrier Asset Viewer + Direct Delete / Tamper Modal */}
      <CarrierPreviewModal
        target={previewTarget}
        onClose={() => setPreviewTarget(null)}
        onDeletePacket={executeDropSeq}
        onTamperPacket={executeTamperSeq}
        actionBusy={mutatingSeq !== null}
      />
    </PageWrapper>
  );
}

export default function ErasureLabPage() {
  return (
    <Suspense
      fallback={
        <PageWrapper>
          <div className="py-24 flex justify-center">
            <LoadingSpinner size="lg" />
          </div>
        </PageWrapper>
      }
    >
      <ErasureLabContent />
    </Suspense>
  );
}
