'use client';

import React, { useEffect, useState, useCallback } from 'react';
import Link from 'next/link';
import {
  Activity,
  Pause,
  Play,
  RefreshCw,
  Trash2,
  ShieldCheck,
  Eye,
  HardDrive,
  Network,
  AlertTriangle,
  Image as ImageIcon,
  FileText,
  Volume2,
  RotateCcw,
} from 'lucide-react';

import {
  PageWrapper,
  Card,
  Badge,
  Button,
  LoadingSpinner,
  ProgressBar,
  StatGrid,
  EmptyState,
  CarrierPreviewModal,
  CarrierPreviewTarget,
  useToastHelpers,
} from '@/components';
import { useRequireAuth } from '@/lib/auth';
import {
  getWireSessions,
  wireDropPacket,
  wireTamperPacket,
  wireRestorePackets,
  WireSessionSnapshot,
  getErrorMessage,
} from '@/lib/api';

export default function WirePage() {
  const { user, loading: authLoading } = useRequireAuth();
  const { info, error: showError } = useToastHelpers();

  const [sessions, setSessions] = useState<WireSessionSnapshot[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [isMonitoring, setIsMonitoring] = useState(true);
  const [loading, setLoading] = useState(true);
  const [droppingKey, setDroppingKey] = useState<string | null>(null);
  const [filterMode, setFilterMode] = useState<'all' | 'on_disk' | 'deleted' | 'ch0' | 'ch1' | 'ch2'>('all');
  const [previewTarget, setPreviewTarget] = useState<CarrierPreviewTarget | null>(null);

  const fetchSessions = useCallback(async () => {
    try {
      const res = await getWireSessions();
      setSessions(res.sessions);
      setSelectedId((prev) => {
        if (prev && res.sessions.some((s) => s.session_id === prev)) return prev;
        return res.sessions[0]?.session_id || null;
      });
    } catch {
      // silent during background poll
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!user) return;
    fetchSessions();
  }, [user, fetchSessions]);

  useEffect(() => {
    if (!user || !isMonitoring) return;
    const interval = setInterval(fetchSessions, 1500);
    return () => clearInterval(interval);
  }, [user, isMonitoring, fetchSessions]);

  const activeSession = sessions.find((s) => s.session_id === selectedId) || sessions[0] || null;

  const handleDropFromWire = async (sessionId: string, seq: number) => {
    const key = `${sessionId}_${seq}`;
    setDroppingKey(key);
    try {
      const updated = await wireDropPacket(sessionId, seq);
      setSessions((prev) => prev.map((s) => (s.session_id === sessionId ? updated : s)));
      setPreviewTarget(null);
      info('Packet Dropped on Wire', `Deleted slot #${seq} from ${updated.channel_dir}/`);
    } catch (err) {
      showError('Drop failed', getErrorMessage(err));
    } finally {
      setDroppingKey(null);
    }
  };

  const handleTamperFromWire = async (sessionId: string, seq: number) => {
    const key = `${sessionId}_${seq}`;
    setDroppingKey(key);
    try {
      const updated = await wireTamperPacket(sessionId, seq);
      setSessions((prev) => prev.map((s) => (s.session_id === sessionId ? updated : s)));
      setPreviewTarget(null);
      info('Packet Tampered on Wire', `Overwrote slot #${seq} media_id on disk`);
    } catch (err) {
      showError('Tamper failed', getErrorMessage(err));
    } finally {
      setDroppingKey(null);
    }
  };

  const handleRestoreOnWire = async (sessionId: string, sequences?: number[]) => {
    const key = sequences ? `${sessionId}_${sequences[0]}` : `${sessionId}_all`;
    setDroppingKey(key);
    try {
      const updated = await wireRestorePackets(sessionId, sequences);
      setSessions((prev) => prev.map((s) => (s.session_id === sessionId ? updated : s)));
      info(
        sequences ? `Restored Packet #${sequences[0]}` : 'Restored All Packets',
        `Restored ${updated.restored_files?.length || 0} file(s) onto ${updated.channel_dir}/`,
      );
    } catch (err) {
      showError('Restore failed', getErrorMessage(err));
    } finally {
      setDroppingKey(null);
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

  const totalSurviving = sessions.reduce((acc, s) => acc + s.surviving_count, 0);
  const totalDropped = sessions.reduce((acc, s) => acc + s.dropped_count, 0);
  const totalTampered = sessions.reduce((acc, s) => acc + s.tampered_count, 0);

  const filteredPackets = (activeSession?.packets || []).filter((pkt) => {
    if (filterMode === 'on_disk') return pkt.status !== 'deleted';
    if (filterMode === 'deleted') return pkt.status === 'deleted';
    if (filterMode === 'ch0') return pkt.channel_id % 3 === 0;
    if (filterMode === 'ch1') return pkt.channel_id % 3 === 1;
    if (filterMode === 'ch2') return pkt.channel_id % 3 === 2;
    return true;
  });

  const modalityCounts = (activeSession?.packets || []).reduce(
    (acc, pkt) => {
      if (pkt.status !== 'deleted') {
        const key = pkt.modality === 'image' || pkt.modality === 'audio' ? pkt.modality : 'text';
        acc[key] += 1;
      }
      return acc;
    },
    { image: 0, text: 0, audio: 0 },
  );

  return (
    <PageWrapper
      title="Live Wire"
      description="Real-time monitor of carrier packet files in storage/shared_channel with asset inspection and live drop controls"
      action={
        <div className="flex items-center gap-2">
          <Button
            variant={isMonitoring ? 'secondary' : 'success'}
            size="sm"
            onClick={() => setIsMonitoring(!isMonitoring)}
            leftIcon={isMonitoring ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
          >
            {isMonitoring ? 'Live Polling' : 'Paused'}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={fetchSessions}
            leftIcon={<RefreshCw className="w-3.5 h-3.5" />}
          >
            Refresh
          </Button>
        </div>
      }
    >
      <div className="space-y-6 pb-12">
        <StatGrid
          columns={4}
          stats={[
            { label: 'Active Sessions', value: sessions.length, variant: 'primary', icon: <HardDrive className="w-4 h-4" /> },
            { label: 'Packet Files on Disk', value: totalSurviving, variant: 'success', icon: <Activity className="w-4 h-4" /> },
            { label: 'Dropped (Deleted)', value: totalDropped, variant: totalDropped > 0 ? 'error' : 'neutral', icon: <Trash2 className="w-4 h-4" /> },
            { label: 'Tampered Files', value: totalTampered, variant: totalTampered > 0 ? 'warning' : 'neutral', icon: <AlertTriangle className="w-4 h-4" /> },
          ]}
        />

        {loading ? (
          <Card className="py-20">
            <LoadingSpinner size="lg" />
          </Card>
        ) : sessions.length === 0 ? (
          <Card>
            <EmptyState
              icon={<Network className="w-10 h-10" />}
              title="The wire is quiet"
              description="Send a message in Covert Chat or transmit a session in RS-ECC Lab to watch real packet JSON files arrive here."
              action={
                <Link href="/erasure" className="btn-primary">
                  <ShieldCheck className="w-4 h-4" />
                  Transmit in RS-ECC Lab
                </Link>
              }
            />
          </Card>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Session Selector + Dispersion Column */}
            <div className="lg:col-span-4 space-y-4">
              <Card title="Wire Sessions" subtitle="Newest transmissions first">
                <div className="space-y-2 max-h-[420px] overflow-y-auto pr-1">
                  {sessions.map((s) => {
                    const active = activeSession?.session_id === s.session_id;
                    return (
                      <button
                        key={s.session_id}
                        type="button"
                        onClick={() => setSelectedId(s.session_id)}
                        className={`w-full text-left p-3.5 rounded-xl border transition-all ${
                          active
                            ? 'bg-primary-muted/50 border-primary shadow-subtle'
                            : 'bg-surface-muted/40 border-border hover:bg-surface-muted'
                        }`}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-mono text-body-xs font-semibold text-foreground truncate">
                            {s.session_id}
                          </span>
                          <Badge variant={s.dropped_count > 0 ? 'error' : 'success'} size="sm">
                            {s.surviving_count}/{s.total_items}
                          </Badge>
                        </div>
                        <div className="text-body-xs text-muted mt-1 flex items-center justify-between">
                          <span>
                            @{s.sender_username} &rarr; @{s.recipient_username}
                          </span>
                          <span className="font-mono uppercase">{s.engine_mode}</span>
                        </div>
                        <div className="text-[11px] font-mono text-muted/80 mt-1 flex items-center justify-between">
                          <span>R={s.ecc_parity_bytes}B &middot; {s.mode_used}</span>
                          <span>
                            {s.timestamp
                              ? new Date(s.timestamp * 1000).toLocaleTimeString([], {
                                  hour: '2-digit',
                                  minute: '2-digit',
                                  second: '2-digit',
                                })
                              : '--'}
                          </span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </Card>

              {activeSession && (
                <>
                  <Card title="Channel Dispersion" subtitle="Surviving files per conduit">
                    <div className="space-y-3">
                      {['0', '1', '2'].map((ch) => {
                        const count = activeSession.channel_counts?.[ch] || 0;
                        const maxVal = Math.max(
                          1,
                          ...Object.values(activeSession.channel_counts || { '0': 1 }),
                        );
                        return (
                          <ProgressBar
                            key={ch}
                            label={`Channel ${Number(ch) + 1}`}
                            value={count}
                            max={maxVal}
                            showLabel
                            size="sm"
                            variant="primary"
                          />
                        );
                      })}
                    </div>
                  </Card>

                  <Card title="Modality Mix" subtitle="Surviving multimodal carriers">
                    <div className="grid grid-cols-3 gap-2 text-center">
                      <div className="p-2.5 rounded-xl bg-surface-muted/60 border border-border">
                        <ImageIcon className="w-4 h-4 text-info mx-auto mb-1" />
                        <div className="font-mono text-body-sm font-semibold text-foreground">
                          {modalityCounts.image}
                        </div>
                        <div className="text-[10px] font-mono text-muted">IMAGE</div>
                      </div>
                      <div className="p-2.5 rounded-xl bg-surface-muted/60 border border-border">
                        <FileText className="w-4 h-4 text-success mx-auto mb-1" />
                        <div className="font-mono text-body-sm font-semibold text-foreground">
                          {modalityCounts.text}
                        </div>
                        <div className="text-[10px] font-mono text-muted">TEXT</div>
                      </div>
                      <div className="p-2.5 rounded-xl bg-surface-muted/60 border border-border">
                        <Volume2 className="w-4 h-4 text-purple mx-auto mb-1" />
                        <div className="font-mono text-body-sm font-semibold text-foreground">
                          {modalityCounts.audio}
                        </div>
                        <div className="text-[10px] font-mono text-muted">AUDIO</div>
                      </div>
                    </div>
                  </Card>
                </>
              )}
            </div>

            {/* Active Session On-Disk Packet Feed */}
            <div className="lg:col-span-8">
              {activeSession && (
                <Card
                  title={`Session ${activeSession.session_id}`}
                  subtitle={`${activeSession.channel_dir}/  ·  ${activeSession.surviving_count} surviving, ${activeSession.dropped_count} deleted, ${activeSession.tampered_count} tampered`}
                  action={
                    <div className="flex items-center gap-2">
                      {(activeSession.dropped_count > 0 || activeSession.tampered_count > 0) && (
                        <Button
                          variant="success"
                          size="sm"
                          disabled={droppingKey !== null}
                          onClick={() => handleRestoreOnWire(activeSession.session_id)}
                          leftIcon={<RotateCcw className="w-3.5 h-3.5" />}
                        >
                          Restore All
                        </Button>
                      )}
                      <Link
                        href={`/erasure?session=${encodeURIComponent(activeSession.session_id)}`}
                        className="btn-primary btn-sm"
                      >
                        <ShieldCheck className="w-3.5 h-3.5" />
                        Recover in RS-ECC Lab
                      </Link>
                    </div>
                  }
                >
                  {/* Filter Pills */}
                  <div className="flex flex-wrap items-center gap-1.5 mb-3">
                    {[
                      { id: 'all', label: `All (${activeSession.packets.length})` },
                      { id: 'on_disk', label: `On Disk (${activeSession.surviving_count})` },
                      { id: 'deleted', label: `Deleted (${activeSession.dropped_count})` },
                      { id: 'ch0', label: 'Ch 1' },
                      { id: 'ch1', label: 'Ch 2' },
                      { id: 'ch2', label: 'Ch 3' },
                    ].map((f) => (
                      <button
                        key={f.id}
                        type="button"
                        onClick={() => setFilterMode(f.id as typeof filterMode)}
                        className={`px-2.5 py-1 rounded-lg text-body-xs font-medium border transition-colors ${
                          filterMode === f.id
                            ? 'bg-primary text-white border-primary'
                            : 'bg-surface-muted/50 text-muted border-border hover:text-foreground'
                        }`}
                      >
                        {f.label}
                      </button>
                    ))}
                  </div>

                  <div className="overflow-x-auto max-h-[600px] overflow-y-auto rounded-xl border border-border">
                    <table className="w-full text-left text-xs font-mono border-collapse">
                      <thead className="sticky top-0 bg-surface-muted border-b border-border text-muted uppercase">
                        <tr>
                          <th className="px-3 py-2.5">Seq</th>
                          <th className="px-3 py-2.5">Status</th>
                          <th className="px-3 py-2.5">Carrier ID (click to view)</th>
                          <th className="px-3 py-2.5">Conduit</th>
                          <th className="px-3 py-2.5">Disk File</th>
                          <th className="px-3 py-2.5 text-right">Actions</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border/70">
                        {filteredPackets.map((pkt) => {
                          const isDeleted = pkt.status === 'deleted';
                          const isTampered = pkt.status === 'tampered';
                          const busy = droppingKey === `${activeSession.session_id}_${pkt.seq}`;
                          return (
                            <tr
                              key={pkt.seq}
                              className={
                                isDeleted
                                  ? 'bg-error-muted/35 text-error'
                                  : isTampered
                                  ? 'bg-warning-muted/40'
                                  : 'hover:bg-surface-muted/40'
                              }
                            >
                              <td className="px-3 py-2.5 font-semibold">
                                #{String(pkt.seq).padStart(3, '0')}
                              </td>
                              <td className="px-3 py-2.5">
                                {isDeleted ? (
                                  <Badge variant="error" size="sm">DELETED</Badge>
                                ) : isTampered ? (
                                  <Badge variant="warning" size="sm">TAMPERED</Badge>
                                ) : (
                                  <Badge variant="success" size="sm">ON DISK</Badge>
                                )}
                              </td>
                              <td className="px-3 py-2.5 max-w-[180px] truncate">
                                {isDeleted || !pkt.media_id ? (
                                  <span className="italic opacity-80">missing sequence gap</span>
                                ) : (
                                  <button
                                    type="button"
                                    onClick={() =>
                                      setPreviewTarget({
                                        mediaId: pkt.media_id!,
                                        packetPath: pkt.rel_path,
                                        seq: pkt.seq,
                                        channelId: pkt.channel_id,
                                        role: pkt.role,
                                        status: pkt.status,
                                      })
                                    }
                                    className="text-foreground hover:text-primary font-semibold underline decoration-border hover:decoration-primary underline-offset-2 truncate max-w-full inline-flex items-center gap-1"
                                    title="View corpus file, path, SHA-256, and packet JSON"
                                  >
                                    <span className="truncate">{pkt.media_id}</span>
                                  </button>
                                )}
                              </td>
                              <td className="px-3 py-2.5 text-muted">Ch {pkt.channel_id + 1}</td>
                              <td className="px-3 py-2.5 max-w-[160px] truncate text-muted">
                                {pkt.filename || '—'}
                              </td>
                              <td className="px-3 py-2.5 text-right">
                                {isDeleted ? (
                                  <button
                                    type="button"
                                    disabled={busy}
                                    onClick={() => handleRestoreOnWire(activeSession.session_id, [pkt.seq])}
                                    className="px-2 py-1 rounded-lg bg-success-muted hover:bg-success hover:text-white border border-success/25 text-success inline-flex items-center gap-1 transition-colors"
                                    title="Restore deleted packet file back to disk"
                                  >
                                    <RotateCcw className="w-3 h-3" />
                                    Restore
                                  </button>
                                ) : pkt.media_id ? (
                                  <div className="inline-flex items-center gap-1.5">
                                    {isTampered && (
                                      <button
                                        type="button"
                                        disabled={busy}
                                        onClick={() => handleRestoreOnWire(activeSession.session_id, [pkt.seq])}
                                        className="px-2 py-1 rounded-lg bg-success-muted hover:bg-success hover:text-white border border-success/25 text-success inline-flex items-center gap-1 transition-colors"
                                        title="Revert tampered carrier back to original"
                                      >
                                        <RotateCcw className="w-3 h-3" />
                                        Revert
                                      </button>
                                    )}
                                    <button
                                      type="button"
                                      onClick={() =>
                                        setPreviewTarget({
                                          mediaId: pkt.media_id!,
                                          packetPath: pkt.rel_path,
                                          seq: pkt.seq,
                                          channelId: pkt.channel_id,
                                          role: pkt.role,
                                          status: pkt.status,
                                        })
                                      }
                                      className="px-2 py-1 rounded-lg bg-surface-muted hover:bg-surface-hover border border-border text-foreground inline-flex items-center gap-1"
                                      title="View carrier asset and packet JSON"
                                    >
                                      <Eye className="w-3 h-3 text-primary" />
                                      View
                                    </button>
                                    <button
                                      type="button"
                                      disabled={busy}
                                      onClick={() => handleDropFromWire(activeSession.session_id, pkt.seq)}
                                      className="px-2 py-1 rounded-lg bg-error-muted hover:bg-error hover:text-white border border-error/25 text-error inline-flex items-center gap-1 transition-colors"
                                      title="Physically delete this packet JSON from disk"
                                    >
                                      <Trash2 className="w-3 h-3" />
                                      Drop
                                    </button>
                                  </div>
                                ) : null}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </Card>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Unified Carrier Asset + Wire Packet Viewer */}
      <CarrierPreviewModal
        target={previewTarget}
        onClose={() => setPreviewTarget(null)}
        onDeletePacket={
          activeSession ? (seq) => handleDropFromWire(activeSession.session_id, seq) : undefined
        }
        onTamperPacket={
          activeSession ? (seq) => handleTamperFromWire(activeSession.session_id, seq) : undefined
        }
        actionBusy={droppingKey !== null}
      />
    </PageWrapper>
  );
}
