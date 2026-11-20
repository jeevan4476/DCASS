'use client';

import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X,
  ExternalLink,
  Copy,
  Check,
  Image as ImageIcon,
  FileText,
  Volume2,
  HardDrive,
  FileCode,
  ShieldCheck,
  Trash2,
  Shuffle,
  AlertCircle,
  AlertTriangle,
} from 'lucide-react';

import { getMediaInfo, getMediaPacket, getApiBase, MediaInfoResponse, getErrorMessage } from '@/lib/api';
import { Badge, Button, LoadingSpinner } from './UI';

export interface CarrierPreviewTarget {
  mediaId: string;
  filePath?: string;
  packetPath?: string | null;
  seq?: number;
  channelId?: number;
  role?: string;
  status?: 'on_disk' | 'deleted' | 'tampered' | string;
}

interface CarrierPreviewModalProps {
  target: CarrierPreviewTarget | null;
  onClose: () => void;
  onDeletePacket?: (seq: number) => Promise<void> | void;
  onTamperPacket?: (seq: number) => Promise<void> | void;
  actionBusy?: boolean;
}

function formatBytes(bytes: number | null): string {
  if (bytes === null || bytes === undefined) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function CarrierPreviewModal({
  target,
  onClose,
  onDeletePacket,
  onTamperPacket,
  actionBusy = false,
}: CarrierPreviewModalProps) {
  const [info, setInfo] = useState<MediaInfoResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [activeTab, setActiveTab] = useState<'asset' | 'packet'>('asset');
  const [packetData, setPacketData] = useState<unknown | null>(null);
  const [loadingPacket, setLoadingPacket] = useState(false);
  const [copiedField, setCopiedField] = useState<string | null>(null);

  useEffect(() => {
    if (!target) {
      setInfo(null);
      setPacketData(null);
      setError(null);
      setActiveTab('asset');
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);
    setInfo(null);
    setActiveTab('asset');

    getMediaInfo(target.mediaId, target.filePath)
      .then((res) => {
        if (!cancelled) setInfo(res);
      })
      .catch(async (err) => {
        // Fallback for an already-running backend started before /api/media/info was added
        const status = (err as { response?: { status?: number } })?.response?.status;
        if (status === 404) {
          try {
            const url = `${getApiBase()}/media/file?media_id=${encodeURIComponent(target.mediaId)}${
              target.filePath ? `&path=${encodeURIComponent(target.filePath)}` : ''
            }`;
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const ct = resp.headers.get('content-type') || '';
            if (ct.startsWith('image/')) {
              if (!cancelled) {
                setInfo({
                  media_id: target.mediaId,
                  modality: 'image',
                  content: '',
                  file_path: target.filePath || '',
                  exists_on_disk: true,
                  size_bytes: null,
                  sha256: null,
                  bytes_modified: 0,
                  is_image: true,
                  is_audio_playable: false,
                  file_url: url,
                  metadata: {},
                });
              }
              return;
            }
            const data = await resp.json();
            if (!cancelled) {
              setInfo({
                media_id: data.media_id || target.mediaId,
                modality: data.modality || 'text',
                content: data.content || '',
                file_path: data.file_path || target.filePath || '',
                exists_on_disk: Boolean(data.file_path),
                size_bytes: null,
                sha256: null,
                bytes_modified: 0,
                is_image: false,
                is_audio_playable: false,
                file_url: url,
                metadata: data.metadata || {},
              });
            }
            return;
          } catch {
            // fall through to standard error below
          }
        }
        if (!cancelled) setError(getErrorMessage(err, 'Could not load carrier metadata'));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    if (target.packetPath) {
      setLoadingPacket(true);
      setPacketData(null);
      getMediaPacket(target.packetPath)
        .then((data) => {
          if (!cancelled) setPacketData(data);
        })
        .catch(() => {
          if (!cancelled) setPacketData(null);
        })
        .finally(() => {
          if (!cancelled) setLoadingPacket(false);
        });
    } else {
      setPacketData(null);
    }

    return () => {
      cancelled = true;
    };
  }, [target]);

  const copyText = (key: string, value: string) => {
    navigator.clipboard.writeText(value);
    setCopiedField(key);
    setTimeout(() => setCopiedField(null), 1500);
  };

  const resolvedFileUrl = info
    ? `${getApiBase()}/media/file?media_id=${encodeURIComponent(info.media_id)}${
        info.file_path ? `&path=${encodeURIComponent(info.file_path)}` : ''
      }`
    : '';

  return (
    <AnimatePresence>
      {target && (
        <div
          className="fixed inset-0 z-[450] flex items-center justify-center p-4 bg-foreground/35 backdrop-blur-[2px]"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, y: 12, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.97 }}
            transition={{ duration: 0.18 }}
            className="bg-surface border border-border rounded-2xl w-full max-w-2xl max-h-[90vh] flex flex-col shadow-modal overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="px-5 py-4 border-b border-border bg-surface-muted/60 flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  {target.seq !== undefined && (
                    <span className="font-mono text-xs px-2 py-0.5 rounded-md bg-surface border border-border text-muted">
                      Packet #{target.seq}
                    </span>
                  )}
                  <h3 className="font-mono text-body-sm font-semibold text-foreground truncate">
                    {target.mediaId}
                  </h3>
                  {info && (
                    <Badge variant="info" size="sm">
                      {info.modality.toUpperCase()}
                    </Badge>
                  )}
                  {target.status === 'tampered' && (
                    <Badge variant="warning" size="sm">
                      TAMPERED
                    </Badge>
                  )}
                </div>
                <p className="text-body-xs text-muted mt-0.5">
                  Unmodified public corpus carrier &middot; live disk inspection
                </p>
              </div>
              <button
                type="button"
                onClick={onClose}
                className="p-1.5 rounded-lg text-muted hover:text-foreground hover:bg-surface-hover shrink-0"
                aria-label="Close carrier viewer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Optional Tabs when packet JSON is also available */}
            {target.packetPath && (
              <div className="px-5 pt-3 pb-0 border-b border-border flex items-center gap-2 bg-surface">
                <button
                  type="button"
                  onClick={() => setActiveTab('asset')}
                  className={`px-3 py-2 text-body-xs font-medium border-b-2 transition-colors flex items-center gap-1.5 ${
                    activeTab === 'asset'
                      ? 'border-primary text-foreground font-semibold'
                      : 'border-transparent text-muted hover:text-foreground'
                  }`}
                >
                  <HardDrive className="w-3.5 h-3.5" />
                  Corpus Asset &amp; Location
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab('packet')}
                  className={`px-3 py-2 text-body-xs font-medium border-b-2 transition-colors flex items-center gap-1.5 ${
                    activeTab === 'packet'
                      ? 'border-primary text-foreground font-semibold'
                      : 'border-transparent text-muted hover:text-foreground'
                  }`}
                >
                  <FileCode className="w-3.5 h-3.5" />
                  Wire Packet JSON
                </button>
              </div>
            )}

            {/* Body */}
            <div className="flex-1 overflow-y-auto p-5 space-y-4 bg-background">
              {activeTab === 'asset' ? (
                loading ? (
                  <div className="py-16 flex flex-col items-center gap-3">
                    <LoadingSpinner size="md" />
                    <span className="text-body-xs text-muted">
                      Resolving carrier asset and computing live SHA-256...
                    </span>
                  </div>
                ) : error ? (
                  <div className="p-4 rounded-xl bg-error-muted border border-error/30 text-error text-body-sm flex items-center gap-2.5">
                    <AlertCircle className="w-4 h-4 shrink-0" />
                    <span>{error}</span>
                  </div>
                ) : info ? (
                  <>
                    {/* Media Asset Preview */}
                    <div className="rounded-xl border border-border bg-surface overflow-hidden">
                      {info.is_image ? (
                        <div className="bg-surface-muted/60 p-3 flex flex-col items-center gap-2">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={resolvedFileUrl}
                            alt={info.media_id}
                            className="max-h-[300px] w-auto rounded-lg object-contain border border-border shadow-subtle"
                          />
                          {info.content && (
                            <p className="text-body-xs text-muted text-center max-w-lg px-2">
                              &ldquo;{info.content}&rdquo;
                            </p>
                          )}
                        </div>
                      ) : info.modality === 'audio' ? (
                        <div className="p-4 space-y-3">
                          <div className="flex items-center justify-between gap-2">
                            <span className="flex items-center gap-2 text-purple text-body-xs font-semibold uppercase tracking-wider">
                              <Volume2 className="w-4 h-4" />
                              Audio Carrier Transcript
                            </span>
                            {typeof info.metadata?.duration === 'number' && (
                              <Badge variant="neutral" size="sm">
                                {Number(info.metadata.duration).toFixed(2)}s &middot;{' '}
                                {String(info.metadata.source || 'libretta')}
                              </Badge>
                            )}
                          </div>
                          {info.is_audio_playable && (
                            <audio controls className="w-full" src={resolvedFileUrl} />
                          )}
                          <div className="p-3.5 rounded-xl bg-surface-muted/60 border border-border text-body-sm text-foreground leading-relaxed whitespace-pre-wrap max-h-48 overflow-y-auto">
                            {info.content || '(No transcript stored for this audio clip)'}
                          </div>
                        </div>
                      ) : (
                        <div className="p-4 space-y-2.5">
                          <div className="flex items-center justify-between gap-2">
                            <span className="flex items-center gap-2 text-success text-body-xs font-semibold uppercase tracking-wider">
                              <FileText className="w-4 h-4" />
                              Corpus Text Passage
                            </span>
                            {info.metadata?.source && (
                              <Badge variant="neutral" size="sm">
                                {String(info.metadata.source)}
                              </Badge>
                            )}
                          </div>
                          <div className="p-3.5 rounded-xl bg-surface-muted/60 border border-border text-body-sm text-foreground leading-relaxed whitespace-pre-wrap max-h-56 overflow-y-auto">
                            {info.content || '(No text content available)'}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* File Location & Live Integrity */}
                    <div className="rounded-xl border border-border bg-surface p-4 space-y-3">
                      <div className="flex items-center justify-between gap-2 flex-wrap">
                        <span className="text-body-xs font-semibold uppercase tracking-wider text-muted flex items-center gap-1.5">
                          <ShieldCheck className="w-3.5 h-3.5 text-success" />
                          Zero-Modification Disk Record
                        </span>
                        <Badge variant="success" size="sm" dot>
                          0 bytes modified
                        </Badge>
                      </div>

                      {target.status === 'tampered' && (
                        <div className="flex items-start gap-2 p-2.5 rounded-lg bg-warning-muted border border-warning/30 text-body-xs text-warning">
                          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                          <span>
                            Corpus asset itself: <strong>0 bytes modified</strong>. The tampering
                            is in the <strong>wire packet JSON on disk</strong> for this slot, which
                            was rewritten to point to a different <code>media_id</code>.
                          </span>
                        </div>
                      )}

                      <div className="space-y-2 text-xs font-mono">
                        <div className="p-2.5 rounded-lg bg-surface-muted/60 border border-border flex items-center justify-between gap-2">
                          <div className="min-w-0">
                            <span className="text-[10px] text-muted uppercase block">
                              Corpus File Location
                            </span>
                            <span className="text-foreground break-all">
                              {info.file_path || '(In-index metadata item)'}
                            </span>
                          </div>
                          {info.file_path && (
                            <button
                              type="button"
                              onClick={() => copyText('path', info.file_path)}
                              className="p-1.5 rounded-lg hover:bg-surface border border-transparent hover:border-border text-muted hover:text-foreground shrink-0"
                              title="Copy file path"
                            >
                              {copiedField === 'path' ? (
                                <Check className="w-3.5 h-3.5 text-success" />
                              ) : (
                                <Copy className="w-3.5 h-3.5" />
                              )}
                            </button>
                          )}
                        </div>

                        {info.sha256 && (
                          <div className="p-2.5 rounded-lg bg-surface-muted/60 border border-border flex items-center justify-between gap-2">
                            <div className="min-w-0">
                              <span className="text-[10px] text-muted uppercase block">
                                Live Disk SHA-256 ({formatBytes(info.size_bytes)})
                              </span>
                              <span className="text-foreground break-all">{info.sha256}</span>
                            </div>
                            <button
                              type="button"
                              onClick={() => copyText('sha', info.sha256!)}
                              className="p-1.5 rounded-lg hover:bg-surface border border-transparent hover:border-border text-muted hover:text-foreground shrink-0"
                              title="Copy SHA-256"
                            >
                              {copiedField === 'sha' ? (
                                <Check className="w-3.5 h-3.5 text-success" />
                              ) : (
                                <Copy className="w-3.5 h-3.5" />
                              )}
                            </button>
                          </div>
                        )}
                      </div>

                      <div className="flex items-center justify-between pt-1 text-body-xs">
                        <span className="text-muted">
                          {info.exists_on_disk ? 'Verified present on disk' : 'Metadata-indexed corpus record'}
                        </span>
                        <a
                          href={resolvedFileUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary hover:underline inline-flex items-center gap-1 font-medium"
                        >
                          Open raw asset <ExternalLink className="w-3.5 h-3.5" />
                        </a>
                      </div>
                    </div>
                  </>
                ) : null
              ) : (
                <div className="space-y-3">
                  <div className="p-3 rounded-xl bg-surface border border-border flex items-center justify-between gap-2 font-mono text-xs">
                    <span className="text-foreground truncate">{target.packetPath}</span>
                    {target.packetPath && (
                      <button
                        type="button"
                        onClick={() => copyText('pkt', target.packetPath!)}
                        className="p-1.5 rounded-lg hover:bg-surface-muted text-muted hover:text-foreground shrink-0"
                      >
                        {copiedField === 'pkt' ? (
                          <Check className="w-3.5 h-3.5 text-success" />
                        ) : (
                          <Copy className="w-3.5 h-3.5" />
                        )}
                      </button>
                    )}
                  </div>
                  {loadingPacket ? (
                    <div className="py-12 flex justify-center">
                      <LoadingSpinner size="md" />
                    </div>
                  ) : packetData ? (
                    <pre className="p-4 rounded-xl bg-surface border border-border font-mono text-xs text-foreground overflow-x-auto max-h-[50vh]">
                      {JSON.stringify(packetData, null, 2)}
                    </pre>
                  ) : (
                    <div className="p-6 text-center text-body-xs text-muted">
                      Packet JSON file is no longer on disk.
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Footer with optional Delete / Tamper wire actions */}
            {(onDeletePacket || onTamperPacket) &&
              target.seq !== undefined &&
              target.status !== 'deleted' && (
                <div className="px-5 py-3.5 border-t border-border bg-surface-muted/60 flex flex-wrap items-center justify-between gap-3">
                  <span className="text-body-xs text-muted">
                    On-disk packet actions for slot <code className="text-foreground">#{target.seq}</code>
                  </span>
                  <div className="flex items-center gap-2">
                    {onTamperPacket && (
                      <Button
                        variant="secondary"
                        size="sm"
                        loading={actionBusy}
                        onClick={() => onTamperPacket(target.seq!)}
                        leftIcon={<Shuffle className="w-3.5 h-3.5 text-warning" />}
                      >
                        Tamper on Disk
                      </Button>
                    )}
                    {onDeletePacket && (
                      <Button
                        variant="danger"
                        size="sm"
                        loading={actionBusy}
                        onClick={() => onDeletePacket(target.seq!)}
                        leftIcon={<Trash2 className="w-3.5 h-3.5" />}
                      >
                        Delete File from Wire
                      </Button>
                    )}
                  </div>
                </div>
              )}
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
