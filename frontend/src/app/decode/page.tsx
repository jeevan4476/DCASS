'use client';

import React, { useState, useEffect, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import {
  Unlock,
  Shield,
  Brain,
  Copy,
  RotateCcw,
  AlertCircle,
  CheckCircle2,
  XCircle,
  Eye,
  Trash2,
  Undo2,
  Info,
} from 'lucide-react';

import { decodeSequence, DecodeResponse, getErrorMessage } from '@/lib/api';
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

const PARITY_OPTIONS = [4, 8, 12, 16];

function DecodeContent() {
  const searchParams = useSearchParams();
  const { success, error: showError } = useToastHelpers();

  const [idsInput, setIdsInput] = useState('');
  const [originalSlotBackup, setOriginalSlotBackup] = useState<Record<number, string>>({});
  const [mode, setMode] = useState<'exact_vcp' | 'dssc'>('exact_vcp');
  const [diversityMode, setDiversityMode] = useState<'best' | 'round_robin' | 'balanced'>('round_robin');
  const [modalities, setModalities] = useState<string[]>(['image', 'text', 'audio']);
  const [parityBytes, setParityBytes] = useState<number>(8);
  const [sessionKeyHex, setSessionKeyHex] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DecodeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [previewTarget, setPreviewTarget] = useState<CarrierPreviewTarget | null>(null);

  useEffect(() => {
    const idsParam = searchParams.get('ids');
    if (idsParam) {
      // searchParams.get() already percent-decodes once; decoding again would
      // corrupt literal '%' sequences (or throw URIError).
      const decodedIds = idsParam;
      setIdsInput(decodedIds);
      const tokens = decodedIds
        .split(/[\s,]+/)
        .map((s) => s.trim())
        .filter(Boolean);
      const backup: Record<number, string> = {};
      tokens.forEach((tok, idx) => {
        backup[idx] = tok;
      });
      setOriginalSlotBackup(backup);
    }
    const modeParam = searchParams.get('mode');
    if (modeParam === 'exact_vcp' || modeParam === 'dssc') setMode(modeParam);
    // Prefer the sessionStorage handoff from the encode page (never in the
    // URL); keep ?key= only as a manual fallback. Read-and-remove so the key
    // does not linger in storage.
    let handoffKey: string | null = null;
    try {
      handoffKey = window.sessionStorage.getItem('dcass_pending_dssc_key');
      if (handoffKey) window.sessionStorage.removeItem('dcass_pending_dssc_key');
    } catch {}
    const keyParam = searchParams.get('key');
    if (handoffKey) setSessionKeyHex(handoffKey);
    else if (keyParam) setSessionKeyHex(keyParam);
    const parityParam = Number(searchParams.get('parity'));
    if (parityParam && parityParam >= 2 && parityParam <= 64) setParityBytes(parityParam);
    const divParam = searchParams.get('diversity');
    if (divParam === 'best' || divParam === 'round_robin' || divParam === 'balanced') {
      setDiversityMode(divParam);
    }
    const modParam = searchParams.get('modalities');
    if (modParam) {
      const parsedMods = modParam
        .split(',')
        .map((m) => m.trim())
        .filter((m) => m === 'image' || m === 'text' || m === 'audio');
      if (parsedMods.length > 0) setModalities(parsedMods);
    }
  }, [searchParams]);

  const parseSlots = (raw: string): { tokens: string[]; slots: Array<string | null>; erasures: number[] } => {
    const tokens = raw
      .split(/[\s,]+/)
      .map((s) => s.trim())
      .filter(Boolean);
    const erasures: number[] = [];
    const slots = tokens.map((tok, idx) => {
      if (tok === '-' || tok.toUpperCase() === 'DROPPED' || tok === '[DROPPED]') {
        erasures.push(idx);
        return null;
      }
      return tok;
    });
    return { tokens, slots, erasures };
  };

  const parsed = parseSlots(idsInput);

  const toggleSlotDrop = (idx: number) => {
    const nextTokens = [...parsed.tokens];
    const current = nextTokens[idx];
    const isDropped =
      current === '-' || current.toUpperCase() === 'DROPPED' || current === '[DROPPED]';
    if (isDropped) {
      const restoreId = originalSlotBackup[idx];
      if (!restoreId) return;
      nextTokens[idx] = restoreId;
    } else {
      setOriginalSlotBackup((prev) => ({ ...prev, [idx]: current }));
      nextTokens[idx] = 'DROPPED';
    }
    setIdsInput(nextTokens.join(','));
  };

  const restoreAllSlots = () => {
    if (Object.keys(originalSlotBackup).length === 0) return;
    const nextTokens = parsed.tokens.map((tok, idx) =>
      tok.toUpperCase() === 'DROPPED' || tok === '-' || tok === '[DROPPED]'
        ? originalSlotBackup[idx] || tok
        : tok,
    );
    setIdsInput(nextTokens.join(','));
  };

  const handleDecode = async () => {
    if (parsed.slots.length === 0) {
      setError('Paste at least one carrier ID.');
      return;
    }
    if (mode === 'dssc' && !sessionKeyHex.trim()) {
      setError('Session key is required for DSSC mode.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await decodeSequence({
        media_ids: parsed.slots,
        mode,
        diversity_mode: diversityMode,
        modalities,
        session_key_hex: mode === 'dssc' ? sessionKeyHex.trim() : undefined,
        use_ecc: true,
        ecc_parity_bytes: parityBytes,
        erasure_positions: parsed.erasures.length > 0 ? parsed.erasures : undefined,
      });
      setResult(response);
      if (response.ecc_success) {
        success('Decoded', `Recovered plaintext in ${response.elapsed_ms.toFixed(1)} ms`);
      } else {
        showError('CRC Rejected', 'Sequence corruption or shifted slots exceeded RS-ECC capacity');
      }
    } catch (err) {
      const msg = getErrorMessage(err, 'Decode failed');
      setError(msg);
      showError('Decode failed', msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageWrapper
      title="Decode"
      description="Reconstruct exact plaintext from carrier IDs with sequence-aligned RS-ECC erasure recovery"
      action={
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            setIdsInput('');
            setOriginalSlotBackup({});
            setSessionKeyHex('');
            setResult(null);
            setError(null);
          }}
          leftIcon={<RotateCcw className="w-3.5 h-3.5" />}
        >
          Clear
        </Button>
      }
    >
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 pb-12">
        <div className="lg:col-span-5 space-y-5">
          <Card title="Decoder Settings">
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-2.5">
                <button
                  type="button"
                  onClick={() => setMode('exact_vcp')}
                  className={`p-3 rounded-xl border text-left transition-all ${
                    mode === 'exact_vcp'
                      ? 'border-primary bg-primary-muted/60 text-foreground'
                      : 'border-border bg-surface-muted/40 text-muted hover:text-foreground'
                  }`}
                >
                  <Shield className="w-4 h-4 text-primary mb-1" />
                  <div className="text-body-xs font-semibold">Exact VCP</div>
                  <div className="text-[11px] text-muted">1 byte / carrier</div>
                </button>
                <button
                  type="button"
                  onClick={() => setMode('dssc')}
                  className={`p-3 rounded-xl border text-left transition-all ${
                    mode === 'dssc'
                      ? 'border-purple bg-purple-muted/60 text-foreground'
                      : 'border-border bg-surface-muted/40 text-muted hover:text-foreground'
                  }`}
                >
                  <Brain className="w-4 h-4 text-purple mb-1" />
                  <div className="text-body-xs font-semibold">DSSC Compact</div>
                  <div className="text-[11px] text-muted">Keyed state spaces</div>
                </button>
              </div>

              {mode === 'dssc' && (
                <div className="space-y-3">
                  <div>
                    <label className="label">Session Key (Hex)</label>
                    <input
                      type="text"
                      value={sessionKeyHex}
                      onChange={(e) => setSessionKeyHex(e.target.value)}
                      placeholder="64-character hex session key"
                      className="input font-mono text-body-xs"
                    />
                  </div>
                  <div>
                    <label className="label">Diversity Strategy</label>
                    <select
                      value={diversityMode}
                      onChange={(e) =>
                        setDiversityMode(e.target.value as 'best' | 'round_robin' | 'balanced')
                      }
                      className="input py-2 text-body-xs font-mono"
                    >
                      <option value="round_robin">round_robin</option>
                      <option value="balanced">balanced</option>
                      <option value="best">best</option>
                    </select>
                  </div>
                </div>
              )}

              <div>
                <label className="label">RS Parity Bytes (R)</label>
                <div className="grid grid-cols-4 gap-2">
                  {PARITY_OPTIONS.map((r) => (
                    <button
                      key={r}
                      type="button"
                      onClick={() => setParityBytes(r)}
                      className={`py-1.5 rounded-xl border font-mono text-xs font-semibold transition-all ${
                        parityBytes === r
                          ? 'border-primary bg-primary-muted text-primary'
                          : 'border-border bg-surface-muted/40 text-muted hover:text-foreground'
                      }`}
                    >
                      R={r}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="label mb-0">Raw Carrier IDs</label>
                  <span className="font-mono text-[11px] text-muted">
                    {parsed.slots.length} slots
                    {parsed.erasures.length > 0 ? ` · ${parsed.erasures.length} dropped` : ''}
                  </span>
                </div>
                <textarea
                  rows={4}
                  value={idsInput}
                  onChange={(e) => setIdsInput(e.target.value)}
                  placeholder="Paste comma- or newline-separated media IDs..."
                  className="input font-mono text-body-xs resize-none"
                  disabled={loading}
                />
                <div className="mt-2 p-2.5 rounded-xl bg-surface-muted/70 border border-border text-[11px] text-muted flex items-start gap-2">
                  <Info className="w-3.5 h-3.5 text-primary shrink-0 mt-0.5" />
                  <span>
                    Deleting an ID from the text box shifts every later carrier left and breaks
                    block alignment. Click <strong>Drop</strong> on any numbered slot below (or
                    replace an ID with <code className="text-foreground">DROPPED</code>) to keep
                    sequence positions intact for real RS-ECC erasure recovery.
                  </span>
                </div>
              </div>

              {/* Interactive Sequence Slots */}
              {parsed.tokens.length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="label mb-0">Sequence Slots (View or Drop)</label>
                    {parsed.erasures.length > 0 && (
                      <button
                        type="button"
                        onClick={restoreAllSlots}
                        className="text-[11px] font-mono text-primary hover:underline inline-flex items-center gap-1"
                      >
                        <Undo2 className="w-3 h-3" />
                        Restore all
                      </button>
                    )}
                  </div>
                  <div className="max-h-52 overflow-y-auto space-y-1.5 pr-1">
                    {parsed.tokens.map((tok, idx) => {
                      const isDropped = parsed.slots[idx] === null;
                      return (
                        <div
                          key={`${idx}_${tok}`}
                          className={`px-2.5 py-1.5 rounded-lg border text-xs font-mono flex items-center justify-between gap-2 ${
                            isDropped
                              ? 'bg-error-muted/50 border-error/40 border-dashed text-error'
                              : 'bg-surface-muted/40 border-border text-foreground'
                          }`}
                        >
                          <div className="flex items-center gap-2 min-w-0">
                            <span className="text-muted text-[11px] w-7 shrink-0">#{idx}</span>
                            {isDropped ? (
                              <span className="truncate font-semibold">DROPPED (erasure slot)</span>
                            ) : (
                              <button
                                type="button"
                                onClick={() => setPreviewTarget({ mediaId: tok, seq: idx })}
                                className="truncate text-left hover:text-primary underline decoration-border hover:decoration-primary underline-offset-2"
                                title="View carrier file and disk path"
                              >
                                {tok}
                              </button>
                            )}
                          </div>
                          <div className="flex items-center gap-1 shrink-0">
                            {!isDropped && (
                              <button
                                type="button"
                                onClick={() => setPreviewTarget({ mediaId: tok, seq: idx })}
                                className="p-1 rounded hover:bg-surface text-muted hover:text-primary"
                                title="View carrier asset"
                              >
                                <Eye className="w-3.5 h-3.5" />
                              </button>
                            )}
                            <button
                              type="button"
                              onClick={() => toggleSlotDrop(idx)}
                              disabled={isDropped && !originalSlotBackup[idx]}
                              className={`px-2 py-0.5 rounded text-[10px] font-semibold border transition-colors ${
                                isDropped
                                  ? 'bg-surface border-border text-foreground hover:border-primary'
                                  : 'bg-error-muted border-error/25 text-error hover:bg-error hover:text-white'
                              }`}
                            >
                              {isDropped ? 'Restore' : 'Drop'}
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              <Button
                variant="primary"
                size="lg"
                className="w-full"
                onClick={handleDecode}
                loading={loading}
                disabled={parsed.slots.length === 0}
                leftIcon={<Unlock className="w-4 h-4" />}
              >
                Decode Sequence
              </Button>

              {error && (
                <div className="p-3 rounded-xl bg-error-muted border border-error/30 text-error text-body-xs flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  <span>{error}</span>
                </div>
              )}
            </div>
          </Card>
        </div>

        <div className="lg:col-span-7 space-y-5">
          {loading ? (
            <Card className="py-20 text-center">
              <LoadingSpinner size="lg" />
              <p className="text-body-sm text-muted mt-4">Running codebook lookup &amp; RS-ECC decode...</p>
            </Card>
          ) : result ? (
            <>
              <Card
                title="Reconstructed Plaintext"
                action={
                  result.ecc_success && (
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => {
                        navigator.clipboard.writeText(result.reconstructed_meaning);
                        success('Copied', 'Recovered plaintext copied');
                      }}
                      leftIcon={<Copy className="w-3.5 h-3.5" />}
                    >
                      Copy
                    </Button>
                  )
                }
              >
                <div
                  className={`p-4 rounded-xl border mb-4 flex items-center gap-2.5 text-body-sm ${
                    result.ecc_success
                      ? 'bg-success-muted border-success/30 text-success'
                      : 'bg-error-muted border-error/30 text-error'
                  }`}
                >
                  {result.ecc_success ? (
                    <CheckCircle2 className="w-5 h-5 shrink-0" />
                  ) : (
                    <XCircle className="w-5 h-5 shrink-0" />
                  )}
                  <span className="font-medium">
                    {result.ecc_success
                      ? `CRC-16 verified${
                          result.ecc_errors_fixed && result.ecc_errors_fixed.length > 0
                            ? ` · RS-ECC repaired ${result.ecc_errors_fixed.length} byte(s)`
                            : ''
                        }`
                      : 'CRC-16 / RS-ECC rejected payload (check parity R, session key, or shifted slot indices)'}
                  </span>
                </div>

                <div className="p-4 rounded-xl bg-surface-muted/60 border border-border font-mono text-body-md text-foreground whitespace-pre-wrap break-words min-h-[88px]">
                  {result.ecc_success ? result.reconstructed_meaning : '(Rejected — unrecoverable)'}
                </div>
              </Card>

              <StatGrid
                columns={4}
                stats={[
                  { label: 'Mode', value: result.mode.toUpperCase(), variant: 'info' },
                  {
                    label: 'Verified',
                    value: `${(result.verification_rate * 100).toFixed(1)}%`,
                    variant: result.verification_rate === 1 ? 'success' : 'warning',
                  },
                  {
                    label: 'Bytes Fixed',
                    value: result.ecc_errors_fixed?.length || 0,
                    variant: 'primary',
                  },
                  { label: 'Elapsed', value: `${result.elapsed_ms.toFixed(1)} ms`, variant: 'neutral' },
                ]}
              />

              <Card title="Carrier Items" subtitle={`${result.items.length} slots inspected — click any carrier to view asset`}>
                <div className="space-y-2 max-h-[380px] overflow-y-auto pr-1">
                  {result.items.map((item, idx) => {
                    const clickable = item.verified && item.media_id !== '[DROPPED]';
                    return (
                      <div
                        key={idx}
                        onClick={() =>
                          clickable &&
                          setPreviewTarget({
                            mediaId: item.media_id,
                            filePath: item.file_path,
                            seq: idx,
                          })
                        }
                        className={`p-3 rounded-xl bg-surface-muted/45 border border-border flex items-center justify-between gap-3 ${
                          clickable ? 'cursor-pointer hover:border-primary/60 hover:bg-surface-muted transition-all' : ''
                        }`}
                      >
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="font-mono text-body-xs text-muted">#{idx}</span>
                            <span className="font-mono text-body-xs font-semibold text-foreground truncate">
                              {item.media_id}
                            </span>
                          </div>
                          <p className="text-body-xs text-muted truncate mt-0.5">{item.content}</p>
                        </div>
                        <div className="flex items-center gap-2 shrink-0">
                          <Badge variant={item.verified ? 'success' : 'error'} size="sm" dot>
                            {item.verified ? item.modality : 'Dropped'}
                          </Badge>
                          {clickable && <Eye className="w-3.5 h-3.5 text-muted" />}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </Card>
            </>
          ) : (
            <Card className="py-20 text-center">
              <Unlock className="w-9 h-9 text-muted/40 mx-auto mb-3" />
              <h3 className="font-serif text-heading-md font-medium text-foreground">
                Ready to decode
              </h3>
              <p className="text-body-sm text-muted mt-1 max-w-sm mx-auto">
                Paste carrier IDs from the Encode page or click <strong>Drop</strong> on any
                sequence slot on the left to test RS-ECC erasure recovery with exact alignment.
              </p>
            </Card>
          )}
        </div>
      </div>

      <CarrierPreviewModal
        target={previewTarget}
        onClose={() => setPreviewTarget(null)}
      />
    </PageWrapper>
  );
}

export default function DecodePage() {
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
      <DecodeContent />
    </Suspense>
  );
}
