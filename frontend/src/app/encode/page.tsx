'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  Lock,
  Brain,
  Shield,
  Copy,
  ExternalLink,
  Key,
  Image as ImageIcon,
  FileText,
  Volume2,
  AlertCircle,
  Eye,
} from 'lucide-react';

import { encodeMessage, EncodeResponse, checkReady, getErrorMessage } from '@/lib/api';
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

export default function EncodePage() {
  const router = useRouter();
  const { success, error: showError } = useToastHelpers();

  const [message, setMessage] = useState('');
  const [mode, setMode] = useState<'exact_vcp' | 'dssc'>('exact_vcp');
  const [parityBytes, setParityBytes] = useState<number>(8);
  const [sessionKeyHex, setSessionKeyHex] = useState('');
  const [diversityMode, setDiversityMode] = useState<'best' | 'round_robin' | 'balanced'>('round_robin');
  const [modalities, setModalities] = useState<string[]>(['image', 'text', 'audio']);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EncodeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [serverReady, setServerReady] = useState<boolean | null>(null);
  const [previewTarget, setPreviewTarget] = useState<CarrierPreviewTarget | null>(null);

  const generateSessionKey = () => {
    const bytes = new Uint8Array(32);
    crypto.getRandomValues(bytes);
    return Array.from(bytes)
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
  };

  const handleModeChange = (newMode: 'exact_vcp' | 'dssc') => {
    setMode(newMode);
    if (newMode === 'dssc' && !sessionKeyHex) {
      setSessionKeyHex(generateSessionKey());
    }
  };

  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const status = await checkReady();
        if (active) setServerReady(status.ready);
      } catch {
        if (active) setServerReady(null);
      }
    };
    poll();
    const timer = setInterval(poll, 3000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  const toggleModality = (mod: string) => {
    if (modalities.includes(mod)) {
      if (modalities.length > 1) setModalities(modalities.filter((m) => m !== mod));
    } else {
      setModalities([...modalities, mod]);
    }
  };

  const handleEncode = async () => {
    if (!message.trim()) {
      setError('Enter a message to encode.');
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await encodeMessage({
        message: message.trim(),
        mode,
        session_key_hex: mode === 'dssc' ? sessionKeyHex : undefined,
        diversity_mode: diversityMode,
        modalities,
        use_ecc: true,
        ecc_parity_bytes: parityBytes,
      });
      setResult(response);
      success(
        'Encoded',
        `Selected ${response.media_ids.length} unmodified carriers in ${response.elapsed_ms.toFixed(1)} ms`,
      );
    } catch (err) {
      const msg = getErrorMessage(err, 'Encoding failed');
      setError(msg);
      showError('Encoding failed', msg);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenDecode = () => {
    if (!result) return;
    const ids = encodeURIComponent(result.media_ids.join(','));
    const url = `/decode?ids=${ids}&mode=${mode}&parity=${result.ecc_parity_bytes || parityBytes}&diversity=${diversityMode}&modalities=${encodeURIComponent(modalities.join(','))}`;
    if (mode === 'dssc' && sessionKeyHex) {
      // Hand the DSSC session key to the decode page via sessionStorage
      // (per-tab, removed on read) instead of the URL, where it would leak
      // into browser history, server logs, and Referer headers.
      try {
        window.sessionStorage.setItem('dcass_pending_dssc_key', sessionKeyHex);
      } catch {}
    }
    router.push(url);
  };

  return (
    <PageWrapper
      title="Encode"
      description="Transform plaintext into clickable unmodified corpus carriers protected by CRC-16 and RS-ECC"
      action={
        <Badge variant={serverReady ? 'success' : 'warning'} dot size="md">
          {serverReady ? 'Engine Ready' : 'Warming Engine'}
        </Badge>
      }
    >
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 pb-12">
        {/* Left: Controls */}
        <div className="lg:col-span-5 space-y-5">
          <Card title="Payload" subtitle={`${message.length} characters`}>
            <textarea
              rows={4}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Enter secret plaintext..."
              className="input font-mono text-body-sm resize-none"
              disabled={loading}
            />
          </Card>

          <Card title="Codec Configuration">
            <div className="space-y-4">
              <div>
                <label className="label">Encoding Mode</label>
                <div className="grid grid-cols-2 gap-2.5">
                  <button
                    type="button"
                    onClick={() => handleModeChange('exact_vcp')}
                    className={`p-3 rounded-xl border text-left transition-all ${
                      mode === 'exact_vcp'
                        ? 'border-primary bg-primary-muted/60 text-foreground'
                        : 'border-border bg-surface-muted/40 text-muted hover:text-foreground'
                    }`}
                  >
                    <Shield className="w-4 h-4 text-primary mb-1" />
                    <div className="text-body-xs font-semibold">Exact VCP</div>
                    <div className="text-[11px] text-muted">8 bits / carrier</div>
                  </button>
                  <button
                    type="button"
                    onClick={() => handleModeChange('dssc')}
                    className={`p-3 rounded-xl border text-left transition-all ${
                      mode === 'dssc'
                        ? 'border-purple bg-purple-muted/60 text-foreground'
                        : 'border-border bg-surface-muted/40 text-muted hover:text-foreground'
                    }`}
                  >
                    <Brain className="w-4 h-4 text-purple mb-1" />
                    <div className="text-body-xs font-semibold">DSSC Compact</div>
                    <div className="text-[11px] text-muted">~13.5 bits / carrier</div>
                  </button>
                </div>
              </div>

              {mode === 'dssc' && (
                <div>
                  <label className="label">Session Key (64-char hex)</label>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={sessionKeyHex}
                      onChange={(e) => setSessionKeyHex(e.target.value)}
                      className="input font-mono text-body-xs flex-1"
                    />
                    <Button
                      variant="secondary"
                      size="sm"
                      type="button"
                      onClick={() => setSessionKeyHex(generateSessionKey())}
                      leftIcon={<Key className="w-3.5 h-3.5" />}
                    >
                      New
                    </Button>
                  </div>
                </div>
              )}

              <div>
                <label className="label">RS-ECC Parity Bytes (R)</label>
                <div className="grid grid-cols-4 gap-2">
                  {PARITY_OPTIONS.map((r) => (
                    <button
                      key={r}
                      type="button"
                      onClick={() => setParityBytes(r)}
                      className={`py-2 rounded-xl border font-mono text-xs font-semibold transition-all ${
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
                <label className="label">Diversity Strategy</label>
                <div className="grid grid-cols-3 gap-2">
                  {(['round_robin', 'balanced', 'best'] as const).map((d) => (
                    <button
                      key={d}
                      type="button"
                      onClick={() => setDiversityMode(d)}
                      className={`py-2 px-2 rounded-xl border text-body-xs font-medium capitalize transition-all ${
                        diversityMode === d
                          ? 'border-primary bg-primary-muted text-primary font-semibold'
                          : 'border-border bg-surface-muted/40 text-muted hover:text-foreground'
                      }`}
                    >
                      {d.replace('_', ' ')}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="label">Allowed Modalities</label>
                <div className="flex flex-wrap gap-2">
                  {[
                    { id: 'image', label: 'Image', icon: ImageIcon },
                    { id: 'text', label: 'Text', icon: FileText },
                    { id: 'audio', label: 'Audio', icon: Volume2 },
                  ].map((m) => {
                    const Icon = m.icon;
                    const active = modalities.includes(m.id);
                    return (
                      <button
                        key={m.id}
                        type="button"
                        onClick={() => toggleModality(m.id)}
                        className={`px-3 py-1.5 rounded-xl border text-body-xs font-medium inline-flex items-center gap-1.5 transition-all ${
                          active
                            ? 'border-primary bg-primary-muted text-primary'
                            : 'border-border bg-surface-muted/40 text-muted'
                        }`}
                      >
                        <Icon className="w-3.5 h-3.5" />
                        {m.label}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            <div className="mt-5 pt-4 border-t border-border space-y-2.5">
              <Button
                variant="primary"
                size="lg"
                className="w-full"
                onClick={handleEncode}
                loading={loading}
                disabled={!message.trim()}
                leftIcon={<Lock className="w-4 h-4" />}
              >
                Encode Sequence
              </Button>
              {result && !loading && (
                <Button
                  variant="secondary"
                  size="md"
                  className="w-full"
                  onClick={handleOpenDecode}
                  leftIcon={<ExternalLink className="w-4 h-4" />}
                >
                  Open in Decode Page
                </Button>
              )}
            </div>

            {error && (
              <div className="mt-4 p-3 rounded-xl bg-error-muted border border-error/30 text-error text-body-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </Card>
        </div>

        {/* Right: Encoded Output */}
        <div className="lg:col-span-7 space-y-5">
          {loading ? (
            <Card className="py-20 text-center">
              <LoadingSpinner size="lg" />
              <p className="text-body-sm text-muted mt-4">Selecting unmodified corpus carriers...</p>
            </Card>
          ) : result ? (
            <>
              <StatGrid
                columns={4}
                stats={[
                  { label: 'Mode', value: result.mode.toUpperCase(), variant: 'info' },
                  { label: 'Carriers', value: result.media_ids.length, variant: 'primary' },
                  { label: 'RS Parity', value: `${result.ecc_parity_bytes ?? parityBytes}B`, variant: 'success' },
                  { label: 'Elapsed', value: `${result.elapsed_ms.toFixed(1)} ms`, variant: 'neutral' },
                ]}
              />

              <Card
                title="Generated Carrier Sequence"
                subtitle={`${result.media_ids.length} unmodified corpus files — click any carrier to view asset & disk location`}
                action={
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => {
                      navigator.clipboard.writeText(result.media_ids.join(','));
                      success('Copied', 'Carrier IDs copied to clipboard');
                    }}
                    leftIcon={<Copy className="w-3.5 h-3.5" />}
                  >
                    Copy IDs
                  </Button>
                }
              >
                <div className="space-y-2 max-h-[540px] overflow-y-auto pr-1">
                  {result.encoded.map((item, idx) => (
                    <button
                      type="button"
                      key={`${item.media_id}_${idx}`}
                      onClick={() =>
                        setPreviewTarget({
                          mediaId: item.media_id,
                          filePath: item.file_path,
                          seq: idx,
                        })
                      }
                      className="w-full text-left p-3 rounded-xl bg-surface-muted/45 hover:bg-surface-muted border border-border hover:border-primary/60 transition-all flex flex-col gap-1 group"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2.5 min-w-0">
                          <span className="font-mono text-body-xs text-muted">#{idx}</span>
                          <span className="font-mono text-body-xs font-semibold text-foreground group-hover:text-primary transition-colors truncate">
                            {item.media_id}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 shrink-0">
                          <Badge variant="info" size="sm">
                            {item.modality}
                          </Badge>
                          <span className="text-[11px] font-mono text-muted group-hover:text-primary inline-flex items-center gap-1">
                            <Eye className="w-3 h-3" />
                            View
                          </span>
                        </div>
                      </div>
                      {item.content && (
                        <p className="text-body-xs text-muted line-clamp-1">{item.content}</p>
                      )}
                      {item.file_path && (
                        <p className="text-[10px] font-mono text-muted/80 truncate">
                          {item.file_path}
                        </p>
                      )}
                    </button>
                  ))}
                </div>
              </Card>
            </>
          ) : (
            <Card className="py-20 text-center">
              <Lock className="w-9 h-9 text-muted/40 mx-auto mb-3" />
              <h3 className="font-serif text-heading-md font-medium text-foreground">
                Ready to encode
              </h3>
              <p className="text-body-sm text-muted mt-1 max-w-sm mx-auto">
                Enter a message on the left to map each CRC-16 + RS-ECC codeword byte onto real
                clickable corpus media.
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
