'use client';

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { motion } from 'framer-motion';
import { Shield, Brain, Eye, Loader2, CheckCircle, AlertCircle, Copy, Key, Trash2, RotateCcw, FileText } from 'lucide-react';

import { decodeSequence, DecodeResponse, getErrorMessage } from '@/lib/api';
import { PageWrapper, Card, Badge, LoadingSpinner, StatCard, useToastHelpers, EmptyState, Tooltip } from '@/components';

const modalityColors = {
  image: 'text-blue-400',
  text: 'text-green-400',
  audio: 'text-purple-400',
};

const modalityIcons = {
  image: '🖼',
  text: '📝',
  audio: '🔊',
};

function DecodeContent() {
  const searchParams = useSearchParams();
  const { success, error: showError } = useToastHelpers();

  const [idsInput, setIdsInput] = useState('');
  const [mode, setMode] = useState<'exact_vcp' | 'dssc'>('exact_vcp');
  const [sessionKeyHex, setSessionKeyHex] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DecodeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const idsParam = searchParams.get('ids');
    if (idsParam) {
      setIdsInput(decodeURIComponent(idsParam));
    }
    const modeParam = searchParams.get('mode');
    if (modeParam === 'exact_vcp' || modeParam === 'dssc') {
      setMode(modeParam);
    }
    const keyParam = searchParams.get('key');
    if (keyParam) {
      setSessionKeyHex(decodeURIComponent(keyParam));
    }
  }, [searchParams]);

  const parseIds = (raw: string): string[] =>
    raw
      .split(/[\s,]+/)
      .map((s) => s.trim())
      .filter(Boolean);

  const handleDecode = async () => {
    const media_ids = parseIds(idsInput);
    if (media_ids.length === 0) {
      setError('Please enter at least one media ID');
      return;
    }

    if (mode === 'dssc' && !sessionKeyHex.trim()) {
      setError('Session key is required for DSSC mode');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setResult(null);

      const response = await decodeSequence({
        media_ids,
        mode,
        session_key_hex: mode === 'dssc' ? sessionKeyHex.trim() : undefined,
        use_ecc: true,
      });

      setResult(response);
      success('Decoded!', `Recovered message with ${(response.verification_rate * 100).toFixed(1)}% verification in ${response.elapsed_ms.toFixed(1)}ms`);
    } catch (err: any) {
      const errMsg = getErrorMessage(err, 'Decode failed');
      setError(errMsg);
      showError('Decode failed', errMsg);
    } finally {
      setLoading(false);
    }
  };

  const copyResult = async () => {
    if (!result) return;
    await navigator.clipboard.writeText(result.reconstructed_meaning);
    success('Copied', 'Decoded message copied to clipboard');
  };

  const clearAll = () => {
    setIdsInput('');
    setSessionKeyHex('');
    setResult(null);
    setError(null);
  };

  const modeOptions = {
    exact_vcp: {
      label: 'Byte-Exact (VCP)',
      description: '1 byte per carrier, Voronoi Codebook Partitioning',
      icon: Shield,
      color: 'primary',
    },
    dssc: {
      label: 'Compact Semantic (DSSC)',
      description: '~15 bits per carrier, Dynamic State Space Compression',
      icon: Brain,
      color: 'purple',
    },
  };

  return (
    <PageWrapper
      title="Decode"
      description="Reconstruct semantic meaning from media IDs using loaded indices"
      action={
        <button onClick={clearAll} className="btn-ghost btn-sm gap-1" disabled={!result && parseIds(idsInput).length === 0}>
          <RotateCcw className="w-4 h-4" />
          Clear
        </button>
      }
    >
      <div className="max-w-6xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Input Panel */}
          <div className="space-y-6">
            <Card>
              <h3 className="text-heading-sm font-semibold mb-4">Decoding Settings</h3>
              <div className="space-y-6">
                {/* Mode Selector */}
                <div>
                  <label className="label">Decoding Mode</label>
                  <div className="grid grid-cols-2 gap-3">
                    {(['exact_vcp', 'dssc'] as const).map((m) => {
                      const opt = modeOptions[m];
                      const Icon = opt.icon;
                      return (
                        <Tooltip key={m} content={opt.description} className="w-full">
                          <button
                            type="button"
                            onClick={() => setMode(m)}
                            className={`w-full relative p-4 rounded-xl border-2 transition-all duration-fast flex flex-col items-center justify-center gap-2 ${
                              mode === m
                                ? m === 'dssc'
                                  ? 'border-purple-500 bg-purple-500/10'
                                  : 'border-primary bg-primary/10'
                                : 'border-border hover:border-primary/50 hover:bg-surface'
                            }`}
                          >
                            <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                              m === 'dssc'
                                ? 'bg-purple-500/20 text-purple-400'
                                : 'bg-primary/20 text-primary'
                            }`}>
                              <Icon className="w-5 h-5" />
                            </div>
                            <span className="font-medium text-body-sm text-center">{opt.label}</span>
                          </button>
                        </Tooltip>
                      );
                    })}
                  </div>
                </div>

                {/* DSSC Session Key Input */}
                {mode === 'dssc' && (
                  <div>
                    <label className="label">Session Key (Hex) \u2014 shared by sender</label>
                    <div className="relative">
                      <input
                        type="text"
                        value={sessionKeyHex}
                        onChange={(e) => setSessionKeyHex(e.target.value)}
                        placeholder="Paste 64-character hex session key..."
                        className="input font-mono text-body-xs pr-12"
                      />
                      {sessionKeyHex.length === 64 && (
                        <Tooltip content="Valid key length">
                          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-success">
                            <CheckCircle className="w-5 h-5" />
                          </span>
                        </Tooltip>
                      )}
                    </div>
                    <p className="text-body-xs text-muted mt-2">{sessionKeyHex.length}/64 characters</p>
                  </div>
                )}

                {/* Media IDs Input */}
                <div>
                  <label className="label flex items-center justify-between">
                    Media Carrier IDs
                    <span className="text-body-xs text-muted font-mono tabular-nums">{parseIds(idsInput).length} IDs</span>
                  </label>
                  <textarea
                    value={idsInput}
                    onChange={(e) => setIdsInput(e.target.value)}
                    placeholder="Paste comma- or newline-separated media IDs, e.g. flickr_00123, wiki_00456..."
                    className="input font-mono text-body-sm h-32 resize-none"
                    disabled={loading}
                  />
                  <p className="text-body-xs text-muted mt-2">
                    Enter media IDs from the encoder output. Supports comma, space, or newline separation.
                  </p>
                </div>
              </div>
            </Card>

            <div className="space-y-3">
              <button
                onClick={handleDecode}
                disabled={loading || parseIds(idsInput).length === 0 || (mode === 'dssc' && !sessionKeyHex.trim())}
                className="btn-primary btn-lg w-full justify-center gap-2"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Decoding...
                  </>
                ) : (
                  <>
                    <Shield className="w-5 h-5" />
                    Decode Sequence
                  </>
                )}
              </button>

              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-4 bg-error/10 border border-error/20 rounded-xl text-error flex items-start gap-3"
                  role="alert"
                >
                  <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium">Decoding failed</p>
                    <p className="text-body-sm mt-1">{error}</p>
                  </div>
                </motion.div>
              )}
            </div>
          </div>

          {/* Results Panel */}
          <div className="space-y-6">
            {loading && (
              <Card>
                <div className="text-center py-12">
                  <LoadingSpinner size="lg" />
                  <p className="text-muted mt-4">Looking up media items and reconstructing meaning...</p>
                </div>
              </Card>
            )}

            {result && !loading && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
              >
                <Card>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-heading-sm font-semibold">Reconstructed Meaning</h3>
                    <button
                      onClick={copyResult}
                      className="btn-ghost btn-sm gap-1"
                    >
                      <Copy className="w-3.5 h-3.5" />
                      Copy
                    </button>
                  </div>
                  <div className="bg-background border border-border rounded-lg p-6 font-medium text-body-lg text-foreground whitespace-pre-wrap min-h-[120px]">
                    {result.reconstructed_meaning || '(empty)'}
                  </div>
                </Card>

                <Card>
                  <h3 className="text-heading-sm font-semibold mb-4">Verification & Recovery</h3>
                  <div className="grid grid-cols-2 gap-4 mb-6">
                    <StatCard
                      label="Mode"
                      value={result.mode.toUpperCase()}
                      variant="info"
                    />
                    <StatCard
                      label="Verification"
                      value={`${(result.verification_rate * 100).toFixed(1)}%`}
                      variant={result.verification_rate === 1 ? 'success' : result.verification_rate > 0.8 ? 'warning' : 'error'}
                    />
                    <StatCard
                      label="ECC Status"
                      value={result.ecc_success ? 'OK (0% BER)' : 'FAILED'}
                      variant={result.ecc_success ? 'success' : 'error'}
                    />
                    <StatCard
                      label="Elapsed"
                      value={`${result.elapsed_ms.toFixed(1)} ms`}
                      variant="primary"
                    />
                    {result.ecc_errors_fixed && result.ecc_errors_fixed.length > 0 && (
                    <StatCard
                      label="Errors Fixed"
                      value={result.ecc_errors_fixed.length}
                      variant="warning"
                    />
                  )}
                  {result.all_verified && (
                    <StatCard
                      label="Source"
                      value="Cached"
                      variant="neutral"
                    />
                  )}
                  </div>

                  {result.ecc_errors_fixed && result.ecc_errors_fixed.length > 0 && (
                    <div className="p-4 bg-warning/10 border border-warning/20 rounded-lg">
                      <p className="font-medium text-warning mb-2 flex items-center gap-2">
                        <AlertCircle className="w-4 h-4" />
                        ECC corrected errors at positions:
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {result.ecc_errors_fixed.map((pos) => (
                          <span key={pos} className="font-mono text-body-xs bg-surface px-2 py-1 rounded border border-border">
                            {pos}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                </Card>

                <Card>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-heading-sm font-semibold">Decoded Items</h3>
                    <span className="text-body-sm text-muted font-mono tabular-nums">{result.items.length} items</span>
                  </div>
                  <div className="space-y-2 max-h-[400px] overflow-y-auto">
                    {result.items.map((item, idx) => (
                      <motion.div
                        key={idx}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.02 }}
                        className="bg-surface/50 border border-border/50 rounded-xl p-3"
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-center gap-3">
                            <span className="text-body-xs text-muted font-mono">#{idx + 1}</span>
                            <span className="font-mono text-sm text-primary">{item.media_id}</span>
                          </div>
                          <Badge variant="info" className="gap-1.5">
                            <span className={
                              item.modality === 'image' ? 'text-blue-400' : item.modality === 'text' ? 'text-emerald-400' : 'text-purple-400'
                            }>
                              {modalityIcons[item.modality as keyof typeof modalityIcons]}
                            </span>
                            <span className="capitalize">{item.modality}</span>
                            <Badge variant={item.verified ? 'success' : 'error'} size="sm" dot>
                              {item.verified ? 'Verified' : 'Failed'}
                            </Badge>
                          </Badge>
                        </div>
                        <div className="text-body-sm text-muted">{item.content || '(no content)'}</div>
                        {item.file_path && (
                          <div className="text-body-xs text-muted font-mono break-all pt-2 border-t border-border/50">
                            File: {item.file_path}
                          </div>
                        )}
                      </motion.div>
                    ))}
                  </div>
                </Card>
              </motion.div>
            )}

            {!result && !loading && (
                          <Card className="text-center py-16">
                            <EmptyState
                              icon={<FileText className="w-12 h-12" />}
                              title="No results yet"
                              description={'Paste media IDs from an encoding operation and click \'Decode Sequence\' to reconstruct the message'}
                            />
                          </Card>
                        )}
          </div>
        </div>
      </div>
    </PageWrapper>
  );
}

export default function DecodePage() {
  return (
    <Suspense fallback={<PageWrapper fullWidth><LoadingSpinner size="lg" /></PageWrapper>}>
      <DecodeContent />
    </Suspense>
  );
}
