'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { Shield, Zap, Brain, Eye, Loader2, CheckCircle, AlertCircle, Copy, ExternalLink, Key, Layers, Radio } from 'lucide-react';

import { encodeMessage, EncodeResponse, checkReady, getErrorMessage } from '@/lib/api';
import { PageWrapper, Card, Badge, LoadingSpinner, StatCard, useToastHelpers, Tooltip } from '@/components';

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

export default function EncodePage() {
  const router = useRouter();
  const { success, error: showError } = useToastHelpers();

  const [message, setMessage] = useState('');
  const [mode, setMode] = useState<'exact_vcp' | 'dssc'>('exact_vcp');
  const [sessionKeyHex, setSessionKeyHex] = useState('');
  const [diversityMode, setDiversityMode] = useState<'best' | 'round_robin' | 'balanced'>('best');
  const [modalities, setModalities] = useState(['image', 'text', 'audio']);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EncodeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [serverReady, setServerReady] = useState<boolean | null>(null);
  const [serverInitializing, setServerInitializing] = useState(false);

  const generateSessionKey = () => {
    const bytes = new Uint8Array(32);
    crypto.getRandomValues(bytes);
    return Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join('');
  };

  const handleModeChange = (newMode: 'exact_vcp' | 'dssc') => {
    setMode(newMode);
    if (newMode === 'dssc' && !sessionKeyHex) {
      setSessionKeyHex(generateSessionKey());
    }
  };

  useEffect(() => {
    let interval: NodeJS.Timeout;
    const checkServerStatus = async () => {
      try {
        const status = await checkReady();
        setServerReady(status.ready);
        setServerInitializing(status.initializing);
        if (status.ready) {
          clearInterval(interval);
        }
      } catch (err) {
        setServerReady(null);
      }
    };
    checkServerStatus();
    interval = setInterval(checkServerStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleEncode = async () => {
    if (!message.trim()) {
      setError('Please enter a message');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setResult(null);

      const response = await encodeMessage({
        message: message.trim(),
        mode,
        session_key_hex: mode === 'dssc' ? sessionKeyHex : undefined,
        diversity_mode: diversityMode,
        modalities,
        use_ecc: true,
      });

      setResult(response);
      success('Encoded!', `Generated ${response.media_ids.length} carriers in ${response.elapsed_ms.toFixed(1)}ms`);
    } catch (err: any) {
      console.error('Encoding error:', err);
      const errMsg = getErrorMessage(err, 'Encoding failed');
      setError(errMsg);
      showError('Encoding failed', errMsg);
    } finally {
      setLoading(false);
    }
  };

  const toggleModality = (mod: string) => {
    if (modalities.includes(mod)) {
      if (modalities.length > 1) {
        setModalities(modalities.filter(m => m !== mod));
      }
    } else {
      setModalities([...modalities, mod]);
    }
  };

  const handleOpenDecode = () => {
    if (!result) return;
    const ids = encodeURIComponent(result.media_ids.join(','));
    let url = `/decode?ids=${ids}&mode=${mode}`;
    if (mode === 'dssc' && sessionKeyHex) {
      url += `&key=${encodeURIComponent(sessionKeyHex)}`;
    }
    router.push(url);
  };

  const copyToClipboard = async (text: string) => {
    await navigator.clipboard.writeText(text);
    success('Copied', 'Media IDs copied to clipboard');
  };

  if (serverInitializing) {
    return (
      <PageWrapper title="Encode" description="Transform messages into semantic media sequences" fullWidth>
        <div className="max-w-4xl mx-auto text-center py-16">
          <LoadingSpinner size="lg" />
          <p className="text-muted mt-4">Initializing engine...</p>
        </div>
      </PageWrapper>
    );
  }

  if (serverReady === false) {
    return (
      <PageWrapper title="Encode" description="Transform messages into semantic media sequences" fullWidth>
        <Card className="max-w-2xl mx-auto text-center py-12">
          <AlertCircle className="w-12 h-12 text-error mx-auto mb-4" />
          <h3 className="text-heading-md font-semibold text-error mb-2">Server Not Ready</h3>
          <p className="text-muted mb-6">The backend engine is not available. Please start the server:</p>
          <code className="bg-surface px-4 py-2 rounded font-mono text-body-sm text-primary block max-w-xs mx-auto">
            python scripts/start_server.py --reload
          </code>
        </Card>
      </PageWrapper>
    );
  }

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
      title="Encode"
      description="Transform messages into semantic media sequences without transmitting"
      action={
        <div className="flex items-center gap-2">
          {serverReady === true && (
            <Badge variant="success" dot size="sm">Engine Ready</Badge>
          )}
          {serverReady === null && (
            <Badge variant="warning" dot size="sm">Unknown</Badge>
          )}
        </div>
      }
    >
      <div className="max-w-6xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Input Panel */}
          <div className="space-y-6">
            <Card>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-heading-sm font-semibold">Message Input</h3>
                <span className="text-body-sm text-muted font-mono tabular-nums">{message.length} chars</span>
              </div>
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Enter your secret message here..."
                className="input h-32 font-mono text-body-sm resize-none"
                disabled={loading}
              />
              <p className="text-body-xs text-muted mt-2">
                Maximum ~1000 characters for VCP mode. DSSC supports longer messages with higher density.
              </p>
            </Card>

            <Card>
              <h3 className="text-heading-sm font-semibold mb-4">Encoding Settings</h3>
              <div className="space-y-6">
                {/* Mode Selector */}
                <div>
                  <label className="label">Steganography Mode</label>
                  <div className="grid grid-cols-2 gap-3">
                    {(['exact_vcp', 'dssc'] as const).map((m) => {
                      const opt = modeOptions[m];
                      const Icon = opt.icon;
                      return (
                        <Tooltip key={m} content={opt.description} className="w-full">
                          <button
                            type="button"
                            onClick={() => handleModeChange(m)}
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
                  {mode === 'dssc' && (
                    <p className="mt-2 text-body-xs text-purple-300 flex items-center gap-1">
                      <Brain className="w-3.5 h-3.5" />
                      DSSC maps multi-bit symbols per carrier using dynamic state spaces (~15 bits/carrier), significantly reducing the required carrier count.
                    </p>
                  )}
                </div>

                {/* DSSC Session Key */}
                {mode === 'dssc' && (
                  <div>
                    <label className="label">Session Key (Hex)</label>
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={sessionKeyHex}
                        onChange={(e) => setSessionKeyHex(e.target.value)}
                        placeholder="64-character hex key (auto-generated if empty)"
                        className="input font-mono text-body-xs flex-1"
                      />
                      <button
                        type="button"
                        onClick={() => setSessionKeyHex(generateSessionKey())}
                        className="btn-secondary gap-2"
                      >
                        <Key className="w-4 h-4" />
                        New Key
                      </button>
                    </div>
                    <p className="text-body-xs text-muted mt-2">Share this key with the recipient for decoding. Must be 64 hex characters.</p>
                  </div>
                )}

                {/* Diversity Mode (exact_vcp only) */}
                {mode === 'exact_vcp' && (
                  <div>
                    <label className="label">Carrier Diversity Mode</label>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                      {(['best', 'round_robin', 'balanced'] as const).map((m) => (
                        <Tooltip key={m} content={
                          m === 'best' ? 'Always pick highest scoring carrier' :
                          m === 'round_robin' ? 'Cycle through modalities evenly' :
                          'Mix of best match and round robin'
                        } className="w-full">
                          <button
                            type="button"
                            onClick={() => setDiversityMode(m)}
                            className={`w-full p-3.5 rounded-xl border-2 transition-all duration-fast text-left ${
                              diversityMode === m
                                ? 'border-primary bg-primary/5'
                                : 'border-border hover:border-primary/50 hover:bg-surface'
                            }`}
                          >
                            <div className="flex items-center gap-2 whitespace-nowrap">
                              <div className={`w-4 h-4 rounded-full border-2 flex items-center justify-center shrink-0 ${
                                diversityMode === m
                                  ? 'border-primary bg-primary'
                                  : 'border-border'
                              }`}>
                                {diversityMode === m && <div className="w-1.5 h-1.5 rounded-full bg-primary-foreground" />}
                              </div>
                              <span className="font-medium text-body-sm capitalize">{m.replace('_', ' ')}</span>
                            </div>
                          </button>
                        </Tooltip>
                      ))}
                    </div>
                  </div>
                )}

                {/* Modalities */}
                <div>
                  <label className="label">Allowed Modalities</label>
                  <div className="flex flex-wrap gap-3">
                    {['image', 'text', 'audio'].map((mod) => (
                      <Tooltip key={mod} content={`${mod.charAt(0).toUpperCase() + mod.slice(1)} media carriers`}>
                        <button
                          type="button"
                          onClick={() => toggleModality(mod)}
                          className={`inline-flex items-center gap-2.5 px-4 py-2.5 rounded-lg border transition-all duration-fast whitespace-nowrap ${
                            modalities.includes(mod)
                              ? 'border-primary bg-primary/10 text-primary'
                              : 'border-border bg-surface hover:border-primary/50 text-muted'
                          }`}
                        >
                          <span className="text-lg leading-none shrink-0">{modalityIcons[mod as keyof typeof modalityIcons]}</span>
                          <span className="font-medium text-body-sm capitalize leading-none">{mod}</span>
                        </button>
                      </Tooltip>
                    ))}
                  </div>
                </div>
              </div>
            </Card>

            <div className="space-y-3">
              <button
                onClick={handleEncode}
                disabled={loading || !message.trim()}
                className="btn-primary btn-lg w-full justify-center gap-2"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Encoding...
                  </>
                ) : (
                  <>
                    <Shield className="w-5 h-5" />
                    Encode & Generate Sequence
                  </>
                )}
              </button>

              {result && !loading && (
                <button
                  onClick={handleOpenDecode}
                  className="btn-success btn-lg w-full justify-center gap-2"
                >
                  <ExternalLink className="w-5 h-5" />
                  Open in Decode Dashboard
                </button>
              )}

              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-4 bg-error/10 border border-error/20 rounded-xl text-error flex items-start gap-3"
                  role="alert"
                >
                  <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium">Encoding failed</p>
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
                  <p className="text-muted mt-4">Chunking message and searching corpus...</p>
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
                  <h3 className="text-heading-sm font-semibold mb-4">Encoding Results</h3>
                  <div className="grid grid-cols-2 gap-4 mb-6">
                    <StatCard label="Mode" value={result.mode.toUpperCase()} variant="info" />
                    <StatCard label="Elapsed" value={`${result.elapsed_ms.toFixed(1)} ms`} variant="primary" />
                    <StatCard label="Carriers" value={result.media_ids.length} variant="success" />
                    <StatCard label="ECC" value="Active (0% BER)" variant="success" />
                    {result.bits_per_carrier && result.mode === 'dssc' && (
                      <StatCard label="Density" value={`${result.bits_per_carrier} bits/carrier`} variant="warning" />
                    )}
                  </div>

                  <div className="flex flex-wrap gap-2">
                    {result.modality_breakdown && Object.entries(result.modality_breakdown).map(([mod, count]) => (
                      <Badge key={mod} variant="info" className="gap-1.5">
                        <span className={
                          mod === 'image' ? 'text-blue-400' : mod === 'text' ? 'text-emerald-400' : 'text-purple-400'
                        }>
                          {modalityIcons[mod as keyof typeof modalityIcons]}
                        </span>
                        <span className="capitalize">{mod}</span>
                        <span className="font-mono bg-surface px-1.5 py-0.5 rounded text-foreground">{count}</span>
                      </Badge>
                    ))}
                  </div>
                </Card>

                <Card>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-heading-sm font-semibold">Media Sequence & File Locations</h3>
                    <button
                      onClick={() => copyToClipboard(result.media_ids.join(','))}
                      className="btn-ghost btn-sm gap-1"
                    >
                      <Copy className="w-3.5 h-3.5" />
                      Copy IDs
                    </button>
                  </div>
                  <div className="space-y-2 max-h-[500px] overflow-y-auto">
                    {result.encoded.map((item, idx) => (
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
                            <span className="font-mono text-body-xs bg-surface px-1.5 py-0.5 rounded text-foreground">
                              {item.score ? (item.score * 100).toFixed(1) + '%' : '—'}
                            </span>
                          </Badge>
                        </div>
                        {item.content && (
                          <div className="text-body-sm text-muted mb-2 line-clamp-2">{item.content}</div>
                        )}
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
                <Shield className="w-12 h-12 text-muted/30 mx-auto mb-4" />
                <h3 className="text-heading-sm font-semibold text-foreground mb-2">No results yet</h3>
                <p className="text-muted">Enter a message and click "Encode" to generate a semantic media sequence</p>
              </Card>
            )}
          </div>
        </div>
      </div>
    </PageWrapper>
  );
}