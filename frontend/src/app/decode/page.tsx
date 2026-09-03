'use client';

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import Navigation from '@/components/Navigation';
import { Card, Badge, LoadingSpinner } from '@/components/UI';
import { decodeSequence, DecodeResponse } from '@/lib/api';

function DecodeContent() {
  const searchParams = useSearchParams();
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
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Decode failed');
    } finally {
      setLoading(false);
    }
  };

  const getModalityColor = (modality: string) => {
    const colors: Record<string, string> = {
      image: 'text-blue-400',
      text: 'text-green-400',
      audio: 'text-purple-400',
    };
    return colors[modality] || 'text-gray-400';
  };

  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold text-primary">Decode Sequence</h1>
          <Badge variant="info">Bob's Dashboard</Badge>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Input Panel */}
          <div className="space-y-6">
            <Card title="Decoding Settings & Media IDs">
              {/* Mode Selector */}
              <div className="mb-4">
                <label className="block text-xs font-semibold text-gray-400 mb-2 uppercase tracking-wide">
                  Decoding Mode
                </label>
                <div className="flex gap-2">
                  {(['exact_vcp', 'dssc'] as const).map((m) => (
                    <button
                      key={m}
                      type="button"
                      onClick={() => setMode(m)}
                      className={`flex-1 px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
                        mode === m
                          ? 'bg-primary text-white shadow-lg shadow-primary/30 border border-primary'
                          : 'bg-gray-800 text-gray-400 hover:bg-gray-700 border border-gray-700'
                      }`}
                    >
                      {m === 'exact_vcp' ? '🔐 Byte-Exact (VCP)' : '🗜️ Compact Semantic (DSSC)'}
                    </button>
                  ))}
                </div>
              </div>

              {/* DSSC Session Key Input */}
              {mode === 'dssc' && (
                <div className="mb-4">
                  <label className="block text-xs font-semibold text-gray-400 mb-1">
                    Session Key (Hex) — shared by sender
                  </label>
                  <input
                    type="text"
                    value={sessionKeyHex}
                    onChange={(e) => setSessionKeyHex(e.target.value)}
                    placeholder="Paste 64-character hex session key..."
                    className="w-full bg-gray-900 border border-gray-700 rounded p-2.5 text-xs text-purple-300 font-mono focus:outline-none focus:border-primary"
                  />
                </div>
              )}

              <label className="block text-xs font-semibold text-gray-400 mb-1">
                Media Carrier IDs
              </label>
              <textarea
                value={idsInput}
                onChange={(e) => setIdsInput(e.target.value)}
                placeholder="Paste comma- or newline-separated media IDs, e.g. flickr_00123, wiki_00456..."
                className="w-full h-32 bg-gray-800 border border-gray-700 rounded-lg p-4 text-white placeholder-gray-500 focus:outline-none focus:border-primary resize-none font-mono text-sm"
              />
              <div className="mt-2 text-sm text-gray-400">
                {parseIds(idsInput).length} IDs detected
              </div>
            </Card>

            <button
              onClick={handleDecode}
              disabled={loading || parseIds(idsInput).length === 0}
              className="w-full bg-primary hover:bg-primary/90 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-semibold py-4 px-6 rounded-lg transition-colors text-lg"
            >
              {loading ? '🔄 Decoding...' : '🔓 Decode Sequence'}
            </button>

            {error && (
              <div className="bg-error/20 border border-error/30 rounded-lg p-4 text-error">
                ❌ {error}
              </div>
            )}
          </div>

          {/* Results Panel */}
          <div className="space-y-6">
            {loading && (
              <Card title="Decoding in Progress">
                <LoadingSpinner />
                <p className="text-center text-gray-400 mt-4">
                  Looking up media items and reconstructing meaning...
                </p>
              </Card>
            )}

            {result && !loading && (
              <>
                <Card title="Reconstructed Meaning">
                  <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
                    <p className="text-white font-medium text-lg whitespace-pre-wrap">
                      {result.reconstructed_meaning || '(empty)'}
                    </p>
                  </div>
                </Card>

                <Card title="Verification & Recovery">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Decoding Mode:</span>
                      <Badge variant="info">{result.mode.toUpperCase()}</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Error Recovery Status:</span>
                      <Badge variant={result.ecc_success ? "success" : "error"}>
                        {result.ecc_success ? "0% BER (100% RECOVERED)" : "DECODING FAILED"}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Verification Rate:</span>
                      <span className="font-mono font-semibold text-success">
                        {(result.verification_rate * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Elapsed Time:</span>
                      <span className="text-white font-mono">{result.elapsed_ms.toFixed(1)} ms</span>
                    </div>
                  </div>
                </Card>

                <Card title="Decoded Items & File Locations">
                  <div className="space-y-2 max-h-96 overflow-y-auto">
                    {result.items.map((item, idx) => (
                      <div key={idx} className="bg-gray-800 rounded p-3 space-y-1">
                        <div className="flex items-start justify-between">
                          <div className="flex items-center space-x-2">
                            <span className="text-xs text-gray-500">#{idx + 1}</span>
                            <span className="font-mono text-sm text-primary">{item.media_id}</span>
                          </div>
                          <Badge variant="info">
                            <span className={getModalityColor(item.modality)}>{item.modality}</span>
                          </Badge>
                        </div>
                        <div className="text-sm text-gray-300">{item.content || '(no content)'}</div>
                        {item.file_path && (
                          <div className="text-xs text-gray-500 font-mono break-all pt-1 border-t border-gray-700/50">
                            File: {item.file_path}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </Card>
              </>
            )}

            {!result && !loading && (
              <Card title="Results">
                <div className="text-center text-gray-500 py-12">
                  Paste media IDs and click "Decode Sequence"
                </div>
              </Card>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}

export default function DecodePage() {
  return (
    <>
      <Navigation />
      <Suspense fallback={<div className="min-h-screen bg-background p-8 text-center text-gray-400">Loading...</div>}>
        <DecodeContent />
      </Suspense>
    </>
  );
}
