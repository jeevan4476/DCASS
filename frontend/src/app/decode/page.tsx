'use client';

import { useState } from 'react';
import Navigation from '@/components/Navigation';
import { Card, Badge, LoadingSpinner } from '@/components/UI';
import { decodeSequence, DecodeResponse } from '@/lib/api';
import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function DecodePage() {
  const [idsInput, setIdsInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [importing, setImporting] = useState(false);
  const [result, setResult] = useState<DecodeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

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

    try {
      setLoading(true);
      setError(null);
      setResult(null);
      const response = await decodeSequence({ media_ids });
      setResult(response);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Decode failed');
    } finally {
      setLoading(false);
    }
  };

  const handleImportFromWire = async () => {
    try {
      setImporting(true);
      setError(null);
      const response = await axios.get(`${API_BASE}/api/wire/packets`);
      const packets = response.data.packets || [];
      if (packets.length === 0) {
        setError('No packets on the wire. Transmit a message first.');
        return;
      }
      // Sort by sequence_number to reconstruct the original order
      const sorted = [...packets].sort(
        (a: any, b: any) => (a.sequence_number ?? 0) - (b.sequence_number ?? 0),
      );
      const ids = sorted.map((p: any) => p.media_id).filter(Boolean);
      setIdsInput(ids.join(', '));
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Wire import failed');
    } finally {
      setImporting(false);
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
    <>
      <Navigation />
      <main className="min-h-screen bg-background">
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-between mb-8">
            <h1 className="text-3xl font-bold text-primary">Decode Sequence</h1>
            <Badge variant="info">Bob's Dashboard</Badge>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Input Panel */}
            <div className="space-y-6">
              <Card title="Media IDs">
                <textarea
                  value={idsInput}
                  onChange={(e) => setIdsInput(e.target.value)}
                  placeholder="Paste comma- or newline-separated media IDs, e.g. flickr_00123, wiki_00456, ..."
                  className="w-full h-40 bg-gray-800 border border-gray-700 rounded-lg p-4 text-white placeholder-gray-500 focus:outline-none focus:border-primary resize-none font-mono text-sm"
                />
                <div className="mt-2 flex items-center justify-between text-sm text-gray-400">
                  <span>{parseIds(idsInput).length} IDs detected</span>
                  <button
                    onClick={handleImportFromWire}
                    disabled={importing}
                    className="text-primary hover:text-primary/80 disabled:text-gray-600"
                  >
                    {importing ? 'Importing…' : '⇣ Import from Wire View'}
                  </button>
                </div>
              </Card>

              <button
                onClick={handleDecode}
                disabled={loading || parseIds(idsInput).length === 0}
                className="w-full bg-primary hover:bg-primary/90 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-semibold py-4 px-6 rounded-lg transition-colors text-lg"
              >
                {loading ? '🔄 Decoding…' : '🔓 Decode Sequence'}
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
                    Looking up media items and reconstructing meaning…
                  </p>
                </Card>
              )}

              {result && !loading && (
                <>
                  <Card title="Reconstructed Meaning">
                    <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
                      <p className="text-white whitespace-pre-wrap">
                        {result.reconstructed_meaning || '(empty)'}
                      </p>
                    </div>
                  </Card>

                  <Card title="Verification">
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400">Verification Rate:</span>
                        <span
                          className={`font-mono font-semibold ${
                            result.verification_rate === 1
                              ? 'text-success'
                              : result.verification_rate > 0.5
                              ? 'text-warning'
                              : 'text-error'
                          }`}
                        >
                          {(result.verification_rate * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400">All Verified:</span>
                        <Badge variant={result.all_verified ? 'success' : 'warning'}>
                          {result.all_verified ? 'YES' : 'NO'}
                        </Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400">Elapsed:</span>
                        <span className="text-white font-mono">
                          {result.elapsed_ms.toFixed(1)} ms
                        </span>
                      </div>
                    </div>
                  </Card>

                  <Card title="Decoded Items">
                    <div className="space-y-2 max-h-96 overflow-y-auto">
                      {result.items.map((item, idx) => (
                        <div key={idx} className="bg-gray-800 rounded p-3">
                          <div className="flex items-start justify-between mb-2">
                            <div className="flex items-center space-x-2">
                              <span className="text-xs text-gray-500">#{idx + 1}</span>
                              <span className="font-mono text-sm text-primary">
                                {item.media_id}
                              </span>
                            </div>
                            <div className="flex items-center space-x-2">
                              <Badge variant="info">
                                <span className={getModalityColor(item.modality)}>
                                  {item.modality}
                                </span>
                              </Badge>
                              <Badge variant={item.verified ? 'success' : 'error'}>
                                {item.verified ? 'VERIFIED' : 'MISSING'}
                              </Badge>
                            </div>
                          </div>
                          <div className="text-sm text-gray-300 line-clamp-2">
                            {item.content || '(no content)'}
                          </div>
                        </div>
                      ))}
                    </div>
                  </Card>
                </>
              )}

              {!result && !loading && (
                <Card title="Results">
                  <div className="text-center text-gray-500 py-12">
                    Paste media IDs (or import from Wire View) and click
                    "Decode Sequence".
                  </div>
                </Card>
              )}
            </div>
          </div>
        </div>
      </main>
    </>
  );
}
