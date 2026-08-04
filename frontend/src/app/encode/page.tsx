'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Navigation from '@/components/Navigation';
import { Card, Badge, LoadingSpinner } from '@/components/UI';
import { encodeMessage, EncodeResponse, checkReady } from '@/lib/api';

export default function EncodePage() {
  const router = useRouter();
  const [message, setMessage] = useState('');
  const [mode, setMode] = useState<'best' | 'round_robin' | 'balanced'>('best');
  const [modalities, setModalities] = useState(['image', 'text', 'audio']);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EncodeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [serverReady, setServerReady] = useState<boolean | null>(null);
  const [serverInitializing, setServerInitializing] = useState(false);

  useEffect(() => {
    const checkServerStatus = async () => {
      try {
        const status = await checkReady();
        setServerReady(status.ready);
        setServerInitializing(status.initializing);
      } catch (err) {
        setServerReady(null);
      }
    };
    checkServerStatus();
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
        modalities,
        use_ecc: true,
      });

      setResult(response);
    } catch (err: any) {
      console.error('Encoding error:', err);
      setError(err.response?.data?.detail || err.message || 'Encoding failed');
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
    const rawHex = result.raw_codeword_hex ? encodeURIComponent(result.raw_codeword_hex) : '';
    router.push(`/decode?ids=${ids}&raw_hex=${rawHex}`);
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
            <h1 className="text-3xl font-bold text-primary">Encode Message</h1>
            <Badge variant="info">Alice's Dashboard</Badge>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Input Panel */}
            <div className="space-y-6">
              <Card title="Message Input">
                <textarea
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="Enter your secret message here..."
                  className="w-full h-32 bg-gray-800 border border-gray-700 rounded-lg p-4 text-white placeholder-gray-500 focus:outline-none focus:border-primary resize-none"
                />
                <div className="mt-2 text-sm text-gray-400">
                  {message.length} characters
                </div>
              </Card>

              <Card title="Encoding Settings">
                <div className="space-y-4">
                  {/* Diversity Mode */}
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Diversity Mode
                    </label>
                    <select
                      value={mode}
                      onChange={(e) => setMode(e.target.value as any)}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg p-3 text-white focus:outline-none focus:border-primary"
                    >
                      <option value="best">Best Match (Highest accuracy)</option>
                      <option value="round_robin">Round Robin (Balanced modalities)</option>
                      <option value="balanced">Balanced (Mix of both)</option>
                    </select>
                  </div>

                  {/* Modalities */}
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Modalities
                    </label>
                    <div className="flex flex-wrap gap-3">
                      {['image', 'text', 'audio'].map((mod) => (
                        <button
                          key={mod}
                          onClick={() => toggleModality(mod)}
                          className={`px-4 py-2 rounded-lg border transition-all ${
                            modalities.includes(mod)
                              ? 'border-primary bg-primary/20 text-primary'
                              : 'border-gray-700 bg-gray-800 text-gray-400 hover:border-gray-600'
                          }`}
                        >
                          <span className="capitalize">{mod}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </Card>

              <button
                onClick={handleEncode}
                disabled={loading || !message.trim()}
                className="w-full bg-primary hover:bg-primary/90 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-semibold py-4 px-6 rounded-lg transition-colors text-lg"
              >
                {loading ? '🔄 Encoding...' : '🔐 Encode & Generate Sequence'}
              </button>

              {result && !loading && (
                <button
                  onClick={handleOpenDecode}
                  className="w-full bg-success hover:bg-success/90 text-white font-semibold py-4 px-6 rounded-lg transition-colors text-lg"
                >
                  🔓 Send to Decode Dashboard
                </button>
              )}

              {error && (
                <div className="bg-error/20 border border-error/30 rounded-lg p-4 text-error">
                  ❌ {error}
                </div>
              )}
            </div>

            {/* Results Panel */}
            <div className="space-y-6">
              {loading && (
                <Card title="Encoding in Progress">
                  <LoadingSpinner />
                  <p className="text-center text-gray-400 mt-4">
                    Chunking message and searching corpus...
                  </p>
                </Card>
              )}

              {result && !loading && (
                <>
                  <Card title="Encoding Results">
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400">Elapsed Time:</span>
                        <span className="text-white font-mono">{result.elapsed_ms.toFixed(1)} ms</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400">Media Items:</span>
                        <span className="text-white font-semibold">{result.media_ids.length}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400">Error Correction (RS-ECC):</span>
                        <Badge variant="success">ACTIVE (0% BER)</Badge>
                      </div>
                    </div>
                  </Card>

                  <Card title="Media Sequence & File Locations">
                    <div className="space-y-3 max-h-96 overflow-y-auto">
                      {result.encoded.map((item, idx) => (
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
                          <div className="text-sm text-gray-300">{item.content}</div>
                          {item.file_path && (
                            <div className="text-xs text-gray-500 font-mono break-all pt-1 border-t border-gray-700/50">
                              📂 {item.file_path}
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
                    Enter a message and click "Encode" to see results
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
