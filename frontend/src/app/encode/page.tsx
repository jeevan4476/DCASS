'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Navigation from '@/components/Navigation';
import { Card, Badge, LoadingSpinner } from '@/components/UI';
import { encodeMessage, EncodeResponse, checkReady } from '@/lib/api';
import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
type TransmissionMode = 'static' | 'rl' | 'gan' | 'auto';

export default function EncodePage() {
  const router = useRouter();
  const [message, setMessage] = useState('');
  const [mode, setMode] = useState<'best' | 'round_robin' | 'balanced'>('best');
  const [transmitMode, setTransmitMode] = useState<TransmissionMode>('auto');
  const [modalities, setModalities] = useState(['image', 'text']);
  const [loading, setLoading] = useState(false);
  const [transmitting, setTransmitting] = useState(false);
  const [result, setResult] = useState<EncodeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [serverReady, setServerReady] = useState<boolean | null>(null);
  const [serverInitializing, setServerInitializing] = useState(false);

  // Check if server is ready on mount
  useEffect(() => {
    const checkServerStatus = async () => {
      try {
        const status = await checkReady();
        setServerReady(status.ready);
        setServerInitializing(status.initializing);
        
        // Only poll if actively initializing
        if (status.initializing) {
          const interval = setInterval(async () => {
            try {
              const updatedStatus = await checkReady();
              setServerReady(updatedStatus.ready);
              setServerInitializing(updatedStatus.initializing);
              
              if (updatedStatus.ready || !updatedStatus.initializing) {
                clearInterval(interval);
              }
            } catch (err) {
              // Server might not be fully started yet
              console.log('Waiting for server...');
            }
          }, 2000); // Poll every 2 seconds
          
          return () => clearInterval(interval);
        }
      } catch (err) {
        console.error('Failed to check server status:', err);
        // If we can't reach the server, show warning but don't block
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

  const handleTransmit = async () => {
    if (!result) return;

    try {
      setTransmitting(true);
      setError(null);

      // Clear existing packets first
      await axios.delete(`${API_BASE}/api/wire/packets`);

      // Transmit the sequence with real-time delays
      // speed_multiplier: 1.0 = real-time, 2.0 = 2x faster, etc.
      await axios.post(`${API_BASE}/api/transmit`, {
        media_ids: result.media_ids,
        mode: transmitMode,
        base_delay: 1.5,  // Shorter delays for demo
        num_channels: 3,
        message: message,
        speed_multiplier: 2.0,  // 2x speed for faster demo
      });

      // Navigate to wire view immediately (packets will appear over time)
      router.push('/wire');
    } catch (err: any) {
      console.error('Transmission error:', err);
      setError(err.response?.data?.detail || err.message || 'Transmission failed');
    } finally {
      setTransmitting(false);
    }
  };

  const handleOpenDecode = () => {
    if (!result) return;
    const ids = encodeURIComponent(result.media_ids.join(','));
    const original = encodeURIComponent(message);
    router.push(`/decode?ids=${ids}&original=${original}`);
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

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Transmission Mode
                    </label>
                    <select
                      value={transmitMode}
                      onChange={(e) => setTransmitMode(e.target.value as TransmissionMode)}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg p-3 text-white focus:outline-none focus:border-primary"
                    >
                      <option value="auto">Auto (RL {'->'} GAN {'->'} Static)</option>
                      <option value="rl">RL Preferred</option>
                      <option value="gan">GAN Preferred</option>
                      <option value="static">Static Only</option>
                    </select>
                    <div className="mt-2 text-xs text-gray-500">
                      Works with indices-only setups. If advanced models are unavailable, the backend can fall back automatically.
                    </div>
                  </div>
                </div>
              </Card>

              {/* Server Status Indicator */}
              {serverInitializing && (
                <div className="bg-yellow-500/20 border border-yellow-500/30 rounded-lg p-4 text-yellow-400">
                  <div className="flex items-center space-x-3">
                    <LoadingSpinner />
                    <div>
                      <div className="font-semibold">Initializing DCASS Engine...</div>
                      <div className="text-sm text-yellow-300">Loading CLIP model and indices. This may take 10-30 seconds.</div>
                    </div>
                  </div>
                </div>
              )}
              
              {serverReady === false && !serverInitializing && (
                <div className="bg-yellow-500/20 border border-yellow-500/30 rounded-lg p-4 text-yellow-400">
                  <div className="font-semibold">⚠️ Models Not Pre-loaded</div>
                  <div className="text-sm text-yellow-300">
                    First encoding request will take 10-30 seconds while models load. Subsequent requests will be fast.
                  </div>
                </div>
              )}
              
              {serverReady === null && (
                <div className="bg-error/20 border border-error/30 rounded-lg p-4 text-error">
                  ❌ Cannot connect to server. Please check if the backend is running on port 8000.
                </div>
              )}

              <button
                onClick={handleEncode}
                disabled={loading || !message.trim() || serverReady === null || serverInitializing}
                className="w-full bg-primary hover:bg-primary/90 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-semibold py-4 px-6 rounded-lg transition-colors text-lg"
              >
                {loading ? '🔄 Encoding...' : 
                 serverInitializing ? '⏳ Initializing models...' :
                 serverReady === null ? '❌ Server offline' :
                 '🔐 Encode & Generate Sequence'}
              </button>

              {result && !loading && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <button
                    onClick={handleTransmit}
                    disabled={transmitting}
                    className="w-full bg-success hover:bg-success/90 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-semibold py-4 px-6 rounded-lg transition-colors text-lg"
                  >
                    {transmitting ? '📡 Transmitting...' : '📡 Transmit on Wire View'}
                  </button>
                  <button
                    onClick={handleOpenDecode}
                    className="w-full bg-gray-800 hover:bg-gray-700 border border-gray-700 text-white font-semibold py-4 px-6 rounded-lg transition-colors text-lg"
                  >
                    🔎 Open in Decode
                  </button>
                </div>
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
                        <span className="text-gray-400">Semantic Chunks:</span>
                        <span className="text-white font-semibold">{result.chunks.length}</span>
                      </div>
                    </div>

                    {/* Modality Breakdown */}
                    <div className="mt-6">
                      <h3 className="text-sm font-medium text-gray-300 mb-3">Modality Breakdown</h3>
                      <div className="space-y-2">
                        {Object.entries(result.modality_breakdown).map(([modality, count]) => (
                          <div key={modality} className="flex items-center justify-between">
                            <span className={`capitalize font-medium ${getModalityColor(modality)}`}>
                              {modality}
                            </span>
                            <div className="flex items-center space-x-3">
                              <div className="w-32 bg-gray-800 rounded-full h-2">
                                <div
                                  className={`h-2 rounded-full ${
                                    modality === 'image' ? 'bg-blue-400' :
                                    modality === 'text' ? 'bg-green-400' : 'bg-purple-400'
                                  }`}
                                  style={{ width: `${(count / result.media_ids.length) * 100}%` }}
                                />
                              </div>
                              <span className="text-white font-mono w-8 text-right">{count}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </Card>

                  <Card title="Semantic Chunks">
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {result.chunks.map((chunk, idx) => (
                        <div key={idx} className="bg-gray-800 rounded p-3">
                          <div className="text-xs text-gray-500 mb-1">Chunk {idx + 1}</div>
                          <div className="text-white">{chunk}</div>
                        </div>
                      ))}
                    </div>
                  </Card>

                  <Card title="Media Sequence">
                    <div className="space-y-2 max-h-96 overflow-y-auto">
                      {result.encoded.map((item, idx) => (
                        <div key={idx} className="bg-gray-800 rounded p-3">
                          <div className="flex items-start justify-between mb-2">
                            <div className="flex items-center space-x-2">
                              <span className="text-xs text-gray-500">#{idx + 1}</span>
                              <span className="font-mono text-sm text-primary">{item.media_id}</span>
                            </div>
                            <div className="flex items-center space-x-2">
                              <Badge variant="info">
                                <span className={getModalityColor(item.modality)}>
                                  {item.modality}
                                </span>
                              </Badge>
                              <span className="text-xs text-gray-400">
                                Score: {(item.score * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                          <div className="text-sm text-gray-300 line-clamp-2">
                            {item.content}
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
