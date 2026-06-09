'use client';

import { useEffect, useState } from 'react';
import Navigation from '@/components/Navigation';
import { Card, StatCard, Badge, LoadingSpinner } from '@/components/UI';
import { getStatus, healthCheck, StatusResponse } from '@/lib/api';

export default function StatusPage() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [health, setHealth] = useState<{ status: string } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchStatus() {
      try {
        setLoading(true);
        setError(null);
        
        const [healthData, statusData] = await Promise.all([
          healthCheck(),
          getStatus(),
        ]);

        setHealth(healthData);
        setStatus(statusData);
      } catch (err: any) {
        console.error('Error fetching status:', err);
        setError(err.message || 'Failed to connect to backend');
      } finally {
        setLoading(false);
      }
    }

    fetchStatus();
    const interval = setInterval(fetchStatus, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, []);

  if (loading && !status) {
    return (
      <>
        <Navigation />
        <main className="min-h-screen bg-background">
          <div className="container mx-auto px-4 py-8">
            <h1 className="text-3xl font-bold text-primary mb-8">System Status</h1>
            <LoadingSpinner />
          </div>
        </main>
      </>
    );
  }

  if (error) {
    return (
      <>
        <Navigation />
        <main className="min-h-screen bg-background">
          <div className="container mx-auto px-4 py-8">
            <h1 className="text-3xl font-bold text-primary mb-8">System Status</h1>
            <div className="bg-error/20 border border-error/30 rounded-lg p-6 text-error">
              <h2 className="text-lg font-semibold mb-2">❌ Connection Error</h2>
              <p>{error}</p>
              <p className="mt-4 text-sm">
                Make sure the backend server is running:
                <code className="block mt-2 bg-gray-900 p-2 rounded">
                  python scripts/start_server.py --reload
                </code>
              </p>
            </div>
          </div>
        </main>
      </>
    );
  }

  const stealthMode = getStealthMode(status?.stealth_models);

  return (
    <>
      <Navigation />
      <main className="min-h-screen bg-background">
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-between mb-8">
            <h1 className="text-3xl font-bold text-primary">System Status</h1>
            <Badge variant={health?.status === 'ok' ? 'success' : 'error'}>
              {health?.status === 'ok' ? '🟢 Online' : '🔴 Offline'}
            </Badge>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <StatCard
              label="Total Corpus Items"
              value={status?.total_items.toLocaleString() || '0'}
              status="success"
            />
            <StatCard
              label="Device"
              value={status?.device.toUpperCase() || 'N/A'}
              status="neutral"
            />
            <StatCard
              label="Indices Loaded"
              value={`${Object.values(status?.indices || {}).filter(i => i.status === 'ok').length}/3`}
              status={Object.values(status?.indices || {}).every(i => i.status === 'ok') ? 'success' : 'warning'}
            />
            <StatCard
              label="Stealth Mode"
              value={stealthMode}
              status={stealthMode === 'Static' ? 'warning' : 'success'}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <Card title="Corpus Indices">
              <div className="space-y-4">
                {status && Object.entries(status.indices).map(([modality, info]) => (
                  <div key={modality} className="flex items-center justify-between">
                    <div>
                      <div className="text-white font-medium capitalize">{modality}</div>
                      <div className="text-sm text-gray-400">
                        {info.status === 'ok' && `${info.count?.toLocaleString()} items`}
                        {info.status === 'missing' && 'Index not built'}
                        {info.status === 'error' && `Error: ${info.error}`}
                      </div>
                    </div>
                    <Badge variant={
                      info.status === 'ok' ? 'success' :
                      info.status === 'missing' ? 'warning' : 'error'
                    }>
                      {info.status === 'ok' ? '✓' : info.status === 'missing' ? '⊘' : '✗'}
                    </Badge>
                  </div>
                ))}
              </div>
            </Card>

            <Card title="Stealth Models">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-white font-medium">GAN Scheduler</div>
                    <div className="text-sm text-gray-400">
                      Temporal Pattern Generator
                    </div>
                  </div>
                  <Badge variant={status?.stealth_models.gan_checkpoint ? 'success' : 'error'}>
                    {status?.stealth_models.gan_checkpoint ? '🟢 Ready' : '🔴 Not Trained'}
                  </Badge>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-white font-medium">RL Agent</div>
                    <div className="text-sm text-gray-400">
                      PPO Policy Optimizer
                    </div>
                  </div>
                  <Badge variant={status?.stealth_models.rl_checkpoint ? 'success' : 'error'}>
                    {status?.stealth_models.rl_checkpoint ? '🟢 Ready' : '🔴 Not Trained'}
                  </Badge>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-white font-medium">Static Fallback</div>
                    <div className="text-sm text-gray-400">
                      NoiseController (Mathematical)
                    </div>
                  </div>
                  <Badge variant="success">
                    🟢 Always Available
                  </Badge>
                </div>
              </div>

              {!status?.stealth_models.gan_checkpoint && !status?.stealth_models.rl_checkpoint && (
                <div className="mt-6 bg-warning/20 border border-warning/30 rounded-lg p-4">
                  <div className="text-warning text-sm">
                    ⚠️ <strong>Dynamic models not trained.</strong> System will use static fallback mode with mathematical noise injection.
                  </div>
                </div>
              )}
            </Card>
          </div>

          <Card title="Backend Configuration">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-400">API Endpoint:</span>
                <span className="ml-2 font-mono text-primary">http://localhost:8000/api</span>
              </div>
              <div>
                <span className="text-gray-400">Default Policy:</span>
                <span className="ml-2 font-mono text-white">round_robin</span>
              </div>
              <div>
                <span className="text-gray-400">Base Delay:</span>
                <span className="ml-2 font-mono text-white">3 seconds</span>
              </div>
              <div>
                <span className="text-gray-400">Channels:</span>
                <span className="ml-2 font-mono text-white">3 channels</span>
              </div>
            </div>
          </Card>
        </div>
      </main>
    </>
  );
}

function getStealthMode(models?: { gan_checkpoint: boolean; rl_checkpoint: boolean }): string {
  if (!models) return 'Unknown';
  if (models.rl_checkpoint) return 'RL';
  if (models.gan_checkpoint) return 'GAN';
  return 'Static';
}
