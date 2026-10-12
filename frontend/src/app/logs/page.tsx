'use client';

import { useEffect, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Database, Server, Cpu, HardDrive, Shield, Activity, AlertTriangle, CheckCircle, RefreshCw, Loader2 } from 'lucide-react';
import { PageWrapper, Card, Badge, LoadingSpinner, StatCard, SkeletonCard, Section, StatGrid } from '@/components';
import { getStatus, checkReady, StatusResponse } from '@/lib/api';

interface SystemMetrics {
  totalItems: number;
  device: string;
  indices: StatusResponse['indices'];
  stealthModels: StatusResponse['stealth_models'];
}

export default function LogsPage() {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [ready, setReady] = useState<{ ready: boolean; initializing: boolean; encoder_loaded: boolean; decoder_loaded: boolean } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setError(null);
      const [statusData, readyData] = await Promise.allSettled([
        getStatus(),
        checkReady(),
      ]);

      if (statusData.status === 'fulfilled') {
        setMetrics({
          totalItems: statusData.value.total_items,
          device: statusData.value.device,
          indices: statusData.value.indices,
          stealthModels: statusData.value.stealth_models,
        });
      }
      if (readyData.status === 'fulfilled') {
        setReady(readyData.value);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to fetch system data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const indicesLoaded = metrics
    ? Object.values(metrics.indices).filter((i) => i.status === 'ok').length
    : 0;
  const indicesTotal = metrics ? Object.keys(metrics.indices).length : 3;

  const modalityColors = {
    image: 'text-blue-400',
    text: 'text-green-400',
    audio: 'text-purple-400',
  };

  if (loading && !metrics) {
    return (
      <PageWrapper title="System Logs" description="Real-time system audit and diagnostics" fullWidth>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {Array.from({ length: 4 }).map((_, i) => (
            <SkeletonCard key={i} lines={2} />
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <SkeletonCard lines={4} className="h-80" />
          <SkeletonCard lines={4} className="h-80" />
        </div>
      </PageWrapper>
    );
  }

  if (error) {
    return (
      <PageWrapper
        title="System Logs"
        description="Real-time system audit and diagnostics"
        action={
          <button onClick={fetchData} className="btn-primary btn-sm gap-2">
            <RefreshCw className="w-4 h-4" />
            Retry
          </button>
        }
      >
        <Card className="max-w-2xl mx-auto">
          <div className="flex items-start gap-4 p-4">
            <AlertTriangle className="w-6 h-6 text-error flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-semibold text-error">Connection Error</h3>
              <p className="text-body-sm text-muted mt-1">{error}</p>
              <p className="text-body-xs text-muted mt-2">
                Ensure the backend is running: <code className="bg-surface px-1.5 py-0.5 rounded text-primary font-mono">python scripts/start_server.py --reload</code>
              </p>
            </div>
          </div>
        </Card>
      </PageWrapper>
    );
  }

  const stats = [
    {
      label: 'Corpus Vectors',
      value: metrics?.totalItems.toLocaleString() || '0',
      variant: 'primary' as const,
      icon: <Database className="w-5 h-5" />,
    },
    {
      label: 'Indices Loaded',
      value: `${indicesLoaded}/${indicesTotal}`,
      variant: indicesLoaded === indicesTotal ? ('success' as const) : ('warning' as const),
      icon: <Shield className="w-5 h-5" />,
    },
    {
      label: 'Device',
      value: metrics?.device?.toUpperCase() || 'N/A',
      variant: 'info' as const,
      icon: <Cpu className="w-5 h-5" />,
    },
    {
      label: 'API Health',
      value: ready?.ready === true ? 'Ready' : ready?.initializing ? 'Initializing' : 'Unknown',
      variant: ready?.ready === true ? ('success' as const) : ('warning' as const),
      icon: <Server className="w-5 h-5" />,
    },
  ];

  return (
    <PageWrapper
      title="System Logs"
      description="Real-time system audit and diagnostics"
      action={
        <button onClick={fetchData} className="btn-ghost btn-sm gap-2">
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      }
    >
      <Section title="System Metrics">
        <StatGrid stats={stats} columns={4} />
      </Section>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <Card title="Corpus Indices" subtitle={`${indicesLoaded}/${indicesTotal} indices loaded`}>
          <div className="space-y-3">
            {metrics && Object.entries(metrics.indices).map(([modality, info]) => (
              <motion.div
                key={modality}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 }}
                className="flex items-center justify-between p-3 bg-surface/50 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <div
                    className="w-10 h-10 rounded-lg flex items-center justify-center"
                    style={{ backgroundColor: `var(--color-${modalityColors[modality as keyof typeof modalityColors].replace('text-', '').replace('-400', '')}-muted)` }}
                  >
                    <span className="text-lg">{modality === 'image' ? '🖼' : modality === 'text' ? '📝' : '🔊'}</span>
                  </div>
                  <div>
                    <p className="font-medium text-foreground capitalize">{modality}</p>
                    <p className="text-body-sm text-muted">
                      {info.status === 'ok' ? `${info.count?.toLocaleString()} vectors` : info.status === 'missing' ? 'Index not built' : `Error: ${info.error}`}
                    </p>
                  </div>
                </div>
                <Badge
                  variant={
                    info.status === 'ok' ? 'success' : info.status === 'missing' ? 'warning' : 'error'
                  }
                  dot
                >
                  {info.status === 'ok' ? 'Loaded' : info.status === 'missing' ? 'Missing' : 'Error'}
                </Badge>
              </motion.div>
            ))}
          </div>
          {indicesLoaded < indicesTotal && (
            <div className="mt-4 p-3 bg-warning/10 border border-warning/20 rounded-lg">
              <p className="text-body-sm text-warning flex items-center gap-2">
                <AlertTriangle className="w-4 h-4" />
                Some indices are missing. Run index build to enable full modality support.
              </p>
            </div>
          )}
        </Card>

        <Card title="Core Modules" subtitle="Engine component status">
          <div className="space-y-3">
            {[
              { name: 'FAISS Vector Index', desc: '512d Multi-Modal Vectors', status: 'ok', detail: metrics?.totalItems.toLocaleString() || '0', icon: <Database className="w-4 h-4" /> },
              { name: 'Voronoi Codebook (VCP)', desc: '256 Centroids, Unit Norm', status: 'ok', detail: '||c||₂ = 1.0', icon: <Shield className="w-4 h-4" /> },
              { name: 'Reed-Solomon ECC', desc: 'GF(2⁸) Berlekamp-Massey', status: 'ok', detail: '0% Bit Error Rate', icon: <Activity className="w-4 h-4" /> },
              { name: 'Stealth Scheduler', desc: 'RL/GAN/Static Fallback', status: metrics?.stealthModels.rl_checkpoint || metrics?.stealthModels.gan_checkpoint ? 'ok' : 'warning', detail: metrics?.stealthModels.rl_checkpoint ? 'RL Ready' : metrics?.stealthModels.gan_checkpoint ? 'GAN Ready' : 'Static Only', icon: <Cpu className="w-4 h-4" /> },
              { name: 'FastAPI Server', desc: 'REST + WebSocket', status: ready?.ready === true ? 'ok' : ready?.initializing ? 'warning' : 'error', detail: 'Port 8000', icon: <Server className="w-4 h-4" /> },
            ].map((module) => (
              <motion.div
                key={module.name}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 }}
                className="flex items-center justify-between p-3 bg-surface/50 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
                    {module.icon}
                  </div>
                  <div>
                    <p className="font-medium text-foreground">{module.name}</p>
                    <p className="text-body-sm text-muted">{module.desc}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-body-sm text-muted font-mono">{module.detail}</span>
                  <Badge
                    variant={module.status === 'ok' ? 'success' : module.status === 'warning' ? 'warning' : 'error'}
                    dot
                    size="sm"
                  >
                    {module.status === 'ok' ? 'Online' : module.status === 'warning' ? 'Degraded' : 'Offline'}
                  </Badge>
                </div>
              </motion.div>
            ))}
          </div>
        </Card>
      </div>

      <Section title="Technical Specifications" columns={2}>
        <Card>
          <dl className="space-y-3">
            {[
              ['Vector Dimension', '512'],
              ['Index Type', 'FAISS IVF (Inverted File)'],
              ['Distance Metric', 'L2 (Euclidean)'],
              ['Codebook Centroids', '256'],
              ['Codebook Norm', 'Unit (||c||₂ = 1.0)'],
              ['ECC Algorithm', 'Reed-Solomon GF(2⁸)'],
              ['ECC Parity Bytes', '8 (configurable)'],
              ['Max Payload/Carrier', '1 byte (VCP) / ~15 bits (DSSC)'],
            ].map(([label, value]) => (
              <div key={label} className="flex justify-between py-2 border-b border-border/50 last:border-0">
                <dt className="text-body-sm text-muted">{label}</dt>
                <dd className="font-mono text-body-sm text-foreground">{value}</dd>
              </div>
            ))}
          </dl>
        </Card>
        <Card>
          <dl className="space-y-3">
            {[
              ['Default Policy', 'round_robin'],
              ['Base Delay', '3 seconds'],
              ['Channels', '3'],
              ['Context Bucket', '3600 seconds'],
              ['Device', metrics?.device?.toUpperCase() || 'N/A'],
              ['Precision', 'FP32'],
              ['Stealth Modes', 'RL → GAN → Static'],
              ['Session Key (DSSC)', '32 bytes (256-bit)'],
            ].map(([label, value]) => (
              <div key={label} className="flex justify-between py-2 border-b border-border/50 last:border-0">
                <dt className="text-body-sm text-muted">{label}</dt>
                <dd className="font-mono text-body-sm text-foreground">{value}</dd>
              </div>
            ))}
          </dl>
        </Card>
      </Section>
    </PageWrapper>
  );
}