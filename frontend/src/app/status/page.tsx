'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import {
  Database,
  Server,
  Cpu,
  HardDrive,
  Network,
  Shield,
  Users,
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Loader2,
  TrendingUp,
  TrendingDown,
  Minus,
  Zap,
  Brain,
  Eye,
  Settings,
  RefreshCw,
  Send,
  Mail,
  Terminal,
} from 'lucide-react';

import { PageWrapper, Section, StatGrid, DataTable, ProgressBar, Tabs, Card, Badge, LoadingSpinner, SkeletonCard, EmptyState } from '@/components';
import { getStatus, healthCheck, StatusResponse, getInbox, listUsers, getWirePackets, clearWirePackets, getTransmissionStatus } from '@/lib/api';
import { useAuth } from '@/lib/auth';
import { useToastHelpers } from '@/components/Toast';

interface SystemMetrics {
  totalItems: number;
  device: string;
  indices: StatusResponse['indices'];
  stealthModels: StatusResponse['stealth_models'];
}

interface UserInfo {
  id: number;
  username: string;
}

interface WirePacket {
  media_id: string;
  channel_id: number;
  sequence_number: number;
  delay_seconds: number;
  timestamp: number;
  mode_used: string;
  filename: string;
}

interface TransmissionStatus {
  active: boolean;
  current: number;
  total: number;
  status: string;
}

export default function StatusPage() {
  const { user } = useAuth();
  const { success, error: showError, info } = useToastHelpers();

  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [health, setHealth] = useState<{ status: string } | null>(null);
  const [users, setUsers] = useState<UserInfo[]>([]);
  const [wirePackets, setWirePackets] = useState<WirePacket[]>([]);
  const [transmissionStatus, setTransmissionStatus] = useState<TransmissionStatus>({
    active: false,
    current: 0,
    total: 0,
    status: 'idle',
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'system' | 'wire' | 'users'>('overview');
  const [refreshing, setRefreshing] = useState(false);

  const fetchAllData = React.useCallback(async (showToast = false) => {
    try {
      if (showToast) setRefreshing(true);
      setError(null);

      const [healthData, statusData, usersData, wireData, txStatusData] = await Promise.allSettled([
        healthCheck(),
        getStatus(),
        listUsers(),
        getWirePackets(),
        getTransmissionStatus(),
      ]);

      if (healthData.status === 'fulfilled') setHealth(healthData.value);
      if (statusData.status === 'fulfilled') {
        setMetrics({
          totalItems: statusData.value.total_items,
          device: statusData.value.device,
          indices: statusData.value.indices,
          stealthModels: statusData.value.stealth_models,
        });
      }
      if (usersData.status === 'fulfilled') setUsers(usersData.value.users);
      if (wireData.status === 'fulfilled') setWirePackets(wireData.value.packets || []);
      if (txStatusData.status === 'fulfilled') setTransmissionStatus(txStatusData.value);

      if (showToast) success('Refreshed', 'System data updated');
    } catch (err: any) {
      setError(err.message || 'Failed to fetch system data');
      if (showToast) showError('Refresh failed', err.message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [success, showError]);

  useEffect(() => {
    fetchAllData();
    const interval = setInterval(() => fetchAllData(false), 10000);
    return () => clearInterval(interval);
  }, [fetchAllData]);

  const stealthMode = metrics?.stealthModels
    ? metrics.stealthModels.rl_checkpoint
      ? 'RL'
      : metrics.stealthModels.gan_checkpoint
      ? 'GAN'
      : 'Static'
    : 'Unknown';

  const indicesLoaded = metrics
    ? Object.values(metrics.indices).filter((i) => i.status === 'ok').length
    : 0;
  const indicesTotal = metrics ? Object.keys(metrics.indices).length : 3;

  const recentPackets = wirePackets.slice(-50).reverse();

  const stats = [
    {
      label: 'Corpus Items',
      value: metrics?.totalItems.toLocaleString() || '0',
      variant: 'primary' as const,
      icon: <Database className="w-5 h-5" />,
      trend: { value: 0, label: '24h' },
    },
    {
      label: 'Indices Loaded',
      value: `${indicesLoaded}/${indicesTotal}`,
      variant: indicesLoaded === indicesTotal ? ('success' as const) : ('warning' as const),
      icon: <Shield className="w-5 h-5" />,
    },
    {
      label: 'Stealth Mode',
      value: stealthMode,
      variant: stealthMode === 'Static' ? ('warning' as const) : ('success' as const),
      icon: <Brain className="w-5 h-5" />,
    },
    {
      label: 'Device',
      value: metrics?.device?.toUpperCase() || 'N/A',
      variant: 'info' as const,
      icon: <Cpu className="w-5 h-5" />,
    },
    {
      label: 'Active Users',
      value: users.length,
      variant: 'success' as const,
      icon: <Users className="w-5 h-5" />,
    },
    {
      label: 'Wire Packets',
      value: wirePackets.length.toLocaleString(),
      variant: 'info' as const,
      icon: <Activity className="w-5 h-5" />,
    },
    {
      label: 'Transmission',
      value: transmissionStatus.active ? 'Active' : 'Idle',
      variant: transmissionStatus.active ? ('success' as const) : ('neutral' as const),
      icon: <Zap className="w-5 h-5" />,
    },
    {
      label: 'API Health',
      value: health?.status === 'ok' ? 'Healthy' : 'Degraded',
      variant: health?.status === 'ok' ? ('success' as const) : ('error' as const),
      icon: <Server className="w-5 h-5" />,
    },
  ];

  if (loading && !metrics) {
    return (
      <PageWrapper title="Panel" description="System monitoring and administration dashboard">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {Array.from({ length: 8 }).map((_, i) => (
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
        title="Panel"
        description="System monitoring and administration dashboard"
        action={
          <button onClick={() => fetchAllData(true)} className="btn-primary btn-sm" disabled={refreshing}>
            <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            Retry
          </button>
        }
      >
        <Card>
          <div className="flex items-center gap-4 p-4">
            <AlertTriangle className="w-6 h-6 text-error flex-shrink-0" />
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

  const modalityColors = {
    image: 'text-blue-400',
    text: 'text-green-400',
    audio: 'text-purple-400',
  };

  const modeColors: Record<string, 'primary' | 'success' | 'warning' | 'error' | 'info' | 'neutral'> = {
    rl: 'success',
    gan: 'info',
    static: 'warning',
    auto: 'primary',
  };

  const getModeColor = (mode: string): 'primary' | 'success' | 'warning' | 'error' | 'info' | 'neutral' => 
    modeColors[mode.toLowerCase()] || 'neutral';

  return (
    <PageWrapper
      title="Panel"
      description="System monitoring, wire telemetry, and administration"
      action={
        <button
          onClick={() => fetchAllData(true)}
          className="btn-secondary btn-sm"
          disabled={refreshing}
        >
          <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      }
    >
      {/* Transmission Progress Banner */}
      {transmissionStatus.active && (
        <motion.div
          className="mb-6"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card className="bg-primary/5 border-primary/20">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-primary animate-pulse-soft" />
                <span className="font-semibold text-primary">Transmission in Progress</span>
              </div>
              <span className="text-body-sm text-muted tabular-nums">
                {transmissionStatus.current} / {transmissionStatus.total} packets
              </span>
            </div>
            <ProgressBar
              value={transmissionStatus.current}
              max={transmissionStatus.total}
              variant="primary"
              size="md"
              animated
            />
          </Card>
        </motion.div>
      )}

      {/* Tabs */}
      <Tabs
        tabs={[
          { id: 'overview', label: 'Overview', icon: <Activity className="w-4 h-4" /> },
          { id: 'system', label: 'System', icon: <Server className="w-4 h-4" /> },
          { id: 'wire', label: 'Wire', icon: <Network className="w-4 h-4" />, count: wirePackets.length },
          { id: 'users', label: 'Users', icon: <Users className="w-4 h-4" />, count: users.length },
        ]}
        activeTab={activeTab}
        onChange={(id: string) => setActiveTab(id as 'overview' | 'system' | 'wire' | 'users')}
        variant="pills"
        className="mb-6"
      />

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <>
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
                        className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                          modality === 'image'
                            ? 'bg-blue-500/10 text-blue-400'
                            : modality === 'text'
                            ? 'bg-emerald-500/10 text-emerald-400'
                            : 'bg-purple-500/10 text-purple-400'
                        }`}
                      >
                        <span className="text-lg">
                          {modality === 'image' ? '🖼' : modality === 'text' ? '📝' : '🔊'}
                        </span>
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

            <Card title="Stealth Models" subtitle="AI-driven scheduling models">
              <div className="space-y-3">
                {[
                  { name: 'RL Agent', desc: 'PPO Policy Optimizer', ready: metrics?.stealthModels.rl_checkpoint, icon: <Brain className="w-4 h-4" /> },
                  { name: 'GAN Scheduler', desc: 'WGAN-GP Temporal Generator', ready: metrics?.stealthModels.gan_checkpoint, icon: <Zap className="w-4 h-4" /> },
                  { name: 'Static Fallback', desc: 'NoiseController (Mathematical)', ready: true, icon: <Shield className="w-4 h-4" /> },
                ].map((model) => (
                  <motion.div
                    key={model.name}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.1 }}
                    className="flex items-center justify-between p-3 bg-surface/50 rounded-lg"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
                        {model.icon}
                      </div>
                      <div>
                        <p className="font-medium text-foreground">{model.name}</p>
                        <p className="text-body-sm text-muted">{model.desc}</p>
                      </div>
                    </div>
                    <Badge variant={model.ready ? 'success' : 'error'} dot>
                      {model.ready ? 'Ready' : 'Not Trained'}
                    </Badge>
                  </motion.div>
                ))}
              </div>
              {!metrics?.stealthModels.gan_checkpoint && !metrics?.stealthModels.rl_checkpoint && (
                <div className="mt-4 p-3 bg-warning/10 border border-warning/20 rounded-lg">
                  <p className="text-body-sm text-warning flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4" />
                    Dynamic models not trained. System uses static fallback with mathematical noise injection.
                  </p>
                </div>
              )}
            </Card>
          </div>

          {/* Quick Actions */}
          <Section title="Quick Actions" columns={4}>
            <Link href="/send" className="block group">
              <Card hover className="text-center py-8 h-full">
                <Send className="w-10 h-10 text-primary mx-auto mb-3 group-hover:scale-110 transition-transform" />
                <h3 className="font-semibold text-foreground">Send Message</h3>
                <p className="text-body-sm text-muted mt-1">Encode and transmit a secret message</p>
              </Card>
            </Link>
            <Link href="/inbox" className="block group">
              <Card hover className="text-center py-8 h-full">
                <Mail className="w-10 h-10 text-success mx-auto mb-3 group-hover:scale-110 transition-transform" />
                <h3 className="font-semibold text-foreground">Check Inbox</h3>
                <p className="text-body-sm text-muted mt-1">View and decode received messages</p>
              </Card>
            </Link>
            <Link href="/encode" className="block group">
              <Card hover className="text-center py-8 h-full">
                <Terminal className="w-10 h-10 text-info mx-auto mb-3 group-hover:scale-110 transition-transform" />
                <h3 className="font-semibold text-foreground">Encode Standalone</h3>
                <p className="text-body-sm text-muted mt-1">Generate media sequences without transmitting</p>
              </Card>
            </Link>
            <Link href="/wire" className="block group">
              <Card hover className="text-center py-8 h-full">
                <Eye className="w-10 h-10 text-warning mx-auto mb-3 group-hover:scale-110 transition-transform" />
                <h3 className="font-semibold text-foreground">Wire View</h3>
                <p className="text-body-sm text-muted mt-1">Real-time transmission telemetry</p>
              </Card>
            </Link>
          </Section>
        </>
      )}

      {/* System Tab */}
      {activeTab === 'system' && (
        <>
          <Section title="Backend Configuration" columns={2}>
            <Card>
              <dl className="space-y-3">
                {[
                  ['API Endpoint', 'http://localhost:8000/api'],
                  ['WebSocket', 'ws://localhost:8000/ws'],
                  ['Default Policy', 'round_robin'],
                  ['Base Delay', '3 seconds'],
                  ['Channels', '3 channels'],
                  ['ECC Parity', '8 bytes (Reed-Solomon)'],
                  ['Context Bucket', '3600 seconds'],
                  ['Max Payload', '256 bytes per carrier'],
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
                  ['Total Vectors', metrics?.totalItems.toLocaleString() || '0'],
                  ['Vector Dimension', '512'],
                  ['Codebook Centroids', '256'],
                  ['Codebook Norm', 'Unit (||c||₂ = 1.0)'],
                  ['Index Type', 'FAISS IVF'],
                  ['Distance Metric', 'L2 (Euclidean)'],
                  ['Device', metrics?.device?.toUpperCase() || 'N/A'],
                  ['Precision', 'FP32'],
                ].map(([label, value]) => (
                  <div key={label} className="flex justify-between py-2 border-b border-border/50 last:border-0">
                    <dt className="text-body-sm text-muted">{label}</dt>
                    <dd className="font-mono text-body-sm text-foreground">{value}</dd>
                  </div>
                ))}
              </dl>
            </Card>
          </Section>

          <Section title="Index Details" columns={1}>
            <Card>
              <DataTable
                columns={[
                  { key: 'modality', header: 'Modality', render: (row: any) => (
                    <span className={`font-medium ${modalityColors[row.modality as keyof typeof modalityColors] || 'text-foreground'}`}>
                      {row.modality.charAt(0).toUpperCase() + row.modality.slice(1)}
                    </span>
                  )},
                  { key: 'status', header: 'Status', render: (row: any) => (
                    <Badge variant={row.status === 'ok' ? 'success' : row.status === 'missing' ? 'warning' : 'error'} dot>
                      {row.status === 'ok' ? 'Loaded' : row.status === 'missing' ? 'Missing' : 'Error'}
                    </Badge>
                  )},
                  { key: 'count', header: 'Vectors', render: (row: any) => (
                    <span className="font-mono tabular-nums">{row.count?.toLocaleString() || '—'}</span>
                  )},
                  { key: 'error', header: 'Error', render: (row: any) => (
                    <span className="text-body-sm text-error max-w-xs truncate block">{row.error || '—'}</span>
                  )},
                ]}
                data={metrics ? Object.entries(metrics.indices).map(([modality, info]) => ({
                  modality,
                  ...info,
                })) : []}
                keyExtractor={(row: any) => row.modality}
                emptyMessage="No index data available"
              />
            </Card>
          </Section>

          <Section title="Stealth Model Details" columns={1}>
            <Card>
              <DataTable
                columns={[
                  { key: 'name', header: 'Model', render: (row: any) => (
                    <div className="flex items-center gap-2">
                      <span className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
                        {row.icon}
                      </span>
                      <span className="font-medium">{row.name}</span>
                    </div>
                  )},
                  { key: 'description', header: 'Description' },
                  { key: 'type', header: 'Type' },
                  { key: 'ready', header: 'Status', render: (row: any) => (
                    <Badge variant={row.ready ? 'success' : 'error'} dot>
                      {row.ready ? 'Ready' : 'Not Trained'}
                    </Badge>
                  )},
                  { key: 'checkpoint', header: 'Checkpoint', render: (row: any) => (
                    <span className="font-mono text-body-xs text-muted">{row.checkpoint}</span>
                  )},
                ]}
                data={[
                  { name: 'RL Agent', description: 'PPO Policy Optimizer', type: 'Reinforcement Learning', ready: metrics?.stealthModels.rl_checkpoint, checkpoint: 'rl_agent.pt', icon: <Brain className="w-4 h-4" /> },
                  { name: 'GAN Scheduler', description: 'WGAN-GP Temporal Generator', type: 'Generative Adversarial', ready: metrics?.stealthModels.gan_checkpoint, checkpoint: 'gan_generator.pt', icon: <Zap className="w-4 h-4" /> },
                  { name: 'Static Fallback', description: 'NoiseController (Mathematical)', type: 'Deterministic', ready: true, checkpoint: 'Built-in', icon: <Shield className="w-4 h-4" /> },
                ]}
                keyExtractor={(row: any) => row.name}
                emptyMessage="No model data available"
              />
            </Card>
          </Section>
        </>
      )}

      {/* Wire Tab */}
      {activeTab === 'wire' && (
        <>
          <Section
            title="Live Transmission Feed"
            subtitle={`${wirePackets.length} packets captured • ${transmissionStatus.active ? 'LIVE' : 'Monitoring'}`}
            action={
              wirePackets.length > 0 && (
                <button
                  onClick={async () => {
                    try {
                      await clearWirePackets();
                      setWirePackets([]);
                      success('Cleared', 'Wire packets cleared successfully');
                    } catch (err: any) {
                      showError('Clear failed', err.message || 'Failed to clear wire packets');
                    }
                  }}
                  className="btn-ghost btn-sm"
                >
                  Clear Feed
                </button>
              )
            }
            columns={1}
          >
            <Card className="overflow-hidden">
              {recentPackets.length === 0 ? (
                <EmptyState
                  icon={<Network className="w-12 h-12" />}
                  title="No packets captured"
                  description="Wire view monitors the shared_channel directory for packet metadata. Start a transmission from the Send page to see packets here."
                  className="py-16"
                />
              ) : (
                <div className="max-h-[500px] overflow-y-auto">
                  {recentPackets.map((packet) => (
                    <motion.div
                      key={packet.filename}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.2 }}
                      className="flex items-center gap-4 p-4 border-b border-border/50 last:border-0 hover:bg-surface/50 transition-colors"
                    >
                      <div className="flex items-center gap-3 min-w-[180px]">
                        <span className="text-body-xs text-muted font-mono">
                          SEQ #{packet.sequence_number.toString().padStart(4, '0')}
                        </span>
                        <span className="font-mono text-sm text-primary">{packet.media_id}</span>
                      </div>
                      <Badge variant="info" className="w-20 text-center">
                        CH-{packet.channel_id}
                      </Badge>
                      <Badge variant={getModeColor(packet.mode_used)} className="w-24 text-center">
                        {packet.mode_used.toUpperCase()}
                      </Badge>
                      <div className="flex-1 text-body-sm text-muted font-mono">
                        {packet.delay_seconds.toFixed(3)}s delay
                      </div>
                      <div className="text-body-xs text-muted font-mono w-32 text-right">
                        {new Date(packet.timestamp * 1000).toLocaleTimeString()}
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </Card>
          </Section>

          <Section title="Channel Distribution" columns={1}>
            <Card>
              {(() => {
                const channelCounts = recentPackets.reduce((acc, p) => {
                  acc[p.channel_id] = (acc[p.channel_id] || 0) + 1;
                  return acc;
                }, {} as Record<number, number>);

                const maxCount = Math.max(...Object.values(channelCounts), 1);

                return (
                  <div className="space-y-4">
                    {Object.entries(channelCounts)
                      .sort(([a], [b]) => Number(a) - Number(b))
                      .map(([channel, count]) => (
                        <div key={channel}>
                          <div className="flex items-center justify-between mb-1.5">
                            <span className="text-body-sm font-medium text-foreground">Channel {channel}</span>
                            <span className="text-body-sm text-muted tabular-nums">{count} packets</span>
                          </div>
                          <ProgressBar
                            value={count}
                            max={maxCount}
                            variant="primary"
                            size="sm"
                            showLabel={false}
                          />
                        </div>
                      ))}
                  </div>
                );
              })()}
            </Card>
          </Section>
        </>
      )}

      {/* Users Tab */}
      {activeTab === 'users' && (
        <>
          <Section title="Registered Users" columns={1}>
            <Card>
              <DataTable
                columns={[
                  { key: 'username', header: 'Username', render: (row: any) => (
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary font-medium">
                        {row.username.charAt(0).toUpperCase()}
                      </div>
                      <span className="font-medium">{row.username}{row.isCurrent ? ' (you)' : ''}</span>
                    </div>
                  )},
                  { key: 'id', header: 'ID', render: (row: any) => <span className="font-mono text-body-sm text-muted">#{row.id}</span> },
                  { key: 'status', header: 'Status', render: () => (
                    <Badge variant="success" dot>Active</Badge>
                  )},
                  { key: 'actions', header: 'Actions', render: (row: any) => (
                    <div className="flex items-center gap-2">
                      {row.isCurrent ? (
                        <Badge variant="neutral" size="sm">Current Session</Badge>
                      ) : (
                        <span className="text-body-xs text-muted">Available as recipient</span>
                      )}
                    </div>
                  )},
                ]}
                data={users.map((u) => ({
                  ...u,
                  isCurrent: user?.id === u.id || user?.username === u.username,
                }))}
                keyExtractor={(row: any) => row.id.toString()}
                emptyMessage="No users registered"
              />
            </Card>
          </Section>

          <Section title="Session Management" columns={1}>
            <Card>
              <p className="text-body-md text-muted mb-4">
                Active sessions are managed through the inbox system. Each message transmission creates a session
                with its own packet stream. Users can decode sessions on-demand from their inbox.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <Link href="/inbox" className="block group">
                  <Card hover className="text-center py-6 h-full">
                    <Mail className="w-10 h-10 text-primary mx-auto mb-2 group-hover:scale-110 transition-transform" />
                    <h4 className="font-semibold text-foreground">View Inbox</h4>
                    <p className="text-body-sm text-muted mt-1">Check received messages</p>
                  </Card>
                </Link>
                <Link href="/send" className="block group">
                  <Card hover className="text-center py-6 h-full">
                    <Send className="w-10 h-10 text-success mx-auto mb-2 group-hover:scale-110 transition-transform" />
                    <h4 className="font-semibold text-foreground">Send Message</h4>
                    <p className="text-body-sm text-muted mt-1">Transmit to another user</p>
                  </Card>
                </Link>
                <Link href="/login" className="block group">
                  <Card hover className="text-center py-6 h-full">
                    <Settings className="w-10 h-10 text-info mx-auto mb-2 group-hover:scale-110 transition-transform" />
                    <h4 className="font-semibold text-foreground">Switch Account</h4>
                    <p className="text-body-sm text-muted mt-1">Log in as Alice or Bob</p>
                  </Card>
                </Link>
              </div>
            </Card>
          </Section>
        </>
      )}
    </PageWrapper>
  );
}