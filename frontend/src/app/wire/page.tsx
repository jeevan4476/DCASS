'use client';

import { useEffect, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Network, Pause, Play, Trash2, Zap, Activity, Minus, Plus, Eye, Settings, RefreshCw } from 'lucide-react';
import { PageWrapper, Card, Badge, LoadingSpinner, ProgressBar, StatCard, EmptyState, useToastHelpers, Section, StatGrid } from '@/components';
import { getWirePackets, clearWirePackets, getTransmissionStatus, WirePacket, TransmissionStatus } from '@/lib/api';

const getModeColor = (mode: string) => {
  switch (mode.toLowerCase()) {
    case 'rl': return 'success';
    case 'gan': return 'info';
    case 'static': return 'warning';
    default: return 'neutral';
  }
};

const modeIcons = {
  rl: <Zap className="w-4 h-4" />,
  gan: <Eye className="w-4 h-4" />,
  static: <Minus className="w-4 h-4" />,
  auto: <Activity className="w-4 h-4" />,
};

export default function WirePage() {
  const { success, error: showError } = useToastHelpers();

  const [packets, setPackets] = useState<WirePacket[]>([]);
  const [isMonitoring, setIsMonitoring] = useState(true);
  const [transmissionStatus, setTransmissionStatus] = useState<TransmissionStatus>({
    active: false,
    current: 0,
    total: 0,
    status: 'idle',
  });
  const [stats, setStats] = useState({
    totalPackets: 0,
    avgDelay: 0,
    channelsUsed: new Set<number>(),
    modeUsed: 'unknown' as string,
  });
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      const [packetsResponse, statusResponse] = await Promise.allSettled([
        getWirePackets(),
        getTransmissionStatus(),
      ]);

      if (packetsResponse.status === 'fulfilled') {
        setPackets(packetsResponse.value.packets || []);
      }
      if (statusResponse.status === 'fulfilled') {
        setTransmissionStatus(statusResponse.value);
      }
    } catch (err) {
      console.error('Error polling data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 500);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Update stats when packets change
  useEffect(() => {
    if (packets.length > 0) {
      const totalDelay = packets.reduce((sum, p) => sum + p.delay_seconds, 0);
      const channels = new Set(packets.map(p => p.channel_id));
      const lastMode = packets[packets.length - 1]?.mode_used || 'unknown';

      setStats({
        totalPackets: packets.length,
        avgDelay: totalDelay / packets.length,
        channelsUsed: channels,
        modeUsed: lastMode,
      });
    }
  }, [packets]);

  const clearPackets = async () => {
    try {
      await clearWirePackets();
      setPackets([]);
      success('Cleared', 'Wire packets cleared successfully');
    } catch (err) {
      showError('Clear failed', 'Failed to clear wire packets');
    }
  };

  const recentPackets = packets.slice(-100).reverse();

  const statCards = [
    {
      label: 'Total Packets',
      value: stats.totalPackets.toLocaleString(),
      variant: 'primary' as const,
      icon: <Activity className="w-5 h-5" />,
    },
    {
      label: 'Avg Delay',
      value: stats.avgDelay > 0 ? `${stats.avgDelay.toFixed(3)}s` : '--',
      variant: 'info' as const,
      icon: <Zap className="w-5 h-5" />,
    },
    {
      label: 'Channels Used',
      value: stats.channelsUsed.size,
      variant: 'success' as const,
      icon: <Network className="w-5 h-5" />,
    },
    {
      label: 'Active Mode',
      value: stats.modeUsed.toUpperCase(),
      variant: getModeColor(stats.modeUsed) as any,
      icon: modeIcons[stats.modeUsed as keyof typeof modeIcons] || <Activity className="w-5 h-5" />,
    },
  ];

  const channelDistribution = () => {
    const channelCounts = recentPackets.reduce((acc, p) => {
      acc[p.channel_id] = (acc[p.channel_id] || 0) + 1;
      return acc;
    }, {} as Record<number, number>);

    const maxCount = Math.max(...Object.values(channelCounts), 1);

    return Object.entries(channelCounts)
      .sort(([a], [b]) => Number(a) - Number(b))
      .map(([channel, count]) => (
        <div key={channel} className="space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-body-sm font-medium text-foreground">Channel {channel}</span>
            <span className="text-body-sm text-muted tabular-nums">{count} packets</span>
          </div>
          <ProgressBar value={count} max={maxCount} variant="primary" size="sm" />
        </div>
      ));
  };

  if (loading) {
    return (
      <PageWrapper title="Wire View" description="Real-time transmission telemetry and packet monitoring" fullWidth>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i} className="bg-surface/50">
              <LoadingSpinner />
            </Card>
          ))}
        </div>
        <Card className="h-96"><LoadingSpinner size="lg" className="h-full" /></Card>
      </PageWrapper>
    );
  }

  return (
    <PageWrapper
      title="Wire View"
      description="Real-time transmission telemetry and packet monitoring"
      action={
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsMonitoring(!isMonitoring)}
            className={`btn-sm gap-2 ${isMonitoring ? 'btn-secondary' : 'btn-success'}`}
          >
            {isMonitoring ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isMonitoring ? 'Pause' : 'Resume'}
          </button>
          <button onClick={fetchData} className="btn-ghost btn-sm gap-2">
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>
      }
    >
      {/* Transmission Progress Banner */}
      {transmissionStatus.active && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6"
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

      {/* Stats Grid */}
      <Section title="Live Statistics">
        <StatGrid stats={statCards} columns={4} />
      </Section>

      {/* Live Feed & Channel Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Section title="Live Feed" subtitle={`${recentPackets.length} packets captured`} columns={1} className="lg:col-span-2">
          <Card className="overflow-hidden">
            {recentPackets.length === 0 ? (
              <EmptyState
                icon={<Network className="w-12 h-12" />}
                title="No packets captured"
                description="Wire view monitors the shared_channel directory for packet metadata. Start a transmission from the Send page to see packets here."
                className="py-16"
              />
            ) : (
              <div className="max-h-[600px] overflow-y-auto overflow-x-auto">
                {recentPackets.map((packet) => (
                  <motion.div
                    key={packet.filename}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.15 }}
                    className="flex items-center gap-4 p-4 border-b border-border/50 last:border-0 hover:bg-surface/50 transition-colors min-w-[540px]"
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
            {recentPackets.length > 0 ? (
              <div className="space-y-4">{channelDistribution()}</div>
            ) : (
              <div className="text-center py-8 text-muted">No data yet</div>
            )}
          </Card>
        </Section>
      </div>

      {/* Info Box */}
      <Section title="How Wire View Works" columns={1}>
        <Card className="bg-surface/50">
          <div className="space-y-3 text-body-sm text-muted">
            <p className="flex items-start gap-2">
              <span className="text-primary font-mono">1.</span>
              <span><strong className="text-foreground">Real-time Monitoring:</strong> Watches the <code className="text-primary font-mono">shared_channel/</code> directory for packet metadata.</span>
            </p>
            <p className="flex items-start gap-2">
              <span className="text-primary font-mono">2.</span>
              <span><strong className="text-foreground">Dynamic Fallback:</strong> System automatically tries RL → GAN → Static scheduling. Badge shows active mode.</span>
            </p>
            <p className="flex items-start gap-2">
              <span className="text-primary font-mono">3.</span>
              <span><strong className="text-foreground">Packet Metadata:</strong> Each JSON contains media_id, channel, sequence, delay, timestamp, and mode.</span>
            </p>
            <p className="flex items-start gap-2">
              <span className="text-primary font-mono">4.</span>
              <span><strong className="text-foreground">Multi-Channel:</strong> Packets distributed across channels to mimic natural human behavior.</span>
            </p>
          </div>
        </Card>
      </Section>
    </PageWrapper>
  );
}