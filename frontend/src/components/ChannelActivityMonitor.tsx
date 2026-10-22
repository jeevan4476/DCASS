'use client';

import { useCallback, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { BellOff, Cloud, Github, Image as ImageIcon, Radio, RadioTower } from 'lucide-react';
import { Card, Badge, EmptyState, ProgressBar, StatGrid } from '@/components';
import { getWirePackets, getTransmissionStatus, WirePacket, TransmissionStatus } from '@/lib/api';

interface ChannelTrack {
  id: number;
  label: string;
  icon: React.ReactNode;
  barClass: string;
  dotClass: string;
}

const CHANNELS: ChannelTrack[] = [
  {
    id: 0,
    label: 'GitHub Gists',
    icon: <Github className="w-4 h-4" />,
    barClass: 'bg-primary/70',
    dotClass: 'bg-primary',
  },
  {
    id: 1,
    label: 'Imgur',
    icon: <ImageIcon className="w-4 h-4" />,
    barClass: 'bg-info/70',
    dotClass: 'bg-info',
  },
  {
    id: 2,
    label: 'Cloud Storage',
    icon: <Cloud className="w-4 h-4" />,
    barClass: 'bg-success/70',
    dotClass: 'bg-success',
  },
];

function laneFor(packet: WirePacket): number {
  const raw = Number.isFinite(packet.channel_id) ? packet.channel_id : 0;
  return Math.abs(raw % CHANNELS.length);
}

export function ChannelActivityMonitor() {
  const [packets, setPackets] = useState<WirePacket[]>([]);
  const [status, setStatus] = useState<TransmissionStatus | null>(null);
  const [lastUpdate, setLastUpdate] = useState<number | null>(null);

  const fetchData = useCallback(async () => {
    const [packetsResponse, statusResponse] = await Promise.allSettled([
      getWirePackets(),
      getTransmissionStatus(),
    ]);
    if (packetsResponse.status === 'fulfilled') {
      setPackets(packetsResponse.value.packets || []);
      setLastUpdate(Date.now());
    }
    if (statusResponse.status === 'fulfilled') {
      setStatus(statusResponse.value);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const sorted = [...packets].sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
  const recent = sorted.slice(-24);
  const meanDelay =
    recent.length > 0 ? recent.reduce((sum, packet) => sum + (packet.delay_seconds || 0), 0) / recent.length : 0;

  const lanePackets = (laneId: number) => recent.filter((packet) => laneFor(packet) === laneId);

  const stats = [
    { label: 'Packets Captured', value: packets.length, variant: 'primary' as const },
    {
      label: 'Mean Scheduled Delay',
      value: meanDelay > 0 ? `${meanDelay.toFixed(2)} s` : 'idle',
      variant: 'info' as const,
    },
    {
      label: 'Active Channels',
      value: new Set(recent.map(laneFor)).size,
      variant: 'success' as const,
    },
    {
      label: 'Transmission',
      value: status?.active ? `${status.current} of ${status.total}` : 'idle',
      variant: status?.active ? ('warning' as const) : ('neutral' as const),
    },
  ];

  return (
    <Card
      title="Multi-Channel Packet Flow"
      subtitle="Live wire telemetry across public carrier channels, scheduled by the stealth pipeline"
      action={
        <Badge variant={status?.active ? 'warning' : 'neutral'} size="sm" dot>
          {status?.active ? 'TRANSMITTING' : 'POLLING'}
        </Badge>
      }
    >
      <div className="mb-6">
        <StatGrid stats={stats} columns={4} />
      </div>

      {status?.active && status.total > 0 && (
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <p className="text-body-sm text-muted flex items-center gap-2">
              <RadioTower className="w-4 h-4 text-warning" />
              Session progress
            </p>
            <p className="text-body-sm font-mono text-foreground tabular-nums">
              {status.current} / {status.total}
            </p>
          </div>
          <ProgressBar value={status.current} max={status.total} variant="warning" animated showLabel={false} />
        </div>
      )}

      {recent.length === 0 ? (
        <EmptyState
          icon={<BellOff className="w-10 h-10" />}
          title="No wire activity yet"
          description="Send a message from the Send page to watch packet pulses travel through the GitHub, Imgur, and Cloud Storage tracks."
        />
      ) : (
        <div className="space-y-5">
          {CHANNELS.map((channel) => {
            const lane = lanePackets(channel.id);
            const laneDelays = lane.map((packet) => packet.delay_seconds || 0);
            const laneMean =
              laneDelays.length > 0 ? laneDelays.reduce((sum, value) => sum + value, 0) / laneDelays.length : 0;
            const pulseDuration = Math.min(8, Math.max(0.7, laneMean || 3));
            const maxDelay = Math.max(0.001, ...laneDelays);

            return (
              <div key={channel.id}>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2 text-body-sm font-medium text-foreground">
                    {channel.icon}
                    {channel.label}
                    <Badge variant="neutral" size="sm">{lane.length} recent</Badge>
                  </div>
                  <p className="text-body-xs text-muted font-mono tabular-nums">
                    {laneMean > 0 ? `${laneMean.toFixed(2)} s mean inter-arrival` : 'no traffic'}
                  </p>
                </div>

                <div className="relative h-10 rounded-lg border border-border bg-surface/40 overflow-hidden">
                  {/* Pulse traveling according to the recorded inter-arrival delay */}
                  {lane.length > 0 && (
                    <motion.div
                      key={`${channel.id}-${lane.length}-${Math.round(laneMean * 100)}`}
                      className={`absolute top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full ${channel.dotClass} shadow-glow-primary`}
                      initial={{ left: '0%', opacity: 0 }}
                      animate={{ left: ['0%', '100%'], opacity: [0, 1, 1, 0] }}
                      transition={{
                        duration: pulseDuration,
                        times: [0, 0.05, 0.9, 1],
                        repeat: Infinity,
                        repeatDelay: 0.2,
                        ease: 'linear',
                      }}
                    />
                  )}

                  {/* Recorded packet markers positioned proportionally on the lane */}
                  {lane.map((packet, index) => (
                    <motion.div
                      key={`${packet.filename}-${index}`}
                      className={`absolute bottom-1 w-1.5 rounded-sm ${channel.barClass}`}
                      style={{
                        left: `${(index / Math.max(1, lane.length - 1)) * 92 + 2}%`,
                        height: `${Math.max(15, ((packet.delay_seconds || 0) / maxDelay) * 80)}%`,
                      }}
                      initial={{ opacity: 0, scaleY: 0 }}
                      animate={{ opacity: 1, scaleY: 1 }}
                      transition={{ delay: index * 0.03, duration: 0.25 }}
                      title={`${packet.media_id}, ${(packet.delay_seconds || 0).toFixed(2)} s delay`}
                    />
                  ))}

                  {lane.length === 0 && (
                    <p className="absolute inset-0 flex items-center justify-center text-body-xs text-muted">
                      idle
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      <p className="mt-5 text-body-xs text-muted flex items-center gap-2">
        <Radio className="w-3.5 h-3.5" />
        Polls the wire API every 2 seconds.
        {lastUpdate ? ` Last refresh ${new Date(lastUpdate).toLocaleTimeString()}.` : ''}
      </p>
    </Card>
  );
}

export default ChannelActivityMonitor;
