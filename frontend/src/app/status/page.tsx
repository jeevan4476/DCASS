'use client';

import React, { useEffect, useState, useCallback } from 'react';
import Link from 'next/link';
import {
  Database,
  Server,
  Cpu,
  ShieldCheck,
  Users,
  Activity,
  Brain,
  Zap,
  RefreshCw,
  AlertTriangle,
  MessageSquare,
  Inbox,
  Image as ImageIcon,
  FileText,
  Volume2,
} from 'lucide-react';

import {
  PageWrapper,
  Section,
  StatGrid,
  DataTable,
  Tabs,
  Card,
  Badge,
  Button,
  SkeletonCard,
  useToastHelpers,
} from '@/components';
import {
  getStatus,
  healthCheck,
  listUsers,
  getWireSessions,
  StatusResponse,
  AuthUser,
  WireSessionSnapshot,
} from '@/lib/api';
import { useAuth } from '@/lib/auth';

export default function StatusPage() {
  const { user } = useAuth();
  const { success, error: showError } = useToastHelpers();

  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [healthy, setHealthy] = useState<boolean>(false);
  const [users, setUsers] = useState<AuthUser[]>([]);
  const [wireSessions, setWireSessions] = useState<WireSessionSnapshot[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'indices' | 'users'>('overview');

  const fetchAll = useCallback(
    async (toast = false) => {
      try {
        if (toast) setRefreshing(true);
        setError(null);
        const [hRes, sRes, uRes, wRes] = await Promise.allSettled([
          healthCheck(),
          getStatus(),
          listUsers(),
          getWireSessions(),
        ]);
        if (hRes.status === 'fulfilled') setHealthy(hRes.value.status === 'ok');
        if (sRes.status === 'fulfilled') setStatus(sRes.value);
        if (uRes.status === 'fulfilled') setUsers(uRes.value.users);
        if (wRes.status === 'fulfilled') setWireSessions(wRes.value.sessions);
        if (toast) success('Refreshed', 'System telemetry updated');
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Failed to fetch status';
        setError(msg);
        if (toast) showError('Refresh failed', msg);
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [success, showError],
  );

  useEffect(() => {
    fetchAll();
    const timer = setInterval(() => fetchAll(false), 12000);
    return () => clearInterval(timer);
  }, [fetchAll]);

  if (loading && !status) {
    return (
      <PageWrapper title="Panel" description="System telemetry, corpus health, and active accounts">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {Array.from({ length: 4 }).map((_, i) => (
            <SkeletonCard key={i} lines={2} />
          ))}
        </div>
      </PageWrapper>
    );
  }

  const indicesLoaded = status
    ? Object.values(status.indices).filter((i) => i.status === 'ok').length
    : 0;
  const indicesTotal = status ? Object.keys(status.indices).length : 3;
  const totalDiskPackets = wireSessions.reduce((acc, s) => acc + s.surviving_count, 0);

  return (
    <PageWrapper
      title="Panel"
      description="System telemetry, FAISS corpus indices, stealth models, and accounts"
      action={
        <Button
          variant="secondary"
          size="sm"
          onClick={() => fetchAll(true)}
          disabled={refreshing}
          leftIcon={<RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} />}
        >
          Refresh
        </Button>
      }
    >
      <div className="space-y-6 pb-12">
        {error && (
          <Card className="border-error/30 bg-error-muted/50">
            <div className="flex items-center gap-3 text-error text-body-sm">
              <AlertTriangle className="w-5 h-5 shrink-0" />
              <span>{error}</span>
            </div>
          </Card>
        )}

        <StatGrid
          columns={4}
          stats={[
            {
              label: 'Corpus Vectors',
              value: status?.total_items?.toLocaleString() || '0',
              variant: 'primary',
              icon: <Database className="w-4 h-4" />,
            },
            {
              label: 'FAISS Indices',
              value: `${indicesLoaded} / ${indicesTotal}`,
              variant: indicesLoaded === indicesTotal ? 'success' : 'warning',
              icon: <ShieldCheck className="w-4 h-4" />,
            },
            {
              label: 'Compute Device',
              value: status?.device?.toUpperCase() || 'N/A',
              variant: 'info',
              icon: <Cpu className="w-4 h-4" />,
            },
            {
              label: 'API Health',
              value: healthy ? 'Healthy' : 'Offline',
              variant: healthy ? 'success' : 'error',
              icon: <Server className="w-4 h-4" />,
            },
          ]}
        />

        <Tabs
          tabs={[
            { id: 'overview', label: 'Overview', icon: <Activity className="w-3.5 h-3.5" /> },
            { id: 'indices', label: 'Corpus & Models', icon: <Database className="w-3.5 h-3.5" /> },
            { id: 'users', label: 'Accounts', icon: <Users className="w-3.5 h-3.5" />, count: users.length },
          ]}
          activeTab={activeTab}
          onChange={(id) => setActiveTab(id as 'overview' | 'indices' | 'users')}
        />

        {activeTab === 'overview' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card title="Multimodal Corpus" subtitle={`${indicesLoaded} of ${indicesTotal} indices loaded`}>
                <div className="space-y-2.5">
                  {status &&
                    Object.entries(status.indices).map(([mod, info]) => (
                      <div
                        key={mod}
                        className="p-3.5 rounded-xl bg-surface-muted/50 border border-border flex items-center justify-between gap-3"
                      >
                        <div className="flex items-center gap-3">
                          <div className="w-9 h-9 rounded-xl bg-surface border border-border flex items-center justify-center">
                            {mod === 'image' ? (
                              <ImageIcon className="w-4 h-4 text-info" />
                            ) : mod === 'audio' ? (
                              <Volume2 className="w-4 h-4 text-purple" />
                            ) : (
                              <FileText className="w-4 h-4 text-success" />
                            )}
                          </div>
                          <div>
                            <div className="text-body-sm font-semibold capitalize text-foreground">
                              {mod}
                            </div>
                            <div className="text-body-xs font-mono text-muted">
                              {info.status === 'ok'
                                ? `${info.count?.toLocaleString()} vectors`
                                : info.error || info.status}
                            </div>
                          </div>
                        </div>
                        <Badge variant={info.status === 'ok' ? 'success' : 'warning'} dot>
                          {info.status === 'ok' ? 'Loaded' : info.status}
                        </Badge>
                      </div>
                    ))}
                </div>
              </Card>

              <Card title="Stealth & Scheduling Models" subtitle="Traffic-timing checkpoints on disk">
                <div className="space-y-2.5">
                  {[
                    {
                      name: 'PPO RL Scheduler',
                      desc: 'Reinforcement learning channel & timing policy',
                      ready: status?.stealth_models?.rl_checkpoint,
                      icon: <Brain className="w-4 h-4 text-primary" />,
                    },
                    {
                      name: 'WGAN-GP Generator',
                      desc: 'Adversarial inter-packet delay generator',
                      ready: status?.stealth_models?.gan_checkpoint,
                      icon: <Zap className="w-4 h-4 text-primary" />,
                    },
                    {
                      name: 'RS-ECC GF(2^8) Codec',
                      desc: 'Berlekamp-Massey + Forney erasure recovery',
                      ready: true,
                      icon: <ShieldCheck className="w-4 h-4 text-success" />,
                    },
                  ].map((m) => (
                    <div
                      key={m.name}
                      className="p-3.5 rounded-xl bg-surface-muted/50 border border-border flex items-center justify-between gap-3"
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-9 h-9 rounded-xl bg-surface border border-border flex items-center justify-center">
                          {m.icon}
                        </div>
                        <div>
                          <div className="text-body-sm font-semibold text-foreground">{m.name}</div>
                          <div className="text-body-xs text-muted">{m.desc}</div>
                        </div>
                      </div>
                      <Badge variant={m.ready ? 'success' : 'warning'} dot>
                        {m.ready ? 'Ready' : 'Fallback'}
                      </Badge>
                    </div>
                  ))}
                </div>
              </Card>
            </div>

            <Section title="Quick Navigation" columns={4}>
              {[
                { href: '/chat', label: 'Covert Chat', desc: 'Send multimodal messages', icon: MessageSquare },
                { href: '/erasure', label: 'RS-ECC Lab', desc: 'Delete packets & recover', icon: ShieldCheck },
                { href: '/wire', label: 'Live Wire', desc: `${totalDiskPackets} packet files on disk`, icon: Activity },
                { href: '/inbox', label: 'Inbox', desc: 'Decode received sessions', icon: Inbox },
              ].map((link) => {
                const Icon = link.icon;
                return (
                  <Link key={link.href} href={link.href} className="block">
                    <Card hover className="h-full">
                      <Icon className="w-5 h-5 text-primary mb-2.5" />
                      <h3 className="font-serif text-heading-sm font-medium text-foreground">
                        {link.label}
                      </h3>
                      <p className="text-body-xs text-muted mt-0.5">{link.desc}</p>
                    </Card>
                  </Link>
                );
              })}
            </Section>
          </div>
        )}

        {activeTab === 'indices' && (
          <Card title="Codec & Vector Store Parameters">
            <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-2">
              {[
                ['Total Corpus Vectors', status?.total_items?.toLocaleString() || '0'],
                ['Voronoi Centroids', '256 (unit-normalized)'],
                ['Embedding Dimension', '512 (CLIP / CLAP)'],
                ['Frame Integrity', 'CRC-16/CCITT-FALSE (5B header)'],
                ['Error & Erasure Bound', '2e + v <= R (GF(2^8) Reed-Solomon)'],
                ['Compute Device', status?.device?.toUpperCase() || 'N/A'],
              ].map(([k, v]) => (
                <div key={k} className="flex items-center justify-between py-2.5 border-b border-border/70">
                  <dt className="text-body-xs text-muted">{k}</dt>
                  <dd className="font-mono text-body-xs font-semibold text-foreground">{v}</dd>
                </div>
              ))}
            </dl>
          </Card>
        )}

        {activeTab === 'users' && (
          <Card title="Registered Accounts" subtitle="Local demo accounts available for routing">
            <DataTable
              columns={[
                {
                  key: 'username',
                  header: 'User',
                  render: (row: AuthUser & { isCurrent: boolean }) => (
                    <div className="flex items-center gap-2.5">
                      <span className="w-7 h-7 rounded-full bg-primary-muted text-primary border border-primary/25 flex items-center justify-center font-serif text-xs font-semibold uppercase">
                        {row.username.slice(0, 2)}
                      </span>
                      <span className="font-medium capitalize text-foreground">
                        {row.username}
                        {row.isCurrent ? ' (you)' : ''}
                      </span>
                    </div>
                  ),
                },
                {
                  key: 'id',
                  header: 'ID',
                  render: (row) => <span className="font-mono text-muted">#{row.id}</span>,
                },
                {
                  key: 'status',
                  header: 'State',
                  render: () => (
                    <Badge variant="success" dot size="sm">
                      Active
                    </Badge>
                  ),
                },
              ]}
              data={users.map((u) => ({
                ...u,
                isCurrent: user?.id === u.id || user?.username === u.username,
              }))}
              keyExtractor={(row) => String(row.id)}
              emptyMessage="Sign in to view registered accounts."
            />
          </Card>
        )}
      </div>
    </PageWrapper>
  );
}
