'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { FlaskConical, GitBranch, Play, ShieldCheck, Timer, Zap } from 'lucide-react';
import {
  PageWrapper,
  Card,
  Badge,
  Button,
  Section,
  StatGrid,
  LoadingSpinner,
} from '@/components';
import { ZeroModVerifier } from '@/components/ZeroModVerifier';
import { LiveSteganalysisLab } from '@/components/LiveSteganalysisLab';
import { ChannelActivityMonitor } from '@/components/ChannelActivityMonitor';
import { runPipelineTrace, PipelineTraceResponse, PipelineTraceStage, getErrorMessage } from '@/lib/api';

const statusVariant = (status: string): 'success' | 'neutral' | 'warning' => {
  if (status === 'completed') return 'success';
  if (status === 'degraded') return 'warning';
  return 'neutral';
};

function formatDetail(detail: Record<string, unknown>): Array<{ key: string; value: string }> {
  return Object.entries(detail).map(([key, value]) => ({
    key: key.replace(/_/g, ' '),
    value: Array.isArray(value) ? value.join(', ') : String(value),
  }));
}

function StageRow({ stage, maxDuration, index }: { stage: PipelineTraceStage; maxDuration: number; index: number }) {
  const widthPct = maxDuration > 0 ? Math.max(2, (stage.duration_ms / maxDuration) * 100) : 2;
  const details = formatDetail(stage.detail);

  return (
    <motion.div
      initial={{ opacity: 0, x: -12 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.05, duration: 0.3, ease: [0.19, 1, 0.22, 1] }}
      className="rounded-xl border border-border bg-surface/40 p-4"
    >
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-lg bg-primary/10 flex items-center justify-center text-primary text-body-xs font-mono">
            {index + 1}
          </div>
          <p className="text-body-sm font-semibold text-foreground">{stage.name}</p>
          <Badge variant={statusVariant(stage.status)} size="sm">{stage.status}</Badge>
        </div>
        <p className="font-mono text-body-sm text-muted tabular-nums">{stage.duration_ms.toFixed(3)} ms</p>
      </div>
      <div className="h-1.5 rounded-full bg-background overflow-hidden mb-3">
        <motion.div
          className="h-full rounded-full bg-primary/60"
          initial={{ width: 0 }}
          animate={{ width: `${widthPct}%` }}
          transition={{ delay: index * 0.05 + 0.15, duration: 0.5, ease: [0.19, 1, 0.22, 1] }}
        />
      </div>
      <div className="flex flex-wrap gap-2">
        {details.map((entry) => (
          <span
            key={entry.key}
            className="inline-flex items-center gap-1.5 rounded-md border border-border bg-background/60 px-2 py-1 text-body-xs"
          >
            <span className="text-muted">{entry.key}</span>
            <span className="font-mono text-foreground/90 max-w-[16rem] truncate">{entry.value}</span>
          </span>
        ))}
      </div>
    </motion.div>
  );
}

export default function ShowcasePage() {
  const [message, setMessage] = useState('Classified transmission for the evaluation panel');
  const [useEcc, setUseEcc] = useState(true);
  const [loading, setLoading] = useState(false);
  const [trace, setTrace] = useState<PipelineTraceResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runTrace = async () => {
    const text = message.trim();
    if (!text) return;
    setLoading(true);
    setError(null);
    setTrace(null);
    try {
      const response = await runPipelineTrace({ message: text, use_ecc: useEcc });
      setTrace(response);
    } catch (err) {
      setError(getErrorMessage(err, 'Pipeline trace failed'));
    } finally {
      setLoading(false);
    }
  };

  const maxDuration = trace ? Math.max(...trace.stages.map((stage) => stage.duration_ms), 0.001) : 0;

  const traceStats = trace
    ? [
        { label: 'Stages Traced', value: trace.stages.length, variant: 'primary' as const },
        { label: 'Carriers Selected', value: trace.carrier_count, variant: 'success' as const },
        {
          label: 'Bits Per Carrier',
          value: trace.bits_per_carrier > 0 ? trace.bits_per_carrier.toFixed(2) : 'pending',
          variant: 'info' as const,
        },
        {
          label: 'Trace Duration',
          value: `${trace.total_duration_ms.toFixed(1)} ms`,
          variant: 'warning' as const,
        },
      ]
    : [];

  return (
    <PageWrapper
      title="Production Showcase"
      description="Live adversarial laboratory: pipeline tracing, zero-modification proofs, and multi-channel telemetry"
      action={
        <Badge variant="primary" size="lg">
          <FlaskConical className="w-3.5 h-3.5 mr-1.5" />
          EVALUATION MODE
        </Badge>
      }
    >
      <Section
        title="Pipeline Inspector"
        subtitle="Granular, timed execution trace of the semantic state-space encode pipeline"
      >
        <Card>
          <div className="flex flex-col lg:flex-row gap-4 mb-5">
            <div className="flex-1">
              <p className="label mb-2">Secret message</p>
              <textarea
                className="input w-full resize-none"
                rows={2}
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                placeholder="Enter the message to trace through the encode pipeline"
              />
            </div>
            <div className="flex flex-col justify-between gap-3 lg:w-64">
              <button
                onClick={() => setUseEcc(!useEcc)}
                className={`flex items-center justify-between rounded-lg border px-3 py-2.5 text-body-sm font-medium transition-all duration-fast ${
                  useEcc
                    ? 'border-success/30 bg-success/10 text-success'
                    : 'border-border text-muted hover:text-foreground hover:bg-surface'
                }`}
              >
                <span className="flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4" />
                  Reed-Solomon ECC
                </span>
                <span className="font-mono text-body-xs">{useEcc ? 'ON' : 'OFF'}</span>
              </button>
              <Button
                variant="primary"
                onClick={runTrace}
                loading={loading}
                leftIcon={<Play className="w-4 h-4" />}
              >
                Run Pipeline Trace
              </Button>
            </div>
          </div>

          {loading && (
            <div className="flex flex-col items-center justify-center py-12 gap-3">
              <LoadingSpinner />
              <p className="text-body-sm text-muted">
                Framing payload, expanding parity, chunking semantics, and mapping vector clusters
              </p>
            </div>
          )}

          {error && !loading && (
            <div className="rounded-xl border border-error/30 bg-error/5 p-4 text-body-sm text-error">{error}</div>
          )}

          {trace && !loading && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, ease: [0.19, 1, 0.22, 1] }}
              className="space-y-5"
            >
              <div className="flex flex-wrap items-center gap-3">
                <Badge variant={trace.engine_warm ? 'success' : 'warning'} size="sm" dot>
                  {trace.engine_warm ? 'Engine warm' : 'Engine cold, cluster mapping skipped'}
                </Badge>
                {trace.engine_warm && trace.mode !== 'unknown' && (
                  <Badge variant="info" size="sm">
                    Mode: {trace.mode}
                  </Badge>
                )}
                {trace.media_ids.length > 0 && (
                  <span className="text-body-xs text-muted font-mono truncate max-w-full">
                    {trace.media_ids.slice(0, 4).join(' / ')}
                    {trace.media_ids.length > 4 ? ` and ${trace.media_ids.length - 4} more` : ''}
                  </span>
                )}
              </div>

              <StatGrid stats={traceStats} columns={4} />

              <div className="space-y-3">
                {trace.stages.map((stage, index) => (
                  <StageRow key={stage.stage_id} stage={stage} maxDuration={maxDuration} index={index} />
                ))}
              </div>

              <div className="rounded-lg border border-border bg-surface/40 px-4 py-3 text-body-xs text-muted flex items-center gap-2">
                <Timer className="w-4 h-4 text-info shrink-0" />
                Every stage is measured with a monotonic high-resolution clock inside the running API process.
                Durations depend on engine warmth and available hardware.
              </div>
            </motion.div>
          )}
        </Card>
      </Section>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mt-6">
        <LiveSteganalysisLab />
        <ZeroModVerifier />
      </div>

      <div className="mt-6">
        <ChannelActivityMonitor />
      </div>

      <Section title="Defense Reference" subtitle="Recorded benchmark anchors for the evaluation panel" className="mt-6">
        <Card>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
            <div className="p-3">
              <p className="stat-value text-success">0.4973–0.4993</p>
              <p className="stat-label mt-1">DCASS ROC AUC (SRNet/ZhuNet/YeNet)</p>
            </div>
            <div className="p-3">
              <p className="stat-value text-primary">0</p>
              <p className="stat-label mt-1">Bytes modified per carrier</p>
            </div>
            <div className="p-3">
              <p className="stat-value text-info">13.50–13.66</p>
              <p className="stat-label mt-1">Coded bits per carrier (user payload lower)</p>
            </div>
            <div className="p-3">
              <p className="stat-value text-warning">0/100</p>
              <p className="stat-label mt-1">Observed swaps, direct engine at 50 threads</p>
            </div>
          </div>
          <div className="mt-5 rounded-lg border border-border bg-surface/40 px-4 py-3 text-body-xs text-muted flex items-center gap-2">
            <GitBranch className="w-4 h-4 text-primary shrink-0" />
            Full methodology: docs/benchmarking/BENCHMARK_DEEP_EXPLANATION.md.
            Raw results: storage/benchmarks, tables in storage/benchmarks/tables.
          </div>
        </Card>
      </Section>

      <p className="mt-6 text-body-xs text-muted flex items-center gap-2">
        <Zap className="w-3.5 h-3.5" />
        If the browser or network fails during the panel session, fall back to the terminal: python scripts/demo/panel_showcase.py
      </p>
    </PageWrapper>
  );
}
