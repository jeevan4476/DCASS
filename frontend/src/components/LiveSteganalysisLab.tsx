'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Activity, Microscope, Play, Target } from 'lucide-react';
import { Card, Badge, Button, LoadingSpinner } from '@/components';
import { runLiveSteganalysis, LiveSteganalysisResponse, getErrorMessage } from '@/lib/api';

type DetectorId = 'sr_net' | 'zhu_net' | 'ye_net';

interface DetectorInfo {
  id: DetectorId;
  name: string;
  reference: string;
  controlAccuracy: number;
  controlAuc: number;
  dcassAuc: number;
}

// Recorded trained benchmark values from storage/benchmarks/component_1_summary.json.
const DETECTORS: DetectorInfo[] = [
  {
    id: 'sr_net',
    name: 'SRNet',
    reference: 'Boroumand et al., IEEE TIFS 2018',
    controlAccuracy: 0.9367,
    controlAuc: 0.9844,
    dcassAuc: 0.4973,
  },
  {
    id: 'zhu_net',
    name: 'ZhuNet',
    reference: 'Zhu et al., IEEE TIFS 2020',
    controlAccuracy: 0.9505,
    controlAuc: 0.986,
    dcassAuc: 0.4993,
  },
  {
    id: 'ye_net',
    name: 'YeNet',
    reference: 'Ye et al., IEEE TIFS 2017',
    controlAccuracy: 0.9397,
    controlAuc: 0.9853,
    dcassAuc: 0.4993,
  },
];

interface ProbabilityRowProps {
  label: string;
  description: string;
  value: number;
  variant: 'info' | 'error' | 'warning';
}

function ProbabilityRow({ label, description, value, variant }: ProbabilityRowProps) {
  const widthPct = Math.max(0, Math.min(100, value * 100));
  const barClass =
    variant === 'error'
      ? 'bg-error'
      : variant === 'warning'
        ? 'bg-warning'
        : 'bg-info';

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-body-sm font-medium text-foreground">{label}</p>
          <p className="text-body-xs text-muted">{description}</p>
        </div>
        <p className="font-mono text-heading-md text-foreground tabular-nums">{widthPct.toFixed(1)}%</p>
      </div>
      <div className="h-2.5 rounded-full bg-surface overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${barClass}`}
          initial={{ width: 0 }}
          animate={{ width: `${widthPct}%` }}
          transition={{ duration: 0.7, ease: [0.19, 1, 0.22, 1] }}
        />
      </div>
    </div>
  );
}

export function LiveSteganalysisLab() {
  const [detector, setDetector] = useState<DetectorId>('sr_net');
  const [sampleCount, setSampleCount] = useState(2);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<LiveSteganalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const selected = DETECTORS.find((entry) => entry.id === detector) ?? DETECTORS[0];

  const runDetection = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await runLiveSteganalysis({ detector, sample_count: sampleCount });
      setResult(response);
    } catch (err) {
      setError(getErrorMessage(err, 'Live detection failed'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card
      title="Live Steganalysis Lab"
      subtitle="Run detector forward passes against clean, LSB stego, and DCASS carriers"
      action={<Badge variant="primary" size="sm" dot>ON DEMAND</Badge>}
    >
      <div className="flex flex-wrap items-end gap-4 mb-5">
        <div>
          <p className="label mb-2">Detector architecture</p>
          <div className="flex flex-wrap gap-2">
            {DETECTORS.map((entry) => (
              <button
                key={entry.id}
                onClick={() => setDetector(entry.id)}
                className={`px-3 py-2 rounded-lg text-body-sm font-medium border transition-all duration-fast ${
                  detector === entry.id
                    ? 'bg-primary/10 text-primary border-primary/30'
                    : 'text-muted border-border hover:text-foreground hover:bg-surface'
                }`}
              >
                {entry.name}
              </button>
            ))}
          </div>
        </div>
        <div>
          <p className="label mb-2">Samples per class</p>
          <div className="flex gap-2">
            {[1, 2, 3, 4].map((count) => (
              <button
                key={count}
                onClick={() => setSampleCount(count)}
                className={`w-9 h-9 rounded-lg text-body-sm font-medium border transition-all duration-fast ${
                  sampleCount === count
                    ? 'bg-primary/10 text-primary border-primary/30'
                    : 'text-muted border-border hover:text-foreground hover:bg-surface'
                }`}
              >
                {count}
              </button>
            ))}
          </div>
        </div>
        <Button
          variant="primary"
          onClick={runDetection}
          loading={loading}
          leftIcon={<Play className="w-4 h-4" />}
        >
          Run Live Detection
        </Button>
      </div>

      <div className="rounded-lg border border-border bg-surface/40 px-4 py-3 mb-5 text-body-xs text-muted flex items-center gap-2">
        <Target className="w-4 h-4 text-info shrink-0" />
        Expected protocol response: clean natural images stay below 10 percent, LSB stego crosses 90 percent, and
        DCASS carriers sit at 50 percent, the blind coin flip point.
      </div>

      {loading && (
        <div className="flex flex-col items-center justify-center py-10 gap-3">
          <LoadingSpinner />
          <p className="text-body-sm text-muted">Loading {selected.name} weights and running the forward pass</p>
        </div>
      )}

      {error && !loading && (
        <div className="rounded-xl border border-error/30 bg-error/5 p-4 text-body-sm text-error mb-4">{error}</div>
      )}

      {result && !loading && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: [0.19, 1, 0.22, 1] }}
          className="space-y-5"
        >
          <div className="space-y-4">
            <ProbabilityRow
              label="Clean Natural Image"
              description="Baseline corpus samples without any embedding"
              value={result.clean_probability}
              variant="info"
            />
            <ProbabilityRow
              label="Traditional LSB Steganography"
              description="Spatial plus minus 1 LSB injection as the positive control"
              value={result.lsb_probability}
              variant="error"
            />
            <ProbabilityRow
              label="DCASS Carrier"
              description="Unmodified corpus carrier selected by semantic state-space coding"
              value={result.dcass_probability}
              variant="warning"
            />
          </div>

          <div className="rounded-lg border border-border bg-surface/40 p-3 text-body-xs text-muted flex items-start gap-2">
            <Activity className="w-4 h-4 text-warning shrink-0 mt-0.5" />
            <span>
              {result.engine_trained
                ? 'The live pass used trained detector weights.'
                : 'The live forward pass uses freshly initialized weights in the running server process, so treat the raw probabilities as a runtime sanity check. The trained benchmark below is the recorded evidence.'}{' '}
              {result.sample_count} samples per class, input size {result.input_size} pixels.
            </span>
          </div>
        </motion.div>
      )}

      <div className="mt-6 border-t border-border pt-5">
        <div className="flex items-center gap-2 mb-3">
          <Microscope className="w-4 h-4 text-primary" />
          <p className="text-body-sm font-semibold text-foreground">Recorded trained benchmark: {selected.name}</p>
        </div>
        <p className="text-body-xs text-muted mb-4">{selected.reference}</p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div className="rounded-lg border border-border bg-surface/40 p-3">
            <p className="stat-label">Positive Control Accuracy</p>
            <p className="stat-value text-info mt-1">{(selected.controlAccuracy * 100).toFixed(2)}%</p>
          </div>
          <div className="rounded-lg border border-border bg-surface/40 p-3">
            <p className="stat-label">Positive Control ROC AUC</p>
            <p className="stat-value text-warning mt-1">{selected.controlAuc.toFixed(4)}</p>
          </div>
          <div className="rounded-lg border border-border bg-surface/40 p-3">
            <p className="stat-label">DCASS ROC AUC</p>
            <p className="stat-value text-success mt-1">{selected.dcassAuc.toFixed(4)}</p>
          </div>
        </div>
      </div>
    </Card>
  );
}

export default LiveSteganalysisLab;
