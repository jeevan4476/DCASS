'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { FileCheck2, Fingerprint, Search, ShieldCheck, ShieldAlert, GitCompare } from 'lucide-react';
import { Card, Badge, Button, Input, EmptyState, LoadingSpinner } from '@/components';
import { verifyCarrier, VerifyCarrierResponse, getErrorMessage } from '@/lib/api';

interface HashCardProps {
  label: string;
  digest: string;
  sizeBytes: number;
  icon: React.ReactNode;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} bytes`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function HashCard({ label, digest, sizeBytes, icon }: HashCardProps) {
  return (
    <div className="rounded-xl border border-border bg-surface/50 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 text-body-sm font-medium text-foreground">
          {icon}
          {label}
        </div>
        <Badge variant="neutral" size="sm">{formatBytes(sizeBytes)}</Badge>
      </div>
      <p className="text-body-xs text-muted mb-1">SHA-256 digest</p>
      <p className="font-mono text-body-xs text-foreground/90 break-all leading-relaxed">{digest}</p>
    </div>
  );
}

export function ZeroModVerifier({ defaultMediaId = 'flickr30k_1000092795' }: { defaultMediaId?: string }) {
  const [mediaId, setMediaId] = useState(defaultMediaId);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<VerifyCarrierResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runVerification = async () => {
    const id = mediaId.trim();
    if (!id) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await verifyCarrier(id);
      setResult(response);
    } catch (err) {
      setError(getErrorMessage(err, 'Verification failed'));
    } finally {
      setLoading(false);
    }
  };

  const verified = result?.integrity === 'verified' && result?.bytes_modified === 0;

  return (
    <Card
      title="Zero-Modification Verifier"
      subtitle="Byte-by-byte carrier integrity against the source corpus"
      action={<Badge variant="success" size="sm" dot>LIVE HASH</Badge>}
    >
      <div className="flex flex-col sm:flex-row gap-3 mb-5">
        <div className="flex-1">
          <Input
            label="Corpus media ID"
            placeholder="flickr30k_1000092795"
            value={mediaId}
            onChange={(event) => setMediaId(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') runVerification();
            }}
            leftIcon={<Search className="w-4 h-4" />}
          />
        </div>
        <div className="flex items-end">
          <Button
            variant="primary"
            onClick={runVerification}
            loading={loading}
            leftIcon={<Fingerprint className="w-4 h-4" />}
          >
            Verify Carrier
          </Button>
        </div>
      </div>

      {!result && !loading && !error && (
        <EmptyState
          icon={<GitCompare className="w-10 h-10" />}
          title="Verify a carrier digest"
          description="Hashes the resolved payload file and cross-checks its modification time against the last recorded transmission. DCASS transmits corpus references, so the corpus file is the transmitted object."
        />
      )}

      {loading && (
        <div className="flex items-center justify-center py-10">
          <LoadingSpinner />
        </div>
      )}

      {error && !loading && (
        <div className="rounded-xl border border-error/30 bg-error/5 p-4 text-body-sm text-error">
          {error}. Check the media ID against the corpus catalogs in storage/data/indices.
        </div>
      )}

      {result && !loading && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: [0.19, 1, 0.22, 1] }}
          className="space-y-4"
        >
          <div
            className={`flex items-center justify-between rounded-xl border p-4 ${
              verified ? 'border-success/30 bg-success/5' : 'border-error/30 bg-error/5'
            }`}
          >
            <div className="flex items-center gap-3">
              {verified ? (
                <ShieldCheck className="w-8 h-8 text-success" />
              ) : (
                <ShieldAlert className="w-8 h-8 text-error" />
              )}
              <div>
                <p className={`font-semibold ${verified ? 'text-success' : 'text-error'}`}>
                  {verified ? 'Zero-Modification Verified' : 'Carrier Divergence Detected'}
                </p>
                <p className="text-body-sm text-muted">
                  Byte difference: {result.bytes_modified} of {result.bytes_total_compared} compared
                  {result.comparison_basis === 'reference-pass-through'
                    ? '. Reference pass-through, the carrier is never copied or altered'
                    : ''}
                </p>
              </div>
            </div>
            <Badge variant={verified ? 'success' : 'error'}>
              {verified ? '100% ZERO-MODIFICATION' : 'MODIFIED'}
            </Badge>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <HashCard
              label="Corpus Original"
              digest={result.corpus_sha256}
              sizeBytes={result.corpus_size_bytes}
              icon={<FileCheck2 className="w-4 h-4 text-info" />}
            />
            <HashCard
              label="Transmitted Carrier"
              digest={result.transmitted_sha256}
              sizeBytes={result.transmitted_size_bytes}
              icon={<FileCheck2 className="w-4 h-4 text-primary" />}
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-body-xs">
            <div className="rounded-lg border border-border bg-surface/40 p-3">
              <p className="stat-label">Media ID</p>
              <p className="font-mono text-foreground mt-1 truncate">{result.media_id}</p>
            </div>
            <div className="rounded-lg border border-border bg-surface/40 p-3">
              <p className="stat-label">Modality</p>
              <p className="text-foreground mt-1">{result.modality || 'unknown'}</p>
            </div>
            <div className="rounded-lg border border-border bg-surface/40 p-3">
              <p className="stat-label">In Shared Channel</p>
              <p className="text-foreground mt-1">
                {result.transmitted_found ? 'packet manifest present' : 'corpus reference only'}
              </p>
            </div>
            <div className="rounded-lg border border-border bg-surface/40 p-3">
              <p className="stat-label">Modified Since Transmission</p>
              <p className="text-foreground mt-1">
                {result.modified_since_transmission === null ||
                result.modified_since_transmission === undefined
                  ? 'no recorded transmission'
                  : result.modified_since_transmission
                    ? 'file touched after send'
                    : 'no, unchanged since send'}
              </p>
            </div>
          </div>

          {result.note && (
            <div className="rounded-lg border border-border bg-surface/40 px-4 py-3 text-body-xs text-muted">
              {result.note}
            </div>
          )}
        </motion.div>
      )}
    </Card>
  );
}

export default ZeroModVerifier;
