'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import {
  MessageSquare,
  ShieldCheck,
  Activity,
  Lock,
  Unlock,
  LayoutGrid,
  Inbox,
  ArrowUpRight,
  Sparkles,
  Layers,
  Network,
  Database,
} from 'lucide-react';
import { PageWrapper, Badge } from '@/components';
import { getStatus, StatusResponse } from '@/lib/api';

const SYSTEM_PILLARS = [
  {
    title: 'Zero-Modification Carriers',
    detail: '245,279 unmodified image, text, and audio items across Flickr30k, Wikipedia, and Libretta.',
    icon: Database,
  },
  {
    title: 'VCP & DSSC Codecs',
    detail: 'Byte-exact Voronoi Codebook Partitioning (8 bpc) and keyed Dynamic State-Space Coding (~13.6 bpc).',
    icon: Layers,
  },
  {
    title: 'Neural Multi-Channel Stealth',
    detail: 'WGAN-GP timing mimicry and PPO routing disperse packets across 3 independent conduits.',
    icon: Network,
  },
  {
    title: 'RS-ECC Erasure Recovery',
    detail: 'CRC-16 framing plus GF(2^8) Reed-Solomon parity recovers messages even after packet loss on disk.',
    icon: ShieldCheck,
  },
];

const WORKSPACE_CARDS = [
  {
    href: '/chat',
    title: 'Covert Chat',
    description: 'Two-way steganographic messenger with multimodal carrier inspection, asset previews, and packet JSON views.',
    icon: MessageSquare,
    tag: 'Messenger',
  },
  {
    href: '/erasure',
    title: 'RS-ECC Lab',
    description: 'Transmit real carrier files onto disk, inspect or delete individual packets, and verify 100% erasure recovery.',
    icon: ShieldCheck,
    tag: 'Fault Tolerance',
  },
  {
    href: '/wire',
    title: 'Live Wire',
    description: 'Monitor real packet JSON files in storage/shared_channel, inspect channel dispersion, and drop packets live.',
    icon: Activity,
    tag: 'Telemetry',
  },
  {
    href: '/encode',
    title: 'Encode',
    description: 'Encode any plaintext into clickable corpus carriers with selectable codec mode and Reed-Solomon parity.',
    icon: Lock,
    tag: 'Codec',
  },
  {
    href: '/decode',
    title: 'Decode',
    description: 'Reconstruct plaintext from carrier sequences with clickable asset inspection and slot-level erasure controls.',
    icon: Unlock,
    tag: 'Codec',
  },
  {
    href: '/status',
    title: 'System Panel',
    description: 'Check live FAISS index counts, WGAN/PPO stealth checkpoints, compute device, and registered accounts.',
    icon: LayoutGrid,
    tag: 'Monitoring',
  },
  {
    href: '/inbox',
    title: 'Inbox',
    description: 'View addressed sessions and decode complete or packet-dropped transmissions directly from disk.',
    icon: Inbox,
    tag: 'Receiver',
  },
];

export default function HomePage() {
  const [status, setStatus] = useState<StatusResponse | null>(null);

  useEffect(() => {
    getStatus()
      .then(setStatus)
      .catch(() => setStatus(null));
  }, []);

  return (
    <PageWrapper>
      <div className="max-w-5xl mx-auto py-4 sm:py-8 space-y-10">
        {/* Editorial Hero */}
        <section className="space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary-muted border border-primary/20 text-primary text-body-xs font-medium">
            <Sparkles className="w-3.5 h-3.5" />
            <span>DCASS &middot; Semantic Covert Channels</span>
          </div>

          <h1 className="font-serif text-3xl sm:text-[2.75rem] font-medium text-foreground leading-[1.14] tracking-tight max-w-3xl">
            Dynamic Context-Aware Semantic Steganography.
          </h1>

          <p className="text-body-lg text-muted max-w-2xl leading-relaxed">
            Communicate covertly by transmitting unmodified references to public images, text
            passages, and audio clips across multiple channels, backed by neural stealth scheduling
            and Reed-Solomon error correction.
          </p>

          <div className="flex flex-wrap items-center gap-3 pt-2">
            <Link href="/chat" className="btn-primary btn-lg">
              <MessageSquare className="w-4 h-4" />
              Open Covert Chat
            </Link>
            <Link href="/erasure" className="btn-secondary btn-lg">
              <ShieldCheck className="w-4 h-4" />
              RS-ECC Lab
            </Link>
            <Link href="/wire" className="btn-secondary btn-lg">
              <Activity className="w-4 h-4" />
              Live Wire
            </Link>
            {status && (
              <Badge variant="success" size="lg" dot className="ml-1">
                {status.total_items.toLocaleString()} vectors &middot; {status.device.toUpperCase()}
              </Badge>
            )}
          </div>
        </section>

        {/* Architecture Pillars */}
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
          {SYSTEM_PILLARS.map((pillar) => {
            const Icon = pillar.icon;
            return (
              <div
                key={pillar.title}
                className="bg-surface border border-border rounded-2xl p-4 shadow-subtle space-y-2"
              >
                <div className="w-8 h-8 rounded-xl bg-surface-muted border border-border flex items-center justify-center text-primary">
                  <Icon className="w-4 h-4" />
                </div>
                <h2 className="font-serif text-[1.05rem] font-medium text-foreground leading-snug">
                  {pillar.title}
                </h2>
                <p className="text-body-xs text-muted leading-relaxed">{pillar.detail}</p>
              </div>
            );
          })}
        </section>

        {/* Workspace Grid */}
        <section className="space-y-4">
          <div>
            <h2 className="section-title">System Modules</h2>
            <p className="section-subtitle">
              Every module runs live against the local FAISS corpus and shared channel filesystem
            </p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {WORKSPACE_CARDS.map((card) => {
              const Icon = card.icon;
              return (
                <Link
                  key={card.href}
                  href={card.href}
                  className="group bg-surface border border-border hover:border-border-strong rounded-2xl p-5 shadow-subtle hover:shadow-elevated transition-all flex flex-col justify-between gap-4"
                >
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="w-9 h-9 rounded-xl bg-surface-muted border border-border flex items-center justify-center text-primary group-hover:bg-primary group-hover:text-white group-hover:border-primary transition-colors">
                        <Icon className="w-4 h-4" />
                      </div>
                      <span className="text-[11px] font-mono text-muted">{card.tag}</span>
                    </div>
                    <div>
                      <h3 className="font-serif text-[1.2rem] font-medium text-foreground group-hover:text-primary transition-colors flex items-center gap-1">
                        <span>{card.title}</span>
                        <ArrowUpRight className="w-4 h-4 opacity-0 -translate-y-0.5 group-hover:opacity-100 group-hover:translate-y-0 transition-all" />
                      </h3>
                      <p className="text-body-sm text-muted mt-1 leading-relaxed">
                        {card.description}
                      </p>
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
        </section>
      </div>
    </PageWrapper>
  );
}
