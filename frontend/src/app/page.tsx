'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { Send, Mail, BarChart2, Terminal, Shield, Network, Sparkles, Zap, Brain, Eye } from 'lucide-react';
import { Navigation, Card, Badge } from '@/components';

const features = [
  {
    title: 'Zero-Modification',
    description: 'No changes to carrier media — resistant to classical steganalysis',
    icon: Shield,
  },
  {
    title: 'Multi-Modal',
    description: 'Support for text, image, and audio semantic encoding',
    icon: Sparkles,
  },
  {
    title: 'Dynamic Fallback',
    description: 'Auto mode tries RL → GAN → Static scheduling',
    icon: Brain,
  },
  {
    title: 'Behavioral Stealth',
    description: 'Human-like transmission patterns with noise injection',
    icon: Eye,
  },
];

const quickActions = [
  { href: '/send', label: 'Send', description: 'Encode and transmit a secret message', icon: Send, color: 'text-primary' },
  { href: '/inbox', label: 'Inbox', description: 'View and decode received messages', icon: Mail, color: 'text-success' },
  { href: '/status', label: 'Panel', description: 'System monitoring and telemetry', icon: BarChart2, color: 'text-info' },
  { href: '/encode', label: 'Encode', description: 'Generate media sequences standalone', icon: Terminal, color: 'text-warning' },
  { href: '/decode', label: 'Decode', description: 'Reconstruct meaning from media IDs', icon: Shield, color: 'text-purple-400' },
  { href: '/wire', label: 'Wire', description: 'Real-time transmission telemetry', icon: Network, color: 'text-primary' },
];

export default function Home() {
  return (
    <>
      <Navigation />
      <main className="min-h-screen bg-background relative overflow-hidden">
        {/* Subtle background pattern */}
        <div className="absolute inset-0 opacity-30" aria-hidden="true">
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--color-primary)/0.03,_transparent_70%)]" />
        </div>

        <div className="container mx-auto px-4 py-16 lg:py-24 relative z-10">
          {/* Hero */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.19, 1, 0.22, 1] }}
            className="max-w-4xl mx-auto text-center mb-16 lg:mb-24"
          >
            <Badge variant="primary" className="mb-6 inline-flex" size="lg">
              <Zap className="w-3 h-3 mr-1" />
              DCASS v0.1.0
            </Badge>
            <h1 className="text-display-xl font-bold text-foreground mb-6">
              Dynamic Context-Aware
              <br />
              <span className="text-primary">Semantic Steganography</span>
            </h1>
            <p className="text-body-lg text-muted mb-10 max-w-2xl mx-auto">
              Zero-modification semantic steganography system with AI-driven stealth.
              Encode messages using semantically aligned media without altering any carrier content.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link href="/send" className="btn-primary btn-lg">
                <Send className="w-5 h-5" />
                Send a Message
              </Link>
              <Link href="/status" className="btn-secondary btn-lg">
                <BarChart2 className="w-5 h-5" />
                View Panel
              </Link>
            </div>
          </motion.div>

          {/* Quick Actions */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.19, 1, 0.22, 1], delay: 0.1 }}
          >
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-16">
              {quickActions.map((action, i) => {
                const Icon = action.icon;
                return (
                  <Link
                    key={action.href}
                    href={action.href}
                    className="card-hover group text-center py-8 px-6 relative overflow-hidden"
                    style={{ transitionDelay: `${i * 50}ms` }}
                  >
                    <div className={`w-14 h-14 rounded-xl mx-auto mb-4 flex items-center justify-center ${action.color} bg-current/10 group-hover:bg-current/20 transition-colors`}>
                      <Icon className="w-7 h-7" />
                    </div>
                    <h3 className="font-semibold text-foreground mb-1">{action.label}</h3>
                    <p className="text-body-sm text-muted">{action.description}</p>
                    {/* Shimmer overlay on hover */}
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-primary/5 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700" />
                  </Link>
                );
              })}
            </div>
          </motion.div>

          {/* Features */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-100px' }}
            transition={{ duration: 0.6, ease: [0.19, 1, 0.22, 1], delay: 0.2 }}
            className="mb-16"
          >
            <div className="text-center mb-10">
              <h2 className="text-heading-xl font-bold text-foreground mb-3">Key Capabilities</h2>
              <p className="text-body-lg text-muted max-w-2xl mx-auto">
                Built on semantic vector spaces with dynamic scheduling for covert communication.
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {features.map((feature, i) => {
                const Icon = feature.icon;
                return (
                  <motion.div
                    key={feature.title}
                    whileHover={{ y: -4, boxShadow: '0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)' }}
                    transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                    className="group"
                  >
                    <Card className="hover:border-primary/50 h-full">
                      <div className="flex items-start gap-4">
                        <motion.div
                          whileHover={{ scale: 1.1, rotate: 6 }}
                          transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                          className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary/20 transition-all duration-300"
                        >
                          <Icon className="w-6 h-6" />
                        </motion.div>
                        <div>
                          <h3 className="font-semibold text-foreground mb-1">{feature.title}</h3>
                          <p className="text-body-sm text-muted">{feature.description}</p>
                        </div>
                      </div>
                    </Card>
                  </motion.div>
                );
              })}
            </div>
          </motion.section>

          {/* Tech Stack / Stats */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-100px' }}
            transition={{ duration: 0.6, ease: [0.19, 1, 0.22, 1], delay: 0.3 }}
          >
            <Card className="relative overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-r from-primary/5 via-transparent to-info/5" />
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center relative z-10">
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.4, type: 'spring', stiffness: 300, damping: 20 }}
                  className="p-4"
                >
                  <p className="stat-value text-primary">512</p>
                  <p className="stat-label mt-1">Vector Dimensions</p>
                </motion.div>
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.45, type: 'spring', stiffness: 300, damping: 20 }}
                  className="p-4"
                >
                  <p className="stat-value text-success">256</p>
                  <p className="stat-label mt-1">Codebook Centroids</p>
                </motion.div>
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.5, type: 'spring', stiffness: 300, damping: 20 }}
                  className="p-4"
                >
                  <p className="stat-value text-info">3</p>
                  <p className="stat-label mt-1">Modalities</p>
                </motion.div>
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.55, type: 'spring', stiffness: 300, damping: 20 }}
                  className="p-4"
                >
                  <p className="stat-value text-warning">0%</p>
                  <p className="stat-label mt-1">Bit Error Rate</p>
                </motion.div>
              </div>
            </Card>
          </motion.section>

          {/* Footer CTA */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.19, 1, 0.22, 1], delay: 0.4 }}
            className="text-center mt-16"
          >
            <p className="text-body-md text-muted mb-4">Ready to try DCASS?</p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link href="/register" className="btn-primary btn-lg">
                Create Free Account
              </Link>
              <Link href="/login" className="btn-secondary btn-lg">
                Sign In
              </Link>
            </div>
          </motion.div>
        </div>
      </main>
    </>
  );
}