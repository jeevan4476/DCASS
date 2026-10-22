'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Navigation from './Navigation';
import { Card, StatCard, EmptyState, Separator } from './UI';

interface PageWrapperProps {
  children: React.ReactNode;
  title?: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
  fullWidth?: boolean;
}

export function PageWrapper({
  children,
  title,
  description,
  action,
  className = '',
  fullWidth = false,
}: PageWrapperProps) {
  return (
    <div className={`min-h-screen bg-background ${className}`}>
      <Navigation />
      <main className={`pt-4 ${fullWidth ? 'px-4' : 'container mx-auto px-4'}`}>
        {(title || description || action) && (
          <motion.header
            className="page-header py-4"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, ease: [0.19, 1, 0.22, 1] }}
          >
            <div>
              {title && <h1 className="page-title">{title}</h1>}
              {description && <p className="page-description mt-1">{description}</p>}
            </div>
            {action && <div>{action}</div>}
          </motion.header>
        )}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: [0.19, 1, 0.22, 1], delay: 0.1 }}
        >
          {children}
        </motion.div>
      </main>
    </div>
  );
}

interface SectionProps {
  children: React.ReactNode;
  title?: string;
  subtitle?: string;
  action?: React.ReactNode;
  className?: string;
  columns?: 1 | 2 | 3 | 4;
  gap?: number;
}

export function Section({
  children,
  title,
  subtitle,
  action,
  className = '',
  columns = 1,
  gap = 6,
}: SectionProps) {
  const gridClasses = {
    1: 'grid-cols-1',
    2: 'grid-cols-1 md:grid-cols-2',
    3: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4',
  };

  const gapClasses: Record<number, string> = {
    2: 'gap-2',
    3: 'gap-3',
    4: 'gap-4',
    6: 'gap-6',
    8: 'gap-8',
  };

  return (
    <motion.section
      className={`${className}`}
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-100px' }}
      transition={{ duration: 0.5, ease: [0.19, 1, 0.22, 1] }}
    >
      {(title || action) && (
        <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4 mb-6">
          <div>
            {title && <h2 className="section-title">{title}</h2>}
            {subtitle && <p className="section-subtitle">{subtitle}</p>}
          </div>
          {action && <div className="mt-4 sm:mt-0">{action}</div>}
        </div>
      )}
      <div className={`grid w-full ${gapClasses[gap] || 'gap-6'} ${gridClasses[columns]}`}>
        {children}
      </div>
    </motion.section>
  );
}

interface StatGridProps {
  stats: Array<{
    label: string;
    value: string | number;
    variant?: 'primary' | 'success' | 'warning' | 'error' | 'info' | 'neutral';
    trend?: { value: number; label: string };
    icon?: React.ReactNode;
  }>;
  columns?: 2 | 3 | 4;
  className?: string;
}

export function StatGrid({ stats, columns = 4, className = '' }: StatGridProps) {
  const gridClasses = {
    2: 'grid-cols-1 sm:grid-cols-2',
    3: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 sm:grid-cols-2 md:grid-cols-2 lg:grid-cols-4',
  };

  return (
    <div className={`grid w-full gap-4 ${gridClasses[columns]} ${className}`}>
      {stats.map((stat, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: [0.19, 1, 0.22, 1], delay: i * 0.05 }}
        >
          <StatCard {...stat} />
        </motion.div>
      ))}
    </div>
  );
}

interface DataTableProps<T> {
  columns: Array<{
    key: string;
    header: string;
    render?: (row: T) => React.ReactNode;
    className?: string;
  }>;
  data: T[];
  keyExtractor: (row: T) => string;
  emptyMessage?: string;
  emptyAction?: React.ReactNode;
  striped?: boolean;
  hoverable?: boolean;
  className?: string;
}

export function DataTable<T>({
  columns,
  data,
  keyExtractor,
  emptyMessage = 'No data available',
  emptyAction,
  striped = true,
  hoverable = true,
  className = '',
}: DataTableProps<T>) {
  if (data.length === 0) {
    return (
      <EmptyState
        title="No data"
        description={emptyMessage}
        action={emptyAction}
        className="py-12"
      />
    );
  }

  return (
    <div className={`overflow-x-auto ${className}`}>
      <table className="w-full text-left">
        <thead>
          <tr className="border-b border-border-strong">
            {columns.map((col) => (
              <th
                key={col.key}
                className={`px-4 py-3 text-body-sm font-semibold text-muted uppercase tracking-wider ${col.className || ''}`}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, rowIndex) => (
            <tr
              key={keyExtractor(row)}
              className={`transition-colors ${striped && rowIndex % 2 === 1 ? 'bg-surface/50' : ''} ${hoverable ? 'hover:bg-surface-hover' : ''}`}
            >
              {columns.map((col) => (
                <td
                  key={col.key}
                  className={`px-4 py-3 text-body-md ${col.className || ''}`}
                >
                  {col.render ? col.render(row) : (row as any)[col.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface ProgressBarProps {
  value: number;
  max?: number;
  variant?: 'primary' | 'success' | 'warning' | 'error';
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  label?: string;
  animated?: boolean;
  className?: string;
}

export function ProgressBar({
  value,
  max = 100,
  variant = 'primary',
  size = 'md',
  showLabel = false,
  label,
  animated = false,
  className = '',
}: ProgressBarProps) {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);

  const variants = {
    primary: 'bg-primary',
    success: 'bg-success',
    warning: 'bg-warning',
    error: 'bg-error',
  };

  const sizes = {
    sm: 'h-1.5',
    md: 'h-2.5',
    lg: 'h-4',
  };

  return (
    <div className={`${className} w-full`}>
      {(showLabel || label) && (
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-body-sm font-medium text-foreground">{label || `${Math.round(percentage)}%`}</span>
          {showLabel && <span className="text-body-sm text-muted tabular-nums">{value} / {max}</span>}
        </div>
      )}
      <div className={`relative w-full bg-border rounded-full overflow-hidden ${sizes[size]}`}>
        <motion.div
          className={`${variants[variant]} h-full rounded-full transition-all duration-500 ease-[cubic-bezier(0.19,1,0.22,1)] ${animated ? 'animate-pulse-soft' : ''}`}
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          style={{ boxShadow: `0 0 8px rgb(var(--color-${variant}-rgb) / 0.4)` }}
        />
      </div>
    </div>
  );
}

interface TabsProps {
  tabs: Array<{ id: string; label: string; icon?: React.ReactNode; count?: number }>;
  activeTab: string;
  onChange: (id: string) => void;
  className?: string;
  variant?: 'line' | 'pills';
}

export function Tabs({ tabs, activeTab, onChange, className = '', variant = 'line' }: TabsProps) {
  return (
    <div className={`flex gap-1 overflow-x-auto max-w-full pb-1 ${variant === 'pills' ? 'bg-surface p-1 rounded-xl' : 'border-b border-border'} ${className}`}>
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          className={`relative flex items-center gap-2 px-3 sm:px-4 py-2 sm:py-2.5 rounded-lg text-body-sm font-medium transition-all duration-fast shrink-0 ${
            activeTab === tab.id
              ? variant === 'line'
                ? 'text-primary'
                : 'bg-background text-foreground shadow-card'
              : 'text-muted hover:text-foreground hover:bg-surface'
          }`}
        >
          {tab.icon && <span>{tab.icon}</span>}
          {tab.label}
          {tab.count !== undefined && (
            <span className={`px-2 py-0.5 text-body-xs rounded-full ${activeTab === tab.id ? 'bg-primary/20 text-primary' : 'bg-border text-muted'}`}>
              {tab.count}
            </span>
          )}
        </button>
      ))}
      {variant === 'line' && (
        <motion.div
          className="absolute bottom-0 h-px bg-primary"
          animate={{
            width: `${(tabs.findIndex(t => t.id === activeTab) + 1) * (100 / tabs.length)}%`,
            x: `${tabs.findIndex(t => t.id === activeTab) * (100 / tabs.length)}%`,
          }}
          transition={{ type: 'spring', stiffness: 500, damping: 30 }}
        />
      )}
    </div>
  );
}