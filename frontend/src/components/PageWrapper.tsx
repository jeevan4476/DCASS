'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Navigation from './Navigation';
import { StatCard, EmptyState } from './UI';

interface PageWrapperProps {
  children: React.ReactNode;
  title?: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
  fullWidth?: boolean;
  hideNav?: boolean;
}

export function PageWrapper({
  children,
  title,
  description,
  action,
  className = '',
  fullWidth = false,
  hideNav = false,
}: PageWrapperProps) {
  if (hideNav) {
    return (
      <div className={`min-h-screen bg-background text-foreground ${className}`}>
        {children}
      </div>
    );
  }

  return (
    <div className={`min-h-screen flex flex-col md:flex-row bg-background text-foreground ${className}`}>
      <Navigation />
      <main className="flex-1 min-w-0 px-4 sm:px-8 lg:px-10 py-6 sm:py-8">
        <div className={fullWidth ? 'w-full' : 'max-w-6xl mx-auto'}>
          {(title || description || action) && (
            <motion.header
              className="page-header"
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.25 }}
            >
              <div className="min-w-0">
                {title && <h1 className="page-title">{title}</h1>}
                {description && <p className="page-description">{description}</p>}
              </div>
              {action && <div className="shrink-0">{action}</div>}
            </motion.header>
          )}
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.25, delay: 0.04 }}
          >
            {children}
          </motion.div>
        </div>
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
  gap = 5,
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
    5: 'gap-5',
    6: 'gap-6',
  };

  return (
    <section className={`mb-8 last:mb-0 ${className}`}>
      {(title || action) && (
        <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-3 mb-4">
          <div>
            {title && <h2 className="section-title">{title}</h2>}
            {subtitle && <p className="section-subtitle">{subtitle}</p>}
          </div>
          {action && <div>{action}</div>}
        </div>
      )}
      <div className={`grid w-full ${gapClasses[gap] || 'gap-5'} ${gridClasses[columns]}`}>
        {children}
      </div>
    </section>
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
    3: 'grid-cols-1 sm:grid-cols-3',
    4: 'grid-cols-2 lg:grid-cols-4',
  };

  return (
    <div className={`grid w-full gap-3.5 ${gridClasses[columns]} ${className}`}>
      {stats.map((stat, i) => (
        <StatCard key={i} {...stat} />
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
  striped = false,
  hoverable = true,
  className = '',
}: DataTableProps<T>) {
  if (data.length === 0) {
    return (
      <EmptyState
        title="No records"
        description={emptyMessage}
        action={emptyAction}
        className="py-10"
      />
    );
  }

  return (
    <div className={`overflow-x-auto ${className}`}>
      <table className="w-full text-left border-collapse">
        <thead>
          <tr className="border-b border-border bg-surface-muted/60">
            {columns.map((col) => (
              <th
                key={col.key}
                className={`px-4 py-2.5 text-[11px] font-mono font-semibold text-muted uppercase tracking-wider ${col.className || ''}`}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-border/70">
          {data.map((row, rowIndex) => (
            <tr
              key={keyExtractor(row)}
              className={`transition-colors ${striped && rowIndex % 2 === 1 ? 'bg-surface-muted/35' : ''} ${hoverable ? 'hover:bg-surface-muted/50' : ''}`}
            >
              {columns.map((col) => (
                <td key={col.key} className={`px-4 py-3 text-body-sm ${col.className || ''}`}>
                  {col.render ? col.render(row) : (row as Record<string, React.ReactNode>)[col.key]}
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
  className = '',
}: ProgressBarProps) {
  const percentage = Math.min(Math.max((value / Math.max(max, 1)) * 100, 0), 100);
  const variants = {
    primary: 'bg-primary',
    success: 'bg-success',
    warning: 'bg-warning',
    error: 'bg-error',
  };
  const sizes = {
    sm: 'h-1.5',
    md: 'h-2',
    lg: 'h-3',
  };

  return (
    <div className={`w-full ${className}`}>
      {(showLabel || label) && (
        <div className="flex items-center justify-between mb-1.5 text-body-xs">
          <span className="font-medium text-foreground">{label || `${Math.round(percentage)}%`}</span>
          {showLabel && (
            <span className="text-muted font-mono tabular-nums">
              {value} / {max}
            </span>
          )}
        </div>
      )}
      <div className={`w-full bg-surface-muted border border-border/60 rounded-full overflow-hidden ${sizes[size]}`}>
        <div
          className={`${variants[variant]} h-full rounded-full transition-all duration-300`}
          style={{ width: `${percentage}%` }}
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

export function Tabs({ tabs, activeTab, onChange, className = '' }: TabsProps) {
  return (
    <div className={`inline-flex flex-wrap gap-1 p-1 rounded-xl bg-surface-muted border border-border ${className}`}>
      {tabs.map((tab) => {
        const active = activeTab === tab.id;
        return (
          <button
            key={tab.id}
            type="button"
            onClick={() => onChange(tab.id)}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-body-xs font-medium transition-all ${
              active
                ? 'bg-surface text-foreground shadow-subtle border border-border'
                : 'text-muted hover:text-foreground border border-transparent'
            }`}
          >
            {tab.icon}
            <span>{tab.label}</span>
            {tab.count !== undefined && (
              <span
                className={`px-1.5 py-0.5 text-[11px] font-mono rounded-md ${
                  active ? 'bg-primary-muted text-primary' : 'bg-surface/70 text-muted'
                }`}
              >
                {tab.count}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
