'use client';

import React from 'react';
import { motion } from 'framer-motion';

interface CardProps {
  title?: string;
  subtitle?: string;
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
  onClick?: () => void;
  action?: React.ReactNode;
}

export function Card({
  title,
  subtitle,
  children,
  className = '',
  hover = false,
  onClick,
  action,
}: CardProps) {
  return (
    <div
      className={`${hover ? 'card-hover' : 'card'} ${className}`}
      onClick={onClick}
      style={onClick ? { cursor: 'pointer' } : undefined}
    >
      {(title || action) && (
        <div className="flex flex-wrap items-start justify-between gap-3 mb-4 pb-3 border-b border-border/70">
          <div className="min-w-0">
            {title && <h2 className="font-serif text-[1.2rem] font-medium text-foreground leading-snug">{title}</h2>}
            {subtitle && <p className="text-body-xs text-muted mt-0.5">{subtitle}</p>}
          </div>
          {action && <div className="shrink-0">{action}</div>}
        </div>
      )}
      <div>{children}</div>
    </div>
  );
}

interface StatCardProps {
  label: string;
  value: string | number;
  variant?: 'primary' | 'success' | 'warning' | 'error' | 'info' | 'neutral';
  icon?: React.ReactNode;
  className?: string;
}

export function StatCard({
  label,
  value,
  variant = 'neutral',
  icon,
  className = '',
}: StatCardProps) {
  const variantColors = {
    primary: 'text-primary',
    success: 'text-success',
    warning: 'text-warning',
    error: 'text-error',
    info: 'text-info',
    neutral: 'text-foreground',
  };

  return (
    <div className={`bg-surface border border-border rounded-2xl p-4 shadow-subtle ${className}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <p className="stat-label truncate">{label}</p>
          <p className={`stat-value ${variantColors[variant]} mt-2 truncate`}>{value}</p>
        </div>
        {icon && <div className="text-muted/60 shrink-0 mt-0.5">{icon}</div>}
      </div>
    </div>
  );
}

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'primary' | 'success' | 'warning' | 'error' | 'info' | 'neutral';
  size?: 'sm' | 'md' | 'lg';
  dot?: boolean;
  className?: string;
}

export function Badge({
  children,
  variant = 'neutral',
  size = 'md',
  dot = false,
  className = '',
}: BadgeProps) {
  const variants = {
    primary: 'badge-primary',
    success: 'badge-success',
    warning: 'badge-warning',
    error: 'badge-error',
    info: 'badge-info',
    neutral: 'bg-surface-muted text-muted border-border',
  };

  const sizes = {
    sm: 'px-2 py-0.5 text-[11px]',
    md: 'px-2.5 py-0.5 text-body-xs',
    lg: 'px-3 py-1 text-body-xs font-semibold',
  };

  const dotColors = {
    primary: 'bg-primary',
    success: 'bg-success',
    warning: 'bg-warning',
    error: 'bg-error',
    info: 'bg-info',
    neutral: 'bg-muted',
  };

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border font-medium ${variants[variant]} ${sizes[size]} ${className}`}>
      {dot && <span className={`w-1.5 h-1.5 rounded-full ${dotColors[variant]}`} />}
      {children}
    </span>
  );
}

export function LoadingSpinner({ size = 'md', className = '' }: { size?: 'sm' | 'md' | 'lg'; className?: string }) {
  const sizes = {
    sm: 'h-4 w-4 border-2',
    md: 'h-8 w-8 border-2',
    lg: 'h-11 w-11 border-[3px]',
  };

  return (
    <div className={`flex justify-center items-center ${className}`}>
      <motion.div
        className={`${sizes[size]} rounded-full border-border-strong border-t-primary`}
        animate={{ rotate: 360 }}
        transition={{ duration: 0.9, repeat: Infinity, ease: 'linear' }}
      />
    </div>
  );
}

export function Skeleton({ className = '', variant = 'text' }: { className?: string; variant?: 'text' | 'circular' | 'rectangular' }) {
  const variants = {
    text: 'h-4 w-3/4',
    circular: 'h-10 w-10 rounded-full',
    rectangular: 'h-16 w-full rounded-xl',
  };
  return <div className={`skeleton ${variants[variant]} ${className}`} />;
}

export function SkeletonCard({ lines = 3, className = '' }: { lines?: number; className?: string }) {
  return (
    <Card className={className}>
      <Skeleton variant="rectangular" className="mb-3" />
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton variant="text" key={i} className="mb-2" />
      ))}
    </Card>
  );
}

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}

export function EmptyState({ icon, title, description, action, className = '' }: EmptyStateProps) {
  return (
    <div className={`flex flex-col items-center justify-center text-center py-12 px-6 ${className}`}>
      {icon && <div className="text-muted/40 mb-3">{icon}</div>}
      <h3 className="font-serif text-heading-md font-medium text-foreground mb-1">{title}</h3>
      {description && <p className="text-body-sm text-muted max-w-md mb-5">{description}</p>}
      {action && <div>{action}</div>}
    </div>
  );
}

export function Separator({ orientation = 'horizontal', className = '' }: { orientation?: 'horizontal' | 'vertical'; className?: string }) {
  return (
    <div
      className={`bg-border ${orientation === 'horizontal' ? 'h-px w-full' : 'w-px h-full'} ${className}`}
      role="separator"
    />
  );
}

interface TooltipProps {
  content: string;
  children: React.ReactElement;
  position?: 'top' | 'bottom' | 'left' | 'right';
  className?: string;
}

export function Tooltip({ content, children, className = '' }: TooltipProps) {
  return (
    <div className={className || 'inline-block'} title={content}>
      {children}
    </div>
  );
}

export function Divider({ className = '', children }: { className?: string; children?: React.ReactNode }) {
  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <Separator />
      {children && <span className="text-body-xs text-muted uppercase tracking-wider">{children}</span>}
      <Separator />
    </div>
  );
}

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger' | 'success';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

export function Button({
  variant = 'primary',
  size = 'md',
  loading = false,
  leftIcon,
  rightIcon,
  className = '',
  children,
  disabled,
  ...props
}: ButtonProps) {
  const variants = {
    primary: 'btn-primary',
    secondary: 'btn-secondary',
    ghost: 'btn-ghost',
    danger: 'btn-danger',
    success: 'btn-success',
  };
  const sizes = {
    sm: 'btn-sm',
    md: '',
    lg: 'btn-lg',
  };

  return (
    <button
      className={`${variants[variant]} ${sizes[size]} ${className}`}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <span className="w-4 h-4 rounded-full border-2 border-current/30 border-t-current animate-spin" />
      ) : leftIcon ? (
        <span className="flex items-center">{leftIcon}</span>
      ) : null}
      {children}
      {rightIcon && !loading && <span className="flex items-center">{rightIcon}</span>}
    </button>
  );
}

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

export function Input({
  label,
  error,
  leftIcon,
  rightIcon,
  className = '',
  id,
  ...props
}: InputProps) {
  const inputId = id || label?.toLowerCase().replace(/\s+/g, '-');

  return (
    <div className="w-full">
      {label && <label htmlFor={inputId} className="label">{label}</label>}
      <div className="relative">
        {leftIcon && (
          <div className="absolute left-3.5 top-1/2 -translate-y-1/2 text-muted pointer-events-none">
            {leftIcon}
          </div>
        )}
        <input
          id={inputId}
          className={`input ${leftIcon ? 'pl-10' : ''} ${rightIcon ? 'pr-10' : ''} ${error ? 'input-error' : ''} ${className}`}
          aria-invalid={error ? 'true' : 'false'}
          {...props}
        />
        {rightIcon && (
          <div className="absolute right-3.5 top-1/2 -translate-y-1/2 text-muted pointer-events-none">
            {rightIcon}
          </div>
        )}
      </div>
      {error && (
        <p className="mt-1 text-body-xs text-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
