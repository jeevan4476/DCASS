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
    <motion.div
      className={`card ${hover ? 'card-hover cursor-pointer' : ''} ${className}`}
      whileHover={hover ? { y: -2, boxShadow: '0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)' } : undefined}
      whileTap={hover ? { scale: 0.98 } : undefined}
      onClick={onClick}
      style={onClick ? { cursor: 'pointer' } : undefined}
    >
      {(title || action) && (
        <div className="flex items-start justify-between mb-4">
          <div>
            {title && <h2 className="section-title">{title}</h2>}
            {subtitle && <p className="section-subtitle">{subtitle}</p>}
          </div>
          {action && <div>{action}</div>}
        </div>
      )}
      <div>{children}</div>
    </motion.div>
  );
}

interface StatCardProps {
  label: string;
  value: string | number;
  variant?: 'primary' | 'success' | 'warning' | 'error' | 'info' | 'neutral';
  trend?: { value: number; label: string };
  icon?: React.ReactNode;
  className?: string;
}

export function StatCard({
  label,
  value,
  variant = 'neutral',
  trend,
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

  const trendColor = trend && trend.value > 0 ? 'text-success' : trend && trend.value < 0 ? 'text-error' : 'text-muted';

  return (
    <Card className={`${className} bg-surface/50 h-full`}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <p className="stat-label truncate">{label}</p>
          <p className={`stat-value ${variantColors[variant]} mt-1 truncate`}>{value}</p>
          {trend && (
            <p className={`text-body-xs ${trendColor} mt-2 flex items-center gap-1 truncate`}>
              {trend.value > 0 ? '↑' : trend.value < 0 ? '↓' : '→'}
              <span>{Math.abs(trend.value)}%</span>
              <span className="text-muted">vs {trend.label}</span>
            </p>
          )}
        </div>
        {icon && <div className="text-muted/50 shrink-0">{icon}</div>}
      </div>
    </Card>
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
  variant = 'info',
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
    neutral: 'bg-border text-muted border-border',
  };

  const sizes = {
    sm: 'px-2 py-0.5 text-body-xs',
    md: 'px-2.5 py-1 text-body-xs',
    lg: 'px-3 py-1.5 text-body-sm',
  };

  return (
    <span className={`inline-flex items-center gap-1.5 ${variants[variant]} ${sizes[size]} ${className}`}>
      {dot && (
        <span
          className={`w-1.5 h-1.5 rounded-full ${{
            primary: 'bg-primary',
            success: 'bg-success',
            warning: 'bg-warning',
            error: 'bg-error',
            info: 'bg-info',
            neutral: 'bg-muted',
          }[variant]}`}
        />
      )}
      {children}
    </span>
  );
}

export function LoadingSpinner({ size = 'md', className = '' }: { size?: 'sm' | 'md' | 'lg'; className?: string }) {
  const sizes = {
    sm: 'h-6 w-6 border-2',
    md: 'h-10 w-10 border-2',
    lg: 'h-14 w-14 border-3',
  };

  return (
    <div className={`flex justify-center items-center ${className}`}>
      <motion.div
        className={`${sizes[size]} rounded-full border-primary/20 border-t-primary animate-spin-slow`}
        animate={{ rotate: 360 }}
        transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
      />
    </div>
  );
}

export function Skeleton({ className = '', variant = 'text' }: { className?: string; variant?: 'text' | 'circular' | 'rectangular' }) {
  const baseClass = 'skeleton';
  const variants = {
    text: 'h-4 w-3/4',
    circular: 'h-12 w-12 rounded-full',
    rectangular: 'h-20 w-full rounded-lg',
  };

  return <div className={`${baseClass} ${variants[variant]} ${className}`} />;
}

export function SkeletonCard({ lines = 3, className = '' }: { lines?: number; className?: string }) {
  return (
    <Card className={className}>
      <Skeleton variant="rectangular" className="mb-4" />
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
    <div className={`flex flex-col items-center justify-center text-center py-16 px-6 ${className}`}>
      {icon && (
        <div className="text-muted/30 text-6xl mb-4 animate-pulse-soft">{icon}</div>
      )}
      <h3 className="text-heading-md text-foreground mb-2">{title}</h3>
      {description && <p className="text-body-md text-muted max-w-sm mb-6">{description}</p>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

interface SeparatorProps {
  orientation?: 'horizontal' | 'vertical';
  className?: string;
}

export function Separator({ orientation = 'horizontal', className = '' }: SeparatorProps) {
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

export function Tooltip({ content, children, position = 'top', className = '' }: TooltipProps) {
  const [visible, setVisible] = React.useState(false);

  const positions = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2',
  };

  const arrows = {
    top: 'border-t-primary',
    bottom: 'border-b-primary',
    left: 'border-l-primary',
    right: 'border-r-primary',
  };

  return (
    <div className={`relative ${className || 'inline-block'}`} onMouseEnter={() => setVisible(true)} onMouseLeave={() => setVisible(false)}>
      {React.cloneElement(children, { 'aria-describedby': visible ? 'tooltip' : undefined })}
      {visible && (
        <motion.div
          id="tooltip"
          className={`absolute ${positions[position]} z-tooltip px-3 py-1.5 bg-surface border border-border-strong rounded-lg text-body-xs text-foreground whitespace-nowrap shadow-modal pointer-events-none`}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          transition={{ duration: 0.15 }}
          role="tooltip"
        >
          {content}
          <div className={`absolute w-0 h-0 border-4 border-transparent ${arrows[position]} ${{
            top: 'bottom-[-8px] left-1/2 -translate-x-1/2',
            bottom: 'top-[-8px] left-1/2 -translate-x-1/2',
            left: 'right-[-8px] top-1/2 -translate-y-1/2',
            right: 'left-[-8px] top-1/2 -translate-y-1/2',
          }[position]}`} />
        </motion.div>
      )}
    </div>
  );
}

export function Divider({ className = '', children }: { className?: string; children?: React.ReactNode }) {
  return (
    <div className={`flex items-center gap-4 ${className}`}>
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
      className={`btn ${variants[variant]} ${sizes[size]} ${className}`}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <svg className="animate-spin -ml-1 mr-2 h-5 w-5" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
        </svg>
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
          <div className="absolute left-3 top-1/2 -translate-y-1/2 text-muted pointer-events-none">
            {leftIcon}
          </div>
        )}
        <input
          id={inputId}
          className={`input ${leftIcon ? 'pl-10' : ''} ${rightIcon ? 'pr-10' : ''} ${error ? 'input-error' : ''} ${className}`}
          aria-invalid={error ? 'true' : 'false'}
          aria-describedby={error ? `${inputId}-error` : undefined}
          {...props}
        />
        {rightIcon && (
          <div className="absolute right-3 top-1/2 -translate-y-1/2 text-muted pointer-events-none">
            {rightIcon}
          </div>
        )}
      </div>
      {error && (
        <p id={`${inputId}-error`} className="mt-1.5 text-body-xs text-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}