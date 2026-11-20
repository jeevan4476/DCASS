/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        background: 'rgb(var(--color-background-rgb) / <alpha-value>)',
        foreground: 'rgb(var(--color-foreground-rgb) / <alpha-value>)',
        surface: 'rgb(var(--color-surface-rgb) / <alpha-value>)',
        'surface-muted': 'rgb(var(--color-surface-muted-rgb) / <alpha-value>)',
        'surface-hover': 'rgb(var(--color-surface-hover-rgb) / <alpha-value>)',
        border: 'rgb(var(--color-border-rgb) / <alpha-value>)',
        'border-strong': 'rgb(var(--color-border-strong-rgb) / <alpha-value>)',
        muted: 'rgb(var(--color-muted-rgb) / <alpha-value>)',
        primary: {
          DEFAULT: 'rgb(var(--color-primary-rgb) / <alpha-value>)',
          foreground: 'rgb(var(--color-primary-foreground-rgb) / <alpha-value>)',
          hover: 'rgb(var(--color-primary-hover-rgb) / <alpha-value>)',
          muted: 'var(--color-primary-muted)',
        },
        success: {
          DEFAULT: 'rgb(var(--color-success-rgb) / <alpha-value>)',
          foreground: 'rgb(var(--color-success-foreground-rgb) / <alpha-value>)',
          muted: 'var(--color-success-muted)',
        },
        warning: {
          DEFAULT: 'rgb(var(--color-warning-rgb) / <alpha-value>)',
          foreground: 'rgb(var(--color-warning-foreground-rgb) / <alpha-value>)',
          muted: 'var(--color-warning-muted)',
        },
        error: {
          DEFAULT: 'rgb(var(--color-error-rgb) / <alpha-value>)',
          foreground: 'rgb(var(--color-error-foreground-rgb) / <alpha-value>)',
          muted: 'var(--color-error-muted)',
        },
        info: {
          DEFAULT: 'rgb(var(--color-info-rgb) / <alpha-value>)',
          foreground: 'rgb(var(--color-info-foreground-rgb) / <alpha-value>)',
          muted: 'var(--color-info-muted)',
        },
        purple: {
          DEFAULT: 'rgb(var(--color-purple-rgb) / <alpha-value>)',
          foreground: 'rgb(var(--color-purple-foreground-rgb) / <alpha-value>)',
          muted: 'var(--color-purple-muted)',
        },
      },
      fontFamily: {
        serif: ['var(--font-serif)', 'Georgia', 'serif'],
        sans: ['var(--font-sans)', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['var(--font-mono)', 'ui-monospace', 'monospace'],
      },
      fontSize: {
        'display-sm': ['2.125rem', { lineHeight: '1.18', letterSpacing: '-0.025em' }],
        'heading-lg': ['1.5rem', { lineHeight: '1.25', letterSpacing: '-0.02em' }],
        'heading-md': ['1.2rem', { lineHeight: '1.35', letterSpacing: '-0.015em' }],
        'heading-sm': ['1.025rem', { lineHeight: '1.4', letterSpacing: '-0.01em' }],
        'body-lg': ['1.0625rem', { lineHeight: '1.6' }],
        'body-md': ['0.9375rem', { lineHeight: '1.6' }],
        'body-sm': ['0.875rem', { lineHeight: '1.5' }],
        'body-xs': ['0.75rem', { lineHeight: '1.45' }],
      },
      boxShadow: {
        subtle: '0 1px 2px rgba(24, 23, 21, 0.04)',
        card: '0 1px 3px rgba(24, 23, 21, 0.05), 0 1px 2px rgba(24, 23, 21, 0.03)',
        elevated: '0 8px 24px -6px rgba(24, 23, 21, 0.08), 0 2px 6px rgba(24, 23, 21, 0.04)',
        modal: '0 24px 60px -12px rgba(24, 23, 21, 0.18)',
      },
      transitionDuration: {
        fast: '140ms',
        normal: '200ms',
        slow: '280ms',
      },
      zIndex: {
        dropdown: '100',
        sticky: '200',
        modal: '300',
        popover: '400',
        tooltip: '500',
        toast: '600',
      },
    },
  },
  plugins: [],
};
