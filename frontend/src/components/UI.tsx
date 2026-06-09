interface CardProps {
  title: string;
  children: React.ReactNode;
  className?: string;
}

export function Card({ title, children, className = '' }: CardProps) {
  return (
    <div className={`bg-gray-900 border border-gray-800 rounded-lg p-6 ${className}`}>
      <h2 className="text-lg font-semibold text-primary mb-4">{title}</h2>
      <div>{children}</div>
    </div>
  );
}

interface StatCardProps {
  label: string;
  value: string | number;
  status?: 'success' | 'error' | 'warning' | 'neutral';
}

export function StatCard({ label, value, status = 'neutral' }: StatCardProps) {
  const statusColors = {
    success: 'text-success',
    error: 'text-error',
    warning: 'text-warning',
    neutral: 'text-gray-300',
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="text-sm text-gray-400 mb-1">{label}</div>
      <div className={`text-2xl font-bold ${statusColors[status]}`}>{value}</div>
    </div>
  );
}

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'success' | 'error' | 'warning' | 'info';
}

export function Badge({ children, variant = 'info' }: BadgeProps) {
  const variants = {
    success: 'bg-success/20 text-success border-success/30',
    error: 'bg-error/20 text-error border-error/30',
    warning: 'bg-warning/20 text-warning border-warning/30',
    info: 'bg-primary/20 text-primary border-primary/30',
  };

  return (
    <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium border ${variants[variant]}`}>
      {children}
    </span>
  );
}

export function LoadingSpinner() {
  return (
    <div className="flex justify-center items-center py-8">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
    </div>
  );
}
