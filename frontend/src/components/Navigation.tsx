'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Menu,
  X,
  LogOut,
  MessageSquare,
  ShieldCheck,
  Activity,
  Lock,
  Unlock,
  LayoutGrid,
  Inbox,
  Sparkles,
  LogIn,
} from 'lucide-react';
import { useAuth } from '@/lib/auth';

export const navigationItems = [
  { href: '/chat', label: 'Chat', icon: MessageSquare, hint: 'Covert messenger' },
  { href: '/erasure', label: 'RS-ECC Lab', icon: ShieldCheck, hint: 'Real packet drops' },
  { href: '/wire', label: 'Live Wire', icon: Activity, hint: 'On-disk channel' },
  { href: '/encode', label: 'Encode', icon: Lock, hint: 'Text to carriers' },
  { href: '/decode', label: 'Decode', icon: Unlock, hint: 'Carriers to text' },
  { href: '/status', label: 'Panel', icon: LayoutGrid, hint: 'System overview' },
  { href: '/inbox', label: 'Inbox', icon: Inbox, hint: 'Received sessions' },
];

function BrandMark() {
  return (
    <Link href="/" className="flex items-center gap-2.5 group" aria-label="DCASS Home">
      <div className="w-8 h-8 rounded-xl bg-primary/15 border border-primary/25 flex items-center justify-center text-primary group-hover:bg-primary group-hover:text-white transition-colors">
        <Sparkles className="w-4 h-4" />
      </div>
      <div className="leading-tight">
        <span className="font-serif text-[1.15rem] font-semibold tracking-tight text-foreground block">
          DCASS
        </span>
        <span className="text-[10px] font-mono uppercase tracking-wider text-muted block">
          Semantic Stego
        </span>
      </div>
    </Link>
  );
}

export function Navigation() {
  const pathname = usePathname();
  const { user, logout } = useAuth();
  const [mobileOpen, setMobileOpen] = useState(false);

  const isActive = (href: string) => {
    if (href === '/') return pathname === '/';
    return pathname === href || pathname.startsWith(`${href}/`);
  };

  return (
    <>
      {/* Desktop Left Sidebar (Claude-inspired warm stone rail) */}
      <aside className="hidden md:flex md:flex-col md:w-[244px] md:shrink-0 h-screen sticky top-0 bg-surface-muted border-r border-border select-none">
        <div className="px-5 pt-5 pb-4 border-b border-border/70">
          <BrandMark />
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto" aria-label="Workspace navigation">
          <div className="px-2.5 pb-1.5 text-[10px] font-mono uppercase tracking-wider text-muted/80">
            Workspace
          </div>
          {navigationItems.map((item) => {
            const Icon = item.icon;
            const active = isActive(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={active ? 'page' : undefined}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-body-sm transition-all duration-fast ${
                  active
                    ? 'bg-surface text-foreground font-semibold border border-border shadow-subtle'
                    : 'text-muted hover:text-foreground hover:bg-surface-hover/70 font-medium border border-transparent'
                }`}
              >
                <Icon
                  className={`w-4 h-4 shrink-0 ${active ? 'text-primary' : 'text-muted'}`}
                  aria-hidden="true"
                />
                <span className="truncate">{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Bottom Account Footer */}
        <div className="p-3 border-t border-border/80">
          {user ? (
            <div className="p-2.5 rounded-xl bg-surface border border-border flex items-center justify-between gap-2">
              <div className="flex items-center gap-2.5 min-w-0">
                <div className="w-8 h-8 rounded-full bg-primary/15 text-primary border border-primary/25 flex items-center justify-center font-serif font-semibold text-sm uppercase shrink-0">
                  {user.username.slice(0, 2)}
                </div>
                <div className="min-w-0 leading-tight">
                  <div className="text-body-xs font-semibold text-foreground truncate capitalize">
                    {user.username}
                  </div>
                  <div className="text-[10px] font-mono text-muted truncate">
                    @{user.username}
                  </div>
                </div>
              </div>
              <button
                type="button"
                onClick={logout}
                title="Sign out"
                className="p-1.5 rounded-lg text-muted hover:text-error hover:bg-error-muted transition-colors shrink-0"
                aria-label="Sign out"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <div className="space-y-1.5">
              <Link href="/login" className="btn-primary btn-sm w-full justify-center">
                <LogIn className="w-3.5 h-3.5" />
                Sign in
              </Link>
              <Link href="/register" className="btn-secondary btn-sm w-full justify-center">
                Create account
              </Link>
            </div>
          )}
        </div>
      </aside>

      {/* Mobile Top Bar */}
      <header className="md:hidden sticky top-0 z-sticky bg-surface-muted/95 backdrop-blur-md border-b border-border px-4 h-14 flex items-center justify-between">
        <BrandMark />
        <div className="flex items-center gap-2">
          {user ? (
            <span className="px-2.5 py-1 rounded-full bg-surface border border-border text-body-xs font-mono text-foreground">
              @{user.username}
            </span>
          ) : (
            <Link href="/login" className="btn-primary btn-sm">
              Sign in
            </Link>
          )}
          <button
            type="button"
            onClick={() => setMobileOpen(true)}
            className="p-2 rounded-lg text-muted hover:text-foreground hover:bg-surface-hover"
            aria-label="Open menu"
          >
            <Menu className="w-5 h-5" />
          </button>
        </div>
      </header>

      {/* Mobile Drawer */}
      <AnimatePresence>
        {mobileOpen && (
          <>
            <motion.div
              className="fixed inset-0 bg-foreground/25 backdrop-blur-[2px] z-[300] md:hidden"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setMobileOpen(false)}
            />
            <motion.aside
              className="fixed inset-y-0 left-0 w-72 md:hidden bg-surface-muted border-r border-border z-[310] flex flex-col shadow-modal"
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={{ type: 'spring', stiffness: 380, damping: 32 }}
            >
              <div className="flex items-center justify-between px-4 h-14 border-b border-border">
                <BrandMark />
                <button
                  type="button"
                  onClick={() => setMobileOpen(false)}
                  className="p-2 rounded-lg text-muted hover:text-foreground hover:bg-surface-hover"
                  aria-label="Close menu"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
                {navigationItems.map((item) => {
                  const Icon = item.icon;
                  const active = isActive(item.href);
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      onClick={() => setMobileOpen(false)}
                      className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-body-sm ${
                        active
                          ? 'bg-surface text-foreground font-semibold border border-border shadow-subtle'
                          : 'text-muted hover:text-foreground hover:bg-surface-hover'
                      }`}
                    >
                      <Icon className={`w-4 h-4 ${active ? 'text-primary' : 'text-muted'}`} />
                      <span>{item.label}</span>
                    </Link>
                  );
                })}
              </nav>
              {user && (
                <div className="p-3 border-t border-border">
                  <button
                    type="button"
                    onClick={() => {
                      setMobileOpen(false);
                      logout();
                    }}
                    className="btn-ghost w-full justify-start text-error hover:bg-error-muted"
                  >
                    <LogOut className="w-4 h-4" />
                    Sign out (@{user.username})
                  </button>
                </div>
              )}
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  );
}

export default Navigation;
