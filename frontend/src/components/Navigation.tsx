'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, X, User, LogOut, LayoutDashboard, Send, Mail, Settings, Terminal, Shield, BarChart2 } from 'lucide-react';
import { useAuth } from '@/lib/auth';

const navigationItems = [
  { href: '/', label: 'Home', icon: LayoutDashboard },
  { href: '/send', label: 'Send', icon: Send },
  { href: '/inbox', label: 'Inbox', icon: Mail },
  { href: '/status', label: 'Panel', icon: BarChart2 },
  { href: '/encode', label: 'Encode', icon: Terminal },
  { href: '/decode', label: 'Decode', icon: Shield },
];

export function Navigation() {
  const pathname = usePathname();
  const { user, logout, loading } = useAuth();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);

  const isActive = (href: string) => {
    if (href === '/') return pathname === '/';
    return pathname.startsWith(href);
  };

  return (
    <>
      {/* Desktop Navigation */}
      <nav className="hidden md:flex glass-strong border-b border-border sticky top-0 z-sticky">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3 lg:gap-8">
              <Link href="/" className="flex items-center gap-2 shrink-0" aria-label="DCASS Home">
                <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center">
                  <span className="text-primary font-bold text-lg">D</span>
                </div>
                <span className="text-heading-md font-bold text-foreground">DCASS</span>
              </Link>

              <div className="flex items-center gap-1 flex-wrap" role="navigation" aria-label="Main navigation">
                {navigationItems.map((item) => {
                  const Icon = item.icon;
                  const active = isActive(item.href);
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      className={`flex items-center gap-1.5 px-2.5 py-1.5 lg:px-3 lg:py-2 rounded-lg text-xs lg:text-body-sm font-medium transition-all duration-fast shrink-0 ${
                        active
                          ? 'bg-primary/10 text-primary border border-primary/20'
                          : 'text-muted hover:text-foreground hover:bg-surface'
                      }`}
                      aria-current={active ? 'page' : undefined}
                    >
                      <Icon className="w-3.5 h-3.5 lg:w-4 lg:h-4" aria-hidden="true" />
                      <span>{item.label}</span>
                    </Link>
                  );
                })}
              </div>
            </div>

            <div className="flex items-center gap-3">
              {user ? (
                <div className="relative">
                  <button
                    onClick={() => setUserMenuOpen(!userMenuOpen)}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg text-body-sm font-medium text-muted hover:text-foreground hover:bg-surface transition-colors"
                    aria-expanded={userMenuOpen}
                    aria-haspopup="true"
                  >
                    <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
                      <User className="w-4 h-4 text-primary" aria-hidden="true" />
                    </div>
                    <span className="hidden sm:block">{user.username}</span>
                  </button>

                  <AnimatePresence>
                    {userMenuOpen && (
                      <>
                        <motion.div
                          className="fixed inset-0 z-dropdown"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          exit={{ opacity: 0 }}
                          onClick={() => setUserMenuOpen(false)}
                          aria-hidden="true"
                        />
                        <motion.div
                          className="absolute right-0 mt-2 w-48 glass-strong border border-border rounded-xl py-2 shadow-modal"
                          initial={{ opacity: 0, y: -10, scale: 0.95 }}
                          animate={{ opacity: 1, y: 0, scale: 1 }}
                          exit={{ opacity: 0, y: -10, scale: 0.95 }}
                          transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                          role="menu"
                        >
                          <div className="px-3 py-2 border-b border-border">
                            <p className="text-body-sm font-medium text-foreground">{user.username}</p>
                            <p className="text-body-xs text-muted">User</p>
                          </div>
                          <Link
                            href="/status"
                            className="flex items-center gap-2 px-3 py-2 text-body-sm text-muted hover:text-foreground hover:bg-surface rounded-none"
                            role="menuitem"
                            onClick={() => setUserMenuOpen(false)}
                          >
                            <LayoutDashboard className="w-4 h-4" />
                            Panel
                          </Link>
                          <Link
                            href="/send"
                            className="flex items-center gap-2 px-3 py-2 text-body-sm text-muted hover:text-foreground hover:bg-surface rounded-none"
                            role="menuitem"
                            onClick={() => setUserMenuOpen(false)}
                          >
                            <Send className="w-4 h-4" />
                            Send Message
                          </Link>
                          <Link
                            href="/inbox"
                            className="flex items-center gap-2 px-3 py-2 text-body-sm text-muted hover:text-foreground hover:bg-surface rounded-none"
                            role="menuitem"
                            onClick={() => setUserMenuOpen(false)}
                          >
                            <Mail className="w-4 h-4" />
                            Inbox
                          </Link>
                          <div className="border-t border-border my-2" />
                          <button
                            onClick={logout}
                            className="flex items-center gap-2 w-full px-3 py-2 text-body-sm text-error hover:bg-error/10 rounded-none"
                            role="menuitem"
                          >
                            <LogOut className="w-4 h-4" />
                            Sign out
                          </button>
                        </motion.div>
                      </>
                    )}
                  </AnimatePresence>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <Link
                    href="/login"
                    className="btn-ghost btn-sm"
                  >
                    Sign in
                  </Link>
                  <Link
                    href="/register"
                    className="btn-primary btn-sm"
                  >
                    Get Started
                  </Link>
                </div>
              )}
            </div>
          </div>
        </div>
      </nav>

      {/* Mobile Navigation */}
      <nav className="md:hidden glass-strong border-b border-border sticky top-0 z-sticky">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between h-14">
            <Link href="/" className="flex items-center gap-2" aria-label="DCASS Home">
              <div className="w-7 h-7 rounded-lg bg-primary/20 flex items-center justify-center">
                <span className="text-primary font-bold text-base">D</span>
              </div>
              <span className="text-heading-sm font-bold text-foreground">DCASS</span>
            </Link>

            <div className="flex items-center gap-2">
              {user ? (
                <div className="relative">
                  <button
                    onClick={() => setUserMenuOpen(!userMenuOpen)}
                    className="flex items-center gap-2 px-2 py-1.5 rounded-lg text-muted hover:text-foreground hover:bg-surface transition-colors"
                    aria-expanded={userMenuOpen}
                    aria-haspopup="true"
                  >
                    <div className="w-7 h-7 rounded-full bg-primary/20 flex items-center justify-center">
                      <User className="w-3.5 h-3.5 text-primary" />
                    </div>
                  </button>

                  <AnimatePresence>
                    {userMenuOpen && (
                      <>
                        <motion.div
                          className="fixed inset-0 z-dropdown"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          exit={{ opacity: 0 }}
                          onClick={() => setUserMenuOpen(false)}
                        />
                        <motion.div
                          className="absolute right-0 mt-1 w-56 glass-strong border border-border rounded-xl py-2 shadow-modal"
                          initial={{ opacity: 0, y: -10, scale: 0.95 }}
                          animate={{ opacity: 1, y: 0, scale: 1 }}
                          exit={{ opacity: 0, y: -10, scale: 0.95 }}
                          transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                          role="menu"
                        >
                          <div className="px-3 py-2 border-b border-border">
                            <p className="text-body-sm font-medium text-foreground">{user.username}</p>
                            <p className="text-body-xs text-muted">User</p>
                          </div>
                          {navigationItems.map((item) => {
                            const Icon = item.icon;
                            return (
                              <Link
                                key={item.href}
                                href={item.href}
                                className="flex items-center gap-2 px-3 py-2 text-body-sm text-muted hover:text-foreground hover:bg-surface"
                                role="menuitem"
                                onClick={() => setUserMenuOpen(false)}
                              >
                                <Icon className="w-4 h-4" />
                                <span>{item.label}</span>
                              </Link>
                            );
                          })}
                          <div className="border-t border-border my-2" />
                          <button
                            onClick={logout}
                            className="flex items-center gap-2 w-full px-3 py-2 text-body-sm text-error hover:bg-error/10"
                            role="menuitem"
                          >
                            <LogOut className="w-4 h-4" />
                            Sign out
                          </button>
                        </motion.div>
                      </>
                    )}
                  </AnimatePresence>
                </div>
              ) : (
                <Link
                  href="/login"
                  className="btn-ghost btn-sm px-3"
                >
                  Sign in
                </Link>
              )}

              <button
                onClick={() => setMobileOpen(true)}
                className="p-2 rounded-lg text-muted hover:text-foreground hover:bg-surface transition-colors"
                aria-label="Open menu"
                aria-expanded={mobileOpen}
              >
                <Menu className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Mobile Drawer */}
      <AnimatePresence>
        {mobileOpen && (
          <>
            <motion.div
              className="fixed inset-0 bg-black/60 z-[300] md:hidden"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setMobileOpen(false)}
              aria-hidden="true"
            />
            <motion.aside
              className="fixed inset-y-0 right-0 w-72 md:hidden bg-surface border-l border-border z-[310] flex flex-col"
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', stiffness: 400, damping: 30 }}
              role="dialog"
              aria-modal="true"
              aria-label="Navigation menu"
            >
              <div className="flex items-center justify-between h-16 px-4 border-b border-border">
                <span className="text-heading-sm font-bold text-foreground">Menu</span>
                <button
                  onClick={() => setMobileOpen(false)}
                  className="p-2 rounded-lg text-muted hover:text-foreground hover:bg-surface-hover transition-colors"
                  aria-label="Close menu"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <nav className="flex-1 p-4 space-y-1 overflow-y-auto" role="navigation" aria-label="Mobile navigation">
                {navigationItems.map((item) => {
                  const Icon = item.icon;
                  const active = isActive(item.href);
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      className={`flex items-center gap-3 px-3 py-3 rounded-xl text-body-md font-medium transition-colors ${
                        active
                          ? 'bg-primary/10 text-primary border border-primary/20'
                          : 'text-muted hover:text-foreground hover:bg-surface-hover'
                      }`}
                      onClick={() => setMobileOpen(false)}
                      aria-current={active ? 'page' : undefined}
                    >
                      <Icon className="w-5 h-5" aria-hidden="true" />
                      <span>{item.label}</span>
                    </Link>
                  );
                })}

                <div className="border-t border-border my-4 pt-4" />
                {user ? (
                  <div className="space-y-1">
                    <div className="px-3 py-2">
                      <p className="text-body-xs text-muted uppercase tracking-wider">Account</p>
                    </div>
                    <button
                      onClick={logout}
                      className="flex items-center gap-3 w-full px-3 py-3 rounded-xl text-body-md font-medium text-error hover:bg-error/10 transition-colors"
                    >
                      <LogOut className="w-5 h-5" />
                      <span>Sign out</span>
                    </button>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <Link
                      href="/login"
                      className="btn-secondary btn-lg w-full justify-center"
                      onClick={() => setMobileOpen(false)}
                    >
                      Sign in
                    </Link>
                    <Link
                      href="/register"
                      className="btn-primary btn-lg w-full justify-center"
                      onClick={() => setMobileOpen(false)}
                    >
                      Create Account
                    </Link>
                  </div>
                )}
              </nav>

              <div className="p-4 border-t border-border">
                <p className="text-body-xs text-muted text-center">
                  DCASS v0.1.0 — Dynamic Context-Aware Semantic Steganography
                </p>
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  );
}

export default Navigation;