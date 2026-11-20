'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send,
  ShieldCheck,
  CheckCheck,
  ArrowLeft,
  X,
  ExternalLink,
  Layers,
  Search,
  ChevronRight,
  FileText,
  Image as ImageIcon,
  Volume2,
  FileCode,
  Sparkles,
  Copy,
  Check,
} from 'lucide-react';

import { useRequireAuth } from '@/lib/auth';
import {
  ChatContact,
  ChatMessage,
  CarrierDetail,
  getChatContacts,
  getChatMessages,
  getChatCarriers,
  sendChatMessage,
  getMediaPacket,
  getApiBase,
  getErrorMessage,
} from '@/lib/api';
import {
  Navigation,
  Badge,
  LoadingSpinner,
  CarrierPreviewModal,
  CarrierPreviewTarget,
  useToastHelpers,
} from '@/components';

const QUICK_PROMPTS = [
  'Classified directive: Proceed with multi-channel window at 0400Z.',
  'Operation Falcon is green. All channels reporting nominal.',
  'Meet at the north gate at 21:00 with the signed ledger.',
];

export default function ChatPage() {
  const { user, loading: authLoading } = useRequireAuth();
  const { error: showError } = useToastHelpers();
  const showErrorRef = useRef(showError);
  showErrorRef.current = showError;

  const [contacts, setContacts] = useState<ChatContact[]>([]);
  const [selectedPeer, setSelectedPeer] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState('');
  const [chatMode, setChatMode] = useState<'dssc' | 'exact_vcp'>('dssc');
  const [sending, setSending] = useState(false);
  const [loadingMessages, setLoadingMessages] = useState(false);

  // Carrier inspector modal
  const [inspectingMessage, setInspectingMessage] = useState<ChatMessage | null>(null);
  const [modalCarriers, setModalCarriers] = useState<CarrierDetail[]>([]);
  const [loadingCarriers, setLoadingCarriers] = useState(false);
  const [carrierFilter, setCarrierFilter] = useState<'all' | '0' | '1' | '2'>('all');
  const [carrierSearch, setCarrierSearch] = useState('');

  // Sub-modals
  const [selectedPacket, setSelectedPacket] = useState<{ path: string; data: unknown } | null>(null);
  const [previewTarget, setPreviewTarget] = useState<CarrierPreviewTarget | null>(null);
  const [copiedPacket, setCopiedPacket] = useState(false);

  const selectedPeerRef = useRef<string | null>(null);
  selectedPeerRef.current = selectedPeer;
  // Tracks which message the carrier inspector modal is currently showing,
  // so a slow fetch for an older message cannot overwrite a newer modal.
  const inspectingMessageIdRef = useRef<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = (smooth = true) => {
    messagesEndRef.current?.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto' });
  };

  const fetchContacts = useCallback(async () => {
    try {
      const resp = await getChatContacts();
      setContacts((prev) => {
        const prevKey = prev.map((c) => `${c.username}_${c.last_timestamp}_${c.last_message}`).join('|');
        const nextKey = resp.contacts.map((c) => `${c.username}_${c.last_timestamp}_${c.last_message}`).join('|');
        return prevKey !== nextKey ? resp.contacts : prev;
      });
      // Only auto-select the first contact when nothing is selected yet —
      // never yank an already-selected peer away during a 3s poll.
      if (resp.contacts.length > 0 && !selectedPeerRef.current) {
        setSelectedPeer(resp.contacts[0].username);
      }
    } catch {
      // silent poll
    }
  }, []);

  const fetchMessages = useCallback(async (peer: string, isInitial = false) => {
    if (!peer) return;
    if (isInitial) setLoadingMessages(true);
    try {
      const resp = await getChatMessages(peer);
      if (selectedPeerRef.current !== peer) return;
      setMessages((prev) => {
        const prevIds = prev.map((m) => m.id).join(',');
        const nextIds = resp.messages.map((m) => m.id).join(',');
        if (prevIds !== nextIds || isInitial) {
          setTimeout(() => scrollToBottom(!isInitial), 30);
          return resp.messages;
        }
        return prev;
      });
    } catch (err) {
      if (isInitial) showErrorRef.current('Chat Error', getErrorMessage(err, 'Could not load messages'));
    } finally {
      if (isInitial) setLoadingMessages(false);
    }
  }, []);

  useEffect(() => {
    if (!user) return;
    fetchContacts();
  }, [user, fetchContacts]);

  useEffect(() => {
    if (!user) return;
    if (selectedPeer) {
      fetchMessages(selectedPeer, true);
    } else {
      setMessages([]);
    }
  }, [user, selectedPeer, fetchMessages]);

  useEffect(() => {
    if (!user) return;
    const interval = setInterval(() => {
      if (selectedPeerRef.current) fetchMessages(selectedPeerRef.current, false);
      fetchContacts();
    }, 3000);
    return () => clearInterval(interval);
  }, [user, fetchMessages, fetchContacts]);

  const handleSend = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const peer = selectedPeerRef.current;
    if (!inputText.trim() || !peer || sending) return;
    const text = inputText.trim();
    setInputText('');
    setSending(true);
    try {
      const sent = await sendChatMessage(peer, text, chatMode);
      // Only append if the user is still viewing this peer — otherwise a
      // slow send that resolved after switching peers would inject the
      // message into the wrong conversation.
      if (selectedPeerRef.current === peer) {
        setMessages((prev) => [...prev, sent]);
        setTimeout(() => scrollToBottom(true), 40);
      }
      fetchContacts();
    } catch (err) {
      showErrorRef.current('Transmission Failed', getErrorMessage(err, 'Could not send message'));
      setInputText(text);
    } finally {
      setSending(false);
    }
  };

  const openCarrierInspector = async (msg: ChatMessage) => {
    setInspectingMessage(msg);
    inspectingMessageIdRef.current = msg.id;
    setCarrierFilter('all');
    setCarrierSearch('');
    if (msg.carriers && msg.carriers.length > 0) {
      setModalCarriers(msg.carriers);
      return;
    }
    setLoadingCarriers(true);
    try {
      const data = await getChatCarriers(msg.id);
      // Guard: a slow fetch for an earlier message must not overwrite the
      // modal state of the message now being inspected.
      if (inspectingMessageIdRef.current !== msg.id) return;
      setModalCarriers(data.carriers || []);
    } catch {
      if (inspectingMessageIdRef.current !== msg.id) return;
      setModalCarriers([]);
    } finally {
      if (inspectingMessageIdRef.current === msg.id) setLoadingCarriers(false);
    }
  };

  const openPacketJson = async (packetPath: string) => {
    try {
      const data = await getMediaPacket(packetPath);
      setSelectedPacket({ path: packetPath, data });
    } catch {
      showErrorRef.current('Packet Error', `Could not read ${packetPath}`);
    }
  };

  const formatTime = (ts: number) => {
    if (!ts) return '';
    return new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  if (authLoading || !user) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <div className="h-[100dvh] max-h-[100dvh] flex flex-col md:flex-row bg-background overflow-hidden">
      <Navigation />

      <div className="flex-1 min-w-0 min-h-0 flex">
        {/* Contacts Rail */}
        <section
          className={`w-full md:w-[290px] lg:w-[310px] shrink-0 border-r border-border bg-surface/70 flex flex-col min-h-0 ${
            selectedPeer ? 'hidden md:flex' : 'flex'
          }`}
        >
          <div className="px-5 py-4 border-b border-border flex items-center justify-between">
            <div>
              <h1 className="font-serif text-[1.25rem] font-medium text-foreground leading-tight">
                Messages
              </h1>
              <p className="text-[11px] font-mono text-muted mt-0.5">
                Signed in as @{user.username}
              </p>
            </div>
            <Badge variant="success" size="sm" dot>
              Stego E2E
            </Badge>
          </div>

          <div className="flex-1 overflow-y-auto divide-y divide-border/60">
            {contacts.map((c) => {
              const active = selectedPeer === c.username;
              return (
                <button
                  key={c.username}
                  type="button"
                  onClick={() => {
                    if (selectedPeer !== c.username) {
                      setMessages([]);
                      setSelectedPeer(c.username);
                    }
                  }}
                  className={`w-full px-4 py-3.5 text-left flex items-center gap-3 transition-colors ${
                    active ? 'bg-primary-muted/60 border-l-[3px] border-primary' : 'hover:bg-surface-muted/60 border-l-[3px] border-transparent'
                  }`}
                >
                  <div className="relative shrink-0">
                    <div className="w-10 h-10 rounded-full bg-surface-muted border border-border-strong flex items-center justify-center font-serif font-semibold text-foreground uppercase">
                      {c.username.slice(0, 2)}
                    </div>
                    {c.online && (
                      <span className="absolute bottom-0 right-0 w-2.5 h-2.5 rounded-full bg-success ring-2 ring-surface" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-semibold text-body-sm text-foreground capitalize truncate">
                        {c.username}
                      </span>
                      {c.last_timestamp > 0 && (
                        <span className="text-[10px] font-mono text-muted shrink-0">
                          {formatTime(c.last_timestamp)}
                        </span>
                      )}
                    </div>
                    <p className="text-body-xs text-muted truncate mt-0.5">{c.last_message}</p>
                  </div>
                </button>
              );
            })}
            {contacts.length === 0 && (
              <div className="p-8 text-center text-body-sm text-muted">
                No other contacts registered yet.
              </div>
            )}
          </div>
        </section>

        {/* Active Conversation Canvas */}
        <section
          className={`flex-1 min-w-0 min-h-0 flex flex-col bg-background ${
            !selectedPeer ? 'hidden md:flex items-center justify-center' : 'flex'
          }`}
        >
          {selectedPeer ? (
            <>
              {/* Conversation Header */}
              <div className="px-5 py-3.5 bg-surface border-b border-border flex items-center justify-between gap-3 shrink-0">
                <div className="flex items-center gap-3 min-w-0">
                  <button
                    type="button"
                    onClick={() => setSelectedPeer(null)}
                    className="md:hidden p-1.5 -ml-1 rounded-lg text-muted hover:text-foreground hover:bg-surface-muted"
                    aria-label="Back"
                  >
                    <ArrowLeft className="w-5 h-5" />
                  </button>
                  <div className="w-9 h-9 rounded-full bg-primary-muted border border-primary/25 flex items-center justify-center font-serif font-semibold text-primary uppercase shrink-0">
                    {selectedPeer.slice(0, 2)}
                  </div>
                  <div className="min-w-0">
                    <h2 className="font-serif text-[1.15rem] font-medium text-foreground capitalize leading-tight truncate">
                      {selectedPeer}
                    </h2>
                    <p className="text-[11px] font-mono text-muted flex items-center gap-1.5 truncate">
                      <ShieldCheck className="w-3.5 h-3.5 text-success shrink-0" />
                      <span>Zero-modification multimodal carriers &middot; RS-ECC protected</span>
                    </p>
                  </div>
                </div>

                {/* Codec Mode Pill Switch */}
                <div className="hidden sm:flex items-center gap-1 p-0.5 rounded-xl bg-surface-muted border border-border">
                  {(['dssc', 'exact_vcp'] as const).map((m) => (
                    <button
                      key={m}
                      type="button"
                      onClick={() => setChatMode(m)}
                      className={`px-2.5 py-1 rounded-lg text-[11px] font-mono transition-all ${
                        chatMode === m
                          ? 'bg-surface text-foreground font-semibold shadow-subtle'
                          : 'text-muted hover:text-foreground'
                      }`}
                    >
                      {m === 'dssc' ? 'DSSC' : 'Exact VCP'}
                    </button>
                  ))}
                </div>
              </div>

              {/* Message Thread */}
              <div className="flex-1 min-h-0 overflow-y-auto px-4 sm:px-8 py-6 space-y-4">
                {loadingMessages && messages.length === 0 ? (
                  <div className="h-full flex items-center justify-center">
                    <LoadingSpinner size="md" />
                  </div>
                ) : messages.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto">
                    <div className="w-12 h-12 rounded-2xl bg-primary-muted border border-primary/20 flex items-center justify-center text-primary mb-3">
                      <Sparkles className="w-5 h-5" />
                    </div>
                    <h3 className="font-serif text-heading-md font-medium text-foreground">
                      Start a covert thread with {selectedPeer}
                    </h3>
                    <p className="text-body-sm text-muted mt-1">
                      Every message is framed with CRC-16 + RS-ECC, mapped to unmodified media
                      files, and written across 3 wire conduits.
                    </p>
                  </div>
                ) : (
                  messages.map((m, idx) => {
                    const isMine = m.is_mine;
                    return (
                      <div
                        key={m.id || idx}
                        className={`flex flex-col ${isMine ? 'items-end' : 'items-start'}`}
                      >
                        <div
                          className={`max-w-[88%] sm:max-w-lg rounded-2xl px-4 py-3 border shadow-subtle ${
                            isMine
                              ? 'bg-[#F6E8DE] text-foreground border-primary/30 rounded-br-sm'
                              : 'bg-surface text-foreground border-border rounded-bl-sm'
                          }`}
                        >
                          <div
                            className={`text-[11px] font-semibold capitalize mb-1 ${
                              isMine ? 'text-primary' : 'text-info'
                            }`}
                          >
                            {isMine ? 'You' : m.sender}
                          </div>
                          <p className="text-body-sm whitespace-pre-wrap break-words leading-relaxed text-foreground">
                            {m.text}
                          </p>
                          <div
                            className={`flex flex-wrap items-center justify-between gap-2 mt-2.5 pt-2 border-t text-[11px] font-mono ${
                              isMine
                                ? 'border-primary/20 text-muted'
                                : 'border-border text-muted'
                            }`}
                          >
                            <div className="flex items-center gap-1.5">
                              <span>{formatTime(m.timestamp)}</span>
                              {isMine && <CheckCheck className="w-3.5 h-3.5 text-primary" />}
                            </div>
                            <div className="flex items-center gap-1.5">
                              <button
                                type="button"
                                onClick={() => openCarrierInspector(m)}
                                className={`px-2 py-0.5 rounded-lg inline-flex items-center gap-1 transition-colors border ${
                                  isMine
                                    ? 'bg-surface/85 hover:bg-surface text-foreground border-primary/20'
                                    : 'bg-surface-muted hover:bg-surface-hover text-foreground border-border'
                                }`}
                              >
                                <Layers className="w-3 h-3 text-primary" />
                                <span>{m.carrier_count} carriers</span>
                                <ChevronRight className="w-3 h-3 text-muted" />
                              </button>
                              <Link
                                href={`/erasure?session=${encodeURIComponent(m.id)}`}
                                className={`px-2 py-0.5 rounded-lg inline-flex items-center gap-1 transition-colors border ${
                                  isMine
                                    ? 'bg-primary text-white hover:bg-primary-hover border-primary'
                                    : 'bg-primary-muted hover:bg-primary/20 text-primary border-primary/25'
                                }`}
                                title="Physically drop packets of this message in RS-ECC Lab"
                              >
                                <ShieldCheck className="w-3 h-3" />
                                <span>Drop Test</span>
                              </Link>
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Prompt Suggestions + Claude-Style Floating Composer */}
              <div className="px-4 sm:px-8 pb-4 pt-2 bg-background border-t border-border/70 shrink-0 space-y-2.5">
                <div className="flex items-center gap-2 overflow-x-auto pb-0.5">
                  <span className="text-[11px] font-mono text-muted shrink-0">Quick:</span>
                  {QUICK_PROMPTS.map((p, i) => (
                    <button
                      key={i}
                      type="button"
                      onClick={() => setInputText(p)}
                      className="px-3 py-1 rounded-full bg-surface hover:bg-surface-muted border border-border text-body-xs text-muted hover:text-foreground shrink-0 truncate max-w-[240px] transition-colors"
                    >
                      {p}
                    </button>
                  ))}
                </div>

                <form
                  onSubmit={handleSend}
                  className="bg-surface border border-border-strong focus-within:border-primary focus-within:ring-2 focus-within:ring-primary/20 rounded-2xl p-2 flex items-center gap-2 shadow-card transition-all"
                >
                  <input
                    type="text"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder={`Write a covert message to @${selectedPeer}...`}
                    disabled={sending}
                    className="flex-1 bg-transparent px-3 py-1.5 text-body-sm text-foreground placeholder:text-muted/70 focus:outline-none"
                  />
                  <button
                    type="submit"
                    disabled={!inputText.trim() || sending}
                    className="btn-primary h-9 px-4 rounded-xl shrink-0"
                    aria-label="Send message"
                  >
                    {sending ? (
                      <LoadingSpinner size="sm" />
                    ) : (
                      <>
                        <Send className="w-3.5 h-3.5" />
                        <span className="hidden sm:inline">Send</span>
                      </>
                    )}
                  </button>
                </form>
              </div>
            </>
          ) : (
            <div className="text-center p-8 text-muted">
              <h3 className="font-serif text-heading-md text-foreground">Select a contact</h3>
              <p className="text-body-sm mt-1">Choose a peer on the left to start messaging.</p>
            </div>
          )}
        </section>
      </div>

      {/* Covert Carrier Inspector Modal */}
      <AnimatePresence>
        {inspectingMessage && (
          <div className="fixed inset-0 z-modal flex items-center justify-center p-4 bg-foreground/30 backdrop-blur-[2px]">
            <motion.div
              initial={{ opacity: 0, y: 12, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8, scale: 0.97 }}
              className="bg-surface border border-border rounded-2xl w-full max-w-4xl max-h-[88vh] flex flex-col shadow-modal overflow-hidden"
            >
              <div className="px-5 py-4 border-b border-border bg-surface-muted/60 flex items-center justify-between gap-3">
                <div>
                  <h3 className="font-serif text-heading-md font-medium text-foreground">
                    Covert Carrier Breakdown
                  </h3>
                  <p className="text-body-xs font-mono text-muted mt-0.5">
                    Session {inspectingMessage.id} &middot; {modalCarriers.length || inspectingMessage.carrier_count} real objects on disk
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <Link
                    href={`/erasure?session=${encodeURIComponent(inspectingMessage.id)}`}
                    className="btn-primary btn-sm"
                  >
                    <ShieldCheck className="w-3.5 h-3.5" />
                    Test Packet Drop
                  </Link>
                  <button
                    type="button"
                    onClick={() => {
                      setInspectingMessage(null);
                      inspectingMessageIdRef.current = null;
                    }}
                    className="p-1.5 rounded-lg text-muted hover:text-foreground hover:bg-surface-hover"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Filter & Search */}
              <div className="px-5 py-3 border-b border-border flex flex-wrap items-center justify-between gap-3 bg-surface">
                <div className="flex items-center gap-1.5 flex-wrap">
                  {[
                    { id: 'all', label: `All (${modalCarriers.length})` },
                    { id: '0', label: `Channel 1 (${modalCarriers.filter((c) => c.channel_id === 0).length})` },
                    { id: '1', label: `Channel 2 (${modalCarriers.filter((c) => c.channel_id === 1).length})` },
                    { id: '2', label: `Channel 3 (${modalCarriers.filter((c) => c.channel_id === 2).length})` },
                  ].map((t) => (
                    <button
                      key={t.id}
                      type="button"
                      onClick={() => setCarrierFilter(t.id as 'all' | '0' | '1' | '2')}
                      className={`px-3 py-1 rounded-lg text-body-xs font-medium border transition-colors ${
                        carrierFilter === t.id
                          ? 'bg-primary text-white border-primary'
                          : 'bg-surface-muted/60 text-muted border-border hover:text-foreground'
                      }`}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>
                <div className="relative w-full sm:w-56">
                  <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
                  <input
                    type="text"
                    value={carrierSearch}
                    onChange={(e) => setCarrierSearch(e.target.value)}
                    placeholder="Filter ID or content..."
                    className="input py-1.5 pl-8 text-body-xs"
                  />
                </div>
              </div>

              {/* Carrier Rows */}
              <div className="flex-1 overflow-y-auto p-4 space-y-2.5 bg-background">
                {loadingCarriers ? (
                  <div className="py-16 flex justify-center">
                    <LoadingSpinner size="md" />
                  </div>
                ) : (
                  modalCarriers
                    .filter((c) => carrierFilter === 'all' || String(c.channel_id) === carrierFilter)
                    .filter(
                      (c) =>
                        !carrierSearch ||
                        c.media_id.toLowerCase().includes(carrierSearch.toLowerCase()) ||
                        (c.content && c.content.toLowerCase().includes(carrierSearch.toLowerCase())),
                    )
                    .map((c) => (
                      <div
                        key={`${c.seq}_${c.media_id}`}
                        className="p-3.5 rounded-xl bg-surface border border-border flex flex-col gap-2"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex items-start gap-3 min-w-0">
                            <span className="font-mono text-body-xs font-semibold text-muted w-7 shrink-0 pt-1">
                              #{String(c.seq).padStart(2, '0')}
                            </span>

                            {c.modality === 'image' ? (
                              <button
                                type="button"
                                onClick={() =>
                                  setPreviewTarget({
                                    mediaId: c.media_id,
                                    filePath: c.file_path,
                                    packetPath: c.packet_file,
                                    seq: c.seq,
                                    channelId: c.channel_id,
                                  })
                                }
                                className="w-11 h-11 rounded-lg border border-border overflow-hidden shrink-0 bg-surface-muted flex items-center justify-center hover:border-primary"
                                title="View full image, path, SHA-256, and packet JSON"
                              >
                                {/* eslint-disable-next-line @next/next/no-img-element */}
                                <img
                                  src={`${getApiBase()}/media/file?media_id=${encodeURIComponent(c.media_id)}&path=${encodeURIComponent(c.file_path)}`}
                                  alt={c.media_id}
                                  className="w-full h-full object-cover"
                                  onError={(e) => {
                                    e.currentTarget.style.display = 'none';
                                  }}
                                />
                              </button>
                            ) : (
                              <button
                                type="button"
                                onClick={() =>
                                  setPreviewTarget({
                                    mediaId: c.media_id,
                                    filePath: c.file_path,
                                    packetPath: c.packet_file,
                                    seq: c.seq,
                                    channelId: c.channel_id,
                                  })
                                }
                                className="w-11 h-11 rounded-lg bg-surface-muted border border-border hover:border-primary flex items-center justify-center shrink-0 transition-colors"
                                title="View carrier asset, file path, SHA-256, and packet JSON"
                              >
                                {c.modality === 'audio' ? (
                                  <Volume2 className="w-4 h-4 text-purple" />
                                ) : (
                                  <FileText className="w-4 h-4 text-success" />
                                )}
                              </button>
                            )}

                            <div className="min-w-0">
                              <div className="flex items-center gap-2 flex-wrap">
                                <button
                                  type="button"
                                  onClick={() =>
                                    setPreviewTarget({
                                      mediaId: c.media_id,
                                      filePath: c.file_path,
                                      packetPath: c.packet_file,
                                      seq: c.seq,
                                      channelId: c.channel_id,
                                    })
                                  }
                                  className="font-mono text-body-xs font-semibold text-foreground hover:text-primary underline decoration-border hover:decoration-primary underline-offset-2 text-left"
                                >
                                  {c.media_id}
                                </button>
                                <Badge variant="info" size="sm">
                                  {c.channel_name || `Channel ${(c.channel_id % 3) + 1}`}
                                </Badge>
                              </div>
                              <p className="text-body-xs text-muted mt-0.5 line-clamp-1">
                                {c.content}
                              </p>
                            </div>
                          </div>

                          {c.packet_file && (
                            <button
                              type="button"
                              onClick={() => openPacketJson(c.packet_file)}
                              className="px-2.5 py-1 rounded-lg bg-surface-muted hover:bg-surface-hover border border-border text-[11px] font-mono text-foreground inline-flex items-center gap-1 shrink-0"
                            >
                              <FileCode className="w-3 h-3 text-primary" />
                              Packet JSON
                            </button>
                          )}
                        </div>
                      </div>
                    ))
                )}
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Packet JSON Sub-modal */}
      {selectedPacket && (
        <div className="fixed inset-0 z-[400] flex items-center justify-center p-4 bg-foreground/30 backdrop-blur-[2px]">
          <div className="bg-surface border border-border rounded-2xl max-w-lg w-full shadow-modal overflow-hidden">
            <div className="px-4 py-3 border-b border-border bg-surface-muted/60 flex items-center justify-between gap-2">
              <span className="font-mono text-xs text-foreground truncate">{selectedPacket.path}</span>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => {
                    navigator.clipboard.writeText(JSON.stringify(selectedPacket.data, null, 2));
                    setCopiedPacket(true);
                    setTimeout(() => setCopiedPacket(false), 1500);
                  }}
                  className="p-1.5 rounded-lg text-muted hover:text-foreground hover:bg-surface-hover"
                >
                  {copiedPacket ? <Check className="w-3.5 h-3.5 text-success" /> : <Copy className="w-3.5 h-3.5" />}
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedPacket(null)}
                  className="p-1.5 rounded-lg text-muted hover:text-foreground hover:bg-surface-hover"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>
            <pre className="p-4 font-mono text-xs overflow-auto max-h-[60vh] bg-background">
              {JSON.stringify(selectedPacket.data, null, 2)}
            </pre>
          </div>
        </div>
      )}

      {/* Unified Carrier Asset + Location + SHA-256 + Packet Modal */}
      <CarrierPreviewModal
        target={previewTarget}
        onClose={() => setPreviewTarget(null)}
      />
    </div>
  );
}
