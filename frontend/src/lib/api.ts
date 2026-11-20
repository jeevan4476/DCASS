/**
 * DCASS API Client
 * 
 * Connects to the FastAPI backend running on localhost:8000
 */

import axios, { AxiosInstance } from 'axios';

// Convention: NEXT_PUBLIC_API_URL is the ORIGIN (no /api suffix).
// The /api prefix lives here so every consumer agrees.
const API_ORIGIN = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
export const API_BASE = `${API_ORIGIN.replace(/\/$/, '')}/api`;

export function getApiBase(): string {
  if (typeof window !== 'undefined') {
    // NOTE: there is deliberately NO ?remote= query-param or localStorage
    // override here. Letting a URL redirect API traffic to an arbitrary
    // origin would exfiltrate the Bearer JWT (and all traffic) to that host.
    const hostname = window.location.hostname;

    // If accessed via a public tunnel (e.g. ngrok, localtunnel, cloudflare)
    // or HTTPS origin with no port: use same-origin /api which Next.js proxies to backend!
    if (
      hostname.includes('ngrok') ||
      hostname.includes('loca.lt') ||
      hostname.includes('trycloudflare') ||
      (window.location.protocol === 'https:' && !window.location.port)
    ) {
      return `${window.location.origin}/api`;
    }

    // Auto-detect LAN host when loaded from mobile or another network device
    if (hostname && hostname !== 'localhost' && hostname !== '127.0.0.1') {
      const protocol = window.location.protocol || 'http:';
      return `${protocol}//${hostname}:8000/api`;
    }
  }
  return API_BASE;
}

// Phase F auth: localStorage-backed JWT. See docs/GAN_RL_INTEGRATION_PLAN.md
// Phase F.6 (R6) for the XSS caveat — fine for a research demo, not for
// a public deployment.
export const AUTH_TOKEN_KEY = 'dcass_token';

export function getErrorMessage(err: unknown, fallback: string = 'Operation failed'): string {
  if (!err) return fallback;
  const anyErr = err as any;
  const detail = anyErr?.response?.data?.detail;
  if (typeof detail === 'string') {
    return detail;
  }
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === 'string') return item;
        if (item && typeof item === 'object') {
          return item.msg || JSON.stringify(item);
        }
        return String(item);
      })
      .join(', ');
  }
  if (detail && typeof detail === 'object') {
    return detail.msg || JSON.stringify(detail);
  }
  if (anyErr?.message && typeof anyErr.message === 'string') {
    return anyErr.message;
  }
  return fallback;
}


export function getAuthToken(): string | null {
  if (typeof window === 'undefined') return null;
  try {
    return window.localStorage.getItem(AUTH_TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setAuthToken(token: string | null): void {
  if (typeof window === 'undefined') return;
  try {
    if (token) {
      window.localStorage.setItem(AUTH_TOKEN_KEY, token);
    } else {
      window.localStorage.removeItem(AUTH_TOKEN_KEY);
    }
  } catch {
    // ignore — private mode / storage disabled
  }
}

function _installAuthInterceptors(instance: AxiosInstance): AxiosInstance {
  instance.interceptors.request.use((config) => {
    config.baseURL = getApiBase();
    config.headers = config.headers ?? {};
    config.headers['ngrok-skip-browser-warning'] = 'true';
    const token = getAuthToken();
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  });
  instance.interceptors.response.use(
    (response) => response,
    (error) => {
      // On 401, clear the stale token so the auth context notices and
      // bounces the user back to /login. Only clear for AUTHENTICATED
      // 401s (we had a token) to avoid an infinite loop on login/register.
      // Skip entirely when already on /login or /register, or when the
      // failing request is itself a login/register call — a bad credential
      // attempt must not wipe the token or trigger a redirect loop.
      if (error?.response?.status === 401 && getAuthToken()) {
        const requestUrl: string = error?.config?.url ?? '';
        const onAuthPage =
          typeof window !== 'undefined' &&
          (window.location.pathname === '/login' || window.location.pathname === '/register');
        const isAuthEndpoint =
          requestUrl.endsWith('/auth/login') || requestUrl.endsWith('/auth/register');
        if (!onAuthPage && !isAuthEndpoint) {
          setAuthToken(null);
          if (typeof window !== 'undefined') {
            window.location.href = '/login';
          }
        }
      }
      return Promise.reject(error);
    },
  );
  return instance;
}

const api = _installAuthInterceptors(
  axios.create({
    baseURL: API_BASE,
    headers: { 'Content-Type': 'application/json' },
    timeout: 30000,
  }),
);

// Longer timeout for encode/decode operations (model loading on first request)
const apiLongTimeout = _installAuthInterceptors(
  axios.create({
    baseURL: API_BASE,
    headers: { 'Content-Type': 'application/json' },
    timeout: 90000,
  }),
);

// ============================================================================
// Types
// ============================================================================

export interface EncodeRequest {
  message: string;
  mode?: 'exact_vcp' | 'dssc';
  session_key_hex?: string;
  diversity_mode?: 'best' | 'round_robin' | 'balanced';
  modalities?: string[];
  use_ecc?: boolean;
  ecc_parity_bytes?: number;
}

export interface EncodeResponse {
  mode: string;
  media_ids: string[];
  carrier_count: number;
  chunks: string[];
  encoded: Array<{
    media_id: string;
    modality: string;
    score: number;
    content: string;
    file_path?: string;
    gdrive_path?: string;
    gdrive_url?: string;
    payload_byte?: number;
    cluster_id?: number;
  }>;
  media_sequence?: Array<{
    id: string;
    media_id: string;
    modality: string;
    content: string;
    score: number;
    file_path?: string;
    gdrive_path?: string;
    gdrive_url?: string;
  }>;
  modality_breakdown: Record<string, number>;
  elapsed_ms: number;
  bits_per_carrier?: number;
  ecc_parity_bytes?: number;
  payload_bytes?: number[];
  context_info?: Record<string, unknown>;
}

export interface DecodeRequest {
  media_ids: Array<string | null>;
  mode?: 'exact_vcp' | 'dssc';
  diversity_mode?: 'best' | 'round_robin' | 'balanced';
  session_key_hex?: string;
  modalities?: string[];
  use_ecc?: boolean;
  ecc_parity_bytes?: number;
  context_epoch_hint?: string;
  erasure_positions?: number[];
}

export interface DecodeResponse {
  mode: string;
  reconstructed_meaning: string;
  items: Array<{
    media_id: string;
    modality: string;
    content: string;
    file_path?: string;
    gdrive_path?: string;
    gdrive_url?: string;
    verified: boolean;
    payload_byte?: number | null;
    cluster_id?: number | null;
  }>;
  decoded?: Array<{
    media_id: string;
    modality: string;
    content: string;
    file_path?: string;
    gdrive_path?: string;
    gdrive_url?: string;
    verified: boolean;
    payload_byte?: number | null;
    cluster_id?: number | null;
  }>;
  verification_rate: number;
  all_verified: boolean;
  elapsed_ms: number;
  ecc_success?: boolean;
  ecc_errors_fixed?: number[];
  erasure_positions?: number[];
  payload_bytes?: number[];
  context_epoch_id?: string;
  cached?: boolean;
}

export interface SearchRequest {
  query: string;
  k?: number;
  modalities?: string[];
}

export interface SearchResponse {
  results: Array<{
    id: string;
    modality: string;
    score: number;
    content: string;
    file_path?: string;
  }>;
  elapsed_ms: number;
}

export interface StatusResponse {
  indices: Record<string, {
    status: string;
    count?: number;
    error?: string;
  }>;
  total_items: number;
  device: string;
  stealth_models: {
    gan_checkpoint: boolean;
    rl_checkpoint: boolean;
  };
}

// ============================================================================
// API Functions
// ============================================================================

export async function healthCheck(): Promise<{ status: string }> {
  const response = await api.get('/health');
  return response.data;
}

export async function checkReady(): Promise<{ ready: boolean; initializing: boolean; encoder_loaded: boolean; decoder_loaded: boolean }> {
  const response = await api.get('/ready');
  return response.data;
}

export async function encodeMessage(request: EncodeRequest): Promise<EncodeResponse> {
  const response = await apiLongTimeout.post('/encode', {
    use_ecc: true,
    mode: 'exact_vcp',
    ...request,
  });
  return response.data;
}

export async function decodeSequence(request: DecodeRequest): Promise<DecodeResponse> {
  const response = await apiLongTimeout.post('/decode', {
    use_ecc: true,
    mode: 'exact_vcp',
    ...request,
  });
  return response.data;
}

export async function searchCorpus(request: SearchRequest): Promise<SearchResponse> {
  const response = await api.post('/search', request);
  return response.data;
}

export async function getStatus(): Promise<StatusResponse> {
  const response = await api.get('/status');
  return response.data;
}

// ============================================================================
// Phase F — Auth, user routing, and inbox
// ============================================================================

export interface AuthUser {
  id: number;
  username: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: AuthUser;
}

export interface UsersListResponse {
  users: AuthUser[];
}

export interface InboxItem {
  session_id: string;
  sender_username?: string | null;
  sender_id?: number | null;
  recipient_username?: string | null;
  timestamp: number;
  mode_used?: string | null;
  total_items: number;
  received_packets: number;
  decoded: boolean;
}

export interface InboxResponse {
  items: InboxItem[];
}

export interface InboxDecodeResponse {
  session_id: string;
  sender_username?: string | null;
  recipient_username?: string | null;
  mode_used?: string | null;
  reconstructed_meaning: string;
  verification_rate: number;
  all_verified: boolean;
  ecc_success: boolean;
  ecc_errors_fixed: number[];
  items: Array<{
    media_id: string;
    modality: string;
    content: string;
    verified: boolean;
  }>;
  cached?: boolean;
  elapsed_ms?: number | null;
}

export interface TransmitRequest {
  media_ids: string[];
  recipient_username: string;
  mode?: 'static' | 'rl' | 'gan' | 'auto';
  engine_mode?: 'exact_vcp' | 'dssc';
  diversity_mode?: 'best' | 'round_robin' | 'balanced';
  session_key_hex?: string;
  base_delay?: number;
  num_channels?: number;
  message?: string;
  speed_multiplier?: number;
}

export interface TransmitResponse {
  success: boolean;
  status: string;
  session_id?: string;
  sender?: string;
  recipient?: string;
  total_packets: number;
  mode_used: string;
  estimated_duration_seconds: number;
  speed_multiplier: number;
  message: string;
}

export async function authLogin(username: string, password: string): Promise<LoginResponse> {
  const response = await api.post('/auth/login', { username, password });
  return response.data;
}

export async function authRegister(username: string, password: string): Promise<LoginResponse> {
  const response = await api.post('/auth/register', { username, password });
  return response.data;
}

export async function authMe(): Promise<AuthUser> {
  const response = await api.get('/auth/me');
  return response.data;
}

export async function listUsers(excludeCurrent: boolean = false): Promise<UsersListResponse> {
  const response = await api.get('/users', {
    params: excludeCurrent ? { exclude_current: true } : undefined,
  });
  return response.data;
}

export async function transmit(req: TransmitRequest): Promise<TransmitResponse> {
  const response = await apiLongTimeout.post('/transmit', req);
  return response.data;
}

export async function getInbox(): Promise<InboxResponse> {
  const response = await api.get('/inbox');
  return response.data;
}

export async function decodeInboxSession(sessionId: string): Promise<InboxDecodeResponse> {
  const response = await apiLongTimeout.post(`/inbox/${sessionId}/decode`);
  return response.data;
}

export async function deleteInboxSession(sessionId: string): Promise<{ success: boolean; deleted: number }> {
  const response = await api.delete(`/inbox/${sessionId}`);
  return response.data;
}

// ============================================================================
// Direct Messenger (WhatsApp / Telegram Stego Experience)
// ============================================================================

export interface ChatContact {
  id: number;
  username: string;
  last_message: string;
  last_timestamp: number;
  online: boolean;
}

export interface CarrierDetail {
  seq: number;
  media_id: string;
  modality: string;
  channel_id: number;
  channel_name: string;
  content: string;
  file_path: string;
  packet_file: string;
  verified: boolean;
}

export interface ChatMessage {
  id: string;
  sender: string;
  recipient: string;
  text: string;
  timestamp: number;
  is_mine: boolean;
  carrier_count: number;
  conduits: string[];
  carriers?: CarrierDetail[];
  status: 'sent' | 'delivered' | 'verified';
  verified: boolean;
}

export async function getChatContacts(): Promise<{ contacts: ChatContact[] }> {
  const response = await api.get('/chat/contacts');
  return response.data;
}

export async function getChatMessages(peer: string): Promise<{ messages: ChatMessage[]; peer: string }> {
  const response = await api.get('/chat/messages', { params: { peer } });
  return response.data;
}

export async function getChatCarriers(session_id: string): Promise<{
  session_id: string;
  carrier_count: number;
  conduits: string[];
  carriers: CarrierDetail[];
}> {
  const response = await api.get('/chat/carriers', { params: { session_id } });
  return response.data;
}

export async function sendChatMessage(recipient_username: string, message: string, mode: string = 'exact_vcp'): Promise<ChatMessage> {
  const response = await apiLongTimeout.post('/chat/send', { recipient_username, message, mode });
  return response.data;
}

// ============================================================================
// Wire / Telemetry
// ============================================================================

export interface WirePacket {
  media_id: string;
  channel_id: number;
  sequence_number: number;
  delay_seconds: number;
  timestamp: number;
  mode_used: string;
  filename: string;
  session_id?: string;
}

export interface WirePacketsResponse {
  packets: WirePacket[];
  count: number;
}

export interface TransmissionStatus {
  active: boolean;
  current: number;
  total: number;
  status: string;
  /** Every active session, keyed by session id. */
  sessions?: Record<string, { active: boolean; current: number; total: number; status: string }>;
  /** How many transmissions are running right now. */
  active_count?: number;
  /** The admission bound on concurrent transmissions. */
  max_concurrent?: number;
}

export async function getWirePackets(params?: { recipient_username?: string; session_id?: string }): Promise<WirePacketsResponse> {
  const response = await api.get('/wire/packets', { params });
  return response.data;
}

export async function clearWirePackets(): Promise<{ success: boolean; deleted: number }> {
  const response = await api.delete('/wire/packets');
  return response.data;
}

export async function getTransmissionStatus(): Promise<TransmissionStatus> {
  const response = await api.get('/transmit/status');
  return response.data;
}

export async function stopTransmission(sessionId?: string): Promise<{ success: boolean; stopped: number }> {
  const url = sessionId ? `/transmit/stop?session_id=${sessionId}` : '/transmit/stop';
  const response = await api.post(url);
  return response.data;
}

// ============================================================================
// Production Showcase: pipeline trace, carrier verification, live steganalysis
// ============================================================================

export interface PipelineTraceStage {
  stage_id: string;
  name: string;
  status: 'completed' | 'skipped' | 'degraded' | string;
  duration_ms: number;
  detail: Record<string, unknown>;
}

export interface PipelineTraceResponse {
  message_preview: string;
  stages: PipelineTraceStage[];
  mode: string;
  carrier_count: number;
  bits_per_carrier: number;
  media_ids: string[];
  engine_warm: boolean;
  total_duration_ms: number;
}

export interface VerifyCarrierResponse {
  media_id: string;
  modality?: string | null;
  catalog?: string | null;
  corpus_path: string;
  corpus_sha256: string;
  corpus_size_bytes: number;
  transmitted_sha256: string;
  transmitted_size_bytes: number;
  transmitted_found: boolean;
  transmitted_packet?: {
    path?: string;
    session_id?: string | null;
    channel?: number | null;
    modified?: number;
  } | null;
  bytes_modified: number;
  bytes_total_compared: number;
  integrity: 'verified' | 'modified' | string;
  comparison_basis?: string;
  modified_since_transmission?: boolean | null;
  verification_method?: string;
  note?: string;
}

export interface LiveSteganalysisResponse {
  detector: string;
  sample_count: number;
  clean_probability: number;
  lsb_probability: number;
  dcass_probability: number;
  per_sample: {
    clean: number[];
    lsb: number[];
    dcass: number[];
  };
  input_size: number;
  engine_trained: boolean;
  note: string;
}

export async function runPipelineTrace(request: {
  message: string;
  use_ecc?: boolean;
  ecc_parity_bytes?: number;
}): Promise<PipelineTraceResponse> {
  const response = await apiLongTimeout.post('/demo/pipeline-trace', request);
  return response.data;
}

export async function verifyCarrier(mediaId: string): Promise<VerifyCarrierResponse> {
  const response = await api.post('/demo/verify-carrier', { media_id: mediaId });
  return response.data;
}

export async function runLiveSteganalysis(request: {
  detector: 'sr_net' | 'zhu_net' | 'ye_net';
  sample_count?: number;
}): Promise<LiveSteganalysisResponse> {
  const response = await apiLongTimeout.post('/demo/live-steganalysis', request);
  return response.data;
}

// ============================================================================
// Live RS-ECC Erasure & Packet-Loss Lab
// ============================================================================

export interface ErasureCarrierInfo {
  seq: number;
  media_id: string;
  modality: string;
  content: string;
  file_path: string;
  role: 'header' | 'payload' | 'parity' | string;
  byte_indices: number[];
  bytes_hex: string[];
  bit_start: number;
  bit_end: number;
  char_preview?: string | null;
  bits?: number;
  family?: string;
}

export interface ErasureEncodeResponse {
  message: string;
  mode: 'exact_vcp' | 'dssc';
  diversity_mode: 'best' | 'round_robin' | 'balanced';
  ecc_parity_bytes: number;
  session_key_hex?: string | null;
  bits_per_carrier: number;
  carrier_count: number;
  header_bytes_len: number;
  message_bytes_len: number;
  framed_bytes_len: number;
  codeword_bytes_len: number;
  max_erasures: number;
  max_errors: number;
  codeword_hex: string[];
  media_ids: string[];
  carriers: ErasureCarrierInfo[];
  elapsed_ms: number;
}

export interface ErasureFaultDetail {
  seq: number;
  fault_type: 'dropped' | 'corrupted';
  original_media_id: string;
  received_media_id: string | null;
  byte_indices: number[];
  original_hex: string[];
  received_hex: string[];
  repaired_hex: string[] | null;
  recovered: boolean;
}

export interface ErasureTestResponse {
  mode: 'exact_vcp' | 'dssc';
  original_message: string;
  reconstructed_message: string;
  unprotected_preview: string;
  ecc_success: boolean;
  crc_verified: boolean;
  exact_match: boolean;
  bit_error_rate: number;
  total_packets: number;
  received_packet_count: number;
  dropped_packet_count: number;
  corrupted_packet_count: number;
  drop_indices: number[];
  corrupt_indices: number[];
  verification_rate: number;
  erased_byte_indices: number[];
  corrupted_byte_indices: number[];
  repaired_byte_indices: number[];
  budget_used: number;
  budget_max: number;
  within_capacity: boolean;
  fault_details: ErasureFaultDetail[];
  elapsed_ms: number;
}

export interface ErasureSweepPoint {
  dropped_packets: number;
  trials: number;
  recovered_exact: number;
  recovery_rate_pct: number;
  false_accepts: number;
  mean_erased_bytes: number;
}

export interface ErasureSweepResponse {
  mode: 'exact_vcp' | 'dssc';
  ecc_parity_bytes: number;
  total_packets: number;
  trials_per_level: number;
  points: ErasureSweepPoint[];
  elapsed_ms: number;
}

export async function erasureEncode(request: {
  message: string;
  mode?: 'exact_vcp' | 'dssc';
  diversity_mode?: 'best' | 'round_robin' | 'balanced';
  ecc_parity_bytes?: number;
  session_key_hex?: string;
}): Promise<ErasureEncodeResponse> {
  const response = await apiLongTimeout.post('/demo/erasure-encode', request);
  return response.data;
}

export async function erasureTest(request: {
  original_message: string;
  media_ids: string[];
  mode?: 'exact_vcp' | 'dssc';
  diversity_mode?: 'best' | 'round_robin' | 'balanced';
  ecc_parity_bytes?: number;
  session_key_hex?: string | null;
  drop_indices: number[];
  corrupt_indices: number[];
}): Promise<ErasureTestResponse> {
  const response = await apiLongTimeout.post('/demo/erasure-test', request);
  return response.data;
}

export async function erasureSweep(request: {
  original_message: string;
  media_ids: string[];
  mode?: 'exact_vcp' | 'dssc';
  diversity_mode?: 'best' | 'round_robin' | 'balanced';
  ecc_parity_bytes?: number;
  session_key_hex?: string | null;
  trials_per_level?: number;
}): Promise<ErasureSweepResponse> {
  const response = await apiLongTimeout.post('/demo/erasure-sweep', request);
  return response.data;
}

// ============================================================================
// Real On-Disk Wire Session API (Transmit, Drop/Unlink, Tamper, Decode)
// ============================================================================

export interface WireSlot {
  seq: number;
  status: 'on_disk' | 'deleted' | 'tampered';
  filename: string | null;
  rel_path: string | null;
  media_id: string | null;
  original_media_id?: string | null;
  modality: string;
  channel_id: number;
  delay_seconds: number;
  timestamp: number | null;
  role: 'header' | 'payload' | 'parity' | string;
  bytes_hex: string[];
  byte_indices?: number[];
}

export interface WireSessionSnapshot {
  session_id: string;
  sender_username: string;
  recipient_username: string;
  channel_dir: string;
  engine_mode: 'exact_vcp' | 'dssc';
  diversity_mode: 'best' | 'round_robin' | 'balanced';
  mode_used: string;
  ecc_parity_bytes: number;
  bits_per_carrier: number;
  total_items: number;
  surviving_count: number;
  dropped_count: number;
  tampered_count: number;
  dropped_sequences: number[];
  tampered_sequences: number[];
  erased_bytes_count?: number | null;
  tampered_bytes_count?: number | null;
  exact_fault_weight?: number | null;
  channel_counts: Record<string, number>;
  timestamp: number;
  decoded: boolean;
  last_decoded_message: string | null;
  packets: WireSlot[];
  elapsed_ms?: number;
  deleted_files?: string[];
  restored_files?: string[];
}

export interface WireSessionDecodeResult {
  session_id: string;
  channel_dir: string;
  engine_mode: 'exact_vcp' | 'dssc';
  ecc_parity_bytes: number;
  total_packets: number;
  surviving_packets: number;
  dropped_packets: number;
  tampered_packets: number;
  dropped_sequences: number[];
  tampered_sequences: number[];
  ecc_success: boolean;
  crc_verified: boolean;
  reconstructed_message: string;
  unprotected_preview: string;
  repaired_byte_indices: number[];
  verification_rate: number;
  elapsed_ms: number;
}

export async function wireSendSession(request: {
  message: string;
  recipient_username?: string;
  engine_mode?: 'exact_vcp' | 'dssc';
  diversity_mode?: 'best' | 'round_robin' | 'balanced';
  ecc_parity_bytes?: number;
  stealth_mode?: 'static' | 'rl' | 'gan' | 'auto';
}): Promise<WireSessionSnapshot> {
  const response = await apiLongTimeout.post('/wire/send', request);
  return response.data;
}

export async function getWireSessions(): Promise<{ sessions: WireSessionSnapshot[] }> {
  const response = await api.get('/wire/sessions');
  return response.data;
}

export async function getWireSession(sessionId: string): Promise<WireSessionSnapshot> {
  const response = await api.get(`/wire/sessions/${sessionId}`);
  return response.data;
}

export async function wireDropPacket(sessionId: string, seq: number): Promise<WireSessionSnapshot> {
  const response = await api.delete(`/wire/sessions/${sessionId}/packets/${seq}`);
  return response.data;
}

export async function wireDropBatch(sessionId: string, sequences: number[]): Promise<WireSessionSnapshot> {
  const response = await api.post(`/wire/sessions/${sessionId}/drop`, { sequences });
  return response.data;
}

export async function wireTamperPacket(sessionId: string, seq: number): Promise<WireSessionSnapshot> {
  const response = await apiLongTimeout.post(`/wire/sessions/${sessionId}/tamper/${seq}`);
  return response.data;
}

export async function wireRestorePackets(
  sessionId: string,
  sequences?: number[],
): Promise<WireSessionSnapshot> {
  const response = await api.post(
    `/wire/sessions/${sessionId}/restore`,
    sequences && sequences.length > 0 ? { sequences } : {},
  );
  return response.data;
}

export async function wireDecodeSession(sessionId: string): Promise<WireSessionDecodeResult> {
  const response = await apiLongTimeout.post(`/wire/sessions/${sessionId}/decode`);
  return response.data;
}

export async function wireDeleteSession(sessionId: string): Promise<{ success: boolean; deleted: number }> {
  const response = await api.delete(`/wire/sessions/${sessionId}`);
  return response.data;
}

export interface MediaInfoResponse {
  media_id: string;
  modality: string;
  content: string;
  file_path: string;
  exists_on_disk: boolean;
  size_bytes: number | null;
  sha256: string | null;
  bytes_modified: number;
  is_image: boolean;
  is_audio_playable: boolean;
  file_url: string;
  metadata: Record<string, string | number | boolean>;
}

export async function getMediaInfo(mediaId: string, path?: string): Promise<MediaInfoResponse> {
  const response = await api.get('/media/info', {
    params: { media_id: mediaId, ...(path ? { path } : {}) },
  });
  return response.data;
}

/**
 * Fetch a wire packet JSON through the authenticated API client (so the
 * Bearer token is attached and the server can authorize the caller) instead
 * of a raw fetch that would hit an unauthenticated endpoint.
 */
export async function getMediaPacket(path: string): Promise<unknown> {
  const response = await api.get('/media/packet', { params: { path } });
  return response.data;
}

export default api;
