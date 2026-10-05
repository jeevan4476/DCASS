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

// Phase F auth: localStorage-backed JWT. See docs/GAN_RL_INTEGRATION_PLAN.md
// Phase F.6 (R6) for the XSS caveat — fine for a research demo, not for
// a public deployment.
export const AUTH_TOKEN_KEY = 'dcass_token';

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
    const token = getAuthToken();
    if (token) {
      config.headers = config.headers ?? {};
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
      if (error?.response?.status === 401 && getAuthToken()) {
        setAuthToken(null);
        if (typeof window !== 'undefined' && window.location.pathname !== '/login') {
          window.location.href = '/login';
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
  media_ids: string[];
  mode?: 'exact_vcp' | 'dssc';
  session_key_hex?: string;
  modalities?: string[];
  use_ecc?: boolean;
  ecc_parity_bytes?: number;
  context_epoch_hint?: string;
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
  payload_bytes?: number[];
  context_epoch_id?: string;
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
  cached: boolean;
}

export interface TransmitRequest {
  media_ids: string[];
  recipient_username: string;
  mode?: 'static' | 'rl' | 'gan' | 'auto';
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

export async function listUsers(): Promise<UsersListResponse> {
  const response = await api.get('/users');
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

export default api;
