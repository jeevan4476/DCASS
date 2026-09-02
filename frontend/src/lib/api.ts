/**
 * DCASS API Client
 * 
 * Connects to the FastAPI backend running on localhost:8000
 */

import axios from 'axios';

// Convention: NEXT_PUBLIC_API_URL is the ORIGIN (no /api suffix).
// The /api prefix lives here so every consumer agrees.
const API_ORIGIN = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
export const API_BASE = `${API_ORIGIN.replace(/\/$/, '')}/api`;

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second default timeout
});

// Longer timeout for encode/decode operations (model loading on first request)
const apiLongTimeout = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 90000, // 90 second timeout for heavy operations
});

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

export default api;
