/**
 * DCASS API Client
 * 
 * Connects to the FastAPI backend running on localhost:8000
 */

import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

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
  mode?: 'best' | 'round_robin' | 'balanced';
  modalities?: string[];
}

export interface EncodeResponse {
  media_ids: string[];
  chunks: string[];
  encoded: Array<{
    media_id: string;
    modality: string;
    score: number;
    content: string;
  }>;
  modality_breakdown: Record<string, number>;
  elapsed_ms: number;
}

export interface DecodeRequest {
  media_ids: string[];
}

export interface DecodeResponse {
  reconstructed_meaning: string;
  items: Array<{
    media_id: string;
    modality: string;
    content: string;
    verified: boolean;
  }>;
  verification_rate: number;
  all_verified: boolean;
  elapsed_ms: number;
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
  const response = await apiLongTimeout.post('/encode', request);
  return response.data;
}

export async function decodeSequence(request: DecodeRequest): Promise<DecodeResponse> {
  const response = await apiLongTimeout.post('/decode', request);
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

export async function getLatestBenchmark(): Promise<any> {
  const response = await api.get('/benchmark/latest');
  return response.data;
}

export default api;
