#!/usr/bin/env node
/**
 * Vertex AI → OpenAI Compatible Proxy (v9 - Streaming Fix + Per-Model Rate Limit)
 *
 * Fixes:
 *  - Proper streaming: stream responses are piped directly, not buffered
 *  - Per-model rate limit tracking: remembers which models are 429-ing
 *  - Better error detection: catches non-JSON in stream bodies
 *  - DeepSeek uses publisher endpoint (not openapi) for better compatibility
 */
const http = require('http');
const https = require('https');
const { URL } = require('url');
const crypto = require('crypto');
const { exec } = require('child_process');

// ─── Config ────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
const DEBUG = process.env.DEBUG === '1';

const API_KEY = process.env.GOOGLE_API_KEY || '';
const ACCESS_TOKEN = process.env.GOOGLE_ACCESS_TOKEN || '';
const REFRESH_TOKEN = process.env.GOOGLE_REFRESH_TOKEN || '';
const OAUTH_CLIENT_ID = process.env.GOOGLE_CLIENT_ID || '';
const OAUTH_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET || '';
const SERVICE_ACCOUNT_JSON = process.env.GOOGLE_SERVICE_ACCOUNT_JSON || '';

const OAUTH_REDIRECT_PORT = parseInt(process.env.OAUTH_REDIRECT_PORT || '8085', 10);
const PROJECT_ID = process.env.GOOGLE_PROJECT_ID || '';
const LOCATION = process.env.GOOGLE_LOCATION || 'global';
const ENDPOINT = process.env.ENDPOINT || 'aiplatform.googleapis.com';
const DEFAULT_MODEL = process.env.DEFAULT_MODEL || 'gemini-3.1-flash-lite';

const MAX_CONCURRENT = parseInt(process.env.MAX_CONCURRENT || '3', 10);
const MAX_RETRIES = parseInt(process.env.MAX_RETRIES || '5', 10);
const BASE_DELAY_MS = parseInt(process.env.BASE_DELAY_MS || '1000', 10);
const MAX_DELAY_MS = parseInt(process.env.MAX_DELAY_MS || '32000', 10);
const CIRCUIT_THRESHOLD = parseInt(process.env.CIRCUIT_THRESHOLD || '10', 10);
const CIRCUIT_RECOVERY_MS = parseInt(process.env.CIRCUIT_RECOVERY_MS || '60000', 10);
const REQUEST_TIMEOUT_MS = parseInt(process.env.REQUEST_TIMEOUT_MS || '120000', 10);

// ─── Model Registry ────────────────────────────────────────────
const MODEL_REGISTRY = {
  'gemini-3.1-flash-lite': { provider: 'google', endpoint: 'openai', location: 'global' },
  'gemini-3.1-pro-preview': { provider: 'google', endpoint: 'openai', location: 'global' },
  'moonshotai/kimi-k2-thinking-maas': { provider: 'moonshot', endpoint: 'openai', location: 'global' },
  'deepseek-ai/deepseek-v3.2-maas': { provider: 'deepseek', endpoint: 'publisher', location: 'global' },
  'deepseek-ai/deepseek-r1-0528-maas': { provider: 'deepseek', endpoint: 'publisher', location: 'us-central1' },
  'claude-haiku-4-5': { provider: 'anthropic', endpoint: 'anthropic', location: 'global' },
  'claude-sonnet-4-6': { provider: 'anthropic', endpoint: 'anthropic', location: 'global' },
};

const MODEL_ALIASES = {
  'gpt-4o-mini': 'gemini-3.1-flash-lite',
  'gpt-4o': 'gemini-3.1-pro-preview',
};

const AVAILABLE_MODELS = Object.keys(MODEL_REGISTRY);

if (!API_KEY && !ACCESS_TOKEN && !REFRESH_TOKEN && !SERVICE_ACCOUNT_JSON) {
  console.error('[FATAL] No auth configured');
  process.exit(1);
}

function log(...args) {
  if (DEBUG) console.error('[DEBUG]', new Date().toISOString(), ...args);
}

// ─── Service Account ───────────────────────────────────────────
let serviceAccount = null;
let cachedToken = { token: null, expiry: 0 };

if (SERVICE_ACCOUNT_JSON) {
  try {
    serviceAccount = JSON.parse(SERVICE_ACCOUNT_JSON);
    if (!serviceAccount.private_key || serviceAccount.private_key.trim() === '') {
      console.error('[WARN] Empty private_key - service account disabled');
      serviceAccount = null;
    } else {
      log('[AUTH]', 'Service account:', serviceAccount.client_email);
    }
  } catch (e) {
    console.error('[FATAL] Invalid GOOGLE_SERVICE_ACCOUNT_JSON:', e.message);
    process.exit(1);
  }
}

function base64UrlEncode(str) {
  return Buffer.from(str).toString('base64')
    .replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function signJWT(header, payload, privateKey) {
  const signer = crypto.createSign('RSA-SHA256');
  signer.update(`${header}.${payload}`);
  return base64UrlEncode(signer.sign(privateKey));
}

async function getOAuthTokenFromServiceAccount() {
  if (!serviceAccount) return null;
  const now = Math.floor(Date.now() / 1000);
  if (cachedToken.token && cachedToken.expiry > now + 60) return cachedToken.token;

  const header = base64UrlEncode(JSON.stringify({ alg: 'RS256', typ: 'JWT' }));
  const claim = base64UrlEncode(JSON.stringify({
    iss: serviceAccount.client_email,
    scope: 'https://www.googleapis.com/auth/cloud-platform',
    aud: 'https://oauth2.googleapis.com/token',
    iat: now,
    exp: now + 3600
  }));
  const signature = signJWT(header, claim, serviceAccount.private_key);
  const jwt = `${header}.${claim}.${signature}`;
  const postData = `grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer&assertion=${encodeURIComponent(jwt)}`;

  return new Promise((resolve, reject) => {
    const req = https.request({
      hostname: 'oauth2.googleapis.com', path: '/token', method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded', 'Content-Length': Buffer.byteLength(postData) }
    }, (res) => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        try {
          const json = JSON.parse(data);
          if (json.access_token) {
            cachedToken = { token: json.access_token, expiry: now + (json.expires_in || 3600) };
            log('[AUTH]', 'Service account token refreshed');
            resolve(json.access_token);
          } else {
            reject(new Error(`OAuth error: ${data}`));
          }
        } catch (e) { reject(new Error(`OAuth parse error: ${data}`)); }
      });
    });
    req.on('error', reject);
    req.write(postData);
    req.end();
  });
}

async function refreshAccessToken() {
  if (!REFRESH_TOKEN || !OAUTH_CLIENT_ID || !OAUTH_CLIENT_SECRET) {
    throw new Error('Missing refresh token or OAuth client credentials');
  }
  const postData = new URLSearchParams({
    grant_type: 'refresh_token',
    refresh_token: REFRESH_TOKEN,
    client_id: OAUTH_CLIENT_ID,
    client_secret: OAUTH_CLIENT_SECRET
  }).toString();

  return new Promise((resolve, reject) => {
    const req = https.request({
      hostname: 'oauth2.googleapis.com', path: '/token', method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded', 'Content-Length': Buffer.byteLength(postData) }
    }, (res) => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        try {
          const json = JSON.parse(data);
          if (json.access_token) {
            cachedToken = { token: json.access_token, expiry: Math.floor(Date.now() / 1000) + (json.expires_in || 3600) };
            log('[AUTH]', 'Refresh token exchanged');
            resolve(json.access_token);
          } else {
            reject(new Error(`Token refresh failed: ${data}`));
          }
        } catch (e) { reject(new Error(`Token refresh parse error: ${data}`)); }
      });
    });
    req.on('error', reject);
    req.write(postData);
    req.end();
  });
}

async function getAuthToken() {
  if (serviceAccount) {
    try { const token = await getOAuthTokenFromServiceAccount(); if (token) return token; }
    catch (err) { log('[AUTH]', 'Service account failed:', err.message); }
  }
  if (REFRESH_TOKEN && OAUTH_CLIENT_ID && OAUTH_CLIENT_SECRET) {
    try {
      const now = Math.floor(Date.now() / 1000);
      if (cachedToken.token && cachedToken.expiry > now + 60) {
        log('[AUTH]', 'Cached access token');
        return cachedToken.token;
      }
      return await refreshAccessToken();
    } catch (err) { log('[AUTH]', 'Refresh token failed:', err.message); }
  }
  if (ACCESS_TOKEN) { log('[AUTH]', 'Static access token'); return ACCESS_TOKEN; }
  return null;
}

// ─── Helpers ───────────────────────────────────────────────────
function resolveModel(requestedModel) {
  const alias = MODEL_ALIASES[requestedModel];
  if (alias) return alias;
  if (MODEL_REGISTRY[requestedModel]) return requestedModel;
  return DEFAULT_MODEL;
}

function getModelInfo(model) {
  return MODEL_REGISTRY[model] || MODEL_REGISTRY[DEFAULT_MODEL] || {
    provider: 'google', endpoint: 'openai', location: 'global'
  };
}

function generateId(prefix = 'chatcmpl') {
  return `${prefix}-${Date.now().toString(16)}${Math.random().toString(36).substring(2, 10)}`;
}

function getVertexUrl(model, info) {
  const apiVersion = info.endpoint === 'anthropic' ? 'v1' : 'v1beta1';
  const endpoint = info.location === 'us-central1' ? 'us-central1-aiplatform.googleapis.com' : ENDPOINT;

  if (info.endpoint === 'openai') {
    return `https://${endpoint}/${apiVersion}/projects/${PROJECT_ID}/locations/${info.location}/endpoints/openapi/chat/completions`;
  }
  if (info.endpoint === 'publisher') {
    // DeepSeek uses publisher endpoint with model in URL
    return `https://${endpoint}/${apiVersion}/projects/${PROJECT_ID}/locations/${info.location}/publishers/deepseek/models/${model}:streamRawPredict`;
  }
  if (info.endpoint === 'anthropic') {
    return `https://${endpoint}/${apiVersion}/projects/${PROJECT_ID}/locations/${info.location}/publishers/anthropic/models/${model}:streamRawPredict`;
  }
  // Legacy Gemini
  if (PROJECT_ID) {
    return `https://${info.location}-aiplatform.googleapis.com/${apiVersion}/projects/${PROJECT_ID}/locations/${info.location}/publishers/google/models/${model}:streamGenerateContent?alt=sse`;
  }
  return `https://aiplatform.googleapis.com/${apiVersion}/publishers/google/models/${model}:streamGenerateContent?alt=sse&key=${API_KEY}`;
}

// ─── Per-Model Rate Limit Tracker ──────────────────────────────
const modelRateLimits = new Map();

function isModelRateLimited(model) {
  const entry = modelRateLimits.get(model);
  if (!entry) return false;
  if (Date.now() > entry.resetAt) {
    modelRateLimits.delete(model);
    return false;
  }
  return true;
}

function recordModelRateLimit(model, retryAfterMs = 30000) {
  modelRateLimits.set(model, { resetAt: Date.now() + retryAfterMs });
  log('[RATE-LIMIT]', model, 'blocked for', retryAfterMs, 'ms');
}

// ─── Circuit Breaker ───────────────────────────────────────────
class CircuitBreaker {
  constructor(threshold, recoveryMs) {
    this.threshold = threshold; this.recoveryMs = recoveryMs;
    this.failures = 0; this.lastFailure = 0; this.state = 'CLOSED';
  }
  recordSuccess() { this.failures = 0; this.state = 'CLOSED'; log('[CIRCUIT]', 'CLOSED'); }
  recordFailure() {
    this.failures++; this.lastFailure = Date.now();
    if (this.failures >= this.threshold) { this.state = 'OPEN'; log('[CIRCUIT]', 'OPENED', this.failures); }
  }
  canExecute() {
    if (this.state === 'CLOSED') return true;
    if (this.state === 'OPEN') {
      if (Date.now() - this.lastFailure >= this.recoveryMs) {
        this.state = 'HALF_OPEN'; this.failures = 0; log('[CIRCUIT]', 'HALF_OPEN'); return true;
      }
      return false;
    }
    return true;
  }
}
const circuitBreaker = new CircuitBreaker(CIRCUIT_THRESHOLD, CIRCUIT_RECOVERY_MS);

// ─── Request Queue ─────────────────────────────────────────────
class RequestQueue {
  constructor(maxConcurrent) { this.maxConcurrent = maxConcurrent; this.running = 0; this.queue = []; }
  async enqueue(fn) {
    return new Promise((resolve, reject) => { this.queue.push({ fn, resolve, reject }); this.process(); });
  }
  async process() {
    if (this.running >= this.maxConcurrent || this.queue.length === 0) return;
    this.running++;
    const { fn, resolve, reject } = this.queue.shift();
    try { resolve(await fn()); } catch (err) { reject(err); }
    finally { this.running--; setImmediate(() => this.process()); }
  }
}
const requestQueue = new RequestQueue(MAX_CONCURRENT);

// ─── Retry ─────────────────────────────────────────────────────
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function jitteredDelay(attempt) {
  const exp = Math.min(Math.pow(2, attempt) * BASE_DELAY_MS, MAX_DELAY_MS);
  return Math.floor(exp + Math.random() * exp * 0.3);
}
async function retryWithBackoff(fn, context, model) {
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    // Check per-model rate limit before attempting
    if (isModelRateLimited(model)) {
      const entry = modelRateLimits.get(model);
      const waitMs = entry.resetAt - Date.now();
      log('[RATE-LIMIT-WAIT]', model, 'waiting', waitMs, 'ms');
      await sleep(waitMs);
    }

    try {
      const result = await fn();
      if (attempt > 0) log('[RETRY-OK]', context, attempt);
      circuitBreaker.recordSuccess();
      return result;
    } catch (err) {
      const is429 = err.statusCode === 429;
      const is5xx = err.statusCode >= 500 && err.statusCode < 600;
      const isRetryable = is429 || is5xx || err.code === 'ECONNRESET' || err.code === 'ETIMEDOUT';

      if (is429) {
        // Extract retry-after if available
        const retryAfter = err.retryAfter || 30000;
        recordModelRateLimit(model, retryAfter);
      }

      if (!isRetryable || attempt >= MAX_RETRIES) { circuitBreaker.recordFailure(); throw err; }
      const delay = jitteredDelay(attempt);
      log('[RETRY]', context, attempt + 1 + '/' + MAX_RETRIES, delay + 'ms', err.statusCode || err.code);
      await sleep(delay);
    }
  }
  throw new Error('Max retries exceeded');
}

// ─── Safe JSON Parse ───────────────────────────────────────────
function safeJsonParse(str, ctx) {
  try { return JSON.parse(str); }
  catch (e) {
    // Don't log huge error HTML pages
    const preview = str.substring(0, 100).replace(/\s+/g, ' ');
    log('[JSON-FAIL]', ctx, preview, '...');
    throw new Error(`JSON parse failed (${ctx}): ${e.message}`);
  }
}

// ─── SSE Parser ────────────────────────────────────────────────
class SseParser {
  constructor() { this.buffer = ''; }
  feed(text) {
    this.buffer += text;
    const events = [];
    const parts = this.buffer.split(/(?:\r?\n){2,}/);
    this.buffer = parts.pop() || '';
    for (const part of parts) {
      if (!part.trim()) continue;
      const lines = part.split(/\r?\n/);
      const dataLines = [];
      for (const line of lines) { if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart()); }
      if (dataLines.length) events.push(dataLines.join('\n'));
    }
    return events;
  }
  flush() {
    const events = [];
    if (this.buffer.trim()) {
      const lines = this.buffer.split(/\r?\n/);
      const dataLines = [];
      for (const line of lines) { if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart()); }
      if (dataLines.length) events.push(dataLines.join('\n'));
    }
    this.buffer = '';
    return events;
  }
}

// ─── Request Builders ──────────────────────────────────────────

function buildGeminiBody(openaiBody) {
  const contents = [];
  let systemInstruction = null;
  const messages = openaiBody.messages || [];
  let i = 0;

  while (i < messages.length) {
    const msg = messages[i];
    if (msg.role === 'system' || msg.role === 'developer') {
      if (typeof msg.content === 'string') systemInstruction = { parts: [{ text: msg.content }] };
      i++; continue;
    }
    if (msg.role === 'tool' || msg.role === 'function') {
      const toolGroup = [];
      while (i < messages.length && (messages[i].role === 'tool' || messages[i].role === 'function')) {
        toolGroup.push(messages[i]); i++;
      }
      const parts = toolGroup.map(t => {
        let responseData;
        if (typeof t.content === 'string') {
          try { responseData = JSON.parse(t.content); } catch { responseData = { result: t.content }; }
        } else if (t.content !== null && typeof t.content === 'object') {
          responseData = t.content;
        } else {
          responseData = { result: String(t.content ?? '') };
        }
        return { functionResponse: { name: t.name || t.function?.name || 'unknown', response: responseData } };
      });
      contents.push({ role: 'user', parts });
      continue;
    }
    const role = msg.role === 'assistant' ? 'model' : 'user';
    let parts = [];
    if (typeof msg.content === 'string') {
      parts.push({ text: msg.content });
    } else if (Array.isArray(msg.content)) {
      for (const item of msg.content) {
        if (item.type === 'text') parts.push({ text: item.text });
        else if (item.type === 'image_url') {
          const imageUrl = item.image_url?.url || '';
          if (imageUrl.startsWith('data:')) {
            const m = imageUrl.match(/^data:([^;]+);base64,(.+)$/);
            if (m) parts.push({ inlineData: { mimeType: m[1], data: m[2] } });
          } else {
            parts.push({ fileData: { mimeType: 'image/jpeg', fileUri: imageUrl } });
          }
        }
      }
    }
    if (msg.tool_calls && msg.tool_calls.length > 0) {
      for (const tc of msg.tool_calls) {
        let args = {};
        if (tc.function?.arguments) {
          if (typeof tc.function.arguments === 'string') {
            try { args = JSON.parse(tc.function.arguments); } catch { args = {}; }
          } else if (typeof tc.function.arguments === 'object') {
            args = tc.function.arguments;
          }
        }
        const fcPart = { functionCall: { name: tc.function?.name || 'unknown', args } };
        const sig = tc.extra_content?.google?.thought_signature || tc.thoughtSignature || tc.thought_signature;
        if (sig) fcPart.thoughtSignature = sig;
        parts.push(fcPart);
      }
    }
    if (msg.function_call) {
      let args = {};
      if (typeof msg.function_call.arguments === 'string') {
        try { args = JSON.parse(msg.function_call.arguments); } catch { args = {}; }
      } else if (typeof msg.function_call.arguments === 'object') {
        args = msg.function_call.arguments;
      }
      parts.push({ functionCall: { name: msg.function_call.name, args } });
    }
    if (parts.length > 0) contents.push({ role, parts });
    i++;
  }

  const generationConfig = {
    temperature: openaiBody.temperature ?? 1,
    maxOutputTokens: openaiBody.max_tokens || 65535,
    topP: openaiBody.top_p ?? 0.95,
    topK: openaiBody.top_k ?? 40,
    candidateCount: openaiBody.n || 1,
    stopSequences: openaiBody.stop ? (Array.isArray(openaiBody.stop) ? openaiBody.stop : [openaiBody.stop]) : undefined
  };
  Object.keys(generationConfig).forEach(k => { if (generationConfig[k] === undefined) delete generationConfig[k]; });

  let tools = null;
  let toolConfig = null;
  if (openaiBody.tools) {
    tools = openaiBody.tools.map(t => ({
      functionDeclarations: [{ name: t.function.name, description: t.function.description, parameters: t.function.parameters }]
    }));
  } else if (openaiBody.functions) {
    tools = openaiBody.functions.map(f => ({
      functionDeclarations: [{ name: f.name, description: f.description, parameters: f.parameters }]
    }));
  }

  if (openaiBody.tool_choice === 'none') {
    toolConfig = { functionCallingConfig: { mode: 'NONE' } };
  } else if (openaiBody.tool_choice === 'auto' || openaiBody.tool_choice == null) {
    toolConfig = { functionCallingConfig: { mode: 'AUTO' } };
  } else if (typeof openaiBody.tool_choice === 'object' && openaiBody.tool_choice?.function?.name) {
    toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [openaiBody.tool_choice.function.name] } };
  }

  const vertexBody = {
    contents,
    generationConfig,
    safetySettings: [
      { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'OFF' },
      { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'OFF' },
      { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'OFF' },
      { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'OFF' }
    ]
  };
  if (systemInstruction) vertexBody.systemInstruction = systemInstruction;
  if (tools) vertexBody.tools = tools;
  if (toolConfig) vertexBody.toolConfig = toolConfig;

  return vertexBody;
}

function buildAnthropicBody(openaiBody) {
  const messages = [];
  let system = null;
  for (const msg of openaiBody.messages || []) {
    if (msg.role === 'system' || msg.role === 'developer') {
      if (typeof msg.content === 'string') system = msg.content;
      continue;
    }
    if (msg.role === 'tool' || msg.role === 'function') {
      messages.push({
        role: 'user',
        content: [{
          type: 'tool_result',
          tool_use_id: msg.tool_call_id || 'unknown',
          content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
        }]
      });
      continue;
    }
    if (msg.role === 'assistant' && msg.tool_calls) {
      const content = [];
      if (msg.content) content.push({ type: 'text', text: msg.content });
      for (const tc of msg.tool_calls) {
        content.push({
          type: 'tool_use',
          id: tc.id,
          name: tc.function.name,
          input: typeof tc.function.arguments === 'string' ? JSON.parse(tc.function.arguments) : tc.function.arguments
        });
      }
      messages.push({ role: 'assistant', content });
      continue;
    }
    if (typeof msg.content === 'string') {
      messages.push({ role: msg.role, content: msg.content });
    } else if (Array.isArray(msg.content)) {
      const anthropicContent = [];
      for (const item of msg.content) {
        if (item.type === 'text') anthropicContent.push({ type: 'text', text: item.text });
        else if (item.type === 'image_url') {
          const imageUrl = item.image_url?.url || '';
          if (imageUrl.startsWith('data:')) {
            const m = imageUrl.match(/^data:([^;]+);base64,(.+)$/);
            if (m) anthropicContent.push({ type: 'image', source: { type: 'base64', media_type: m[1], data: m[2] } });
          }
        }
      }
      messages.push({ role: msg.role, content: anthropicContent });
    }
  }

  const body = {
    anthropic_version: 'vertex-2023-10-16',
    stream: openaiBody.stream !== false,
    max_tokens: openaiBody.max_tokens || 512,
    temperature: openaiBody.temperature ?? 1,
    messages: messages
  };
  if (system) body.system = system;
  if (openaiBody.top_p != null) body.top_p = openaiBody.top_p;
  if (openaiBody.stop) body.stop_sequences = Array.isArray(openaiBody.stop) ? openaiBody.stop : [openaiBody.stop];
  if (openaiBody.tools) {
    body.tools = openaiBody.tools.map(t => ({
      name: t.function.name,
      description: t.function.description,
      input_schema: t.function.parameters
    }));
  }
  return body;
}

function buildDeepSeekBody(openaiBody) {
  // DeepSeek publisher endpoint expects OpenAI format but with model in URL
  // So we pass the body through but remove the model field (it's in the URL)
  const body = { ...openaiBody };
  delete body.model;
  return body;
}

// ─── Response Translators ──────────────────────────────────────

function mapGeminiFinishReason(vertexFinishReason, hasToolCalls) {
  if (!vertexFinishReason) return null;
  const map = {
    'STOP': hasToolCalls ? 'tool_calls' : 'stop',
    'MAX_TOKENS': 'length',
    'SAFETY': 'content_filter',
    'RECITATION': 'content_filter',
    'OTHER': 'stop'
  };
  return map[vertexFinishReason] || 'stop';
}

function geminiToOpenAIStreamingChunk(vertexChunk, id, model, isFirst) {
  const candidate = vertexChunk.candidates?.[0];
  if (!candidate) return null;
  const parts = candidate.content?.parts || [];
  let delta = {};
  let hasToolCalls = false;
  if (isFirst) delta.role = 'assistant';
  for (const part of parts) {
    if (part.text != null) delta.content = part.text;
    if (part.functionCall) {
      hasToolCalls = true;
      const fc = part.functionCall;
      const tc = {
        index: 0,
        id: `call_${Date.now().toString(36)}${Math.random().toString(36).substring(2, 6)}`,
        type: 'function',
        function: { name: fc.name, arguments: JSON.stringify(fc.args || {}) }
      };
      if (part.thoughtSignature || part.thought_signature) {
        tc.extra_content = { google: { thought_signature: part.thoughtSignature || part.thought_signature } };
      }
      delta.tool_calls = [tc];
    }
  }
  return {
    id: `chatcmpl-${id}`,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model: model,
    choices: [{ index: 0, delta: delta, finish_reason: mapGeminiFinishReason(candidate.finishReason, hasToolCalls), logprobs: null }]
  };
}

function geminiToOpenAINonStreaming(events, id, model) {
  let fullText = '';
  const toolCalls = [];
  let finishReason = 'stop';
  let hasToolCalls = false;
  for (const event of events) {
    if (event === '[DONE]') continue;
    try {
      const chunk = safeJsonParse(event, 'gemini-non-stream');
      const candidate = chunk.candidates?.[0];
      if (!candidate) continue;
      for (const part of candidate.content?.parts || []) {
        if (part.text != null) fullText += part.text;
        if (part.functionCall) {
          hasToolCalls = true;
          const fc = part.functionCall;
          const tc = {
            id: `call_${Date.now().toString(36)}${Math.random().toString(36).substring(2, 6)}`,
            type: 'function',
            function: { name: fc.name, arguments: JSON.stringify(fc.args || {}) }
          };
          if (part.thoughtSignature || part.thought_signature) {
            tc.extra_content = { google: { thought_signature: part.thoughtSignature || part.thought_signature } };
          }
          toolCalls.push(tc);
        }
      }
      if (candidate.finishReason) finishReason = mapGeminiFinishReason(candidate.finishReason, hasToolCalls);
    } catch (e) { log('[GEMINI-PARSE-ERR]', e.message); }
  }
  return {
    id: `chatcmpl-${id}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: model,
    choices: [{
      index: 0,
      message: {
        role: 'assistant',
        content: fullText || null,
        refusal: null,
        tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
        function_call: undefined
      },
      finish_reason: finishReason,
      logprobs: null
    }],
    usage: { prompt_tokens: -1, completion_tokens: -1, total_tokens: -1 }
  };
}

function anthropicToOpenAIStreamingChunk(anthropicEvent, id, model) {
  if (anthropicEvent.type === 'content_block_delta' && anthropicEvent.delta?.text) {
    return {
      id: `chatcmpl-${id}`,
      object: 'chat.completion.chunk',
      created: Math.floor(Date.now() / 1000),
      model: model,
      choices: [{ index: 0, delta: { content: anthropicEvent.delta.text }, finish_reason: null, logprobs: null }]
    };
  }
  if (anthropicEvent.type === 'message_stop') {
    return {
      id: `chatcmpl-${id}`,
      object: 'chat.completion.chunk',
      created: Math.floor(Date.now() / 1000),
      model: model,
      choices: [{ index: 0, delta: {}, finish_reason: 'stop', logprobs: null }]
    };
  }
  if (anthropicEvent.type === 'content_block_start' && anthropicEvent.content_block?.type === 'tool_use') {
    const tb = anthropicEvent.content_block;
    return {
      id: `chatcmpl-${id}`,
      object: 'chat.completion.chunk',
      created: Math.floor(Date.now() / 1000),
      model: model,
      choices: [{
        index: 0,
        delta: {
          tool_calls: [{
            index: 0,
            id: tb.id,
            type: 'function',
            function: { name: tb.name, arguments: JSON.stringify(tb.input || {}) }
          }]
        },
        finish_reason: null,
        logprobs: null
      }]
    };
  }
  return null;
}

function anthropicToOpenAINonStreaming(anthropicResponse, id, model) {
  const content = anthropicResponse.content || [];
  let fullText = '';
  const toolCalls = [];
  for (const block of content) {
    if (block.type === 'text') fullText += block.text;
    if (block.type === 'tool_use') {
      toolCalls.push({
        id: block.id,
        type: 'function',
        function: { name: block.name, arguments: JSON.stringify(block.input || {}) }
      });
    }
  }
  return {
    id: `chatcmpl-${id}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: model,
    choices: [{
      index: 0,
      message: {
        role: 'assistant',
        content: fullText || null,
        refusal: null,
        tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
        function_call: undefined
      },
      finish_reason: anthropicResponse.stop_reason === 'tool_use' ? 'tool_calls' : 'stop',
      logprobs: null
    }],
    usage: {
      prompt_tokens: anthropicResponse.usage?.input_tokens || -1,
      completion_tokens: anthropicResponse.usage?.output_tokens || -1,
      total_tokens: (anthropicResponse.usage?.input_tokens || 0) + (anthropicResponse.usage?.output_tokens || 0)
    }
  };
}

// ─── Streaming Request Helper ──────────────────────────────────
// For streaming: pipe response directly, parse SSE on the fly
function makeStreamingRequest(url, options, body, onEvent, onError, onEnd) {
  const parsed = new URL(url);
  const client = parsed.protocol === 'https:' ? https : http;
  const req = client.request(parsed, options, (res) => {
    if (res.statusCode !== 200) {
      let errorData = '';
      res.on('data', c => errorData += c);
      res.on('end', () => {
        const err = new Error(`HTTP ${res.statusCode}: ${errorData.substring(0, 500)}`);
        err.statusCode = res.statusCode;
        // Try to extract retry-after
        const retryAfter = res.headers['retry-after'];
        if (retryAfter) {
          const ms = parseInt(retryAfter) * 1000;
          if (!isNaN(ms)) err.retryAfter = ms;
        }
        onError(err);
      });
      return;
    }

    const decoder = new TextDecoder('utf-8');
    const parser = new SseParser();

    res.on('data', (chunk) => {
      const text = decoder.decode(chunk, { stream: true });
      const events = parser.feed(text);
      for (const event of events) {
        if (event === '[DONE]') continue;
        onEvent(event);
      }
    });

    res.on('end', () => {
      try {
        const events = parser.feed(decoder.decode());
        for (const event of events) {
          if (event === '[DONE]') continue;
          onEvent(event);
        }
        const events2 = parser.flush();
        for (const event of events2) {
          if (event === '[DONE]') continue;
          onEvent(event);
        }
      } catch (e) { log('[STREAM-DECODER-ERR]', e.message); }
      onEnd();
    });

    res.on('error', (err) => { err.statusCode = 0; onError(err); });
  });

  req.on('error', (err) => { err.statusCode = 0; onError(err); });
  req.on('timeout', () => {
    req.destroy();
    const err = new Error('Timeout'); err.statusCode = 0; err.code = 'ETIMEDOUT'; onError(err);
  });
  req.setTimeout(REQUEST_TIMEOUT_MS);
  if (body) req.write(typeof body === 'string' ? body : JSON.stringify(body));
  req.end();
}

// For non-streaming: buffer entire response
function makeRequest(url, options, body) {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const client = parsed.protocol === 'https:' ? https : http;
    const req = client.request(parsed, options, (res) => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        if (res.statusCode !== 200) {
          const err = new Error(`HTTP ${res.statusCode}: ${data.substring(0, 500)}`);
          err.statusCode = res.statusCode;
          err.data = data;
          const retryAfter = res.headers['retry-after'];
          if (retryAfter) {
            const ms = parseInt(retryAfter) * 1000;
            if (!isNaN(ms)) err.retryAfter = ms;
          }
          reject(err);
          return;
        }
        resolve({ statusCode: res.statusCode, data, headers: res.headers });
      });
    });
    req.on('error', (err) => { err.statusCode = 0; reject(err); });
    req.on('timeout', () => { req.destroy(); const err = new Error('Timeout'); err.statusCode = 0; err.code = 'ETIMEDOUT'; reject(err); });
    req.setTimeout(REQUEST_TIMEOUT_MS);
    if (body) req.write(typeof body === 'string' ? body : JSON.stringify(body));
    req.end();
  });
}

// ─── Proxy Dispatch ────────────────────────────────────────────
function buildRequestBody(openaiBody, info) {
  if (info.endpoint === 'anthropic') return buildAnthropicBody(openaiBody);
  if (info.endpoint === 'publisher') return buildDeepSeekBody(openaiBody);
  if (info.endpoint === 'legacy') return buildGeminiBody(openaiBody);
  // openai: pass through
  return openaiBody;
}

function getRequestHeaders(authToken, info, stream) {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${authToken}`
  };
  if (stream) {
    headers['Accept'] = 'text/event-stream';
  } else {
    headers['Accept'] = 'application/json';
  }
  return headers;
}

// ─── OAuth Flow ────────────────────────────────────────────────
let oauthState = null;
function startOAuthFlow() {
  if (!OAUTH_CLIENT_ID || !OAUTH_CLIENT_SECRET) {
    console.error('[OAUTH] Missing GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET');
    return;
  }
  oauthState = crypto.randomBytes(16).toString('hex');
  const redirectUri = `http://localhost:${OAUTH_REDIRECT_PORT}/auth/callback`;
  const scope = encodeURIComponent('https://www.googleapis.com/auth/cloud-platform');
  const authUrl = `https://accounts.google.com/o/oauth2/auth?client_id=${OAUTH_CLIENT_ID}&redirect_uri=${encodeURIComponent(redirectUri)}&response_type=code&scope=${scope}&access_type=offline&prompt=consent&state=${oauthState}`;

  console.log('\n═══════════════════════════════════════════════════════════════');
  console.log('  OPEN BROWSER AND LOG IN:');
  console.log('  ' + authUrl);
  console.log('═══════════════════════════════════════════════════════════════\n');

  const platform = process.platform;
  let cmd;
  if (platform === 'darwin') cmd = `open "${authUrl}"`;
  else if (platform === 'win32') cmd = `start "" "${authUrl}"`;
  else cmd = `xdg-open "${authUrl}"`;
  exec(cmd, (err) => {
    if (err) console.log('[OAUTH] Could not auto-open browser. Open URL manually.');
  });
}

async function exchangeCodeForTokens(code) {
  const redirectUri = `http://localhost:${OAUTH_REDIRECT_PORT}/auth/callback`;
  const postData = new URLSearchParams({
    grant_type: 'authorization_code',
    code: code,
    client_id: OAUTH_CLIENT_ID,
    client_secret: OAUTH_CLIENT_SECRET,
    redirect_uri: redirectUri
  }).toString();

  return new Promise((resolve, reject) => {
    const req = https.request({
      hostname: 'oauth2.googleapis.com', path: '/token', method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded', 'Content-Length': Buffer.byteLength(postData) }
    }, (res) => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        try {
          const json = JSON.parse(data);
          if (json.access_token) {
            console.log('\n═══════════════════════════════════════════════════════════════');
            console.log('  OAUTH SUCCESS! Add these to your environment:');
            console.log('');
            console.log(`  GOOGLE_ACCESS_TOKEN="${json.access_token}"`);
            if (json.refresh_token) {
              console.log(`  GOOGLE_REFRESH_TOKEN="${json.refresh_token}"`);
              console.log('  IMPORTANT: Save the refresh_token! You only get it once.');
            }
            console.log('═══════════════════════════════════════════════════════════════\n');
            resolve(json);
          } else {
            reject(new Error(`Token exchange failed: ${data}`));
          }
        } catch (e) { reject(new Error(`Token exchange parse error: ${data}`)); }
      });
    });
    req.on('error', reject);
    req.write(postData);
    req.end();
  });
}

const oauthServer = http.createServer(async (req, res) => {
  if (req.url.startsWith('/auth/callback')) {
    const parsed = new URL(req.url, `http://localhost:${OAUTH_REDIRECT_PORT}`);
    const code = parsed.searchParams.get('code');
    const state = parsed.searchParams.get('state');
    const error = parsed.searchParams.get('error');

    if (error) {
      res.writeHead(400, { 'Content-Type': 'text/html' });
      res.end(`<h1>OAuth Error</h1><p>${error}</p>`);
      return;
    }
    if (!code) {
      res.writeHead(400, { 'Content-Type': 'text/html' });
      res.end('<h1>Missing code</h1>');
      return;
    }
    if (state !== oauthState) {
      res.writeHead(400, { 'Content-Type': 'text/html' });
      res.end('<h1>Invalid state</h1>');
      return;
    }
    try {
      await exchangeCodeForTokens(code);
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end('<h1>Success!</h1><p>Check proxy console for your tokens.</p>');
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'text/html' });
      res.end(`<h1>Error</h1><p>${err.message}</p>`);
    }
    return;
  }
  res.writeHead(404); res.end('Not found');
});

// ─── Main Proxy Server ─────────────────────────────────────────
const server = http.createServer(async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  if (req.url === '/auth/start') {
    if (!OAUTH_CLIENT_ID || !OAUTH_CLIENT_SECRET) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Missing GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET' }));
      return;
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ message: 'Check proxy console for browser URL' }));
    startOAuthFlow();
    return;
  }

  if (req.url === '/health') {
    const token = await getAuthToken().catch(() => null);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      status: 'ok',
      models: AVAILABLE_MODELS,
      circuit: circuitBreaker.state,
      queue: { running: requestQueue.running, pending: requestQueue.queue.length },
      rate_limits: Object.fromEntries([...modelRateLimits.entries()].map(([k, v]) => [k, { reset_in_ms: Math.max(0, v.resetAt - Date.now()) }])),
      auth: {
        mode: serviceAccount ? 'service_account' : (REFRESH_TOKEN ? 'refresh_token' : (ACCESS_TOKEN ? 'access_token' : 'api_key')),
        token_valid: !!token,
      }
    }));
    return;
  }

  if (req.url === '/v1/models') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      object: 'list',
      data: AVAILABLE_MODELS.map(m => {
        const info = MODEL_REGISTRY[m];
        return {
          id: m,
          object: 'model',
          created: Math.floor(Date.now() / 1000),
          owned_by: info.provider,
          provider: info.provider,
          location: info.location
        };
      })
    }));
    return;
  }

  if (req.url !== '/v1/chat/completions') {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { message: 'Not found', type: 'invalid_request_error', param: null, code: null } }));
    return;
  }

  if (req.method !== 'POST') {
    res.writeHead(405, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { message: 'Method not allowed', type: 'invalid_request_error', param: null, code: null } }));
    return;
  }

  if (!circuitBreaker.canExecute()) {
    res.writeHead(429, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      error: {
        message: `Circuit breaker open. Retry in ${Math.ceil(CIRCUIT_RECOVERY_MS / 1000)}s.`,
        type: 'rate_limit_error',
        param: null,
        code: 'rate_limit_exceeded'
      }
    }));
    return;
  }

  let body = '';
  req.on('data', chunk => body += chunk);
  await new Promise(resolve => req.on('end', resolve));

  log('[REQUEST]', req.url, body.substring(0, 400));

  let openaiBody;
  try {
    openaiBody = safeJsonParse(body, 'request-body');
  } catch (e) {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { message: e.message, type: 'invalid_request_error', param: null, code: null } }));
    return;
  }

  const stream = openaiBody.stream === true;
  const requestId = generateId();

  try {
    const authToken = await getAuthToken();
    if (!authToken && !API_KEY) {
      throw new Error('No valid authentication token available');
    }

    const model = resolveModel(openaiBody.model || DEFAULT_MODEL);
    const modelInfo = getModelInfo(model);
    log('[MODEL]', model, '| provider:', modelInfo.provider, '| endpoint:', modelInfo.endpoint, '| location:', modelInfo.location);

    const url = getVertexUrl(model, modelInfo);
    const requestBody = buildRequestBody(openaiBody, modelInfo);
    const headers = getRequestHeaders(authToken, modelInfo, stream);

    log('[URL]', url.replace(authToken, '***'));

    if (stream) {
      // ─── Streaming: pipe directly ─────────────────────────────
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
      });

      let isFirst = true;
      let hasError = false;

      await requestQueue.enqueue(() =>
        retryWithBackoff(() => new Promise((resolve, reject) => {
          makeStreamingRequest(url, { method: 'POST', headers }, requestBody,
            (event) => {
              if (event === '[DONE]') return;
              try {
                if (modelInfo.endpoint === 'anthropic') {
                  // Anthropic SSE: event: xxx\ndata: {...}
                  // But we already parsed the data line in makeStreamingRequest
                  // event here is just the data payload
                  try {
                    const parsed = safeJsonParse(event, 'anthropic-stream');
                    const chunk = anthropicToOpenAIStreamingChunk(parsed, requestId, model);
                    if (chunk) res.write(`data: ${JSON.stringify(chunk)}\n\n`);
                  } catch (e) {
                    // Might be a non-JSON event line, skip
                    log('[ANTHROPIC-SKIP]', event.substring(0, 100));
                  }
                } else if (modelInfo.endpoint === 'openai' || modelInfo.endpoint === 'publisher') {
                  // OpenAI-native or DeepSeek publisher
                  try {
                    const parsed = safeJsonParse(event, 'openai-stream');
                    if (parsed.model) parsed.model = model;
                    res.write(`data: ${JSON.stringify(parsed)}\n\n`);
                  } catch (e) {
                    log('[OPENAI-SKIP]', event.substring(0, 100));
                  }
                } else {
                  // Legacy Gemini
                  const parsed = safeJsonParse(event, 'gemini-stream');
                  const chunk = geminiToOpenAIStreamingChunk(parsed, requestId, model, isFirst);
                  isFirst = false;
                  if (chunk) res.write(`data: ${JSON.stringify(chunk)}\n\n`);
                }
              } catch (e) {
                log('[STREAM-EVENT-ERR]', e.message, event.substring(0, 200));
              }
            },
            (err) => {
              hasError = true;
              log('[STREAM-ERR]', err.message);
              reject(err);
            },
            () => {
              if (!hasError) resolve();
            }
          );
        }), model, model)
      );

      if (!hasError) {
        res.write(`data: {"id":"chatcmpl-${requestId}","object":"chat.completion.chunk","created":${Math.floor(Date.now() / 1000)},"model":"${model}","choices":[{"index":0,"delta":{},"finish_reason":null,"logprobs":null}]}\n\n`);
        res.write('data: [DONE]\n\n');
      }
      res.end();

    } else {
      // ─── Non-streaming: buffer ────────────────────────────────
      const vertexResponse = await requestQueue.enqueue(() =>
        retryWithBackoff(() => makeRequest(url, { method: 'POST', headers }, requestBody), model, model)
      );

      let response;
      if (modelInfo.endpoint === 'anthropic') {
        const parsed = safeJsonParse(vertexResponse.data, 'anthropic-response');
        response = anthropicToOpenAINonStreaming(parsed, requestId, model);
      } else if (modelInfo.endpoint === 'openai' || modelInfo.endpoint === 'publisher') {
        response = safeJsonParse(vertexResponse.data, 'openai-response');
        if (response.model) response.model = model;
      } else {
        const parser = new SseParser();
        const events = parser.feed(vertexResponse.data);
        const events2 = parser.flush();
        response = geminiToOpenAINonStreaming([...events, ...events2], requestId, model);
      }

      log('[RESPONSE]', JSON.stringify(response).substring(0, 300));
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(response));
    }

  } catch (err) {
    log('[PROXY-ERR]', err.message, err.stack?.substring(0, 300));
    const is429 = err.statusCode === 429;
    const statusCode = is429 ? 429 : 500;
    res.writeHead(statusCode, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      error: {
        message: err.message,
        type: is429 ? 'rate_limit_error' : 'internal_error',
        param: null,
        code: is429 ? 'rate_limit_exceeded' : null
      }
    }));
  }
});

server.listen(PORT, () => {
  console.log(`Vertex AI Multi-Provider Proxy v9 running on port ${PORT}`);
  console.log(`Auth mode: ${serviceAccount ? 'Service Account' : (REFRESH_TOKEN ? 'OAuth Refresh Token' : (ACCESS_TOKEN ? 'OAuth Access Token' : 'API Key'))}`);
  console.log(`Models: ${AVAILABLE_MODELS.join(', ')}`);
  console.log(`Project: ${PROJECT_ID || '(none)'}`);
  console.log(`429 mitigation: concurrent=${MAX_CONCURRENT}, retries=${MAX_RETRIES}, circuit=${CIRCUIT_THRESHOLD}`);
  console.log(`Debug: ${DEBUG ? 'enabled' : 'disabled'}`);
  console.log('');
  console.log('Endpoints:');
  console.log(`  Proxy:     http://localhost:${PORT}/v1/chat/completions`);
  console.log(`  Health:    http://localhost:${PORT}/health`);
  console.log(`  Models:    http://localhost:${PORT}/v1/models`);
  if (OAUTH_CLIENT_ID && OAUTH_CLIENT_SECRET && !REFRESH_TOKEN && !ACCESS_TOKEN) {
    console.log(`  OAuth:     http://localhost:${PORT}/auth/start`);
  }
  console.log('');
});

if (OAUTH_CLIENT_ID && OAUTH_CLIENT_SECRET) {
  oauthServer.listen(OAUTH_REDIRECT_PORT, () => {
    console.log(`OAuth callback server on port ${OAUTH_REDIRECT_PORT}`);
  });
}

process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down...');
  server.close(() => process.exit(0));
});