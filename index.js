#!/usr/bin/env node
/**
 * Vertex AI → OpenAI Compatible Proxy (v7 - Built-in OAuth + Refresh Token)
 *
 * Auth modes (priority order):
 *   1. Client-provided Bearer token (pass-through from Hermes)
 *   2. Service Account JSON (auto-refresh JWT)
 *   3. OAuth 2.0 Refresh Token (auto-refresh access token)
 *   4. Static Access Token (no refresh)
 *   5. API Key (legacy, non-OAuth endpoints only)
 *
 * Built-in OAuth flow:
 *   GET /auth/start → opens browser, returns refresh token
 *   (Run once on your desktop, then copy refresh token to env)
 *
 * Endpoint modes:
 *   A. OpenAI-native: /endpoints/openapi/chat/completions (RECOMMENDED)
 *   B. Legacy Gemini: /models/{model}:streamGenerateContent
 */
const http = require('http');
const https = require('https');
const { URL } = require('url');
const crypto = require('crypto');
const { exec } = require('child_process');

// ─── Config ────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
const DEBUG = process.env.DEBUG === '1';

// Auth (priority: service_account > refresh_token > access_token > api_key)
const API_KEY = process.env.GOOGLE_API_KEY || '';
const ACCESS_TOKEN = process.env.GOOGLE_ACCESS_TOKEN || '';
const REFRESH_TOKEN = process.env.GOOGLE_REFRESH_TOKEN || '';
const OAUTH_CLIENT_ID = process.env.GOOGLE_CLIENT_ID || '';
const OAUTH_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET || '';
const SERVICE_ACCOUNT_JSON = process.env.GOOGLE_SERVICE_ACCOUNT_JSON || '';

// OAuth web server (for built-in flow)
const OAUTH_REDIRECT_PORT = parseInt(process.env.OAUTH_REDIRECT_PORT || '8085', 10);

// Endpoint
const USE_OPENAI_ENDPOINT = process.env.USE_OPENAI_ENDPOINT !== '0';
const PROJECT_ID = process.env.GOOGLE_PROJECT_ID || '';
const LOCATION = process.env.GOOGLE_LOCATION || 'global';
const ENDPOINT = process.env.ENDPOINT || 'aiplatform.googleapis.com';

// Models
const DEFAULT_MODEL = process.env.DEFAULT_MODEL || 'gemini-3.1-flash-lite';
const AVAILABLE_MODELS = (process.env.AVAILABLE_MODELS || 'gemini-3.1-flash-lite,gemini-3.1-pro-preview,moonshotai/kimi-k2-thinking-maas')
  .split(',').map(m => m.trim()).filter(Boolean);
const MODEL_ALIASES = {
  'gpt-4o-mini': 'gemini-3.1-flash-lite',
  'gpt-4o': 'gemini-3.1-pro-preview',
};

// 429 Mitigation
const MAX_CONCURRENT = parseInt(process.env.MAX_CONCURRENT || '3', 10);
const MAX_RETRIES = parseInt(process.env.MAX_RETRIES || '5', 10);
const BASE_DELAY_MS = parseInt(process.env.BASE_DELAY_MS || '1000', 10);
const MAX_DELAY_MS = parseInt(process.env.MAX_DELAY_MS || '32000', 10);
const CIRCUIT_THRESHOLD = parseInt(process.env.CIRCUIT_THRESHOLD || '10', 10);
const CIRCUIT_RECOVERY_MS = parseInt(process.env.CIRCUIT_RECOVERY_MS || '60000', 10);
const REQUEST_TIMEOUT_MS = parseInt(process.env.REQUEST_TIMEOUT_MS || '120000', 10);

if (!API_KEY && !ACCESS_TOKEN && !REFRESH_TOKEN && !SERVICE_ACCOUNT_JSON) {
  console.error('[FATAL] No auth configured. Set one of:');
  console.error('  GOOGLE_SERVICE_ACCOUNT_JSON (recommended for servers)');
  console.error('  GOOGLE_REFRESH_TOKEN + GOOGLE_CLIENT_ID + GOOGLE_CLIENT_SECRET');
  console.error('  GOOGLE_ACCESS_TOKEN (short-lived)');
  console.error('  GOOGLE_API_KEY (legacy)');
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
    log('[AUTH]', 'Service account:', serviceAccount.client_email);
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
  if (cachedToken.token && cachedToken.expiry > now + 60) {
    return cachedToken.token;
  }
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

// ─── OAuth Refresh Token ───────────────────────────────────────
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
            log('[AUTH]', 'Refresh token exchanged, expires in', json.expires_in || 3600);
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

async function getAuthToken(clientBearerToken) {
  if (clientBearerToken && clientBearerToken.startsWith('Bearer ')) {
    const token = clientBearerToken.slice(7).trim();
    if (token && token !== 'dummy-key' && token !== 'sk-test') {
      log('[AUTH]', 'Client Bearer token');
      return token;
    }
  }
  if (serviceAccount) return await getOAuthTokenFromServiceAccount();
  if (REFRESH_TOKEN && OAUTH_CLIENT_ID && OAUTH_CLIENT_SECRET) {
    const now = Math.floor(Date.now() / 1000);
    if (cachedToken.token && cachedToken.expiry > now + 60) {
      log('[AUTH]', 'Cached access token');
      return cachedToken.token;
    }
    return await refreshAccessToken();
  }
  if (ACCESS_TOKEN) return ACCESS_TOKEN;
  return null;
}

// ─── Helpers ───────────────────────────────────────────────────
function resolveModel(requestedModel) {
  const alias = MODEL_ALIASES[requestedModel];
  if (alias) return alias;
  if (AVAILABLE_MODELS.includes(requestedModel)) return requestedModel;
  return DEFAULT_MODEL;
}
function generateId(prefix = 'chatcmpl') {
  return `${prefix}-${Date.now().toString(16)}${Math.random().toString(36).substring(2, 10)}`;
}
function getVertexUrl(model, openaiMode) {
  const apiVersion = 'v1beta1';
  if (openaiMode) {
    if (model.includes('/')) {
      return `https://${ENDPOINT}/${apiVersion}/projects/${PROJECT_ID}/locations/${LOCATION}/endpoints/openapi/chat/completions`;
    }
    return `https://${ENDPOINT}/${apiVersion}/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${model}:streamGenerateContent?alt=sse`;
  }
  if (PROJECT_ID) {
    return `https://${LOCATION}-aiplatform.googleapis.com/${apiVersion}/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${model}:streamGenerateContent?alt=sse`;
  }
  return `https://aiplatform.googleapis.com/${apiVersion}/publishers/google/models/${model}:streamGenerateContent?alt=sse&key=${API_KEY}`;
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
async function retryWithBackoff(fn, context) {
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const result = await fn();
      if (attempt > 0) log('[RETRY-OK]', context, attempt);
      circuitBreaker.recordSuccess();
      return result;
    } catch (err) {
      const isRetryable = err.statusCode === 429 || (err.statusCode >= 500 && err.statusCode < 600)
        || err.code === 'ECONNRESET' || err.code === 'ETIMEDOUT';
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
  catch (e) { log('[JSON-FAIL]', ctx, str.substring(0, 200), e.message); throw new Error(`JSON parse failed (${ctx}): ${e.message}`); }
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

// ─── OpenAI → Vertex (Legacy) ──────────────────────────────────
function openAIToVertex(openaiBody) {
  const requestedModel = openaiBody.model || DEFAULT_MODEL;
  const model = resolveModel(requestedModel);
  log('[MODEL]', requestedModel, '→', model);

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
      log('[TOOL-GROUP]', toolGroup.length, '→ 1 turn');
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
        } else if (item.type === 'input_audio') {
          parts.push({ inlineData: { mimeType: `audio/${item.input_audio?.format || 'wav'}`, data: item.input_audio?.data || '' } });
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
  } else if (openaiBody.function_call === 'none') {
    toolConfig = { functionCallingConfig: { mode: 'NONE' } };
  } else if (typeof openaiBody.function_call === 'object' && openaiBody.function_call?.name) {
    toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [openaiBody.function_call.name] } };
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

  log('[VERTEX-CONTENTS]', contents.length, 'turns');
  contents.forEach((c, idx) => {
    log(`  [${idx}] role=${c.role} parts=${c.parts.length}`, c.parts.map(p => Object.keys(p).join(',')).join(' | '));
  });

  return { model, vertexBody };
}

// ─── Vertex → OpenAI (Legacy) ──────────────────────────────────
function mapFinishReason(vertexFinishReason, hasToolCalls) {
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

function buildOpenAIStreamingChunk(vertexChunk, id, model, isFirst) {
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
    choices: [{ index: 0, delta: delta, finish_reason: mapFinishReason(candidate.finishReason, hasToolCalls), logprobs: null }]
  };
}

function buildOpenAINonStreamingResponse(events, id, model) {
  let fullText = '';
  const toolCalls = [];
  let finishReason = 'stop';
  let hasToolCalls = false;
  for (const event of events) {
    if (event === '[DONE]') continue;
    try {
      const chunk = safeJsonParse(event, 'non-stream-vertex');
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
      if (candidate.finishReason) finishReason = mapFinishReason(candidate.finishReason, hasToolCalls);
    } catch (e) { log('[NON-STREAM-PARSE-ERR]', e.message); }
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

// ─── HTTP Request Helper ───────────────────────────────────────
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

// ─── Proxy Endpoints ───────────────────────────────────────────
async function proxyOpenAIEndpoint(openaiBody, authToken, stream) {
  const model = openaiBody.model || DEFAULT_MODEL;
  const url = getVertexUrl(model, true);
  const headers = {
    'Content-Type': 'application/json',
    'Accept': stream ? 'text/event-stream' : 'application/json',
    'Authorization': `Bearer ${authToken}`
  };
  log('[OPENAI-ENDPOINT]', url.replace(authToken, '***'));
  return await makeRequest(url, { method: 'POST', headers }, openaiBody);
}

async function proxyLegacyEndpoint(openaiBody, authToken, stream) {
  const { model, vertexBody } = openAIToVertex(openaiBody);
  const url = getVertexUrl(model, false);
  const headers = {
    'Content-Type': 'application/json',
    'Accept': 'text/event-stream'
  };
  if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
  log('[LEGACY-ENDPOINT]', url.replace(authToken || API_KEY, '***'));
  return await makeRequest(url, { method: 'POST', headers }, vertexBody);
}

// ════════════════════════════════════════════════════════════════
// BUILT-IN OAUTH 2.0 FLOW (run once to get refresh token)
// ════════════════════════════════════════════════════════════════

let oauthState = null;
let oauthCodePromise = null;

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

  // Try to open browser automatically
  const platform = process.platform;
  let cmd;
  if (platform === 'darwin') cmd = `open "${authUrl}"`;
  else if (platform === 'win32') cmd = `start "" "${authUrl}"`;
  else cmd = `xdg-open "${authUrl}"`;

  exec(cmd, (err) => {
    if (err) console.log('[OAUTH] Could not auto-open browser. Please open the URL manually.');
  });

  oauthCodePromise = new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error('OAuth timeout: no callback received in 5 minutes'));
    }, 300000);

    // Store resolver globally for the callback handler
    global.oauthResolve = (code) => { clearTimeout(timeout); resolve(code); };
    global.oauthReject = (err) => { clearTimeout(timeout); reject(err); };
  });

  return oauthCodePromise;
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
              console.log('');
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

// ─── OAuth Callback Server ─────────────────────────────────────
const oauthServer = http.createServer(async (req, res) => {
  if (req.url.startsWith('/auth/callback')) {
    const parsed = new URL(req.url, `http://localhost:${OAUTH_REDIRECT_PORT}`);
    const code = parsed.searchParams.get('code');
    const state = parsed.searchParams.get('state');
    const error = parsed.searchParams.get('error');

    if (error) {
      res.writeHead(400, { 'Content-Type': 'text/html' });
      res.end(`<h1>OAuth Error</h1><p>${error}</p><p>${parsed.searchParams.get('error_description') || ''}</p>`);
      if (global.oauthReject) global.oauthReject(new Error(`OAuth error: ${error}`));
      return;
    }

    if (!code) {
      res.writeHead(400, { 'Content-Type': 'text/html' });
      res.end('<h1>Missing code</h1>');
      if (global.oauthReject) global.oauthReject(new Error('Missing authorization code'));
      return;
    }

    if (state !== oauthState) {
      res.writeHead(400, { 'Content-Type': 'text/html' });
      res.end('<h1>Invalid state</h1>');
      if (global.oauthReject) global.oauthReject(new Error('Invalid state parameter'));
      return;
    }

    try {
      await exchangeCodeForTokens(code);
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end('<h1>Success!</h1><p>You can close this tab and check the proxy console for your tokens.</p>');
      if (global.oauthResolve) global.oauthResolve(code);
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'text/html' });
      res.end(`<h1>Error</h1><p>${err.message}</p>`);
      if (global.oauthReject) global.oauthReject(err);
    }
    return;
  }

  res.writeHead(404);
  res.end('Not found');
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

  // OAuth start endpoint
  if (req.url === '/auth/start') {
    if (!OAUTH_CLIENT_ID || !OAUTH_CLIENT_SECRET) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Missing GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET' }));
      return;
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ message: 'Check proxy console for browser URL' }));
    startOAuthFlow().catch(err => console.error('[OAUTH]', err.message));
    return;
  }

  if (req.url === '/health') {
    const token = await getAuthToken(null).catch(() => null);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      status: 'ok',
      models: AVAILABLE_MODELS,
      circuit: circuitBreaker.state,
      queue: { running: requestQueue.running, pending: requestQueue.queue.length },
      auth: {
        mode: serviceAccount ? 'service_account' : (REFRESH_TOKEN ? 'refresh_token' : (ACCESS_TOKEN ? 'access_token' : (API_KEY ? 'api_key' : 'none'))),
        token_valid: !!token,
        endpoint_mode: USE_OPENAI_ENDPOINT ? 'openai_native' : 'legacy_gemini'
      }
    }));
    return;
  }

  if (req.url === '/v1/models') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      object: 'list',
      data: AVAILABLE_MODELS.map(m => ({
        id: m,
        object: 'model',
        created: Math.floor(Date.now() / 1000),
        owned_by: 'google'
      }))
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
  const clientAuth = req.headers['authorization'] || '';

  try {
    const authToken = await getAuthToken(clientAuth);
    if (!authToken && !API_KEY) {
      throw new Error('No valid authentication token available');
    }

    const useOpenAIEndpoint = USE_OPENAI_ENDPOINT && authToken && PROJECT_ID;
    log('[MODE]', useOpenAIEndpoint ? 'openai_native' : 'legacy_gemini');

    const vertexResponse = await requestQueue.enqueue(() =>
      retryWithBackoff(() => {
        if (useOpenAIEndpoint) {
          return proxyOpenAIEndpoint(openaiBody, authToken, stream);
        } else {
          return proxyLegacyEndpoint(openaiBody, authToken, stream);
        }
      }, openaiBody.model || DEFAULT_MODEL)
    );

    if (stream) {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
      });

      if (useOpenAIEndpoint) {
        const parser = new SseParser();
        const events = parser.feed(vertexResponse.data);
        const events2 = parser.flush();
        const allEvents = [...events, ...events2];
        for (const event of allEvents) {
          if (event === '[DONE]') { res.write('data: [DONE]\n\n'); continue; }
          try {
            const chunk = safeJsonParse(event, 'openai-native-stream');
            if (chunk.model) chunk.model = openaiBody.model || DEFAULT_MODEL;
            res.write(`data: ${JSON.stringify(chunk)}\n\n`);
          } catch (e) { log('[NATIVE-STREAM-ERR]', e.message, event.substring(0, 200)); }
        }
        res.end();
      } else {
        const parser = new SseParser();
        const events = parser.feed(vertexResponse.data);
        const events2 = parser.flush();
        const allEvents = [...events, ...events2];
        let isFirst = true;
        for (const event of allEvents) {
          if (event === '[DONE]') continue;
          try {
            const vertexChunk = safeJsonParse(event, 'legacy-stream');
            const openaiChunk = buildOpenAIStreamingChunk(vertexChunk, requestId, openaiBody.model || DEFAULT_MODEL, isFirst);
            isFirst = false;
            if (openaiChunk) res.write(`data: ${JSON.stringify(openaiChunk)}\n\n`);
          } catch (e) { log('[LEGACY-STREAM-ERR]', e.message, event.substring(0, 200)); }
        }
        res.write(`data: {"id":"chatcmpl-${requestId}","object":"chat.completion.chunk","created":${Math.floor(Date.now() / 1000)},"model":"${openaiBody.model || DEFAULT_MODEL}","choices":[{"index":0,"delta":{},"finish_reason":null,"logprobs":null}]}\n\n`);
        res.write('data: [DONE]\n\n');
        res.end();
      }
    } else {
      if (useOpenAIEndpoint) {
        const data = safeJsonParse(vertexResponse.data, 'openai-native-response');
        log('[NATIVE-RESPONSE]', JSON.stringify(data).substring(0, 300));
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(data));
      } else {
        const parser = new SseParser();
        const events = parser.feed(vertexResponse.data);
        const events2 = parser.flush();
        const allEvents = [...events, ...events2];
        log('[LEGACY-EVENTS]', allEvents.length);
        const response = buildOpenAINonStreamingResponse(allEvents, requestId, openaiBody.model || DEFAULT_MODEL);
        log('[LEGACY-RESPONSE]', JSON.stringify(response).substring(0, 300));
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(response));
      }
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

// ─── Start Servers ─────────────────────────────────────────────
server.listen(PORT, () => {
  console.log(`Vertex AI OpenAI Proxy v7 running on port ${PORT}`);
  console.log(`Auth mode: ${serviceAccount ? 'Service Account' : (REFRESH_TOKEN ? 'OAuth Refresh Token' : (ACCESS_TOKEN ? 'OAuth Access Token' : 'API Key'))}`);
  console.log(`Endpoint mode: ${USE_OPENAI_ENDPOINT ? 'OpenAI-native' : 'Legacy Gemini'}`);
  console.log(`Models: ${AVAILABLE_MODELS.join(', ')}`);
  console.log(`Project: ${PROJECT_ID || '(none)'}, Location: ${LOCATION}`);
  console.log(`429 mitigation: concurrent=${MAX_CONCURRENT}, retries=${MAX_RETRIES}, circuit=${CIRCUIT_THRESHOLD}`);
  console.log(`Debug: ${DEBUG ? 'enabled' : 'disabled'}`);
  console.log('');
  console.log('Endpoints:');
  console.log(`  Proxy:     http://localhost:${PORT}/v1/chat/completions`);
  console.log(`  Health:    http://localhost:${PORT}/health`);
  console.log(`  Models:    http://localhost:${PORT}/v1/models`);
  if (OAUTH_CLIENT_ID && OAUTH_CLIENT_SECRET && !REFRESH_TOKEN && !ACCESS_TOKEN) {
    console.log(`  OAuth:     http://localhost:${PORT}/auth/start  (run this to get tokens)`);
  }
  console.log('');
});

// Start OAuth callback server if needed
if (OAUTH_CLIENT_ID && OAUTH_CLIENT_SECRET) {
  oauthServer.listen(OAUTH_REDIRECT_PORT, () => {
    console.log(`OAuth callback server on port ${OAUTH_REDIRECT_PORT}`);
  });
}

process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down...');
  server.close(() => process.exit(0));
});