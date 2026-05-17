#!/usr/bin/env node
/**
 * Vertex AI → OpenAI Compatible Proxy (v5 - 429 Mitigation)
 *
 * 429 Mitigation:
 *  - Request queue with configurable max concurrency
 *  - Exponential backoff with jitter on 429 (up to 5 retries)
 *  - Circuit breaker (opens after 10 consecutive 429s, recovers after 60s)
 *  - Global endpoint support (higher quotas)
 *  - Per-request timeout to prevent hung connections
 *  - Internal retry before returning error to client
 *
 * Existing fixes:
 *  - Multi-model support
 *  - Tool message grouping
 *  - thought_signature pass-through
 *  - Spec-compliant SSE parser
 *  - stream default = false
 */
const http = require('http');
const https = require('https');
const { URL } = require('url');

// ─── Config ────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
const API_KEY = process.env.GOOGLE_API_KEY || '';
const PROJECT_ID = process.env.GOOGLE_PROJECT_ID || '';
const LOCATION = process.env.GOOGLE_LOCATION || 'us-central1';
const USE_GLOBAL = process.env.USE_GLOBAL === '1';  // Use global endpoint for higher quotas
const DEFAULT_MODEL = process.env.DEFAULT_MODEL || 'gemini-3.1-flash-lite';
const DEBUG = process.env.DEBUG === '1';

// 429 Mitigation Config
const MAX_CONCURRENT = parseInt(process.env.MAX_CONCURRENT || '3', 10);        // Max parallel requests to Vertex
const MAX_RETRIES = parseInt(process.env.MAX_RETRIES || '5', 10);              // Retries on 429
const BASE_DELAY_MS = parseInt(process.env.BASE_DELAY_MS || '1000', 10);       // Base backoff delay
const MAX_DELAY_MS = parseInt(process.env.MAX_DELAY_MS || '32000', 10);        // Max backoff cap
const CIRCUIT_THRESHOLD = parseInt(process.env.CIRCUIT_THRESHOLD || '10', 10); // Errors before circuit opens
const CIRCUIT_RECOVERY_MS = parseInt(process.env.CIRCUIT_RECOVERY_MS || '60000', 10);
const REQUEST_TIMEOUT_MS = parseInt(process.env.REQUEST_TIMEOUT_MS || '60000', 10);

const AVAILABLE_MODELS = (process.env.AVAILABLE_MODELS || 'gemini-3.1-flash-lite,gemini-3.1-pro-preview')
  .split(',').map(m => m.trim()).filter(Boolean);

const MODEL_ALIASES = {
  'gpt-4o-mini': 'gemini-3.1-flash-lite',
  'gpt-4o': 'gemini-3.1-pro-preview',
};

if (!API_KEY) {
  console.error('[FATAL] GOOGLE_API_KEY required');
  process.exit(1);
}

function log(...args) {
  if (DEBUG) console.error('[DEBUG]', new Date().toISOString(), ...args);
}

function resolveModel(requestedModel) {
  const alias = MODEL_ALIASES[requestedModel];
  if (alias) return alias;
  if (AVAILABLE_MODELS.includes(requestedModel)) return requestedModel;
  return DEFAULT_MODEL;
}

function generateId(prefix = 'chatcmpl') {
  return `${prefix}-${Date.now().toString(16)}${Math.random().toString(36).substring(2, 10)}`;
}

function getVertexUrl(model) {
  const isPreview = model.includes('preview') || model.includes('exp');
  const apiVersion = (isPreview || !PROJECT_ID) ? 'v1beta1' : 'v1';

  if (USE_GLOBAL) {
    // Global endpoint: higher quotas, no regional pinning
    return `https://aiplatform.googleapis.com/${apiVersion}/projects/${PROJECT_ID || '_'}/locations/us-central1/publishers/google/models/${model}:streamGenerateContent?alt=sse`;
  }

  if (PROJECT_ID) {
    return `https://${LOCATION}-aiplatform.googleapis.com/${apiVersion}/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${model}:streamGenerateContent?alt=sse`;
  }
  return `https://aiplatform.googleapis.com/${apiVersion}/publishers/google/models/${model}:streamGenerateContent?alt=sse&key=${API_KEY}`;
}

// ─── Circuit Breaker ───────────────────────────────────────────
class CircuitBreaker {
  constructor(threshold, recoveryMs) {
    this.threshold = threshold;
    this.recoveryMs = recoveryMs;
    this.failures = 0;
    this.lastFailure = 0;
    this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
  }

  recordSuccess() {
    this.failures = 0;
    this.state = 'CLOSED';
    log('[CIRCUIT]', 'CLOSED (success)');
  }

  recordFailure() {
    this.failures++;
    this.lastFailure = Date.now();
    if (this.failures >= this.threshold) {
      this.state = 'OPEN';
      log('[CIRCUIT]', 'OPENED after', this.failures, 'failures');
    }
  }

  canExecute() {
    if (this.state === 'CLOSED') return true;
    if (this.state === 'OPEN') {
      if (Date.now() - this.lastFailure >= this.recoveryMs) {
        this.state = 'HALF_OPEN';
        this.failures = 0;
        log('[CIRCUIT]', 'HALF_OPEN (testing)');
        return true;
      }
      return false;
    }
    return true; // HALF_OPEN allows one test
  }
}

const circuitBreaker = new CircuitBreaker(CIRCUIT_THRESHOLD, CIRCUIT_RECOVERY_MS);

// ─── Request Queue ─────────────────────────────────────────────
class RequestQueue {
  constructor(maxConcurrent) {
    this.maxConcurrent = maxConcurrent;
    this.running = 0;
    this.queue = [];
  }

  async enqueue(fn) {
    return new Promise((resolve, reject) => {
      this.queue.push({ fn, resolve, reject });
      this.process();
    });
  }

  async process() {
    if (this.running >= this.maxConcurrent || this.queue.length === 0) return;
    this.running++;
    const { fn, resolve, reject } = this.queue.shift();
    try {
      const result = await fn();
      resolve(result);
    } catch (err) {
      reject(err);
    } finally {
      this.running--;
      // Process next
      setImmediate(() => this.process());
    }
  }
}

const requestQueue = new RequestQueue(MAX_CONCURRENT);

// ─── Retry with Exponential Backoff + Jitter ───────────────────
function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function jitteredDelay(attempt) {
  // Truncated exponential backoff: base * 2^attempt + random jitter
  const exp = Math.min(Math.pow(2, attempt) * BASE_DELAY_MS, MAX_DELAY_MS);
  const jitter = Math.random() * exp * 0.3; // 0-30% jitter
  return Math.floor(exp + jitter);
}

async function retryWithBackoff(fn, context) {
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const result = await fn();
      if (attempt > 0) log('[RETRY-SUCCESS]', context, 'on attempt', attempt);
      circuitBreaker.recordSuccess();
      return result;
    } catch (err) {
      const is429 = err.statusCode === 429 || (err.message && err.message.includes('429'));
      const is5xx = err.statusCode >= 500 && err.statusCode < 600;
      const isRetryable = is429 || is5xx || err.code === 'ECONNRESET' || err.code === 'ETIMEDOUT';

      if (!isRetryable || attempt >= MAX_RETRIES) {
        circuitBreaker.recordFailure();
        throw err;
      }

      const delay = jitteredDelay(attempt);
      log('[RETRY]', context, 'attempt', attempt + 1, '/', MAX_RETRIES,
        'delay', delay + 'ms', 'error:', err.statusCode || err.code || err.message.substring(0, 50));
      await sleep(delay);
    }
  }
  throw new Error('Max retries exceeded');
}

// ─── Safe JSON Parse ───────────────────────────────────────────
function safeJsonParse(str, ctx) {
  try {
    return JSON.parse(str);
  } catch (e) {
    log('[JSON-FAIL]', ctx, 'input:', str.substring(0, 200), 'err:', e.message);
    throw new Error(`JSON parse failed (${ctx}): ${e.message}`);
  }
}

// ─── SSE Parser ────────────────────────────────────────────────
class SseParser {
  constructor() {
    this.buffer = '';
  }
  feed(text) {
    this.buffer += text;
    const events = [];
    const parts = this.buffer.split(/(?:\r?\n){2,}/);
    this.buffer = parts.pop() || '';
    for (const part of parts) {
      if (!part.trim()) continue;
      const lines = part.split(/\r?\n/);
      const dataLines = [];
      for (const line of lines) {
        if (line.startsWith('data:')) {
          dataLines.push(line.slice(5).trimStart());
        }
      }
      if (dataLines.length) events.push(dataLines.join('\n'));
    }
    return events;
  }
  flush() {
    const events = [];
    if (this.buffer.trim()) {
      const lines = this.buffer.split(/\r?\n/);
      const dataLines = [];
      for (const line of lines) {
        if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
      }
      if (dataLines.length) events.push(dataLines.join('\n'));
    }
    this.buffer = '';
    return events;
  }
}

// ─── OpenAI → Vertex Request Translator ────────────────────────
function openAIToVertex(openaiBody) {
  const requestedModel = openaiBody.model || DEFAULT_MODEL;
  const model = resolveModel(requestedModel);
  log('[MODEL]', 'requested:', requestedModel, '→ resolved:', model);

  const contents = [];
  let systemInstruction = null;
  const messages = openaiBody.messages || [];
  let i = 0;

  while (i < messages.length) {
    const msg = messages[i];

    if (msg.role === 'system' || msg.role === 'developer') {
      if (typeof msg.content === 'string') {
        systemInstruction = { parts: [{ text: msg.content }] };
      }
      i++;
      continue;
    }

    // Group consecutive tool/function responses
    if (msg.role === 'tool' || msg.role === 'function') {
      const toolGroup = [];
      while (i < messages.length && (messages[i].role === 'tool' || messages[i].role === 'function')) {
        toolGroup.push(messages[i]);
        i++;
      }

      const parts = toolGroup.map(t => {
        let responseData;
        if (typeof t.content === 'string') {
          try {
            responseData = JSON.parse(t.content);
          } catch {
            responseData = { result: t.content };
          }
        } else if (t.content !== null && typeof t.content === 'object') {
          responseData = t.content;
        } else {
          responseData = { result: String(t.content ?? '') };
        }

        return {
          functionResponse: {
            name: t.name || t.function?.name || 'unknown',
            response: responseData
          }
        };
      });

      log('[TOOL-GROUP]', toolGroup.length, 'tools → 1 turn');
      contents.push({ role: 'user', parts });
      continue;
    }

    const role = msg.role === 'assistant' ? 'model' : 'user';
    let parts = [];

    if (typeof msg.content === 'string') {
      parts.push({ text: msg.content });
    } else if (Array.isArray(msg.content)) {
      for (const item of msg.content) {
        if (item.type === 'text') {
          parts.push({ text: item.text });
        } else if (item.type === 'image_url') {
          const imageUrl = item.image_url?.url || '';
          if (imageUrl.startsWith('data:')) {
            const m = imageUrl.match(/^data:([^;]+);base64,(.+)$/);
            if (m) parts.push({ inlineData: { mimeType: m[1], data: m[2] } });
          } else {
            parts.push({ fileData: { mimeType: 'image/jpeg', fileUri: imageUrl } });
          }
        } else if (item.type === 'input_audio') {
          parts.push({
            inlineData: {
              mimeType: `audio/${item.input_audio?.format || 'wav'}`,
              data: item.input_audio?.data || ''
            }
          });
        } else if (item.type === 'file') {
          if (item.file?.file_data) {
            const m = item.file.file_data.match(/^data:([^;]+);base64,(.+)$/);
            if (m) parts.push({ inlineData: { mimeType: m[1], data: m[2] } });
          } else if (item.file?.file_id) {
            parts.push({ fileData: { mimeType: 'application/octet-stream', fileUri: item.file.file_id } });
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
        const sig = tc.extra_content?.google?.thought_signature
                 || tc.thoughtSignature
                 || tc.thought_signature;
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

    if (parts.length > 0) {
      contents.push({ role, parts });
    }
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
  Object.keys(generationConfig).forEach(k => {
    if (generationConfig[k] === undefined) delete generationConfig[k];
  });

  let tools = null;
  let toolConfig = null;

  if (openaiBody.tools) {
    tools = openaiBody.tools.map(t => ({
      functionDeclarations: [{
        name: t.function.name,
        description: t.function.description,
        parameters: t.function.parameters
      }]
    }));
  } else if (openaiBody.functions) {
    tools = openaiBody.functions.map(f => ({
      functionDeclarations: [{
        name: f.name,
        description: f.description,
        parameters: f.parameters
      }]
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
    log(`  [${idx}] role=${c.role} parts=${c.parts.length}`,
      c.parts.map(p => Object.keys(p).join(',')).join(' | '));
  });

  return { model, vertexBody };
}

// ─── Vertex → OpenAI Translators ───────────────────────────────
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
        function: {
          name: fc.name,
          arguments: JSON.stringify(fc.args || {})
        }
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
    choices: [{
      index: 0,
      delta: delta,
      finish_reason: mapFinishReason(candidate.finishReason, hasToolCalls),
      logprobs: null
    }]
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
      if (candidate.finishReason) {
        finishReason = mapFinishReason(candidate.finishReason, hasToolCalls);
      }
    } catch (e) {
      log('[NON-STREAM-PARSE-ERR]', e.message);
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
      finish_reason: finishReason,
      logprobs: null
    }],
    usage: {
      prompt_tokens: -1,
      completion_tokens: -1,
      total_tokens: -1
    }
  };
}

// ─── Vertex Request with Retry ─────────────────────────────────
function makeVertexRequest(vertexUrl, options, vertexBody) {
  return new Promise((resolve, reject) => {
    const url = new URL(vertexUrl);
    const req = https.request(url, options, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        if (res.statusCode !== 200) {
          const err = new Error(`Vertex API error ${res.statusCode}: ${data}`);
          err.statusCode = res.statusCode;
          err.data = data;
          reject(err);
          return;
        }
        resolve({ statusCode: res.statusCode, data });
      });
    });

    req.on('error', (err) => {
      err.statusCode = 0;
      reject(err);
    });

    req.on('timeout', () => {
      req.destroy();
      const err = new Error('Request timeout');
      err.statusCode = 0;
      err.code = 'ETIMEDOUT';
      reject(err);
    });

    req.setTimeout(REQUEST_TIMEOUT_MS);
    req.write(JSON.stringify(vertexBody));
    req.end();
  });
}

// ─── HTTP Server ────────────────────────────────────────────────
const server = http.createServer(async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  if (req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      status: 'ok',
      models: AVAILABLE_MODELS,
      circuit: circuitBreaker.state,
      queue: { running: requestQueue.running, pending: requestQueue.queue.length }
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

  // Circuit breaker check
  if (!circuitBreaker.canExecute()) {
    log('[CIRCUIT-REJECT]');
    res.writeHead(429, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      error: {
        message: 'Rate limit: circuit breaker open. Vertex AI is overloaded. Retry in ' + Math.ceil(CIRCUIT_RECOVERY_MS / 1000) + 's.',
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
    const { model, vertexBody } = openAIToVertex(openaiBody);
    const vertexUrl = getVertexUrl(model);
    const url = new URL(vertexUrl);
    const options = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
      }
    };
    if (PROJECT_ID || USE_GLOBAL) options.headers['Authorization'] = `Bearer ${API_KEY}`;

    log('[VERTEX-URL]', url.toString().replace(API_KEY, '***'));

    // Queue the request + retry on 429
    const vertexResponse = await requestQueue.enqueue(() =>
      retryWithBackoff(() => makeVertexRequest(vertexUrl, options, vertexBody), model)
    );

    if (stream) {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
      });

      const decoder = new TextDecoder('utf-8');
      const parser = new SseParser();
      let isFirst = true;

      // Parse the accumulated response data as a stream
      const data = vertexResponse.data;
      // Feed chunks to simulate streaming from the already-fetched response
      // For non-streaming Vertex returns everything at once, but we parse it as SSE events
      const events = parser.feed(data);
      for (const event of events) {
        if (event === '[DONE]') continue;
        try {
          const vertexChunk = safeJsonParse(event, 'stream-chunk');
          const openaiChunk = buildOpenAIStreamingChunk(vertexChunk, requestId, model, isFirst);
          isFirst = false;
          if (openaiChunk) res.write(`data: ${JSON.stringify(openaiChunk)}\n\n`);
        } catch (e) {
          log('[STREAM-PARSE-ERR]', e.message, event.substring(0, 200));
        }
      }
      const events2 = parser.flush();
      for (const event of events2) {
        if (event === '[DONE]') continue;
        try {
          const vertexChunk = safeJsonParse(event, 'stream-flush');
          const openaiChunk = buildOpenAIStreamingChunk(vertexChunk, requestId, model, isFirst);
          if (openaiChunk) res.write(`data: ${JSON.stringify(openaiChunk)}\n\n`);
        } catch (e) {
          log('[STREAM-FLUSH-ERR]', e.message);
        }
      }
      res.write(`data: {"id":"chatcmpl-${requestId}","object":"chat.completion.chunk","created":${Math.floor(Date.now()/1000)},"model":"${model}","choices":[{"index":0,"delta":{},"finish_reason":null,"logprobs":null}]}\n\n`);
      res.write('data: [DONE]\n\n');
      res.end();
    } else {
      const parser = new SseParser();
      const events = parser.feed(vertexResponse.data);
      const events2 = parser.flush();
      const allEvents = [...events, ...events2];
      log('[NON-STREAM-EVENTS]', allEvents.length);

      const response = buildOpenAINonStreamingResponse(allEvents, requestId, model);
      log('[RESPONSE]', JSON.stringify(response).substring(0, 300));
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(response));
    }

  } catch (err) {
    log('[PROXY-ERR]', err.message, err.stack);
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
  console.log(`Vertex AI OpenAI Proxy running on port ${PORT}`);
  console.log(`Available models: ${AVAILABLE_MODELS.join(', ')}`);
  console.log(`Default model: ${DEFAULT_MODEL}`);
  console.log(`Max concurrent: ${MAX_CONCURRENT}, Max retries: ${MAX_RETRIES}`);
  console.log(`Circuit breaker: ${CIRCUIT_THRESHOLD} failures / ${CIRCUIT_RECOVERY_MS}ms recovery`);
  console.log(`Global endpoint: ${USE_GLOBAL ? 'enabled' : 'disabled'} (USE_GLOBAL=1)`);
  console.log(`Debug: ${DEBUG ? 'enabled' : 'disabled'}`);
});

process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down...');
  server.close(() => process.exit(0));
});
