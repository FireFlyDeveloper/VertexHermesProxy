#!/usr/bin/env node
/**
 * Vertex AI → OpenAI Compatible Proxy (v4 - Multi-Model Support)
 *
 * Supported models:
 *   - gemini-3.1-flash-lite
 *   - gemini-3.1-pro-preview
 *
 * Critical fixes:
 *  1. Group consecutive role="tool" messages into ONE Vertex user turn
 *  2. functionResponse.response = raw object (not wrapped in {result:...})
 *  3. Spec-compliant SSE parser
 *  4. stream default = false (OpenAI spec)
 *  5. thought_signature bidirectional pass-through
 *  6. Multi-model support with /v1/models listing
 */
const http = require('http');
const https = require('https');
const { URL } = require('url');

// ─── Config ────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
const API_KEY = process.env.GOOGLE_API_KEY || '';
const PROJECT_ID = process.env.GOOGLE_PROJECT_ID || '';
const LOCATION = process.env.GOOGLE_LOCATION || 'us-central1';
const DEFAULT_MODEL = process.env.DEFAULT_MODEL || 'gemini-3.1-flash-lite';
const DEBUG = process.env.DEBUG === '1';

// Multi-model configuration
const AVAILABLE_MODELS = (process.env.AVAILABLE_MODELS || 'gemini-3.1-flash-lite,gemini-3.1-pro-preview')
  .split(',')
  .map(m => m.trim())
  .filter(Boolean);

// Model aliases (OpenAI-style names → Vertex names)
const MODEL_ALIASES = {
  'gpt-4o-mini': 'gemini-3.1-flash-lite',
  'gpt-4o': 'gemini-3.1-pro-preview',
  // Add more aliases as needed
};

if (!API_KEY) {
  console.error('[FATAL] GOOGLE_API_KEY required');
  process.exit(1);
}

function log(...args) {
  if (DEBUG) console.error('[DEBUG]', ...args);
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
  // Preview models require v1beta1 even on project endpoints
  const isPreview = model.includes('preview') || model.includes('exp');
  const apiVersion = (isPreview || !PROJECT_ID) ? 'v1beta1' : 'v1';

  if (PROJECT_ID) {
    return `https://${LOCATION}-aiplatform.googleapis.com/${apiVersion}/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${model}:streamGenerateContent?alt=sse`;
  }
  return `https://aiplatform.googleapis.com/${apiVersion}/publishers/google/models/${model}:streamGenerateContent?alt=sse&key=${API_KEY}`;
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

    // System / Developer
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

    // Normal user / assistant
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

    // tool_calls
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

    // Deprecated function_call
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

  // Health check
  if (req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok', models: AVAILABLE_MODELS }));
    return;
  }

  // Models list (OpenAI-compatible)
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

  // OpenAI default: stream = false
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
    if (PROJECT_ID) options.headers['Authorization'] = `Bearer ${API_KEY}`;

    log('[VERTEX-URL]', url.toString().replace(API_KEY, '***'));
    log('[VERTEX-BODY]', JSON.stringify(vertexBody).substring(0, 1000));

    const vertexReq = https.request(url, options, (vertexRes) => {
      log('[VERTEX-STATUS]', vertexRes.statusCode);

      if (vertexRes.statusCode !== 200) {
        let errorData = '';
        vertexRes.on('data', chunk => errorData += chunk);
        vertexRes.on('end', () => {
          log('[VERTEX-ERROR]', errorData.substring(0, 400));
          res.writeHead(vertexRes.statusCode, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            error: { message: `Vertex API error: ${errorData}`, type: 'api_error', param: null, code: vertexRes.statusCode }
          }));
        });
        return;
      }

      if (stream) {
        res.writeHead(200, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive'
        });

        const decoder = new TextDecoder('utf-8');
        const parser = new SseParser();
        let isFirst = true;

        vertexRes.on('data', (chunk) => {
          const events = parser.feed(decoder.decode(chunk, { stream: true }));
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
        });

        vertexRes.on('end', () => {
          try {
            const events = parser.feed(decoder.decode());
            for (const event of events) {
              if (event === '[DONE]') continue;
              try {
                const vertexChunk = safeJsonParse(event, 'stream-end');
                const openaiChunk = buildOpenAIStreamingChunk(vertexChunk, requestId, model, isFirst);
                if (openaiChunk) res.write(`data: ${JSON.stringify(openaiChunk)}\n\n`);
              } catch (e) {
                log('[STREAM-END-ERR]', e.message);
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
          } catch (e) {
            log('[DECODER-FLUSH-ERR]', e.message);
          }
          res.write(`data: {"id":"chatcmpl-${requestId}","object":"chat.completion.chunk","created":${Math.floor(Date.now()/1000)},"model":"${model}","choices":[{"index":0,"delta":{},"finish_reason":null,"logprobs":null}]}\n\n`);
          res.write('data: [DONE]\n\n');
          res.end();
        });

        vertexRes.on('error', (err) => {
          log('[VERTEX-STREAM-ERR]', err.message);
          res.end();
        });
      } else {
        const decoder = new TextDecoder('utf-8');
        const parser = new SseParser();

        vertexRes.on('data', chunk => {
          parser.feed(decoder.decode(chunk, { stream: true }));
        });

        vertexRes.on('end', () => {
          try { parser.feed(decoder.decode()); } catch (e) { log('[DECODER-FLUSH]', e.message); }
          const events = parser.flush();
          log('[NON-STREAM-EVENTS]', events.length);

          try {
            const response = buildOpenAINonStreamingResponse(events, requestId, model);
            log('[RESPONSE]', JSON.stringify(response).substring(0, 300));
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(response));
          } catch (e) {
            log('[BUILD-RESPONSE-ERR]', e.message, e.stack);
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: { message: e.message, type: 'internal_error', param: null, code: null } }));
          }
        });
      }
    });

    vertexReq.on('error', (err) => {
      log('[VERTEX-REQ-ERR]', err.message);
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { message: err.message, type: 'api_error', param: null, code: null } }));
    });

    vertexReq.write(JSON.stringify(vertexBody));
    vertexReq.end();

  } catch (err) {
    log('[PROXY-ERR]', err.message, err.stack);
    res.writeHead(500, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { message: err.message, type: 'internal_error', param: null, code: null } }));
  }
});

server.listen(PORT, () => {
  console.log(`Vertex AI OpenAI Proxy running on port ${PORT}`);
  console.log(`Available models: ${AVAILABLE_MODELS.join(', ')}`);
  console.log(`Default model: ${DEFAULT_MODEL}`);
  console.log(`Endpoint: http://localhost:${PORT}/v1/chat/completions`);
  console.log(`Models list: http://localhost:${PORT}/v1/models`);
  console.log(`Debug: ${DEBUG ? 'enabled' : 'disabled'} (DEBUG=1)`);
});

process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down...');
  server.close(() => process.exit(0));
});
