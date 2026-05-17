#!/usr/bin/env node
/**
 * Vertex AI → OpenAI Compatible Proxy
 * Complete implementation matching OpenAI Chat Completions API spec
 * 
 * Fixes:
 *  - stream default = false (OpenAI spec compliance)
 *  - thought_signature support for Gemini 3.x tool calling
 *  - finish_reason = "tool_calls" when appropriate
 *  - \r\n-safe SSE parsing
 *  - Full OpenAI request/response schema compliance
 */
const http = require('http');
const https = require('https');
const { URL } = require('url');

// ─── Configuration ─────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
const API_KEY = process.env.GOOGLE_API_KEY || '';
const PROJECT_ID = process.env.GOOGLE_PROJECT_ID || '';
const LOCATION = process.env.GOOGLE_LOCATION || 'us-central1';
const DEFAULT_MODEL = process.env.DEFAULT_MODEL || 'gemini-3.1-flash-lite';

if (!API_KEY) {
  console.error('Error: GOOGLE_API_KEY environment variable is required');
  process.exit(1);
}

// ─── Helpers ───────────────────────────────────────────────────
function generateId(prefix = 'chatcmpl') {
  return `${prefix}-${Date.now().toString(16)}${Math.random().toString(36).substring(2, 10)}`;
}

function getVertexUrl(model) {
  if (PROJECT_ID) {
    return `https://${LOCATION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${model}:streamGenerateContent?alt=sse`;
  }
  return `https://aiplatform.googleapis.com/v1beta1/publishers/google/models/${model}:streamGenerateContent?alt=sse&key=${API_KEY}`;
}

// ─── OpenAI → Vertex Request Translator ────────────────────────
function openAIToVertex(openaiBody) {
  const model = openaiBody.model || DEFAULT_MODEL;
  const contents = [];
  let systemInstruction = null;

  for (const msg of openaiBody.messages || []) {
    // Map developer role to system for Vertex
    if (msg.role === 'system' || msg.role === 'developer') {
      if (typeof msg.content === 'string') {
        systemInstruction = { parts: [{ text: msg.content }] };
      }
      continue;
    }

    const role = msg.role === 'assistant' ? 'model' : 'user';
    let parts = [];

    // Handle string content
    if (typeof msg.content === 'string') {
      parts.push({ text: msg.content });
    }
    // Handle array content (multimodal)
    else if (Array.isArray(msg.content)) {
      for (const item of msg.content) {
        if (item.type === 'text') {
          parts.push({ text: item.text });
        } else if (item.type === 'image_url') {
          const imageUrl = item.image_url?.url || '';
          if (imageUrl.startsWith('data:')) {
            const match = imageUrl.match(/^data:([^;]+);base64,(.+)$/);
            if (match) {
              parts.push({ inlineData: { mimeType: match[1], data: match[2] } });
            }
          } else {
            parts.push({ fileData: { mimeType: 'image/jpeg', fileUri: imageUrl } });
          }
        } else if (item.type === 'input_audio') {
          parts.push({ inlineData: { mimeType: `audio/${item.input_audio?.format || 'wav'}`, data: item.input_audio?.data || '' } });
        } else if (item.type === 'file') {
          if (item.file?.file_data) {
            const match = item.file.file_data.match(/^data:([^;]+);base64,(.+)$/);
            if (match) {
              parts.push({ inlineData: { mimeType: match[1], data: match[2] } });
            }
          } else if (item.file?.file_id) {
            parts.push({ fileData: { mimeType: 'application/octet-stream', fileUri: item.file.file_id } });
          }
        }
      }
    }

    // Handle tool_calls (function calls from assistant)
    if (msg.tool_calls && msg.tool_calls.length > 0) {
      for (const tc of msg.tool_calls) {
        const fcPart = {
          functionCall: {
            name: tc.function.name,
            args: JSON.parse(tc.function.arguments || '{}')
          }
        };
        // CRITICAL: Pass thought_signature back to Vertex for Gemini 3.x
        const sig = tc.extra_content?.google?.thought_signature || tc.thoughtSignature || tc.thought_signature;
        if (sig) {
          fcPart.thoughtSignature = sig;
        }
        parts.push(fcPart);
      }
    }

    // Handle tool response (function response from user/tool role)
    if (msg.role === 'tool' && msg.content) {
      parts = [{
        functionResponse: {
          name: msg.name || 'unknown',
          response: {
            result: typeof msg.content === 'string' ? JSON.parse(msg.content) : msg.content
          }
        }
      }];
    }

    // Handle deprecated function_call
    if (msg.function_call) {
      parts.push({
        functionCall: {
          name: msg.function_call.name,
          args: JSON.parse(msg.function_call.arguments || '{}')
        }
      });
    }

    // Handle deprecated function role response
    if (msg.role === 'function' && msg.content) {
      parts = [{
        functionResponse: {
          name: msg.name || 'unknown',
          response: {
            result: typeof msg.content === 'string' ? JSON.parse(msg.content) : msg.content
          }
        }
      }];
    }

    if (parts.length > 0) {
      contents.push({ role, parts });
    }
  }

  // Build generation config
  const generationConfig = {
    temperature: openaiBody.temperature ?? 1,
    maxOutputTokens: openaiBody.max_tokens || 65535,
    topP: openaiBody.top_p ?? 0.95,
    topK: openaiBody.top_k ?? 40,
    candidateCount: openaiBody.n || 1,
    stopSequences: openaiBody.stop ? (Array.isArray(openaiBody.stop) ? openaiBody.stop : [openaiBody.stop]) : undefined
  };
  Object.keys(generationConfig).forEach(key => {
    if (generationConfig[key] === undefined) delete generationConfig[key];
  });

  // Build tools config
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
  }
  // Handle deprecated functions
  else if (openaiBody.functions) {
    tools = openaiBody.functions.map(f => ({
      functionDeclarations: [{
        name: f.name,
        description: f.description,
        parameters: f.parameters
      }]
    }));
  }

  // Handle tool_choice / function_call
  if (openaiBody.tool_choice === 'none') {
    toolConfig = { functionCallingConfig: { mode: 'NONE' } };
  } else if (openaiBody.tool_choice === 'auto' || openaiBody.tool_choice === undefined) {
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

  return { model, vertexBody };
}

// ─── Vertex → OpenAI Response Translators ──────────────────────

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

function buildOpenAIStreamingChunk(vertexChunk, id, model, isFirst, accumulatedToolCalls) {
  const candidate = vertexChunk.candidates?.[0];
  if (!candidate) return null;

  const content = candidate.content;
  const finishReason = candidate.finishReason;
  let delta = {};
  let hasToolCalls = false;

  // First chunk should set role
  if (isFirst) {
    delta.role = 'assistant';
  }

  // Handle text parts
  if (content?.parts) {
    for (let i = 0; i < content.parts.length; i++) {
      const part = content.parts[i];
      if (part.text !== undefined) {
        delta.content = part.text;
      }
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
        // Include thought_signature for Gemini 3.x compatibility
        if (part.thoughtSignature || part.thought_signature) {
          tc.extra_content = {
            google: {
              thought_signature: part.thoughtSignature || part.thought_signature
            }
          };
        }
        delta.tool_calls = [tc];
      }
    }
  }

  const mappedFinish = mapFinishReason(finishReason, hasToolCalls);

  const chunk = {
    id: `chatcmpl-${id}`,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model: model,
    choices: [{
      index: 0,
      delta: delta,
      finish_reason: mappedFinish,
      logprobs: null
    }]
  };

  return `data: ${JSON.stringify(chunk)}\n\n`;
}

function buildOpenAIFinalChunk(model, id) {
  const chunk = {
    id: `chatcmpl-${id}`,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model: model,
    choices: [{
      index: 0,
      delta: {},
      finish_reason: null,
      logprobs: null
    }]
  };
  return `data: ${JSON.stringify(chunk)}\n\n`;
}

function buildOpenAINonStreamingResponse(vertexData, id, model) {
  const lines = vertexData.split('\n').filter(l => l.trim().startsWith('data: '));
  let fullText = '';
  let toolCalls = [];
  let finishReason = 'stop';
  let hasToolCalls = false;

  for (const line of lines) {
    try {
      const chunk = JSON.parse(line.slice(6).trim());
      const candidate = chunk.candidates?.[0];
      if (!candidate) continue;

      const parts = candidate.content?.parts || [];
      for (const part of parts) {
        if (part.text !== undefined) {
          fullText += part.text;
        }
        if (part.functionCall) {
          hasToolCalls = true;
          const fc = part.functionCall;
          const tc = {
            id: `call_${Date.now().toString(36)}${Math.random().toString(36).substring(2, 6)}`,
            type: 'function',
            function: {
              name: fc.name,
              arguments: JSON.stringify(fc.args || {})
            }
          };
          // Include thought_signature
          if (part.thoughtSignature || part.thought_signature) {
            tc.extra_content = {
              google: {
                thought_signature: part.thoughtSignature || part.thought_signature
              }
            };
          }
          toolCalls.push(tc);
        }
      }

      if (candidate.finishReason) {
        finishReason = mapFinishReason(candidate.finishReason, hasToolCalls);
      }
    } catch {}
  }

  const response = {
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

  return response;
}

// ─── HTTP Server ────────────────────────────────────────────────
const server = http.createServer(async (req, res) => {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  // Health check / models list
  if (req.url === '/health' || req.url === '/v1/models') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      object: 'list',
      data: [{
        id: DEFAULT_MODEL,
        object: 'model',
        created: Math.floor(Date.now() / 1000),
        owned_by: 'google'
      }]
    }));
    return;
  }

  // Only handle chat completions
  if (req.url !== '/v1/chat/completions') {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      error: {
        message: 'Not found',
        type: 'invalid_request_error',
        param: null,
        code: null
      }
    }));
    return;
  }

  if (req.method !== 'POST') {
    res.writeHead(405, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      error: {
        message: 'Method not allowed',
        type: 'invalid_request_error',
        param: null,
        code: null
      }
    }));
    return;
  }

  // Parse request body
  let body = '';
  req.on('data', chunk => body += chunk);
  await new Promise(resolve => req.on('end', resolve));

  let openaiBody;
  try {
    openaiBody = JSON.parse(body);
  } catch (e) {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      error: {
        message: 'Invalid JSON in request body',
        type: 'invalid_request_error',
        param: null,
        code: null
      }
    }));
    return;
  }

  // CRITICAL FIX: OpenAI default is stream=false
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

    if (PROJECT_ID) {
      options.headers['Authorization'] = `Bearer ${API_KEY}`;
    }

    const vertexReq = https.request(url, options, (vertexRes) => {
      if (vertexRes.statusCode !== 200) {
        let errorData = '';
        vertexRes.on('data', chunk => errorData += chunk);
        vertexRes.on('end', () => {
          res.writeHead(vertexRes.statusCode, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            error: {
              message: `Vertex API error: ${errorData}`,
              type: 'api_error',
              param: null,
              code: vertexRes.statusCode
            }
          }));
        });
        return;
      }

      if (stream) {
        // SSE streaming response
        res.writeHead(200, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive'
        });

        const decoder = new TextDecoder();
        let buffer = '';
        let isFirst = true;

        vertexRes.on('data', (chunk) => {
          buffer += decoder.decode(chunk, { stream: true });
          // CRITICAL FIX: Handle \r\n and \n line endings
          const lines = buffer.split(/\r?\n/);
          buffer = lines.pop();

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith('data: ')) continue;
            const json = trimmed.slice(6).trim();
            if (!json || json === '[DONE]') continue;

            try {
              const vertexChunk = JSON.parse(json);
              const openaiChunk = buildOpenAIStreamingChunk(vertexChunk, requestId, model, isFirst);
              isFirst = false;
              if (openaiChunk) res.write(openaiChunk);
            } catch (e) {
              // Skip malformed chunks
            }
          }
        });

        vertexRes.on('end', () => {
          res.write(buildOpenAIFinalChunk(model, requestId));
          res.write('data: [DONE]\n\n');
          res.end();
        });

        vertexRes.on('error', (err) => {
          console.error('Vertex stream error:', err);
          res.end();
        });
      } else {
        // Non-streaming response
        let data = '';
        vertexRes.on('data', chunk => data += chunk);
        vertexRes.on('end', () => {
          const response = buildOpenAINonStreamingResponse(data, requestId, model);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(response));
        });
      }
    });

    vertexReq.on('error', (err) => {
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        error: {
          message: err.message,
          type: 'api_error',
          param: null,
          code: null
        }
      }));
    });

    vertexReq.write(JSON.stringify(vertexBody));
    vertexReq.end();

  } catch (err) {
    console.error('Proxy error:', err);
    res.writeHead(500, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      error: {
        message: err.message,
        type: 'internal_error',
        param: null,
        code: null
      }
    }));
  }
});

server.listen(PORT, () => {
  console.log(`Vertex AI OpenAI Proxy running on port ${PORT}`);
  console.log(`Default model: ${DEFAULT_MODEL}`);
  console.log(`Endpoint: http://localhost:${PORT}/v1/chat/completions`);
  console.log(`Health check: http://localhost:${PORT}/health`);
});

process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down...');
  server.close(() => process.exit(0));
});