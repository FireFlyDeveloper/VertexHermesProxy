const http = require('http');
const https = require('https');
const { config, log } = require('./config');
const { getAuthToken } = require('./auth');
const {
  resolveModel,
  getModelInfo,
  getVertexUrl,
  safeJsonParse
} = require('./models');
const { requestQueue, retryWithBackoff } = require('./queue');
const {
  geminiToOpenAIStreamingChunk,
  geminiToOpenAINonStreaming,
  anthropicToOpenAIStreamingChunk,
  anthropicToOpenAINonStreaming,
  buildRequestBody,
  getRequestHeaders
} = require('./translators');

// SSE Parser
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
      for (const line of lines) {
        if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
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

// Non-streaming request
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
    req.on('timeout', () => {
      req.destroy();
      const err = new Error('Timeout');
      err.statusCode = 0;
      err.code = 'ETIMEDOUT';
      reject(err);
    });
    req.setTimeout(config.REQUEST_TIMEOUT_MS);
    if (body) req.write(typeof body === 'string' ? body : JSON.stringify(body));
    req.end();
  });
}

// Streaming request with proper error handling
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
      } catch (e) {
        log('[STREAM-DECODER-ERR]', e.message);
      }
      onEnd();
    });

    res.on('error', (err) => {
      err.statusCode = 0;
      onError(err);
    });
  });

  req.on('error', (err) => {
    err.statusCode = 0;
    onError(err);
  });

  req.on('timeout', () => {
    req.destroy();
    const err = new Error('Timeout');
    err.statusCode = 0;
    err.code = 'ETIMEDOUT';
    onError(err);
  });

  req.setTimeout(config.REQUEST_TIMEOUT_MS);
  if (body) req.write(typeof body === 'string' ? body : JSON.stringify(body));
  req.end();

  return req;
}

// Process non-streaming request
async function processNonStreaming(req, res, openaiBody, requestId) {
  const authToken = await getAuthToken();
  if (!authToken && !config.API_KEY) {
    throw new Error('No valid authentication token available');
  }

  const model = resolveModel(openaiBody.model || config.DEFAULT_MODEL);
  const modelInfo = getModelInfo(model);
  log('[MODEL]', model, '| provider:', modelInfo.provider, '| endpoint:', modelInfo.endpoint, '| location:', modelInfo.location);

  const url = getVertexUrl(model, modelInfo);
  const requestBody = buildRequestBody(openaiBody, modelInfo);
  const headers = getRequestHeaders(authToken, modelInfo, false);

  log('[URL]', url.replace(authToken, '***'));

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

// Process streaming request - FIXED: headers sent only once
async function processStreaming(req, res, openaiBody, requestId) {
  let headersSent = false;
  let requestAborted = false;

  // Handle client disconnect
  req.on('close', () => {
    requestAborted = true;
    log('[STREAM] Client disconnected');
  });

  try {
    const authToken = await getAuthToken();
    if (!authToken && !config.API_KEY) {
      throw new Error('No valid authentication token available');
    }

    const model = resolveModel(openaiBody.model || config.DEFAULT_MODEL);
    const modelInfo = getModelInfo(model);
    log('[MODEL]', model, '| provider:', modelInfo.provider, '| endpoint:', modelInfo.endpoint, '| location:', modelInfo.location);

    const url = getVertexUrl(model, modelInfo);
    const requestBody = buildRequestBody(openaiBody, modelInfo);
    const headers = getRequestHeaders(authToken, modelInfo, true);

    log('[URL]', url.replace(authToken, '***'));

    // Send headers only once, before any async operations
    // Use writeHead to set status and headers together
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': '*'
    });
    headersSent = true;

    let isFirst = true;
    let hasError = false;
    let streamRequest = null;

    await requestQueue.enqueue(() =>
      retryWithBackoff(() => new Promise((resolve, reject) => {
        streamRequest = makeStreamingRequest(url, { method: 'POST', headers }, requestBody,
          (event) => {
            if (requestAborted) {
              if (streamRequest) streamRequest.destroy();
              return;
            }
            if (event === '[DONE]') return;
            try {
              if (modelInfo.provider === 'google') {
                // Gemini streaming
                try {
                  const parsed = safeJsonParse(event, 'gemini-stream');
                  const chunk = geminiToOpenAIStreamingChunk(parsed, requestId, model, isFirst);
                  isFirst = false;
                  if (chunk && !requestAborted) {
                    res.write(`data: ${JSON.stringify(chunk)}\n\n`);
                  }
                } catch (e) {
                  log('[GEMINI-STREAM-ERR]', e.message, event.substring(0, 200));
                }
              } else if (modelInfo.provider === 'anthropic') {
                // Anthropic streaming (if needed)
                try {
                  const parsed = safeJsonParse(event, 'anthropic-stream');
                  const chunk = anthropicToOpenAIStreamingChunk(parsed, requestId, model);
                  if (chunk && !requestAborted) {
                    res.write(`data: ${JSON.stringify(chunk)}\n\n`);
                  }
                } catch {
                  log('[ANTHROPIC-SKIP]', event.substring(0, 100));
                }
              } else {
                // OpenAI-compatible (DeepSeek, Moonshot, Qwen, Zhipu)
                try {
                  const parsed = safeJsonParse(event, 'openai-stream');
                  // Ensure model field is set
                  if (parsed.model) parsed.model = model;
                  if (!requestAborted) {
                    res.write(`data: ${JSON.stringify(parsed)}\n\n`);
                  }
                } catch {
                  log('[OPENAI-SKIP]', event.substring(0, 100));
                }
              }
            } catch (e) {
              log('[STREAM-EVENT-ERR]', e.message, event.substring(0, 200));
            }
          },
          (err) => {
            if (!requestAborted) {
              hasError = true;
              log('[STREAM-ERR]', err.message);
              reject(err);
            } else {
              resolve();
            }
          },
          () => {
            if (!requestAborted) resolve();
            else resolve();
          }
        );
      }), model, model)
    );

    if (!hasError && !requestAborted) {
      // Send final chunk and DONE
      res.write(`data: {"id":"chatcmpl-${requestId}","object":"chat.completion.chunk","created":${Math.floor(Date.now() / 1000)},"model":"${model}","choices":[{"index":0,"delta":{},"finish_reason":null,"logprobs":null}]}\n\n`);
      res.write('data: [DONE]\n\n');
    }

    if (!requestAborted) {
      res.end();
    }

  } catch (err) {
    log('[STREAM-PROXY-ERR]', err.message);
    // Only send error response if headers haven't been sent yet
    if (!headersSent && !requestAborted) {
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
    } else if (!requestAborted) {
      // Headers already sent, try to send error as SSE
      try {
        res.write(`data: ${JSON.stringify({ error: { message: err.message } })}\n\n`);
        res.end();
      } catch (writeErr) {
        // Ignore write errors after headers sent
        log('[STREAM-CLOSE-ERR]', writeErr.message);
      }
    }
  }
}

module.exports = {
  processNonStreaming,
  processStreaming,
  SseParser,
  makeRequest,
  makeStreamingRequest,
};