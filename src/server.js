const http = require('http');
const { config, log } = require('./config');
const { getAuthToken, startOAuthFlow, exchangeCodeForTokens, getOAuthState } = require('./auth');
const { AVAILABLE_MODELS, generateId, safeJsonParse } = require('./models');
const { circuitBreaker, requestQueue, getRateLimits } = require('./queue');
const { processNonStreaming, processStreaming } = require('./streaming');

// OAuth callback server
let oauthServer = null;

function createOAuthServer() {
  const server = http.createServer(async (req, res) => {
    if (req.url.startsWith('/auth/callback')) {
      const parsed = new URL(req.url, `http://localhost:${config.OAUTH_REDIRECT_PORT}`);
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
      if (state !== getOAuthState()) {
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
    res.writeHead(404);
    res.end('Not found');
  });
  return server;
}

function createProxyServer() {
  const server = http.createServer(async (req, res) => {
    // Set CORS headers
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
      if (!config.OAUTH_CLIENT_ID || !config.OAUTH_CLIENT_SECRET) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Missing GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET' }));
        return;
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ message: 'Check proxy console for browser URL' }));
      startOAuthFlow();
      return;
    }

    // Health endpoint
    if (req.url === '/health') {
      const token = await getAuthToken().catch(() => null);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        status: 'ok',
        models: AVAILABLE_MODELS,
        circuit: circuitBreaker.getState(),
        queue: requestQueue.getStats(),
        rate_limits: getRateLimits(),
        auth: {
          mode: config.SERVICE_ACCOUNT_JSON ? 'service_account' : 
                (config.REFRESH_TOKEN ? 'refresh_token' : 
                (config.ACCESS_TOKEN ? 'access_token' : 'api_key')),
          token_valid: !!token,
        }
      }));
      return;
    }

    // Models endpoint
    if (req.url === '/v1/models') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        object: 'list',
        data: AVAILABLE_MODELS.map(m => {
          const { MODEL_REGISTRY } = require('./models');
          const info = MODEL_REGISTRY[m];
          return {
            id: m,
            object: 'model',
            created: Math.floor(Date.now() / 1000),
            owned_by: info?.provider || 'unknown',
            provider: info?.provider || 'unknown',
            location: info?.location || 'global'
          };
        })
      }));
      return;
    }

    // Only accept /v1/chat/completions
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

    // Check circuit breaker
    if (!circuitBreaker.canExecute()) {
      res.writeHead(429, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        error: {
          message: `Circuit breaker open. Retry in ${Math.ceil(config.CIRCUIT_RECOVERY_MS / 1000)}s.`,
          type: 'rate_limit_error',
          param: null,
          code: 'rate_limit_exceeded'
        }
      }));
      return;
    }

    // Read body
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
      if (stream) {
        await processStreaming(req, res, openaiBody, requestId);
      } else {
        await processNonStreaming(req, res, openaiBody, requestId);
      }
    } catch (err) {
      log('[PROXY-ERR]', err.message, err.stack?.substring(0, 300));
      // Only send error response if headers haven't been sent
      if (!res.headersSent) {
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
    }
  });

  return server;
}

function startServers() {
  const proxyServer = createProxyServer();
  
  proxyServer.listen(config.PORT, () => {
    console.log(`Vertex AI Multi-Provider Proxy v10 running on port ${config.PORT}`);
    const authMode = config.SERVICE_ACCOUNT_JSON ? 'Service Account' : 
                     (config.REFRESH_TOKEN ? 'OAuth Refresh Token' : 
                     (config.ACCESS_TOKEN ? 'OAuth Access Token' : 'API Key'));
    console.log(`Auth mode: ${authMode}`);
    console.log(`Models: ${AVAILABLE_MODELS.join(', ')}`);
    console.log(`Project: ${config.PROJECT_ID || '(none)'}`);
    console.log(`429 mitigation: concurrent=${config.MAX_CONCURRENT}, retries=${config.MAX_RETRIES}, circuit=${config.CIRCUIT_THRESHOLD}`);
    console.log(`Debug: ${config.DEBUG ? 'enabled' : 'disabled'}`);
    console.log('');
    console.log('Endpoints:');
    console.log(`  Proxy:     http://localhost:${config.PORT}/v1/chat/completions`);
    console.log(`  Health:    http://localhost:${config.PORT}/health`);
    console.log(`  Models:    http://localhost:${config.PORT}/v1/models`);
    if (config.OAUTH_CLIENT_ID && config.OAUTH_CLIENT_SECRET && !config.REFRESH_TOKEN && !config.ACCESS_TOKEN) {
      console.log(`  OAuth:     http://localhost:${config.PORT}/auth/start`);
    }
    console.log('');
  });

  if (config.OAUTH_CLIENT_ID && config.OAUTH_CLIENT_SECRET) {
    oauthServer = createOAuthServer();
    oauthServer.listen(config.OAUTH_REDIRECT_PORT, () => {
      console.log(`OAuth callback server on port ${config.OAUTH_REDIRECT_PORT}`);
    });
  }
}

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down...');
  if (oauthServer) oauthServer.close();
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down...');
  if (oauthServer) oauthServer.close();
  process.exit(0);
});

module.exports = { startServers };