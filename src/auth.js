const https = require('https');
const crypto = require('crypto');
const { exec } = require('child_process');
const { config, log } = require('./config');

let serviceAccount = null;
let cachedToken = { token: null, expiry: 0 };

// Parse service account if provided
if (config.SERVICE_ACCOUNT_JSON) {
  try {
    serviceAccount = JSON.parse(config.SERVICE_ACCOUNT_JSON);
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
        } catch { reject(new Error(`OAuth parse error: ${data}`)); }
      });
    });
    req.on('error', reject);
    req.write(postData);
    req.end();
  });
}

async function refreshAccessToken() {
  if (!config.REFRESH_TOKEN || !config.OAUTH_CLIENT_ID || !config.OAUTH_CLIENT_SECRET) {
    throw new Error('Missing refresh token or OAuth client credentials');
  }
  const postData = new URLSearchParams({
    grant_type: 'refresh_token',
    refresh_token: config.REFRESH_TOKEN,
    client_id: config.OAUTH_CLIENT_ID,
    client_secret: config.OAUTH_CLIENT_SECRET
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
        } catch { reject(new Error(`Token refresh parse error: ${data}`)); }
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
  if (config.REFRESH_TOKEN && config.OAUTH_CLIENT_ID && config.OAUTH_CLIENT_SECRET) {
    try {
      const now = Math.floor(Date.now() / 1000);
      if (cachedToken.token && cachedToken.expiry > now + 60) {
        log('[AUTH]', 'Cached access token');
        return cachedToken.token;
      }
      return await refreshAccessToken();
    } catch (err) { log('[AUTH]', 'Refresh token failed:', err.message); }
  }
  if (config.ACCESS_TOKEN) { log('[AUTH]', 'Static access token'); return config.ACCESS_TOKEN; }
  return null;
}

// OAuth flow
let oauthState = null;

function startOAuthFlow() {
  if (!config.OAUTH_CLIENT_ID || !config.OAUTH_CLIENT_SECRET) {
    console.error('[OAUTH] Missing GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET');
    return;
  }
  oauthState = crypto.randomBytes(16).toString('hex');
  const redirectUri = `http://localhost:${config.OAUTH_REDIRECT_PORT}/auth/callback`;
  const scope = encodeURIComponent('https://www.googleapis.com/auth/cloud-platform');
  const authUrl = `https://accounts.google.com/o/oauth2/auth?client_id=${config.OAUTH_CLIENT_ID}&redirect_uri=${encodeURIComponent(redirectUri)}&response_type=code&scope=${scope}&access_type=offline&prompt=consent&state=${oauthState}`;

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
  return authUrl;
}

async function exchangeCodeForTokens(code) {
  const redirectUri = `http://localhost:${config.OAUTH_REDIRECT_PORT}/auth/callback`;
  const postData = new URLSearchParams({
    grant_type: 'authorization_code',
    code: code,
    client_id: config.OAUTH_CLIENT_ID,
    client_secret: config.OAUTH_CLIENT_SECRET,
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
        } catch { reject(new Error(`Token exchange parse error: ${data}`)); }
      });
    });
    req.on('error', reject);
    req.write(postData);
    req.end();
  });
}

function getOAuthState() {
  return oauthState;
}

module.exports = {
  getAuthToken,
  startOAuthFlow,
  exchangeCodeForTokens,
  getOAuthState,
};