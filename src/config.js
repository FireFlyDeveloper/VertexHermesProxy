const path = require('path');

// Environment variables with defaults
const config = {
  // Server
  PORT: parseInt(process.env.PORT || '3000', 10),
  DEBUG: process.env.DEBUG === '1',
  
  // Auth
  API_KEY: process.env.GOOGLE_API_KEY || '',
  ACCESS_TOKEN: process.env.GOOGLE_ACCESS_TOKEN || '',
  REFRESH_TOKEN: process.env.GOOGLE_REFRESH_TOKEN || '',
  OAUTH_CLIENT_ID: process.env.GOOGLE_CLIENT_ID || '',
  OAUTH_CLIENT_SECRET: process.env.GOOGLE_CLIENT_SECRET || '',
  SERVICE_ACCOUNT_JSON: process.env.GOOGLE_SERVICE_ACCOUNT_JSON || '',
  
  // OAuth
  OAUTH_REDIRECT_PORT: parseInt(process.env.OAUTH_REDIRECT_PORT || '8085', 10),
  
  // Vertex AI
  PROJECT_ID: process.env.GOOGLE_PROJECT_ID || '',
  LOCATION: process.env.GOOGLE_LOCATION || 'global',
  ENDPOINT: process.env.ENDPOINT || 'aiplatform.googleapis.com',
  DEFAULT_MODEL: process.env.DEFAULT_MODEL || 'gemini-3.1-flash-lite',
  
  // Rate limiting & retries
  MAX_CONCURRENT: parseInt(process.env.MAX_CONCURRENT || '3', 10),
  MAX_RETRIES: parseInt(process.env.MAX_RETRIES || '5', 10),
  BASE_DELAY_MS: parseInt(process.env.BASE_DELAY_MS || '1000', 10),
  MAX_DELAY_MS: parseInt(process.env.MAX_DELAY_MS || '32000', 10),
  CIRCUIT_THRESHOLD: parseInt(process.env.CIRCUIT_THRESHOLD || '10', 10),
  CIRCUIT_RECOVERY_MS: parseInt(process.env.CIRCUIT_RECOVERY_MS || '60000', 10),
  REQUEST_TIMEOUT_MS: parseInt(process.env.REQUEST_TIMEOUT_MS || '120000', 10),
};

// Validate required config
if (!config.API_KEY && !config.ACCESS_TOKEN && !config.REFRESH_TOKEN && !config.SERVICE_ACCOUNT_JSON) {
  console.error('[FATAL] No auth configured. Set one of: GOOGLE_API_KEY, GOOGLE_ACCESS_TOKEN, GOOGLE_REFRESH_TOKEN, GOOGLE_SERVICE_ACCOUNT_JSON');
  process.exit(1);
}

function log(...args) {
  if (config.DEBUG) {
    console.error('[DEBUG]', new Date().toISOString(), ...args);
  }
}

module.exports = { config, log };