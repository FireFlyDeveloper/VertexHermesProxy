const { config } = require('./config');

// Model registry
const MODEL_REGISTRY = {
  // Google Gemini
  'gemini-3.1-flash-lite':      { provider: 'google',    endpoint: 'openai',    location: 'global' },
  'gemini-3.1-pro-preview':     { provider: 'google',    endpoint: 'openai',    location: 'global' },
  // Moonshot
  'moonshotai/kimi-k2-thinking-maas': { provider: 'moonshot', endpoint: 'openai', location: 'global' },
  // DeepSeek
  'deepseek-ai/deepseek-v3.2-maas':   { provider: 'deepseek', endpoint: 'openai', location: 'global' },
  'deepseek-ai/deepseek-v3.1-maas':   { provider: 'deepseek', endpoint: 'openai', location: 'us-west2' },
  'deepseek-ai/deepseek-r1-0528-maas':{ provider: 'deepseek', endpoint: 'publisher', location: 'us-central1' },
  // Anthropic Claude
  'claude-haiku-4-5':           { provider: 'anthropic', endpoint: 'anthropic', location: 'global' },
  'claude-sonnet-4-6':          { provider: 'anthropic', endpoint: 'anthropic', location: 'global' },
  // Zhipu AI (GLM)
  'zai-org/glm-4.7-maas':       { provider: 'zhipu',     endpoint: 'openai',    location: 'global' },
  // Alibaba Qwen
  'qwen/qwen3-235b-a22b-instruct-2507-maas': { provider: 'qwen', endpoint: 'openai', location: 'us-south1' },
};

const MODEL_ALIASES = {
  'gpt-4o-mini': 'gemini-3.1-flash-lite',
  'gpt-4o': 'gemini-3.1-pro-preview',
};

const AVAILABLE_MODELS = Object.keys(MODEL_REGISTRY);

function resolveModel(requestedModel) {
  const alias = MODEL_ALIASES[requestedModel];
  if (alias) return alias;
  if (MODEL_REGISTRY[requestedModel]) return requestedModel;
  return config.DEFAULT_MODEL;
}

function getModelInfo(model) {
  return MODEL_REGISTRY[model] || MODEL_REGISTRY[config.DEFAULT_MODEL] || {
    provider: 'google', endpoint: 'openai', location: 'global'
  };
}

function getVertexUrl(model, info) {
  const apiVersion = info.endpoint === 'anthropic' ? 'v1' : 'v1beta1';
  
  let endpoint;
  if (info.location === 'global') {
    endpoint = config.ENDPOINT;
  } else if (info.location.includes('-')) {
    endpoint = `${info.location}-aiplatform.googleapis.com`;
  } else {
    endpoint = config.ENDPOINT;
  }

  if (info.endpoint === 'openai') {
    return `https://${endpoint}/${apiVersion}/projects/${config.PROJECT_ID}/locations/${info.location}/endpoints/openapi/chat/completions`;
  }
  if (info.endpoint === 'publisher') {
    return `https://${endpoint}/${apiVersion}/projects/${config.PROJECT_ID}/locations/${info.location}/publishers/deepseek/models/${model}:streamRawPredict`;
  }
  if (info.endpoint === 'anthropic') {
    return `https://${endpoint}/${apiVersion}/projects/${config.PROJECT_ID}/locations/${info.location}/publishers/anthropic/models/${model}:streamRawPredict`;
  }
  // Legacy Gemini
  if (config.PROJECT_ID) {
    return `https://${endpoint}/${apiVersion}/projects/${config.PROJECT_ID}/locations/${info.location}/publishers/google/models/${model}:streamGenerateContent?alt=sse`;
  }
  return `https://aiplatform.googleapis.com/${apiVersion}/publishers/google/models/${model}:streamGenerateContent?alt=sse&key=${config.API_KEY}`;
}

function generateId(prefix = 'chatcmpl') {
  return `${prefix}-${Date.now().toString(16)}${Math.random().toString(36).substring(2, 10)}`;
}

function safeJsonParse(str, ctx) {
  try { 
    return JSON.parse(str); 
  } catch (e) {
    const preview = str.substring(0, 100).replace(/\s+/g, ' ');
    console.error('[JSON-FAIL]', ctx, preview, '...');
    throw new Error(`JSON parse failed (${ctx}): ${e.message}`);
  }
}

module.exports = {
  MODEL_REGISTRY,
  MODEL_ALIASES,
  AVAILABLE_MODELS,
  resolveModel,
  getModelInfo,
  getVertexUrl,
  generateId,
  safeJsonParse,
};