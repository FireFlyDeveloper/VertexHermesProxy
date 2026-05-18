const { config, log } = require('./config');

// Per-model rate limit tracker
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

function getRateLimits() {
  return Object.fromEntries([...modelRateLimits.entries()].map(([k, v]) => [k, { reset_in_ms: Math.max(0, v.resetAt - Date.now()) }]));
}

// Circuit breaker
class CircuitBreaker {
  constructor(threshold, recoveryMs) {
    this.threshold = threshold;
    this.recoveryMs = recoveryMs;
    this.failures = 0;
    this.lastFailure = 0;
    this.state = 'CLOSED';
  }
  
  recordSuccess() {
    this.failures = 0;
    this.state = 'CLOSED';
    log('[CIRCUIT]', 'CLOSED');
  }
  
  recordFailure() {
    this.failures++;
    this.lastFailure = Date.now();
    if (this.failures >= this.threshold) {
      this.state = 'OPEN';
      log('[CIRCUIT]', 'OPENED', this.failures);
    }
  }
  
  canExecute() {
    if (this.state === 'CLOSED') return true;
    if (this.state === 'OPEN') {
      if (Date.now() - this.lastFailure >= this.recoveryMs) {
        this.state = 'HALF_OPEN';
        this.failures = 0;
        log('[CIRCUIT]', 'HALF_OPEN');
        return true;
      }
      return false;
    }
    return true;
  }
  
  getState() {
    return this.state;
  }
}

const circuitBreaker = new CircuitBreaker(config.CIRCUIT_THRESHOLD, config.CIRCUIT_RECOVERY_MS);

// Request queue
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
      resolve(await fn());
    } catch (err) {
      reject(err);
    } finally {
      this.running--;
      setImmediate(() => this.process());
    }
  }
  
  getStats() {
    return { running: this.running, pending: this.queue.length };
  }
}

const requestQueue = new RequestQueue(config.MAX_CONCURRENT);

// Retry helpers
function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function jitteredDelay(attempt) {
  const exp = Math.min(Math.pow(2, attempt) * config.BASE_DELAY_MS, config.MAX_DELAY_MS);
  return Math.floor(exp + Math.random() * exp * 0.3);
}

async function retryWithBackoff(fn, context, model) {
  for (let attempt = 0; attempt <= config.MAX_RETRIES; attempt++) {
    // Check per-model rate limit before attempting
    if (isModelRateLimited(model)) {
      const entry = modelRateLimits.get(model);
      const waitMs = entry.resetAt - Date.now();
      log('[RATE-LIMIT-WAIT]', model, 'waiting', waitMs, 'ms');
      await sleep(Math.max(0, waitMs));
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
        const retryAfter = err.retryAfter || 30000;
        recordModelRateLimit(model, retryAfter);
      }

      if (!isRetryable || attempt >= config.MAX_RETRIES) {
        circuitBreaker.recordFailure();
        throw err;
      }
      const delay = jitteredDelay(attempt);
      log('[RETRY]', context, attempt + 1 + '/' + config.MAX_RETRIES, delay + 'ms', err.statusCode || err.code);
      await sleep(delay);
    }
  }
  throw new Error('Max retries exceeded');
}

module.exports = {
  modelRateLimits,
  isModelRateLimited,
  recordModelRateLimit,
  getRateLimits,
  circuitBreaker,
  requestQueue,
  retryWithBackoff,
  sleep,
};