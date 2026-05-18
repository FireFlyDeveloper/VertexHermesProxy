const { log } = require('./config');

function mapGeminiFinishReason(vertexFinishReason, hasToolCalls) {
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

function geminiToOpenAIStreamingChunk(vertexChunk, id, model, isFirst) {
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
    choices: [{ index: 0, delta: delta, finish_reason: mapGeminiFinishReason(candidate.finishReason, hasToolCalls), logprobs: null }]
  };
}

function geminiToOpenAINonStreaming(events, id, model) {
  let fullText = '';
  const toolCalls = [];
  let finishReason = 'stop';
  let hasToolCalls = false;
  for (const event of events) {
    if (event === '[DONE]') continue;
    try {
      const chunk = JSON.parse(event);
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
      if (candidate.finishReason) finishReason = mapGeminiFinishReason(candidate.finishReason, hasToolCalls);
    } catch (e) { log('[GEMINI-PARSE-ERR]', e.message); }
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

function anthropicToOpenAIStreamingChunk(anthropicEvent, id, model) {
  if (anthropicEvent.type === 'content_block_delta' && anthropicEvent.delta?.text) {
    return {
      id: `chatcmpl-${id}`,
      object: 'chat.completion.chunk',
      created: Math.floor(Date.now() / 1000),
      model: model,
      choices: [{ index: 0, delta: { content: anthropicEvent.delta.text }, finish_reason: null, logprobs: null }]
    };
  }
  if (anthropicEvent.type === 'message_stop') {
    return {
      id: `chatcmpl-${id}`,
      object: 'chat.completion.chunk',
      created: Math.floor(Date.now() / 1000),
      model: model,
      choices: [{ index: 0, delta: {}, finish_reason: 'stop', logprobs: null }]
    };
  }
  if (anthropicEvent.type === 'content_block_start' && anthropicEvent.content_block?.type === 'tool_use') {
    const tb = anthropicEvent.content_block;
    return {
      id: `chatcmpl-${id}`,
      object: 'chat.completion.chunk',
      created: Math.floor(Date.now() / 1000),
      model: model,
      choices: [{
        index: 0,
        delta: {
          tool_calls: [{
            index: 0,
            id: tb.id,
            type: 'function',
            function: { name: tb.name, arguments: JSON.stringify(tb.input || {}) }
          }]
        },
        finish_reason: null,
        logprobs: null
      }]
    };
  }
  return null;
}

function anthropicToOpenAINonStreaming(anthropicResponse, id, model) {
  const content = anthropicResponse.content || [];
  let fullText = '';
  const toolCalls = [];
  for (const block of content) {
    if (block.type === 'text') fullText += block.text;
    if (block.type === 'tool_use') {
      toolCalls.push({
        id: block.id,
        type: 'function',
        function: { name: block.name, arguments: JSON.stringify(block.input || {}) }
      });
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
      finish_reason: anthropicResponse.stop_reason === 'tool_use' ? 'tool_calls' : 'stop',
      logprobs: null
    }],
    usage: {
      prompt_tokens: anthropicResponse.usage?.input_tokens || -1,
      completion_tokens: anthropicResponse.usage?.output_tokens || -1,
      total_tokens: (anthropicResponse.usage?.input_tokens || 0) + (anthropicResponse.usage?.output_tokens || 0)
    }
  };
}

function buildGeminiBody(openaiBody) {
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

  return vertexBody;
}

function buildAnthropicBody(openaiBody) {
  const messages = [];
  let system = null;
  for (const msg of openaiBody.messages || []) {
    if (msg.role === 'system' || msg.role === 'developer') {
      if (typeof msg.content === 'string') system = msg.content;
      continue;
    }
    if (msg.role === 'tool' || msg.role === 'function') {
      messages.push({
        role: 'user',
        content: [{
          type: 'tool_result',
          tool_use_id: msg.tool_call_id || 'unknown',
          content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
        }]
      });
      continue;
    }
    if (msg.role === 'assistant' && msg.tool_calls) {
      const content = [];
      if (msg.content) content.push({ type: 'text', text: msg.content });
      for (const tc of msg.tool_calls) {
        content.push({
          type: 'tool_use',
          id: tc.id,
          name: tc.function.name,
          input: typeof tc.function.arguments === 'string' ? JSON.parse(tc.function.arguments) : tc.function.arguments
        });
      }
      messages.push({ role: 'assistant', content });
      continue;
    }
    if (typeof msg.content === 'string') {
      messages.push({ role: msg.role, content: msg.content });
    } else if (Array.isArray(msg.content)) {
      const anthropicContent = [];
      for (const item of msg.content) {
        if (item.type === 'text') anthropicContent.push({ type: 'text', text: item.text });
        else if (item.type === 'image_url') {
          const imageUrl = item.image_url?.url || '';
          if (imageUrl.startsWith('data:')) {
            const m = imageUrl.match(/^data:([^;]+);base64,(.+)$/);
            if (m) anthropicContent.push({ type: 'image', source: { type: 'base64', media_type: m[1], data: m[2] } });
          }
        }
      }
      messages.push({ role: msg.role, content: anthropicContent });
    }
  }

  const body = {
    anthropic_version: 'vertex-2023-10-16',
    stream: openaiBody.stream !== false,
    max_tokens: openaiBody.max_tokens || 512,
    temperature: openaiBody.temperature ?? 1,
    messages: messages
  };
  if (system) body.system = system;
  if (openaiBody.top_p != null) body.top_p = openaiBody.top_p;
  if (openaiBody.stop) body.stop_sequences = Array.isArray(openaiBody.stop) ? openaiBody.stop : [openaiBody.stop];
  if (openaiBody.tools) {
    body.tools = openaiBody.tools.map(t => ({
      name: t.function.name,
      description: t.function.description,
      input_schema: t.function.parameters
    }));
  }
  return body;
}

function buildDeepSeekBody(openaiBody) {
  const body = { ...openaiBody };
  delete body.model;
  return body;
}

function buildRequestBody(openaiBody, info) {
  if (info.endpoint === 'anthropic') return buildAnthropicBody(openaiBody);
  if (info.endpoint === 'publisher') return buildDeepSeekBody(openaiBody);
  if (info.endpoint === 'legacy') return buildGeminiBody(openaiBody);
  return openaiBody;
}

function getRequestHeaders(authToken, info, stream) {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${authToken}`
  };
  if (stream) {
    headers['Accept'] = 'text/event-stream';
  } else {
    headers['Accept'] = 'application/json';
  }
  return headers;
}

module.exports = {
  geminiToOpenAIStreamingChunk,
  geminiToOpenAINonStreaming,
  anthropicToOpenAIStreamingChunk,
  anthropicToOpenAINonStreaming,
  buildRequestBody,
  getRequestHeaders,
};