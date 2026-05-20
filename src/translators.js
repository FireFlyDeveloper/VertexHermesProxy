const { log } = require('./config');

function geminiToOpenAIStreamingChunk(vertexChunk, id, model, isFirst) {
  const candidate = vertexChunk.candidates?.[0];
  if (!candidate) return null;

  const parts = candidate.content?.parts || [];
  let delta = {};
  let toolCalls = [];

  if (isFirst) delta.role = 'assistant';

  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];

    if (part.text != null) {
      delta.content = part.text;
    }

    if (part.functionCall) {
      const fc = part.functionCall;
      const toolCall = {
        index: i,
        id: `call_${Date.now().toString(36)}${Math.random().toString(36).substring(2, 10)}`,
        type: 'function',
        function: {
          name: fc.name,
          arguments: JSON.stringify(fc.args || {})
        }
      };

      // CRITICAL: Capture thought_signature from Gemini response
      if (part.thoughtSignature) {
        // Store in extra_content as required by OpenAI compatibility
        toolCall.extra_content = {
          google: {
            thought_signature: part.thoughtSignature
          }
        };
        // Also store directly for easier access
        toolCall.thought_signature = part.thoughtSignature;
      }

      toolCalls.push(toolCall);
    }
  }

  if (toolCalls.length > 0) {
    delta.tool_calls = toolCalls;
  }

  // Determine finish reason
  let finishReason = null;
  if (candidate.finishReason) {
    const finishReasonMap = {
      'STOP': 'stop',
      'MAX_TOKENS': 'length',
      'SAFETY': 'content_filter',
      'RECITATION': 'content_filter',
      'OTHER': 'stop'
    };
    finishReason = finishReasonMap[candidate.finishReason] || 'stop';

    // If there are tool calls and finish reason is stop, it should be tool_calls
    if (toolCalls.length > 0 && finishReason === 'stop') {
      finishReason = 'tool_calls';
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
      finish_reason: finishReason,
      logprobs: null
    }]
  };
}

function geminiToOpenAINonStreaming(events, id, model) {
  let fullText = '';
  const toolCalls = [];
  let finishReason = 'stop';

  for (const event of events) {
    if (event === '[DONE]') continue;
    try {
      const chunk = JSON.parse(event);
      const candidate = chunk.candidates?.[0];
      if (!candidate) continue;

      const parts = candidate.content?.parts || [];
      for (const part of parts) {
        if (part.text != null) {
          fullText += part.text;
        }
        if (part.functionCall) {
          const fc = part.functionCall;
          const toolCall = {
            id: `call_${Date.now().toString(36)}${Math.random().toString(36).substring(2, 10)}`,
            type: 'function',
            function: {
              name: fc.name,
              arguments: JSON.stringify(fc.args || {})
            }
          };

          // CRITICAL: Capture thought_signature
          if (part.thoughtSignature) {
            toolCall.extra_content = {
              google: {
                thought_signature: part.thoughtSignature
              }
            };
            toolCall.thought_signature = part.thoughtSignature;
          }

          toolCalls.push(toolCall);
        }
      }

      if (candidate.finishReason) {
        const finishReasonMap = {
          'STOP': 'stop',
          'MAX_TOKENS': 'length',
          'SAFETY': 'content_filter',
          'RECITATION': 'content_filter',
          'OTHER': 'stop'
        };
        finishReason = finishReasonMap[candidate.finishReason] || 'stop';

        if (toolCalls.length > 0 && finishReason === 'stop') {
          finishReason = 'tool_calls';
        }
      }
    } catch (e) {
      log('[GEMINI-PARSE-ERR]', e.message);
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

function buildGeminiRequestBody(openaiBody) {
  const contents = [];
  let systemInstruction = null;

  const messages = openaiBody.messages || [];

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];

    if (msg.role === 'system' || msg.role === 'developer') {
      systemInstruction = { parts: [{ text: msg.content }] };
      continue;
    }

    // Handle assistant messages with tool_calls (Gemini response with thought_signatures)
    if (msg.role === 'assistant' && msg.tool_calls) {
      const parts = [];

      // Add text content if present
      if (msg.content) {
        parts.push({ text: msg.content });
      }

      // Add function calls with thought signatures
      for (let j = 0; j < msg.tool_calls.length; j++) {
        const tc = msg.tool_calls[j];
        let args = {};
        if (tc.function?.arguments) {
          try {
            args = typeof tc.function.arguments === 'string'
              ? JSON.parse(tc.function.arguments)
              : tc.function.arguments;
          } catch {
            args = {};
          }
        }

        const functionCallPart = {
          functionCall: {
            name: tc.function?.name || 'unknown',
            args: args
          }
        };

        // CRITICAL: Preserve thought_signature for Gemini
        // The signature is stored in extra_content.google.thought_signature
        if (tc.extra_content?.google?.thought_signature) {
          functionCallPart.thoughtSignature = tc.extra_content.google.thought_signature;
        } else if (tc.thought_signature) {
          functionCallPart.thoughtSignature = tc.thought_signature;
        } else if (tc.thoughtSignature) {
          functionCallPart.thoughtSignature = tc.thoughtSignature;
        }

        parts.push(functionCallPart);
      }

      contents.push({ role: 'model', parts });
      continue;
    }

    // Handle tool responses
    if (msg.role === 'tool') {
      let responseData;
      try {
        responseData = typeof msg.content === 'string'
          ? JSON.parse(msg.content)
          : msg.content;
      } catch {
        responseData = { result: msg.content };
      }

      const functionResponsePart = {
        functionResponse: {
          name: msg.name || msg.tool_call_id || 'unknown',
          response: responseData
        }
      };

      contents.push({ role: 'user', parts: [functionResponsePart] });
      continue;
    }

    // Regular user/assistant messages
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
            const match = imageUrl.match(/^data:([^;]+);base64,(.+)$/);
            if (match) {
              parts.push({ inlineData: { mimeType: match[1], data: match[2] } });
            }
          } else if (imageUrl.startsWith('gs://')) {
            parts.push({ fileData: { mimeType: 'image/jpeg', fileUri: imageUrl } });
          } else {
            parts.push({ fileData: { mimeType: 'image/jpeg', fileUri: imageUrl } });
          }
        }
      }
    }

    if (parts.length > 0) {
      contents.push({ role, parts });
    }
  }

  const generationConfig = {
    temperature: openaiBody.temperature ?? 1,
    maxOutputTokens: openaiBody.max_tokens || 8192,
    topP: openaiBody.top_p ?? 0.95,
    topK: openaiBody.top_k ?? 40,
  };

  // Remove undefined values
  Object.keys(generationConfig).forEach(key => {
    if (generationConfig[key] === undefined) delete generationConfig[key];
  });

  const requestBody = {
    contents,
    generationConfig,
  };

  if (systemInstruction) {
    requestBody.systemInstruction = systemInstruction;
  }

  // Add tools if present
  if (openaiBody.tools && openaiBody.tools.length > 0) {
    requestBody.tools = openaiBody.tools.map(tool => ({
      functionDeclarations: [{
        name: tool.function.name,
        description: tool.function.description,
        parameters: tool.function.parameters
      }]
    }));
  }

  // Add tool config
  if (openaiBody.tool_choice) {
    if (openaiBody.tool_choice === 'none') {
      requestBody.toolConfig = { functionCallingConfig: { mode: 'NONE' } };
    } else if (openaiBody.tool_choice === 'auto' || openaiBody.tool_choice === 'required') {
      requestBody.toolConfig = { functionCallingConfig: { mode: 'AUTO' } };
    } else if (typeof openaiBody.tool_choice === 'object' && openaiBody.tool_choice.function) {
      requestBody.toolConfig = {
        functionCallingConfig: {
          mode: 'ANY',
          allowedFunctionNames: [openaiBody.tool_choice.function.name]
        }
      };
    }
  }

  // Add safety settings (disabled for maximum compatibility)
  requestBody.safetySettings = [
    { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'OFF' },
    { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'OFF' },
    { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'OFF' },
    { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'OFF' }
  ];

  return requestBody;
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
  if (info.provider === 'google') return buildGeminiRequestBody(openaiBody);
  if (info.provider === 'anthropic') return buildAnthropicBody(openaiBody);
  if (info.endpoint === 'publisher') return buildDeepSeekBody(openaiBody);
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