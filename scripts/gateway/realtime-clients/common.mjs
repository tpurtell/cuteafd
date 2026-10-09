// Shared headless gate, not a replacement transport: SDKs own the sockets.
import { createRequire } from 'node:module';
import { resolve } from 'node:path';

export function dependency(name) {
  const root = process.env.CUTEAFD_RT_NODE_ROOT;
  if (!root) throw new Error('Set CUTEAFD_RT_NODE_ROOT to the scratch npm directory');
  return createRequire(resolve(root, 'package.json'))(name);
}

export function options() {
  const o = { model: 'default', keyEnv: 'CUTEAFD_GATEWAY_KEY', timeout: 30, audio: true };
  const fields = { '--url': 'url', '--model': 'model', '--key': 'keyEnv', '--key-env': 'keyEnv', '--timeout': 'timeout' };
  for (let i = 2; i < process.argv.length; ++i) {
    const arg = process.argv[i];
    if (arg === '--skip-audio') o.audio = false;
    else if (arg === '--native') o.native = true;
    else if (fields[arg] && process.argv[i + 1]) o[fields[arg]] = process.argv[++i];
    else throw new Error('Unknown or missing argument');
  }
  const u = new URL(o.url);
  if (!['ws:', 'wss:'].includes(u.protocol) || u.username || u.password || u.search || u.hash ||
      u.pathname !== '/v1/realtime' || u.hostname === 'openai.com' || u.hostname.endsWith('.openai.com')) {
    throw new Error('Use an explicit non-OpenAI ws(s)://HOST/v1/realtime URL without credentials/query');
  }
  o.key = process.env[o.keyEnv];
  if (!o.key) throw new Error('Set the credential environment variable');
  o.timeout = Number(o.timeout);
  if (!Number.isFinite(o.timeout) || o.timeout <= 0) throw new Error('Invalid timeout');
  return o;
}

export const toolDefinition = { type: 'function', name: 'get_time', description: 'Return a fixed test time.',
  parameters: { type: 'object', properties: {}, additionalProperties: false } };
export const timeResult = '{"time":"2000-01-01T00:00:00Z"}';
export const sessionConfig = { type: 'realtime', output_modalities: ['text'],
  audio: { input: { format: { type: 'audio/pcm', rate: 24000 }, turn_detection: null } },
  tools: [toolDefinition], tool_choice: 'auto' };

export function inbox(key) {
  const queued = [];
  let wake;
  let failure;
  return {
    fail() { failure = new Error('SDK/server protocol error'); wake?.(); },
    push(e) {
      if (!e || typeof e.type !== 'string') { this.fail(); return; }
      console.log(e.type);
      if (['response.output_text.delta', 'response.text.delta'].includes(e.type)) {
        console.log(String(e.delta ?? '').split(key).join('<redacted>'));
      }
      if (e.type === 'error' || (e.type === 'response.done' && e.response?.status !== 'completed')) {
        this.fail(); return;
      }
      queued.push(e); wake?.();
    },
    async until(type) {
      for (;;) {
        if (failure) throw failure;
        while (queued.length) {
          const e = queued.shift();
          if (e.type === type) return e;
        }
        await new Promise(r => { wake = r; });
        wake = undefined;
      }
    },
  };
}

export async function scenario(send, received, { audio, automaticTools = false, toolCalls = () => 0 }) {
  const user = text => send({ type: 'conversation.item.create', item: { type: 'message', role: 'user',
    content: [{ type: 'input_text', text }] } });
  user('Say hello in one short sentence.');
  send({ type: 'response.create', response: { tool_choice: 'none' } });
  const text = await received.until('response.done');
  if (!text.response.output?.some(i => i.type === 'message')) throw new Error('Missing text output');
  user('Call get_time now.');
  send({ type: 'response.create', response: { tool_choice: { type: 'function', name: 'get_time' } } });
  const response = await received.until('response.done');
  const calls = response.response.output?.filter(i => i.type === 'function_call') ?? [];
  if (calls.length !== 1 || calls[0].name !== 'get_time' || !calls[0].call_id) {
    throw new Error('Missing or invalid get_time call');
  }
  const argumentsObject = JSON.parse(calls[0].arguments);
  if (!argumentsObject || Array.isArray(argumentsObject) || typeof argumentsObject !== 'object' ||
      Object.keys(argumentsObject).length !== 0) throw new Error('Invalid get_time arguments');
  if (!automaticTools) {
    send({ type: 'conversation.item.create', item: { type: 'function_call_output',
      call_id: calls[0].call_id, output: timeResult } });
    send({ type: 'response.create', response: { tool_choice: 'none' } });
  }
  const final = await received.until('response.done');
  if (!final.response.output?.some(i => i.type === 'message')) throw new Error('Missing tool result response');
  if (automaticTools && toolCalls() !== 1) throw new Error('Agents SDK did not execute exactly one tool');
  if (audio) {
    send({ type: 'input_audio_buffer.append', audio: Buffer.alloc(12000).toString('base64') });
    send({ type: 'input_audio_buffer.commit' });
    await received.until('input_audio_buffer.committed');
  }
}

export async function bounded(label, run) {
  let timer;
  try {
    const o = options();
    await Promise.race([run(o), new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error('Timeout')), o.timeout * 1000);
    })]);
    console.log(`PASS ${label}`);
  } catch {
    // SDK errors may contain credentials or endpoint URLs; never dump them.
    console.error(`FAIL ${label}`);
    process.exitCode = 1;
  } finally {
    clearTimeout(timer);
    // A stalled handshake must not keep a failed gate alive indefinitely.
    if (process.exitCode) process.exit(process.exitCode);
  }
}
