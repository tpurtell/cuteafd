#!/usr/bin/env node
/** README: official openai-node GA Realtime headless gate (7.31.0, Apache-2.0).
 * npm install --prefix "$SCRATCH/node" openai@7.31.0 @openai/agents-realtime@0.20.0
 * CUTEAFD_RT_NODE_ROOT="$SCRATCH/node" CUTEAFD_GATEWAY_KEY=<local-key> node
 * openai-node.mjs --url wss://localhost:8443/v1/realtime --model default
 * [--skip-audio] [--timeout 30] [--key NAME_OF_ENV_VAR] [--native]
 * --native selects OpenAIRealtimeWebSocket (browser-style subprotocol auth).
 * This current SDK enforces wss, including on custom endpoints. Use a trusted
 * local TLS certificate (NODE_EXTRA_CA_CERTS for a local test CA), not disabled
 * certificate checks. No microphone, ephemeral-key request, or hosted call.
 */
import { bounded, dependency, inbox, scenario, sessionConfig } from './common.mjs';

await bounded('openai-node', async o => {
  if (!o.url.startsWith('wss:')) throw new Error('openai-node requires wss');
  const OpenAI = dependency('openai').OpenAI;
  const { OpenAIRealtimeWS } = dependency('openai/realtime/ws');
  const { OpenAIRealtimeWebSocket } = dependency('openai/realtime/websocket');
  const Transport = o.native ? OpenAIRealtimeWebSocket : OpenAIRealtimeWS;
  const client = new OpenAI({ apiKey: o.key, baseURL: o.url.replace('wss:', 'https:').replace(/\/realtime$/, '') });
  const rt = new Transport({ model: o.model, buildRealtimeURL: () => {
    const u = new URL(o.url); u.searchParams.set('model', o.model); return u;
  } }, client);
  const received = inbox(o.key);
  rt.on('event', e => received.push(e));
  rt.on('error', () => received.fail());
  if (o.native) rt.socket.addEventListener('close', () => received.fail());
  else rt.socket.on('close', () => received.fail());
  try {
    await received.until('session.created');
    rt.send({ type: 'session.update', session: sessionConfig });
    await received.until('session.updated');
    await scenario(e => rt.send(e), received, o);
  } finally {
    rt.close();
  }
});
