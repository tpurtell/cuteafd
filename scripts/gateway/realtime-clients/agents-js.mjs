#!/usr/bin/env node
/** README: real @openai/agents-realtime RealtimeAgent/RealtimeSession gate.
 * Install: npm install --prefix "$SCRATCH/node" openai@7.31.0 @openai/agents-realtime@0.20.0
 * Run: CUTEAFD_RT_NODE_ROOT="$SCRATCH/node" CUTEAFD_GATEWAY_KEY=<local-key> node
 * agents-js.mjs --url ws://127.0.0.1:8080/v1/realtime --model default
 * [--skip-audio] [--timeout 30] [--key NAME_OF_ENV_VAR]
 * MIT. WebSocket transport, no microphone/speaker or ephemeral key. Tracing is
 * explicitly disabled; tools are executed by RealtimeSession, not our runner.
 */
import { bounded, dependency, inbox, scenario, timeResult } from './common.mjs';

await bounded('agents-js', async o => {
  const { RealtimeAgent, RealtimeSession, tool } = dependency('@openai/agents-realtime');
  let executed = 0;
  const agent = new RealtimeAgent({ name: 'Local gate', instructions: 'Answer briefly. Call get_time when requested.',
    tools: [tool({ name: 'get_time', description: 'Return a fixed test time.',
      parameters: { type: 'object', properties: {}, additionalProperties: false },
      execute: async () => { ++executed; return timeResult; } })] });
  const session = new RealtimeSession(agent, { transport: 'websocket', model: o.model, tracingDisabled: true,
    config: { outputModalities: ['text'], audio: { input: {
      format: { type: 'audio/pcm', rate: 24000 }, transcription: null, turnDetection: null } } } });
  const received = inbox(o.key);
  session.on('transport_event', e => received.push(e));
  session.on('error', () => received.fail());
  session.transport.on('connection_change', state => { if (state === 'disconnected') received.fail(); });
  try {
    const url = new URL(o.url); url.searchParams.set('model', o.model);
    await session.connect({ apiKey: o.key, url: url.toString(), model: o.model });
    await received.until('session.updated');
    await scenario(e => session.transport.sendEvent(e), received,
      { ...o, automaticTools: true, toolCalls: () => executed });
  } finally {
    session.close();
  }
});
