#!/usr/bin/env python3
"""C1/C4 streaming gaps under an 8K text admission or four-image burst.

Adapted from parity-v2c's image-smoke driver. Gaps are between nonempty SSE
emission deltas (a speculative emission may contain several tokens), not GPU
kernel timing. Encode windows exclude the pair straddling encode completion.
"""
import argparse
import base64
import concurrent.futures
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import threading
import time
import urllib.request
import zlib

MODEL = 'deepseek-ai/DeepSeek-V4.1-Flash'


def percentile(values, q=0.99):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(q * len(values)) - 1)]


def gaps(stamps, start, end, contained=False):
    # Encode is half-open. Never assign a gap ending after encoder completion
    # to the encoder: that gap includes image LM prefill in the old scheduler.
    pairs = zip(stamps, stamps[1:])
    return [(b - a) * 1000 for a, b in pairs
            if (start <= a < b < end if contained else b >= start and a <= end)]


def summarize(values):
    return {'max_ms': max(values, default=None), 'p99_ms': percentile(values),
            'count': len(values)}


def stream(url, body, notify=None, api_key=None):
    request = urllib.request.Request(url + '/v1/chat/completions',
        data=json.dumps(dict(body, stream=True, stream_options={'include_usage': True})).encode(),
        headers={'Content-Type': 'application/json',
                 **({'Authorization': 'Bearer ' + api_key} if api_key else {})})
    started = time.time()
    stamps, chunks, usage = [], [], None
    with urllib.request.urlopen(request, timeout=900) as response:
        for line in response:
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                break
            event = json.loads(data)
            if event.get('error'):
                raise RuntimeError(event['error'])
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                text = delta.get('content', '') + delta.get('reasoning_content', '')
                if text:
                    stamps.append(time.time())
                    chunks.append(text)
                    if notify and len(stamps) == 12:
                        notify()
    text = ''.join(chunks)
    return {'started': started, 'finished': time.time(), 'stamps': stamps,
            'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest(), 'usage': usage,
            'ttft_ms': (stamps[0] - started) * 1000 if stamps else None}


def text_body(index, budget, label='decode-share'):
    return {'model': MODEL, 'messages': [{'role': 'user', 'content':
        f'{label} stream {index}. Write a complete Python AVL balanced binary search tree '
        'implementation with insertion deletion traversal and validation. Include exactly '
        '100 fully implemented, independently named unittest methods covering rotations, '
        'deletion edge cases, fuzz tests and differential checks. Include all code, no prose, '
        'and do not abbreviate or omit any test.'}], 'temperature': 0,
        'thinking': {'type': 'disabled'}, 'max_tokens': budget}


def png(side, seed):
    def chunk(kind, body):
        return struct.pack('>I', len(body)) + kind + body + struct.pack('>I', zlib.crc32(kind + body))
    raw = (b'\0' + bytes([seed]) * (side * 3)) * side
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', side, side, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')


def image_body(index, budget=16):
    return {'model': MODEL, 'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': f'Decode share burst image {index}. What color is this image? One word.'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png(1344, index)).decode()}}
    ]}], 'thinking': {'type': 'disabled'}, 'temperature': 0, 'max_tokens': budget}


def strip_ansi(logs):
    return re.sub(r'\x1b\[[0-9;]*m', '', logs)


def completed_prefill_wave(logs):
    return next((line for line in strip_ansi(logs).splitlines()
                 if 'V4.1 prefill wave drained' in line and 'complete=false' in line), None)


def lifecycle(url, container, tokenizer_path, api_key, require_wave=False):
    spec = importlib.util.spec_from_file_location('native_api',
        Path(__file__).resolve().parents[2] / 'qualify-ds41-native-api.py')
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    ids = tokenizer.encode(Path(__file__).read_text() * 64, add_special_tokens=False).ids
    long = api.payload('Lifecycle canceled prefill. ' +
        tokenizer.decode(ids[:8000], skip_special_tokens=False), True)
    long['max_tokens'] = 512
    ready = [threading.Event() for _ in range(4)]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        streams = [pool.submit(stream, url, text_body(i, 512, 'lifecycle-fixed'),
                               ready[i].set, api_key) for i in range(4)]
        deadline = time.monotonic() + 120
        for signal in ready:
            assert signal.wait(max(0, deadline - time.monotonic())), 'lifecycle streams not ready'
        started = time.monotonic()
        cancel_started_wall = time.time()
        cancel_wave = None
        with api.open_request(url, long, api_key=api_key):
            cutoff = time.monotonic() + 5
            while time.monotonic() < cutoff:
                logs = subprocess.check_output(['docker', 'logs', '--since',
                    f'{cancel_started_wall:.9f}', container], stderr=subprocess.STDOUT,
                    text=True, timeout=10)
                cancel_wave = completed_prefill_wave(logs)
                if cancel_wave is not None:
                    break
                time.sleep(0.025)
        closed = time.monotonic()
        closed_wall = time.time()
        results = [future.result(timeout=180) for future in streams]
    assert all(r['stamps'] and r['finished'] >= closed_wall for r in results), \
        'peer streams did not survive cancellation'
    simple = api.payload('What is 2 + 2? Answer with just the number.')
    with api.open_request(url, simple, api_key=api_key) as response:
        recovery = json.load(response)
    assert recovery['choices'][0]['message']['content'].strip() == '4'
    continued = api.payload('Lifecycle prefix fixed. Write a complete Python AVL tree with insertion and deletion.', True)
    first = api.stream_case(url, continued, api_key=api_key)
    again = api.stream_case(url, continued, api_key=api_key)
    assert again['usage']['prompt_cache_hit_tokens'] == again['usage']['prompt_tokens'], again['usage']
    continued['messages'].append({'role': 'assistant', 'content': first['text']})
    continued['messages'].append({'role': 'user', 'content': 'Add a validation method. Output code only.'})
    turn = api.stream_case(url, continued, api_key=api_key)
    assert turn['text'] and turn['usage']['prompt_cache_hit_tokens'] > 0, turn['usage']
    logs = strip_ansi(subprocess.check_output(['docker', 'logs', '--since',
        f'{cancel_started_wall:.9f}', container], stderr=subprocess.STDOUT, text=True, timeout=10))
    waves = [line for line in logs.splitlines() if completed_prefill_wave(line)]
    observed = cancel_wave is not None
    return {'cancel_before_first_delta_s': closed - started, 'surviving_streams': results,
            'canceled_prefill_completed_wave_observed': observed,
            'wave_observed_before_close': cancel_wave, 'completed_wave_rows': waves,
            'post_cancel_recovery': recovery, 'prefix_first': first, 'prefix_restore': again,
            'prefix_continuation': turn, 'passed': observed or not require_wave,
            'scope': 'cancel before first delta; four peers survive; restore and continuation use retained prefix'}


def encoder_windows(logs, start, end):
    logs = strip_ansi(logs)
    windows = []
    prefill = []
    for line in logs.splitlines():
        if 'V4.1 prefill wave starts' in line:
            match = re.search(r'started_unix_ms=([0-9.]+)', line)
            if match:
                prefill.append(float(match[1]) / 1000)
        if 'V4.1 asynchronous image encoder roundtrip' not in line:
            continue
        fields = dict(re.findall(r'(tokens|started_unix_ms|finished_unix_ms|owner_ms|roundtrip_ms)=([0-9.]+)', line))
        if 'started_unix_ms' not in fields:
            continue
        a, b = [float(fields[k]) / 1000 for k in ('started_unix_ms', 'finished_unix_ms')]
        if a >= start and b <= end:
            windows.append(dict(start=a, end=b, telemetry=fields))
    # Another request can enter LM prefill while a later image is encoding.
    # Cut there too: isolated encoder interference cannot include any LM wave.
    for window in windows:
        window['end'] = min(window['end'], min((p for p in prefill if p >= window['start']), default=window['end']))
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--container', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--api-key-file', type=Path, help='Read bearer key without putting it in argv or evidence')
    parser.add_argument('--case', choices=['text', 'images', 'lifecycle'], required=True)
    parser.add_argument('--require-completed-wave', action='store_true',
                        help='Fail lifecycle qualification unless cancellation follows a drained shared wave')
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--context-file', type=Path,
                        help='Freeze the 8K text source across benchmark revisions')
    parser.add_argument('--budget', type=int, default=2048)
    parser.add_argument('--concurrency', type=int, choices=[1, 4], default=4)
    parser.add_argument('--warm', action='store_true')
    parser.add_argument('--prompt-label', default='decode-share',
                        help='Use the same label across arms and a fresh label for each measured case')
    args = parser.parse_args()
    api_key = args.api_key_file.read_text().strip() if args.api_key_file else None
    if args.case == 'lifecycle':
        if args.tokenizer is None:
            parser.error('--tokenizer required for lifecycle qualification')
        report = lifecycle(args.url, args.container, args.tokenizer, api_key,
                           require_wave=args.require_completed_wave)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'passed': report['passed'],
                          'completed_wave_observed': report['canceled_prefill_completed_wave_observed']}), flush=True)
        assert report['passed'], 'cancellation did not follow a completed shared wave'
        return
    def request(body, notify=None):
        return stream(args.url, body, notify, api_key=api_key)
    injected = [image_body(i) for i in range(4)]
    if args.case == 'text':
        if args.tokenizer is None:
            parser.error('--tokenizer required for an 8K text prompt')
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        context = (args.context_file or Path(__file__)).read_text() * 64
        ids = tokenizer.encode(context, add_special_tokens=False).ids
        assert len(ids) >= 8000
        injected = [{'model': MODEL, 'messages': [{'role': 'user', 'content':
            tokenizer.decode(ids[:8000], skip_special_tokens=False) + '\nIgnore the code above. Count from 1 to 20, separated by commas. Output only the numbers.'}],
            'temperature': 0, 'thinking': {'type': 'disabled'}, 'max_tokens': 64}]
    for body in injected:
        content = body['messages'][0]['content']
        if isinstance(content, str):
            body['messages'][0]['content'] = args.prompt_label + '. ' + content
        else:
            content[0]['text'] = args.prompt_label + '. ' + content[0]['text']
    if args.warm:
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            list(pool.map(lambda i: request(text_body(i + 20, 128, args.prompt_label + '-warm')), range(4)))
        warm = json.loads(json.dumps(injected[0]))
        content = warm['messages'][0]['content']
        if isinstance(content, str):
            warm['messages'][0]['content'] = 'Warm shape only. ' + content
        else:
            content[0]['text'] = 'Warm shape only. ' + content[0]['text']
            # Keep the measured image out of the embedding cache as well.
            content[1]['image_url']['url'] = 'data:image/png;base64,' + base64.b64encode(png(1344, 9)).decode()
        request(warm)
    events = [threading.Event() for _ in range(args.concurrency)]
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        decoding = [pool.submit(request, text_body(i, args.budget, args.prompt_label), event.set)
                    for i, event in enumerate(events)]
        for event in events:
            assert event.wait(120), 'text did not reach injection point'
        start = time.time()
        prefilling = [pool.submit(request, body) for body in injected]
        prompts = [future.result() for future in prefilling]
        end = time.time()
        texts = [future.result() for future in decoding]
    logs = subprocess.check_output(['docker', 'logs', '--since', f'{start:.9f}', args.container],
                                   stderr=subprocess.STDOUT, text=True)
    logs = strip_ansi(logs)
    windows = encoder_windows(logs, start, end) if args.case == 'images' else []
    if args.case == 'images':
        assert len(windows) == 4, windows
    metrics = []
    for result in texts:
        stamps = result['stamps']
        assert stamps and stamps[-1] >= max(p['stamps'][0] for p in prompts), 'text ended before injected prefills'
        encode = [gap for w in windows for gap in gaps(stamps, w['start'], w['end'], contained=True)]
        prefill_start = min((w['end'] for w in windows), default=start)
        metrics.append({'normal': summarize(gaps(stamps, result['started'], start, contained=True)),
                        'burst': summarize(gaps(stamps, start, end)),
                        'prefill': summarize(gaps(stamps, prefill_start, end)),
                        'encode': summarize(encode)})
    waves = [float(value) for value in re.findall(r'wave_ms=([0-9.]+)', logs)]
    report = {'case': args.case, 'concurrency': args.concurrency, 'burst_start': start, 'burst_end': end,
              'text_streams': texts, 'injected': prompts, 'metrics': metrics,
              'encode_windows': windows, 'prefill_wave_ms': summarize(waves),
              'scope': 'nonempty SSE emission gaps; encoder-only pairs are wholly inside its half-open window',
              'passed': all(p['stamps'] for p in prompts)}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('case', 'metrics', 'prefill_wave_ms', 'passed')}), flush=True)


if __name__ == '__main__':
    main()
