#!/usr/bin/env python3
"""Validate and seal original-quality fidelity logits, without an engine run."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(1 << 20):
            h.update(data)
    return h.hexdigest()


def lse(values):
    peak = np.max(values)
    return peak + np.log(np.exp(values - peak).sum(dtype=np.float64))


def validate_reference_only(root, *, seal=False):
    """Validate an oracle dataset without implying engine parity or calibration."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python/reference'))
    from fidelity_windows import load_set, validate_qualification, canonical, validate_text_source
    root = root.resolve()
    golden = root / 'golden'
    manifest = load_set(golden / 'windows.json')
    validate_text_source(manifest)
    meta = json.loads((golden / 'meta.json').read_text())
    for key in ('family', 'checkpoint', 'set_sha256'):
        if meta.get(key) != manifest[key]:
            raise ValueError(f'golden/set {key} mismatch')
    identity = meta['snapshot_identity']
    if any(identity.get(key) != value for key, value in manifest['reference_snapshot'].items()):
        raise ValueError('golden snapshot identity mismatch')
    proof = meta.get('prefix_qualification')
    validate_qualification(proof, manifest, identity)
    probes = list(golden.glob('prefix-gate-*/meta.json'))
    if len(probes) != 1:
        raise ValueError('reference requires one raw prefix probe')
    probe_root = probes[0].parent
    probe_meta = json.loads(probes[0].read_text())
    probe_set = load_set(probe_root / 'windows.json')
    source = next((w for w in manifest['windows'] if w['id'] == proof.get('source_window')), None)
    if (probe_meta.get('set_sha256') != probe_set['set_sha256']
            or probe_meta.get('family') != manifest['family'] or probe_meta.get('checkpoint') != manifest['checkpoint']):
        raise ValueError('prefix probe metadata mismatch')
    if source is None or probe_meta.get('snapshot_identity') != identity:
        raise ValueError('prefix source/identity mismatch')
    from fidelity_windows import prefix_comparison
    arrays = []
    if len(probe_set['windows']) != 2 or len(probe_meta['windows']) != 2:
        raise ValueError('prefix probe coverage mismatch')
    for i, (window, entry) in enumerate(zip(probe_set['windows'], probe_meta['windows'])):
        size = proof['lengths'][i]
        if window['tokens'] != source['tokens'][:size] or entry['positions'] != list(range(proof['score_from'], size)):
            raise ValueError('prefix probe pinning mismatch')
        folder = (probe_root / entry['path']).resolve()
        if not folder.is_relative_to(probe_root.resolve()):
            raise ValueError('prefix probe path escapes root')
        if np.fromfile(folder / 'tokens.bin', dtype='<i4').tolist() != window['tokens']:
            raise ValueError('prefix probe tokens mismatch')
        path = folder / 'logits.bin'
        if digest(path) != proof['logits_sha256'][i] or path.stat().st_size != (size - proof['score_from']) * proof['vocab'] * 4:
            raise ValueError('prefix probe hash/extent mismatch')
        arrays.append(np.memmap(path, dtype='<f4', mode='r', shape=(size - proof['score_from'], proof['vocab'])))
    observed = prefix_comparison(*arrays)
    if not observed['passed'] or any(observed[k] != proof[k] for k in observed):
        raise ValueError('raw prefix qualification failed')
    if meta.get('mtp') is not False or meta.get('use_cache') is not False or meta.get('attention') != 'eager':
        raise ValueError('reference execution policy differs')
    entries = {w['id']: w for w in meta['windows']}
    expected = {w['id'] for w in manifest['windows']}
    if len(entries) != len(meta['windows']) or set(entries) != expected:
        raise ValueError('golden window coverage mismatch')
    rows = json.loads((root / 'full/rows.json').read_text())
    compact = json.loads((root / 'reference.json').read_text())
    if (rows.get('schema') != 'cuteafd.fidelity.rows/1' or rows.get('dtype') != '<f2'
            or rows.get('kind') != 'log_softmax' or compact.get('schema') != 'cuteafd.fidelity.reference/2'):
        raise ValueError('converted reference format mismatch')
    for document in (rows, compact):
        if any(document.get(k) != manifest[k] for k in ('family', 'checkpoint', 'set_sha256')):
            raise ValueError('converted reference identity mismatch')
    if (compact.get('quick_windows') != manifest['quick_windows']
            or compact.get('source', {}).get('snapshot_identity') != identity
            or compact.get('source', {}).get('prefix_qualification') != proof):
        raise ValueError('compact reference provenance mismatch')
    row_entries = {w['id']: w for w in rows['windows']}
    compact_entries = {w['id']: w for w in compact['windows']}
    if (len(row_entries) != len(rows['windows']) or len(compact_entries) != len(compact['windows'])
            or set(row_entries) != expected or set(compact_entries) != expected):
        raise ValueError('converted reference coverage mismatch')
    def contained(base, name):
        path = (base / name).resolve()
        if not path.is_relative_to(base.resolve()):
            raise ValueError('dataset path escapes its root')
        return path
    total, count = 0.0, 0
    for window in manifest['windows']:
        entry, full, summary = entries[window['id']], row_entries[window['id']], compact_entries[window['id']]
        positions = list(range(window['score_from'], len(window['tokens'])))
        folder = contained(golden, entry['path'])
        vocab = entry['vocab']
        if entry['positions'] != positions or vocab != rows['vocab'] or vocab != compact['vocab']:
            raise ValueError('window geometry mismatch')
        if np.fromfile(folder / 'tokens.bin', dtype='<i4').tolist() != window['tokens']:
            raise ValueError('golden tokens differ from pinned windows')
        if json.loads((folder / 'meta.json').read_text()) != entry:
            raise ValueError('per-window meta mismatch')
        if full['shape'] != [len(positions), vocab] or full['positions'] != positions:
            raise ValueError('full row shape mismatch')
        full_path = contained(root / 'full', full['path'])
        if digest(full_path) != full['sha256'] or full_path.stat().st_size != len(positions) * vocab * 2:
            raise ValueError('full row file hash/extent mismatch')
        if (folder / 'logits.bin').stat().st_size != len(positions) * vocab * 4:
            raise ValueError('golden logits extent mismatch')
        logits = np.memmap(folder / 'logits.bin', dtype='<f4', mode='r', shape=(len(positions), vocab))
        stored = np.memmap(full_path, dtype='<f2', mode='r', shape=logits.shape)
        if any(summary.get(k) != window[k] for k in ('tokens', 'roles', 'score_from', 'block', 'bucket')):
            raise ValueError('compact window pinning mismatch')
        if len(summary['positions']) != len(positions):
            raise ValueError('compact position coverage mismatch')
        loss = 0.0
        for i, (row, pos) in enumerate(zip(logits, positions)):
            values = row.astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError('non-finite golden logits')
            logp = values - lse(values)
            if not np.array_equal(logp.astype('<f2'), stored[i]):
                raise ValueError('full rows differ from golden log-softmax')
            record = summary['positions'][i]
            top = record['top']
            k = len(top)
            if not 1 <= k <= vocab or k != summary['top_k']:
                raise ValueError('invalid compact top-k')
            threshold = np.partition(values, vocab - k)[vocab - k]
            better = np.flatnonzero(values > threshold)
            order = np.concatenate((better, np.flatnonzero(values == threshold)[:k - len(better)]))
            order = order[np.lexsort((order, -values[order]))].tolist()
            if [item['id'] for item in top] != order or any(abs(item['lp'] - logp[item['id']]) > 0.000006 for item in top):
                raise ValueError('compact top probabilities mismatch')
            tail = math.log(max(1.0 - float(np.exp(logp[order]).sum()), 1e-30))
            if abs(record['tail_lp'] - tail) > 0.000006:
                raise ValueError('compact tail probability mismatch')
            target = window['tokens'][pos]
            if record['pos'] != pos or record['next'] != target or abs(record['next_lp'] - logp[target]) > 0.000006:
                raise ValueError('compact next-token probability mismatch')
            loss -= float(logp[target])
        if not math.isclose(entry['nll'], loss / len(positions), abs_tol=1e-8):
            raise ValueError('window NLL mismatch')
        total += loss
        count += len(positions)
    if count != meta['scored_tokens'] or not math.isclose(meta['nll'], total / count, abs_tol=1e-8):
        raise ValueError('aggregate NLL mismatch')
    excluded = {'full.sha256.json', 'golden-seal.json', 'reference-validation.json'}
    def inventory(base):
        paths = [p for p in sorted(base.rglob('*')) if p.is_file() and p.name not in excluded]
        if any(not p.resolve().is_relative_to(base.resolve()) for p in paths):
            raise ValueError('sealed file escapes dataset root')
        return [{'path': str(p.relative_to(base)), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in paths]
    if seal:
        for name in ('full.sha256.json', 'golden-seal.json'):
            if (root / name).exists():
                raise ValueError('refusing to overwrite an immutable seal')
        (root / 'full.sha256.json').write_bytes(canonical({'schema': 'cuteafd.fidelity.full-seal/1',
            'set_sha256': manifest['set_sha256'], 'files': inventory(root / 'full')}) + b'\n')
        (root / 'golden-seal.json').write_bytes(canonical({'schema': 'cuteafd.fidelity.golden-seal/1',
            'set_sha256': manifest['set_sha256'], 'snapshot_identity': identity,
            'full_seal_sha256': digest(root / 'full.sha256.json'), 'files': inventory(root)}) + b'\n')
    full_seal = json.loads((root / 'full.sha256.json').read_text())
    golden_seal = json.loads((root / 'golden-seal.json').read_text())
    if (full_seal['set_sha256'] != manifest['set_sha256'] or full_seal['files'] != inventory(root / 'full')
            or golden_seal['set_sha256'] != manifest['set_sha256'] or golden_seal['snapshot_identity'] != identity
            or golden_seal['full_seal_sha256'] != digest(root / 'full.sha256.json')
            or golden_seal['files'] != inventory(root)):
        raise ValueError('dataset seal mismatch')
    return dict(schema='cuteafd.fidelity.reference-validation/1', passed=True, engine_parity=False,
                windows=len(entries), scored_tokens=count, nll=total / count,
                set_sha256=manifest['set_sha256'], golden_seal_sha256=digest(root / 'golden-seal.json'),
                full_seal_sha256=digest(root / 'full.sha256.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='root with golden/, full/, reference.json')
    parser.add_argument('--seal', action='store_true', help='create immutable seals after checking the reference')
    args = parser.parse_args()
    result = validate_reference_only(args.source, seal=args.seal)
    (args.source / 'reference-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
