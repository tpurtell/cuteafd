"""Placement-input publications: a checkout's inputs fingerprint and the publication that matches it.

The Rust consumers (cuteafd_loader::plan::inputs) accept a publication only when
its recorded sparkinfer_revision, source_tree_sha256, engine_sources_sha256 (the
compiled tier) and selection_sources_sha256 (what installed tests select and
admit; absent in publications made before the two tiers) equal what the
checkout computes. Those values are the publication's inputs key. This module computes a checkout's key with that checkout's own
exporter code, and finds a publication in a store that records the same key,
so a gate validates against exactly the compiled facts of its own checkout.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat

STORES = {'all': 'placement-inputs', 'v4': 'v4-placement-inputs'}
FINGERPRINT = ('sparkinfer_revision', 'source_tree_sha256', 'engine_sources_sha256', 'selection_sources_sha256')
COMPILED = FINGERPRINT[:3]
# The fingerprint fields precede "exports" (KB into a ~100 MB INPUTS.json).
HEAD_BYTES = 1 << 20


def inputs_key(fingerprint):
    """Canonical digest of a fingerprint; equal keys mean equal fingerprints."""
    canonical = json.dumps({key: fingerprint.get(key) for key in FINGERPRINT}, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()


def checkout_fingerprints(root):
    """{'all': ..., 'v4': ...} computed by the checkout's own exporter source hashing.

    Runs the checkout's scripts/build/export-placement-inputs.py source_hashes()
    (and, for checkouts before the two tiers, compare-sm-exports.py
    v4_placement_sources()), the functions the Rust validator mirrors, against
    root. Standard library only; no CUDA, no fork import.
    """
    root = Path(root).resolve()
    previous = os.environ.get('CUTEAFD_SOURCE_ROOT')
    os.environ['CUTEAFD_SOURCE_ROOT'] = str(root)  # compare-sm-exports.py's ROOT
    try:
        spec = importlib.util.spec_from_file_location(f'placement_publisher_{os.getpid()}_{id(root)}',
                                                      root / 'scripts/build/export-placement-inputs.py')
        publisher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(publisher)
        if Path(publisher.ROOT).resolve() != root or Path(publisher.compare.ROOT).resolve() != root:
            raise ValueError(f'exporter resolved a different checkout than {root}')
        lock = json.loads((root / 'third_party/sparkinfer.lock.json').read_text())
        pin = {'sparkinfer_revision': lock['revision'], 'source_tree_sha256': lock['source_tree_sha256']}
        if hasattr(publisher, 'compiled_hashes'):
            return {kind: {**pin, **publisher.source_hashes(kind)} for kind in STORES}
        return {'all': {**pin, 'engine_sources_sha256': publisher.source_hashes()},
                'v4': {**pin, 'engine_sources_sha256': publisher.compare.v4_placement_sources()}}
    finally:
        if previous is None:
            os.environ.pop('CUTEAFD_SOURCE_ROOT', None)
        else:
            os.environ['CUTEAFD_SOURCE_ROOT'] = previous


def recorded_fingerprint(publication):
    """The fingerprint a publication's INPUTS.json records, read from its head; None if unreadable."""
    try:
        with open(Path(publication) / 'INPUTS.json', 'rb') as stream:
            head = stream.read(HEAD_BYTES).decode('utf-8', errors='replace')
    except OSError:
        return None
    # json.dumps(indent=2) writes these fields before "exports"; parse the prefix as an object.
    cut = head.find('\n  "exports":')
    try:
        data = json.loads(head[:cut].rstrip().rstrip(',') + '\n}') if cut > 0 else json.loads(head)
    except ValueError:
        return None
    if not isinstance(data, dict) or not all(key in data for key in COMPILED):
        return None
    return {'engine_commit': data.get('engine_commit'), **{key: data.get(key) for key in FINGERPRINT}}


def publications(store):
    """Real publication directories directly in a store, newest first."""
    store = Path(store)
    found = []
    try:
        entries = list(store.iterdir())
    except OSError:
        return found
    for entry in entries:
        try:
            info = entry.lstat()
            inputs = (entry / 'INPUTS.json').lstat()
        except OSError:
            continue
        if stat.S_ISDIR(info.st_mode) and stat.S_ISREG(inputs.st_mode) and not entry.name.startswith('.'):
            found.append((inputs.st_mtime, entry))
    return [entry for _, entry in sorted(found, reverse=True)]


def resolve(store, fingerprint):
    """The publication in store whose recorded inputs key equals fingerprint's, preferring current.

    Returns (path, how) or (None, None). The validator still rechecks every
    field and checksum; this only chooses which publication to hand it.
    """
    store = Path(store)
    key = inputs_key(fingerprint)
    current = store / 'current'
    candidates = []
    if current.is_symlink():
        target = Path(os.path.realpath(current))
        if target.parent == Path(os.path.realpath(store)):
            candidates.append((target, 'current'))
    candidates += [(path, 'matching publication') for path in publications(store)]
    for path, how in candidates:
        recorded = recorded_fingerprint(path)
        if recorded is not None and inputs_key(recorded) == key:
            return path, how
    return None, None


def differing(expected, recorded):
    """Fingerprint fields and source paths that differ, for refusal messages."""
    if recorded is None:
        return ['unreadable INPUTS.json']
    fields = [key for key in FINGERPRINT[:2] if expected.get(key) != recorded.get(key)]
    paths = []
    for tier in FINGERPRINT[2:]:
        ours, theirs = expected.get(tier) or {}, recorded.get(tier) or {}
        if expected.get(tier) is None or recorded.get(tier) is None:
            if expected.get(tier) != recorded.get(tier):
                fields.append(tier)
            continue
        paths += sorted(path for path in ours.keys() | theirs.keys() if ours.get(path) != theirs.get(path))
    return fields + paths


def compiled_matches(expected, recorded):
    """Only the selection tier differs: a re-attested publication suffices (no compile)."""
    return recorded is not None and all(expected.get(key) == recorded.get(key) for key in COMPILED)


def resolve_compiled(store, fingerprint):
    """A publication whose compiled tier matches (a --reattest-from source), preferring current."""
    store = Path(store)
    current = store / 'current'
    candidates = [Path(os.path.realpath(current))] if current.is_symlink() else []
    for path in [*candidates, *publications(store)]:
        if compiled_matches(fingerprint, recorded_fingerprint(path)):
            return path
    return None
