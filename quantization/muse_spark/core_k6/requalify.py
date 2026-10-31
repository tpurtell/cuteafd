"""Fresh pilot covariance capture using verified native BF16 teacher states.

Only five explicitly named native hidden-state banks and the embedding witness
can be read from the old pilot. Old Hessians, expert captures, candidates and
quantized reconstructions never enter this path. All covariance observations,
expert rows, source outputs and native/staged proofs are produced again.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
import time

import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'quantization'))
from kimi_k3.uniform_k2.storage import (atomic_json, canonical, read_header,
    sha256_file, verify_receipt)
from . import numerical

SCHEMA = 'cuteafd-muse-native-reference-origin-v1'
PILOT_SCHEMA = 'cuteafd-muse-spark-core-pilot-v1'
LAYERS = (0, 1, 61)
BANKS = ('frontier-embedding', 'frontier-000', 'frontier-001', 'frontier-060', 'frontier-061')
INPUT_BANK = {0: 'frontier-embedding', 1: 'frontier-000', 61: 'frontier-060'}
NATIVE_WIDTH = 8192
NATIVE_CODE = (
    'quantization/muse_spark/core_k6/numerical.py',
    'quantization/muse_spark/core_k6/source.py',
    'quantization/muse_spark/prepare_corpus.py',
    'quantization/kimi_k3/prepare_corpus.py',
    'quantization/kimi_k3/core_k6/pilot.py',
)
CAPTURE_CODE = (
    'quantization/muse_spark/core_k6/requalify.py',
    'quantization/muse_spark/core_k6/pilot.py',
    'quantization/muse_spark/core_k6/codec.py',
)
NATIVE_FIELDS = ('schema', 'source_repo', 'source_revision', 'source_config_sha256',
    'source_index_sha256', 'token_manifest_sha256', 'native', 'records', 'witness')


def _digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('native reference metadata is missing or unsafe')
    result = json.loads(path.read_text())
    canonical(result)
    return result


def _native_path(root, relative):
    """A strict allowlist prevents accidental access to old numerical caches."""
    parts = Path(relative).parts
    if (len(parts) != 2 or parts[0] not in BANKS
            or not (re.fullmatch(r'[0-9]{3}\.safetensors', parts[1])
                or (parts[0] == 'frontier-embedding' and parts[1] == 'witness.safetensors'))):
        raise ValueError('old reference path is outside the native BF16 allowlist')
    root = Path(root)
    path = root / relative
    if root.is_symlink() or path.parent.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError('native teacher payload is missing or unsafe')
    return path


def _record_rows(identity):
    records = identity.get('records', {})
    if set(records) != {'calibration', 'heldout'}:
        raise ValueError('native reference must retain both exact corpus splits')
    rows = []
    for split in ('calibration', 'heldout'):
        if not records[split]:
            raise ValueError('empty native reference split')
        for row in records[split]:
            if (not isinstance(row.get('id'), str) or not row['id'] or type(row.get('tokens')) is not int
                    or row['tokens'] <= 0 or not re.fullmatch(r'[0-9a-f]{64}', row.get('ids_sha256', ''))):
                raise ValueError('native reference row identity is malformed')
            rows.append((split, row))
    if len({r['id'] for _, r in rows}) != len(rows):
        raise ValueError('duplicate native reference record identities')
    return rows


def _source_receipt(root, relative, old_identity, tokens):
    path = _native_path(root, relative)
    receipt_path = path.with_suffix('.receipt.json')
    receipt = _json(receipt_path)
    if receipt.get('identity') != old_identity or receipt.get('filename') != path.name:
        raise ValueError('native BF16 teacher receipt producer differs')
    verify_receipt(path, receipt, checksum=True)
    header, _ = read_header(path)
    tensors = {name: value for name, value in header.items() if name != '__metadata__'}
    if (set(tensors) != {'hidden'} or tensors['hidden']['dtype'] != 'BF16'
            or tensors['hidden']['shape'] != [1, tokens, NATIVE_WIDTH]):
        raise ValueError('old reference is not a native BF16 hidden-state bank')
    return dict(relative=relative, sha256=receipt['sha256'], file_identity=receipt['file_identity'],
        receipt_sha256=sha256_file(receipt_path), tokens=tokens)


def native_origin(oldroot, current_identity, repo=REPO):
    """Read-only audit of the native source, tokenizer, prompt IDs and banks."""
    oldroot, repo = Path(oldroot).expanduser().absolute(), Path(repo).resolve()
    if oldroot.is_symlink() or not oldroot.is_dir():
        raise ValueError('native teacher root must be an existing regular directory')
    report_path = oldroot / 'qualification.json'
    report = _json(report_path)
    old = report.get('identity', {})
    if (report.get('schema') != PILOT_SCHEMA or report.get('status') != 'passed'
            or old.get('schema') != PILOT_SCHEMA or current_identity.get('schema') != PILOT_SCHEMA
            or old.get('recipe', {}).get('native_reference') is not True):
        raise ValueError('old report does not attest a completed native BF16 reference capture')
    for field in NATIVE_FIELDS:
        if field not in old or current_identity.get(field) != old[field]:
            raise ValueError('native teacher source/tokenizer/runtime identity differs: ' + field)
    if _json(oldroot / 'identity.json') != old:
        raise ValueError('native teacher identity file differs from its report')
    code = {}
    for relative in NATIVE_CODE:
        actual = sha256_file(repo / relative)
        if old.get('code', {}).get(relative) != actual or current_identity.get('code', {}).get(relative) != actual:
            raise ValueError('native teacher adapter/source/corpus code differs: ' + relative)
        code[relative] = actual
    rows = _record_rows(old)
    witness = old['witness']
    if (type(witness.get('tokens')) is not int or witness['tokens'] != len(witness.get('input_ids', []))
            or not 2048 < witness['tokens'] <= 4096):
        raise ValueError('native long-window witness identity differs')
    current = _json(oldroot / 'capture-current.json')
    if current.get('identity') != old or current.get('layer') != 61 or current.get('directory') != 'frontier-061':
        raise ValueError('native teacher capture did not finish all 62 original BF16 layers')
    manifests = {}
    for bank in BANKS[1:]:
        path = oldroot / bank / 'manifest.json'
        manifest = _json(path)
        if (manifest.get('identity') != old or manifest.get('layer') != int(bank[-3:])
                or manifest.get('directory') != bank):
            raise ValueError('native teacher bank commitment differs')
        manifests[bank] = sha256_file(path)
    if _json(oldroot / 'frontier-061/manifest.json') != current:
        raise ValueError('final native teacher bank differs from capture-current')
    files = []
    for bank in BANKS:
        for index, (_, row) in enumerate(rows):
            files.append(_source_receipt(oldroot, f'{bank}/{index:03d}.safetensors', old, row['tokens']))
    files.append(_source_receipt(oldroot, 'frontier-embedding/witness.safetensors', old, witness['tokens']))
    native = {field: old[field] for field in NATIVE_FIELDS}
    native['code'] = code
    return dict(schema=SCHEMA, teacher_root=str(oldroot), old_report_sha256=sha256_file(report_path),
        old_native_identity_sha256=_digest(native), native_code=code, bank_manifest_sha256=manifests,
        native_files=files, policy='native-BF16-hidden-states-only; fresh-covariances-expert-rows-and-candidates')


def _device():
    return torch.device('cuda:0')


def _teacher(root, origin, relative):
    entries = [entry for entry in origin['native_files'] if entry['relative'] == relative]
    if len(entries) != 1:
        raise ValueError('native teacher file is not in the immutable origin inventory')
    entry = entries[0]
    path = _native_path(root, relative)
    receipt_path = path.with_suffix('.receipt.json')
    if sha256_file(receipt_path) != entry['receipt_sha256']:
        raise ValueError('native teacher receipt changed after origin qualification')
    verify_receipt(path, entry, checksum=True)
    tensors = load_file(str(path), device='cpu')
    if set(tensors) != {'hidden'}:
        raise ValueError('native teacher payload contains non-hidden tensors')
    hidden = tensors['hidden']
    if (hidden.dtype != torch.bfloat16 or tuple(hidden.shape) != (1, entry['tokens'], NATIVE_WIDTH)
            or not torch.isfinite(hidden).all()):
        raise ValueError('invalid native BF16 teacher state')
    return hidden, entry


def _bytes(tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8)


def _equal_native(actual, expected):
    if (actual.dtype != torch.bfloat16 or actual.shape != expected.shape
            or not torch.isfinite(actual).all() or not torch.equal(_bytes(actual), _bytes(expected))):
        raise ValueError('fresh native source output differs byte-for-byte from the BF16 teacher')


def _proof_status(proof, *, window=False):
    if proof.get('status') != 'passed' or proof.get('byte_exact') is not True:
        raise ValueError('fresh native/staged byte proof is missing')
    if window and (proof.get('tokens', 0) <= 2048 or proof.get('visible_tail_tokens') != 2048 or proof.get('hessian_rows') != 0):
        raise ValueError('fresh long sliding-window proof is missing')


def _capture_widths(source, core):
    return {name: source.config['moe_hidden_size'] if name.endswith('post_expert_proj')
        else source.config['hidden_size'] for name in core}


def _check_fresh_split(pilot, directory, split, identity, records, source):
    root = directory / split
    manifest = _json(root / 'manifest.json')
    counts = manifest.get('counts', {})
    expected_rows = sum(len(row['input_ids']) for row in records[split])
    if manifest.get('identity') != identity or set(counts) != set(pilot.CORE) or any(value != expected_rows for value in counts.values()):
        raise ValueError('fresh native Hessian manifest loses or duplicates observations')
    widths = _capture_widths(source, pilot.CORE)
    for name in pilot.CORE:
        raw = pilot.read_tensor(root / (name + '.safetensors'), identity)
        H = raw.get('H')
        if (set(raw) != {'H'} or H.dtype != torch.float32 or tuple(H.shape) != (widths[name],) * 2
                or not torch.isfinite(H).all() or not H.diagonal().sum() > 0):
            raise ValueError('fresh native covariance is malformed or nonfinite')
    experts = pilot.read_tensor(root / 'experts.safetensors', identity)
    x, selected = experts.get('inputs'), experts.get('selected')
    topk, count = source.config['num_experts_per_tok'], source.config['num_local_experts']
    if (set(experts) != {'inputs', 'selected'} or x.dtype != torch.bfloat16
            or tuple(x.shape) != (expected_rows, source.config['moe_hidden_size'])
            or selected.dtype != torch.int64 or tuple(selected.shape) != (expected_rows, topk)
            or not torch.isfinite(x).all() or (selected < 0).any() or (selected >= count).any()
            or selected.sort(dim=1).values.diff(dim=1).eq(0).any()):
        raise ValueError('fresh native natural expert observations differ')


def _check_completed(pilot, source, records, root, identity, oldroot, origin, layer, previous_proofs):
    directory = root / f'frontier-{layer:03d}'
    manifest = _json(directory / 'manifest.json')
    if (manifest.get('identity') != identity or manifest.get('layer') != layer
            or manifest.get('directory') != directory.name or manifest.get('native_reference_origin') != origin
            or manifest.get('fresh_covariances') is not True):
        raise ValueError('completed native requalification producer differs')
    batches = [(split, row) for split in ('calibration', 'heldout') for row in records[split]]
    checks = manifest.get('native_output_checks', [])
    if len(checks) != len(batches):
        raise ValueError('completed fresh source-output comparisons are incomplete')
    for index, (split, row) in enumerate(batches):
        relative = f'frontier-{layer:03d}/{index:03d}.safetensors'
        expected, entry = _teacher(oldroot, origin, relative)
        fresh = pilot.read_tensor(directory / f'{index:03d}.safetensors', identity)
        if set(fresh) != {'hidden'}:
            raise ValueError('fresh source output contains non-hidden tensors')
        _equal_native(fresh['hidden'], expected)
        proof = checks[index]
        if (proof != dict(index=index, split=split, id=row['id'], tokens=len(row['input_ids']),
                teacher_relative=relative, teacher_sha256=entry['sha256'],
                teacher_receipt_sha256=entry['receipt_sha256'], byte_exact=True,
                tensor_sha256=hashlib.sha256(_bytes(expected).numpy().tobytes()).hexdigest())):
            raise ValueError('fresh native output comparison provenance differs')
        receipt = _json((directory / f'{index:03d}.safetensors').with_suffix('.receipt.json'))
        if receipt.get('native_reference_output') != proof:
            raise ValueError('fresh source-output receipt lost its teacher provenance')
    for split in ('calibration', 'heldout'):
        _check_fresh_split(pilot, directory, split, identity, records, source)
    proofs = manifest.get('proofs', {})
    for key, proof in previous_proofs.items():
        if proofs.get(key) != proof:
            raise ValueError('fresh proof prefix changed between selected layers')
    _proof_status(proofs.get(str(layer), {}))
    if proofs[str(layer)].get('output_sha256') != checks[0]['tensor_sha256']:
        raise ValueError('native/staged proof does not match the fresh source output')
    if layer == 0:
        _proof_status(proofs.get('sliding_window', {}), window=True)
    return manifest


@torch.inference_mode()
def capture_from_native(source, records, newroot, newidentity, oldroot):
    """Rerun selected original blocks; capture fresh H and natural expert rows."""
    from . import pilot
    root, oldroot = Path(newroot).expanduser().absolute(), Path(oldroot).expanduser().absolute()
    if root == oldroot or root.is_relative_to(oldroot) or oldroot.is_relative_to(root):
        raise ValueError('fresh pilot and native teacher trees must be disjoint')
    if root.is_symlink() or not root.is_dir() or _json(root / 'identity.json') != newidentity:
        raise ValueError('fresh pilot identity must be committed before native requalification')
    for relative in CAPTURE_CODE:
        if newidentity.get('code', {}).get(relative) != sha256_file(REPO / relative):
            raise ValueError('fresh native capture code differs from committed identity: ' + relative)
    origin = native_origin(oldroot, newidentity, REPO)
    if newidentity.get('native_reference_origin') != origin:
        raise ValueError('fresh pilot identity does not bind its audited native teacher')
    actual_records = {split: [dict(id=row['id'], tokens=len(row['input_ids']),
        ids_sha256=hashlib.sha256(canonical(row['input_ids']).encode()).hexdigest()) for row in records[split]]
        for split in ('calibration', 'heldout')}
    if actual_records != newidentity['records'] or set(records) != {'calibration', 'heldout'}:
        raise ValueError('fresh capture records differ from native teacher corpus identity')
    batches = [(split, row) for split in ('calibration', 'heldout') for row in records[split]]
    current_path = root / 'capture-current.json'
    original_current = _json(current_path) if current_path.exists() else None
    if original_current is not None and (original_current.get('identity') != newidentity or original_current.get('layer') not in LAYERS):
        raise ValueError('fresh native capture recovery identity differs')
    current = dict(identity=newidentity, layer=-1, directory='native-reference', proofs={})
    device = _device()
    rotary = None
    for layer in LAYERS:
        directory = root / f'frontier-{layer:03d}'
        completed = directory / 'manifest.json'
        if completed.exists():
            current = _check_completed(pilot, source, records, root, newidentity, oldroot, origin,
                layer, current['proofs'])
            if original_current is None or original_current['layer'] < layer:
                atomic_json(current_path, current)
            continue
        if original_current is not None and original_current['layer'] >= layer:
            raise ValueError('capture-current points beyond a missing fresh layer commitment')
        started = time.monotonic()
        pilot.preserve_partial(directory)
        if rotary is None:
            rotary = numerical.rotary(source, device=device)
        block = numerical.load_block(source, layer, device=device, include_experts=True)
        checks = []
        try:
            for split in ('calibration', 'heldout'):
                accumulator = pilot.Capture(block)  # Current K128 capture; no old H is opened.
                try:
                    for index, (batch_split, row) in enumerate(batches):
                        if batch_split != split:
                            continue
                        input_relative = f'{INPUT_BANK[layer]}/{index:03d}.safetensors'
                        native_input, _ = _teacher(oldroot, origin, input_relative)
                        hidden = native_input.to(device)
                        output = numerical.native_forward(block, dict(hidden=hidden), rotary)['hidden']
                        output_relative = f'frontier-{layer:03d}/{index:03d}.safetensors'
                        expected, entry = _teacher(oldroot, origin, output_relative)
                        _equal_native(output, expected)
                        proof = dict(index=index, split=split, id=row['id'], tokens=len(row['input_ids']),
                            teacher_relative=output_relative, teacher_sha256=entry['sha256'],
                            teacher_receipt_sha256=entry['receipt_sha256'], byte_exact=True,
                            tensor_sha256=hashlib.sha256(_bytes(output).numpy().tobytes()).hexdigest())
                        checks.append(proof)
                        path = directory / f'{index:03d}.safetensors'
                        receipt = pilot.save_tensor(path, dict(hidden=output), newidentity)
                        receipt['native_reference_output'] = proof
                        atomic_json(path.with_suffix('.receipt.json'), receipt)
                        del hidden, output, native_input, expected
                    accumulator.close()
                    accumulator.save(directory / split, newidentity)
                finally:
                    accumulator.close()
                del accumulator
            proofs = dict(current['proofs'])
            hidden, _ = _teacher(oldroot, origin, f'{INPUT_BANK[layer]}/000.safetensors')
            proofs[str(layer)] = pilot.byte_proof(block, hidden.to(device), rotary)
            _proof_status(proofs[str(layer)])
            if layer == 0:
                witness, _ = _teacher(oldroot, origin, 'frontier-embedding/witness.safetensors')
                proofs['sliding_window'] = pilot.window_proof(block, witness.to(device), rotary)
                _proof_status(proofs['sliding_window'], window=True)
                del witness
            current = dict(identity=newidentity, layer=layer, directory=directory.name, proofs=proofs,
                fresh_covariances=True, native_reference_origin=origin,
                native_output_checks=sorted(checks, key=lambda p: p['index']))
            atomic_json(completed, current)
            current = _check_completed(pilot, source, records, root, newidentity, oldroot, origin,
                layer, {} if layer == 0 else {key:value for key,value in proofs.items() if key != str(layer)})
            atomic_json(current_path, current)
            print(canonical(dict(event='pilot_native_requalified', layer=layer,
                seconds=time.monotonic() - started, old_native_identity_sha256=origin['old_native_identity_sha256'],
                fresh_covariances=True, source_outputs_byte_exact=True)), flush=True)
        finally:
            del block
            pilot.compact()
    return current
