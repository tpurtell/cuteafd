#!/usr/bin/env python3
"""Stage a sealed oracle as top-1024 reference-only data; never upload or calibrate."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
from safetensors.numpy import save_file

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'python/reference'))
from fidelity_windows import canonical, validate_public_metadata, validate_public_text, validate_text_source
from checkpoint_tokenizer import load_tokenizer, tokenizer_identity

QUICK = ['legacy', 'a00', 'a04', 'a08', 'a20', 'c00', 'd00', 'e00']
POLICY_EXPECT = dict(top1_min=.90, kl_max=.06,
                     tripwires=dict(confident_top1_min=.96, top3_min=.97,
                                    confident_drop_margin=.01, top3_drop_margin=.005))
CAVEAT = ('The source shared text panel contains repeated unrelated source-text splices in 33 scored suffixes; '
          'see cuteafd issue #302. These are teacher-forced reference rows, '
          'not evidence of coherent free-running generation. NLL is not a cross-model quality ranking.')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def contained(root, relative):
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(p in ('.', '..') for p in path.parts):
        raise ValueError('unsafe input path')
    result = root / path
    if not result.resolve().is_relative_to(root.resolve()) or result.is_symlink():
        raise ValueError('input escapes sealed root')
    return result


def compact_tensors(full, window):
    """Match existing publication storage, including F16 boundary-tie ordering."""
    n, vocab = full.shape
    if vocab <= 1024 or n != len(window['tokens']) - window['score_from']:
        raise ValueError('invalid full-row geometry')
    ids = np.empty((n, 1024), dtype='<u4')
    probabilities = np.empty((n, 1024), dtype='<f2')
    tails, next_lps = np.empty(n, dtype='<f4'), np.empty(n, dtype='<f4')
    for i in range(n):
        raw = full[i].astype(np.float64)
        if not np.isfinite(raw).all():
            raise ValueError('non-finite oracle row')
        threshold = np.partition(raw, vocab - 1024)[-1024]
        above = np.flatnonzero(raw > threshold)
        tied = np.flatnonzero(raw == threshold)[:1024 - len(above)]
        order = np.concatenate((above, tied))
        order = order[np.lexsort((order, -raw[order]))]
        mask = np.ones(vocab, dtype=bool)
        mask[order] = False
        tail = raw[mask]
        peak = np.max(tail)
        tails[i] = peak + np.log(np.exp(tail - peak).sum(dtype=np.float64))
        peak = np.max(raw)
        normalizer = peak + np.log(np.exp(raw - peak).sum(dtype=np.float64))
        target = window['tokens'][window['score_from'] + i]
        if not 0 <= target < vocab:
            raise ValueError('next token outside vocabulary')
        next_lps[i] = raw[target] - normalizer
        ids[i], probabilities[i] = order, full[i, order]
    start = window['score_from']
    return dict(positions=np.arange(start, start + n, dtype='<u4'),
                next_token_ids=np.array(window['tokens'][start:], dtype='<u4'),
                roles=np.array([r == 'gen' for r in window['roles'][start:]], dtype='u1'),
                top_ids=ids, top_log_probs=probabilities, tail_log_mass=tails,
                next_token_log_prob=next_lps)


def standard_generated_rows(panel):
    """Report natural row counts using the consumer's fixed split, never resample."""
    source = (REPO / 'rust/crates/cuteafd-bench/src/fidelity_dataset.rs').read_text()
    windows = {window['id']: window for window in panel['windows']}
    counts = {}
    for name in ('decode', 'prefill'):
        match = re.search(r'pub const STANDARD_' + name.upper() + r':[^=]+=\s*\[(.*?)\];', source, re.S)
        ids = re.findall(r'"([a-e][0-9]{2}|legacy)"', match.group(1))
        counts[name] = sum(window['roles'][window['score_from']:].count('gen')
                           for window in (windows[wid] for wid in ids))
    small, large = min(counts.values()), max(counts.values())
    return dict(tier='standard-v2', decode_generated_positions=counts['decode'],
                prefill_generated_positions=counts['prefill'],
                max_min_ratio=large / small if small else None, informational_only=True)


def standard_generated_note(counts):
    ratio = counts['max_min_ratio']
    ratio_text = f'{ratio:.6f}' if ratio is not None else 'undefined (zero generated rows)'
    return (f'Standard-v2 generated scored positions: decode {counts["decode_generated_positions"]:,}; '
            f'prefill {counts["prefill_generated_positions"]:,}; max/min ratio {ratio_text}. '
            'Informational only: fixed window IDs are not resampled; each half reports per-position rates.\n')


def reference_role_metrics(validation):
    counts = [validation['scored_tokens_' + role] for role in ('gen', 'ctx')]
    if (any(not isinstance(count, int) or count < 0 for count in counts)
            or sum(counts) != validation['scored_tokens']):
        raise ValueError('reference role counts mismatch')
    for role, count in zip(('gen', 'ctx'), counts):
        value = validation['nll_' + role]
        if (count and (value is None or not np.isfinite(value) or value < 0)
                or not count and value is not None):
            raise ValueError('reference role NLL mismatch')
    top1 = validation['gen_top1']
    if (counts[0] and (top1 is None or not np.isfinite(top1) or not 0 <= top1 <= 1)
            or not counts[0] and top1 is not None):
        raise ValueError('reference generated top-1 mismatch')
    weighted = sum((validation['nll_' + role] or 0) * count
                   for role, count in zip(('gen', 'ctx'), counts)) / sum(counts)
    if not np.isclose(weighted, validation['nll'], rtol=1e-9, atol=1e-8):
        raise ValueError('reference role aggregate NLL mismatch')
    return {key: validation[key] for key in
            ('nll_gen', 'nll_ctx', 'gen_top1', 'scored_tokens_gen', 'scored_tokens_ctx')}


def reference_role_note(metrics):
    def number(value):
        return f'{value:.9f}' if value is not None else 'n/a (no rows)'
    top1 = f'{100 * metrics["gen_top1"]:.4f}%' if metrics['gen_top1'] is not None else 'n/a (no rows)'
    return (f'Generated-row NLL {number(metrics["nll_gen"])} nat/token '
            f'({metrics["scored_tokens_gen"]:,} rows); context-row NLL {number(metrics["nll_ctx"])} nat/token '
            f'({metrics["scored_tokens_ctx"]:,} rows); reference top-1 agreement with generated text {top1}.\n'
            'Aggregate NLL includes trailing-context rows appended after short completions; '
            'these can be out-of-distribution and make aggregate NLL non-comparable across models. '
            'They remain valid teacher-forced fidelity probes: engine and reference score the same tokens.\n')


def own_text(panel):
    try:
        validate_text_source(panel)
    except ValueError:
        return False
    return True


def panel_caveat(panel):
    caveat = '' if own_text(panel) else CAVEAT
    if panel.get('retokenized_text_source'):
        caveat += ' ' + panel['retokenized_text_source']['caveat']
        caveat += ' Provisional chat windows were re-rendered with the target chat template; all windows were retokenized. This is not a Step-5-generated panel.'
    if any(w.get('tokenizer_length_extension') for w in panel['windows']):
        caveat += ' Legacy window: extended for tokenizer length. Window 0 is not byte-comparable with other families legacy windows.'
    return caveat


def sparse_launch_note(meta):
    launch = meta.get('sparse_attention_launch')
    if launch is None:
        return ''
    if (launch.get('official_launch_configuration') is not False or launch.get('num_stages') != 1
            or launch.get('block_regions') != 16 or launch.get('kernel') != '_sparse_attn_prefill_kernel'):
        raise ValueError('unsupported sparse reference launch configuration')
    return ('This is not the official sparse-attention launch configuration: the adapter sets num_stages=1; '
            'official region/head/dimension tiles and region-loop order remain unchanged.\n')


KIMI_F32_ROOT = {
    'arithmetic': 'Reference computed entirely in F32 from the native checkpoint: exact group-32 MXFP4 to F32; '
                  'shipped BF16/F32 dense tensors upcast to F32; F32 activations, block history, router, norms, '
                  'KDA recurrence and eager attention; TF32 disabled in Torch and Triton, highest matmul precision; '
                  'one expert matrix dequantized at a time from a per-layer packed GPU cache; fixed-M128 linears; '
                  'MTP/cache off.\n\nCheckpoint-format note: the release zero-pads native KDA A_log from 96 active heads '
                  'to 128. Each affected tensor is narrowed to the first 96 only after all 32 unused F32 values '
                  'are verified bit-exact positive zero. Per-layer shapes and tail SHA256 checks are recorded in '
                  'reference_provenance.gates.kda_a_log_zero_padding; all other tensor shapes remain strict.',
    'source': 'official checkpoint Python decoder; exact E2M1/E8M0 group-32 MXFP4 decoded directly to F32',
}
KIMI_F32_PRECISION = 'native MXFP4 experts and shipped BF16/F32 dense tensors, computed in F32'


def reference_precision(meta, panel):
    precision = panel.get('reference_snapshot', {}).get('precision')
    if isinstance(precision, str) and precision.strip():
        return precision
    if meta.get('root_arithmetic') == KIMI_F32_ROOT:
        return KIMI_F32_PRECISION
    raise ValueError('missing reference precision: unrecognized sealed root arithmetic')


def root_description(meta, panel):
    """Card wording for the logit root: (root label, arithmetic line, pinned
    source). A golden that records its reference arithmetic (V4.1: official
    FP8/MXFP4 root, no BF16 master) states the root precision from the set
    manifest's reference snapshot; dense BF16 goldens keep their wording."""
    root = meta.get('root_arithmetic')
    if root is None:
        return ('BF16 root', 'Checkpoint-native BF16/F32 tensors and family activation arithmetic, bounded attention, '
                'fixed GEMM rows 128, MTP/cache off.', 'Transformers source')
    return (f'root ({reference_precision(meta, panel)})', root['arithmetic'], root['source'])


def validate_runtime_platform(runtime):
    required = {'schema', 'image', 'image_sha256', 'compute_capability',
                'tokenizers_dist_sha256', 'tokenizers_hash_kind', 'dependency_install',
                'tokenizers_native_sha256', 'tokenizers_record_sha256'}
    optional = {'tokenizers_wheel_archive_sha256'}
    if (not required <= runtime.keys() or runtime.keys() - required - optional
            or runtime['schema'] != 'cuteafd.fidelity.platform/1'
            or runtime['tokenizers_hash_kind'] != 'sha256 of canonical RECORD-verified installed file inventory'):
        raise ValueError('invalid reference runtime schema')
    for key in required | optional:
        if key.endswith('_sha256') and key in runtime:
            value = runtime[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('invalid reference runtime digest')
    capability = runtime['compute_capability']
    if (not isinstance(capability, list) or len(capability) != 2
            or any(type(v) is not int or v < 0 for v in capability)
            or not isinstance(runtime['image'], str) or not runtime['image']
            or not isinstance(runtime['dependency_install'], str) or not runtime['dependency_install']):
        raise ValueError('invalid reference runtime identity')
    validate_public_metadata(runtime)
    return copy.deepcopy(runtime)


def public_panel(panel):
    public = copy.deepcopy(panel)
    # Keep immutable generation identity, not deployment arguments or local paths.
    server = public.get('generation_server', {})
    public['generation_server'] = {k: server[k] for k in ('build', 'family', 'model') if k in server}
    public['publication_provenance'] = {
        **public.get('publication_provenance', {}),
        'source_set_sha256': panel['set_sha256'],
        'identity': 'tokens/roles/score geometry unchanged; deployment settings omitted',
    }
    validate_public_metadata(public)
    return public


def package(source, out, config, tokenizer_path, license_path):
    if out.exists() or not config or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.' for c in config) or config in ('.', '..'):
        raise ValueError('output must be fresh and config must be a safe component')
    read = lambda name: json.loads(contained(source, name).read_text())
    meta, panel = read('golden/meta.json'), read('golden/windows.json')
    validate_text_source(panel)
    validation, seal, full_seal = read('reference-validation.json'), read('golden-seal.json'), read('full.sha256.json')
    rows = read('full/rows.json')
    set_id = panel['set_sha256']
    caveat = panel_caveat(panel)
    launch_note = sparse_launch_note(meta)
    if panel['family'] == 'step5' and not launch_note:
        raise ValueError('Step-5 reference must disclose its sparse launch configuration')
    if not (validation['passed'] is True and validation['engine_parity'] is False
            and validation['set_sha256'] == seal['set_sha256'] == full_seal['set_sha256'] == meta['set_sha256'] == rows['set_sha256'] == set_id
            and validation['golden_seal_sha256'] == digest(source / 'golden-seal.json')
            and validation['full_seal_sha256'] == seal['full_seal_sha256'] == digest(source / 'full.sha256.json')
            and validation['scored_tokens'] == meta['scored_tokens'] == 32768
            and np.isfinite(meta['nll']) and abs(validation['nll'] - meta['nll']) <= 1e-8
            and len(panel['windows']) == 64
            and meta['mtp'] is False and meta['use_cache'] is False and meta['attention'] == 'eager'):
        raise ValueError('unsealed or mismatched oracle evidence')
    inventory = {e['path']: e for e in seal['files']}
    for name in ('golden/meta.json', 'golden/windows.json', 'full/rows.json'):
        if digest(source / name) != inventory[name]['sha256']:
            raise ValueError('sealed metadata differs')
    proof = meta['prefix_qualification']
    if (proof.get('passed') is not True or proof.get('finite') is not True or proof.get('rows') != 512
            or proof.get('different_rows') != [] or proof.get('argmax_disagreements') != 0
            or proof.get('lengths') != [576, 640] or proof.get('score_from') != 64
            or proof.get('fixed_rows') != 128 or proof.get('set_sha256') != set_id
            or proof.get('snapshot_identity') != meta['snapshot_identity']):
        raise ValueError('invalid reference prefix qualification')
    for value in meta['source_pin'].values():
        if not isinstance(value, str) or len(value) not in (40, 64) or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('invalid source pin')
    if rows['dtype'] != '<f2' or rows['kind'] != 'log_softmax':
        raise ValueError('unsupported oracle rows')
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer_pin = tokenizer_identity(tokenizer_path.parent)
    if (tokenizer_pin['tokenizer_sha256'] != meta['snapshot_identity']['tokenizer_sha256']
            or tokenizer_pin.get('tokenizer_files') != meta['snapshot_identity'].get('tokenizer_files')):
        raise ValueError('reference tokenizer differs')
    for window in panel['windows']:
        if window.get('media') or len(window['tokens']) - window['score_from'] != 512:
            raise ValueError('only the sealed text panel is supported')
        text = (tokenizer.decode(window['tokens']) if tokenizer_pin.get('tokenizer_kind') else
                tokenizer.decode(window['tokens'], skip_special_tokens=False))
        validate_public_text(text, scored_text=True)
    precision, arithmetic, runtime_source = root_description(meta, panel)
    public = public_panel(panel)
    destination = out / config
    destination.mkdir(parents=True)
    (destination / 'windows.json').write_bytes(canonical(public) + b'\n')
    shutil.copyfile(license_path, destination / 'LICENSE')
    entries = {e['id']: e for e in rows['windows']}
    sealed_rows = {e['path']: e for e in full_seal['files']}
    if len(entries) != 64 or set(entries) != {w['id'] for w in panel['windows']}:
        raise ValueError('oracle coverage differs')
    files = []
    for window in panel['windows']:
        wid = window['id']
        if not wid or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in wid):
            raise ValueError('unsafe window id')
        entry = entries[wid]
        path = contained(source / 'full', entry['path'])
        if (entry['positions'] != list(range(window['score_from'], len(window['tokens'])))
                or entry['shape'] != [512, rows['vocab']] or path.stat().st_size != 512 * rows['vocab'] * 2
                or digest(path) != entry['sha256'] or entry['sha256'] != sealed_rows[entry['path']]['sha256']):
            raise ValueError('full oracle checksum or geometry differs')
        full = np.memmap(path, dtype='<f2', mode='r', shape=(512, rows['vocab']))
        tensors = compact_tensors(full, window)
        target = destination / f'{wid}.safetensors'
        save_file(tensors, str(target))
        files.append(dict(window=wid, path=target.name, bytes=target.stat().st_size,
                          sha256=digest(target), shape=[512, 1024], source_full_sha256=entry['sha256']))
        print(f'packaged {config}/{wid}', flush=True)
    provenance = {k: meta[k] for k in ('snapshot_identity', 'source_pin', 'platform', 'tokenizers', 'torch',
                                      'transformers', 'attention', 'mtp', 'use_cache', 'gates', 'native_fp32_parameters')}
    if meta.get('root_arithmetic') is not None:
        provenance['root_arithmetic'] = meta['root_arithmetic']
        provenance['reference_precision'] = reference_precision(meta, panel)
    if launch_note:
        provenance['sparse_attention_launch'] = meta['sparse_attention_launch']
    if digest(source / 'platform.json') != inventory['platform.json']['sha256']:
        raise ValueError('sealed runtime provenance differs')
    runtime = read('platform.json')
    provenance['runtime'] = validate_runtime_platform(runtime)
    generated_rows = standard_generated_rows(panel)
    role_metrics = reference_role_metrics(validation)
    qualification = dict(schema='cuteafd.fidelity.qualification/1', kind='reference', qualifies=True,
                         **role_metrics, standard_generated_rows=generated_rows,
                         engine_parity=False, qualification_scope='reference logits, storage and prefix invariance',
                         set_sha256=set_id, nll=meta['nll'], scored_tokens=meta['scored_tokens'],
                         prefix_qualification=meta['prefix_qualification'], reference_validation=validation,
                         reference_provenance=provenance, panel_caveat=caveat)
    validate_public_metadata(qualification)
    (destination / 'qualification.json').write_bytes(canonical(qualification) + b'\n')
    snapshot = meta['snapshot_identity']
    manifest = dict(schema='cuteafd.fidelity.dataset/1', kind='reference-top1024-plus-tail', config=config,
                    family=panel['family'], checkpoint=panel['checkpoint'], root_checkpoint=dict(id=snapshot['checkpoint'], revision=snapshot['snapshot_revision']),
                    set_sha256=set_id, vocab=rows['vocab'], top_k=1024,
                    expect=copy.deepcopy(POLICY_EXPECT),
                    expect_provenance='declared policy floors; independent of engine measurements',
                    windows_sha256=digest(destination / 'windows.json'), qualification_sha256=digest(destination / 'qualification.json'),
                    files=files, licence_files=[dict(path='LICENSE', sha256=digest(destination / 'LICENSE'))],
                    packaging_source_commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                    packaging_source_sha256=digest(Path(__file__)),
                    publication_status='REFERENCE',
                    text_checkpoint=panel['generation_arm']['checkpoint'], panel_caveat=caveat)
    if launch_note:
        manifest['reference_execution'] = dict(sparse_attention_launch=meta['sparse_attention_launch'])
    manifest['reference_sha256'] = hashlib.sha256(canonical(manifest)).hexdigest()
    validate_public_metadata(manifest)
    (destination / 'manifest.json').write_bytes(canonical(manifest) + b'\n')
    panel_description = ('Own-model text: fixed prompts, checkpoint chat template, and real fixture tool execution.\n'
                         if own_text(panel) else
                         'Provisional foreign-text panel: chat windows use the target chat template; all 64 windows were retokenized with character-span roles.\n'
                         if panel.get('retokenized_text_source') else
                         'All 64 windows, tokens, roles and score geometry are unchanged from the pinned shared panel.\n')
    text_description = (f'Text source: `{panel["generation_arm"].get("text_source", "local")}` from the scored model.\n'
                        if own_text(panel) else
                        f'Text was generated by `{panel["generation_arm"]["checkpoint"]}`, not by the reference root.\n')
    card = (f'# {config}\n\nNumerical fidelity reference dataset, not a training corpus or a general quality ranking.\n\n'
            f'Official {precision}: `{snapshot["checkpoint"]}` at `{snapshot["snapshot_revision"]}`.\n'
            f'{text_description}'
            f'{panel_description}'
            'The bench quick subset is legacy,a00,a04,a08,a20,c00,d00,e00 (4,096 rows).\n\n'
            '## Qualification\n\n'
            f'32,768 scored positions; reference NLL {meta["nll"]:.15f} nat/token.\n'
            f'{reference_role_note(role_metrics)}'
            f'{standard_generated_note(generated_rows)}'
            'The 576/640-token prefix probes have 512 byte-identical common F32 rows and zero argmax disagreements.\n'
            f'{arithmetic}\n'
            f'{launch_note}'
            f'Pinned {runtime_source}, tokenizer runtime, platform and oracle seal hashes are in qualification.json.\n'
            'No engine run is required to finish or publish this dataset.\n'
            'Declared policy floors: top-1 >=90%, KL <=0.06 nat, confident top-1 >=96%, top-3 >=97%.\n\n'
            '## Storage\n\n'
            'One safetensors file per window: U32 positions, next_token_ids and top_ids; U8 roles; F16 top_log_probs; '
            'F32 tail_log_mass and next_token_log_prob. Top-1024 selection uses the sealed full F16 rows with lower-ID boundary ties.\n'
            'Tail and target masses normalize the F16 oracle; no paired engine equivalence is asserted.\n\n'
            + (f'## Panel Caveat\n\n{caveat}\n\n' if caveat else '') +
            '## Provenance And Privacy\n\n'
            f'Source set: `{set_id}`. Raw deployment settings and local paths are omitted.\n'
            'Public text and metadata pass credential/private-key/personal-data screening. Original public MIT notice is preserved verbatim.\n'
            'Reference data is MIT licensed; the official checkpoint remains subject to its own upstream license.\n')
    validate_public_text(card, scored_text=True)
    (destination / 'README.md').write_text(card)
    # This is an index entry, not a replacement for the shared live configs.json.
    index = dict(name=config, path=f'{config}/manifest.json', sha256=digest(destination / 'manifest.json'))
    (out / 'config-entry.json').write_bytes(canonical(index) + b'\n')
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'out', 'tokenizer', 'license'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.source, args.out, args.config, args.tokenizer, args.license), indent=2))
    print('Staged only; no upload or engine qualification performed.')


if __name__ == '__main__':
    main()
