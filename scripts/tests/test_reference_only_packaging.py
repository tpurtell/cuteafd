"""Reference-only publication storage and privacy contracts, without hardware."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('reference_packaging', ROOT / 'scripts/bench/package-reference-only-fidelity.py')
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


def test_policy_tripwires_do_not_newly_fail_current_cards():
    import json
    import re
    policy = packager.POLICY_EXPECT
    assert policy == dict(top1_min=.90, kl_max=.06,
                          tripwires=dict(confident_top1_min=.96, top3_min=.97,
                                         confident_drop_margin=.01, top3_drop_margin=.005))
    cards = set(re.findall(r'benchmarks/[^\s\"<>)]*/card.svg', (ROOT / 'README.md').read_text()))
    assert cards
    reports = {str(Path(name).with_name('report.json')) for name in cards}
    checked = 0
    for name in reports:
        report = json.loads((ROOT / name).read_text())
        for check in report.get('baseline', {}).get('quality', {}).get('checks', []):
            if check.get('id') != 'fidelity':
                continue
            metrics = check['metrics']
            generated = metrics.get('verdict', {}).get('generated', metrics)
            if generated.get('confident_top1') is None or generated.get('top3_contained') is None:
                continue
            assert generated['confident_top1'] >= policy['tripwires']['confident_top1_min'], name
            assert generated['top3_contained'] >= policy['tripwires']['top3_min'], name
            checked += 1
    assert checked == len(reports)


def test_golden_runtime_producer_passes_packager_validator(tmp_path, monkeypatch):
    import base64
    import hashlib
    import importlib.metadata
    import json
    import sys
    from types import SimpleNamespace
    from fidelity_windows import collect_runtime_platform, finish_golden

    files = {'tokenizers/fake.so': b'native', 'tokenizers-1.dist-info/WHEEL': b'Wheel-Version: 1.0\n'}
    rows = []
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        encoded = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip('=')
        rows.append(f'{name},sha256={encoded},{len(content)}\n')
    record = ''.join(rows)
    metadata = {'RECORD': record}
    dist = SimpleNamespace(read_text=lambda name: metadata.get(name), locate_file=lambda name: tmp_path / name)
    monkeypatch.setattr(importlib.metadata, 'distribution', lambda name: dist)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda _: (12, 1))))
    monkeypatch.setenv('CUTEAFD_REFERENCE_IMAGE', 'reference:test')
    monkeypatch.setenv('CUTEAFD_REFERENCE_IMAGE_SHA256', 'sha256:' + 'a' * 64)
    runtime = collect_runtime_platform()
    assert packager.validate_runtime_platform(runtime) == runtime
    assert 'tokenizers_wheel_archive_sha256' not in runtime
    assert 'tokenizers_wheel_sha256' not in runtime
    metadata['direct_url.json'] = json.dumps({'url': 'https://example.org/tokenizers.whl',
                                            'archive_info': {'hashes': {'sha256': 'b' * 64}}})
    runtime = collect_runtime_platform()
    assert runtime['tokenizers_wheel_archive_sha256'] == 'b' * 64
    assert packager.validate_runtime_platform(runtime) == runtime
    out = tmp_path / 'golden'
    out.mkdir()
    finish_golden(out, dict(set_sha256='test', family='test', checkpoint='test'), [])
    assert packager.validate_runtime_platform(json.loads((tmp_path / 'platform.json').read_text())) == runtime
    assert (out / 'platform.json').read_bytes() == (tmp_path / 'platform.json').read_bytes()
    with pytest.raises(ValueError, match='schema'):
        packager.validate_runtime_platform({**runtime, 'tokenizers_wheel_sha256': 'c' * 64})
    (tmp_path / 'tokenizers/fake.so').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='differs from RECORD'):
        collect_runtime_platform()


def test_top1024_storage_uses_stable_f16_ties_and_normalized_target():
    rows = np.full((2, 1100), -8., dtype='<f2')
    rows[:, 1099] = -1.
    window = dict(tokens=[0, 1099, 1098], roles=['ctx', 'gen', 'ctx'], score_from=1)
    result = packager.compact_tensors(rows, window)
    assert result['top_ids'][0].tolist() == [1099] + list(range(1023))
    assert result['roles'].tolist() == [1, 0]
    assert result['positions'].tolist() == [1, 2]
    assert result['next_token_ids'].tolist() == [1099, 1098]
    for i, target in enumerate([1099, 1098]):
        raw = rows[i].astype(np.float64)
        expected = raw[target] - np.log(np.exp(raw).sum())
        assert result['next_token_log_prob'][i] == np.float32(expected)
        tail = np.delete(raw, result['top_ids'][i])
        assert result['tail_log_mass'][i] == np.float32(np.log(np.exp(tail).sum()))
        assert np.array_equal(result['top_log_probs'][i], rows[i, result['top_ids'][i]])
    rows[0, 0] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        packager.compact_tensors(rows, window)


def test_public_panel_preserves_tokens_and_generation_identity_not_deployment():
    source = dict(set_sha256='sealed', windows=[dict(tokens=[1, 2], roles=['ctx', 'gen'], score_from=1)],
                  generation_server=dict(build=dict(commit='a' * 40), model='Qwen/model', family='qwen4',
                                         settings=[dict(name='snapshot', value='/home/tj/private')]))
    public = packager.public_panel(source)
    assert public['windows'] == source['windows']
    assert public['generation_server'] == dict(build=dict(commit='a' * 40), model='Qwen/model', family='qwen4')
    assert 'settings' in source['generation_server']
    assert public['publication_provenance']['source_set_sha256'] == 'sealed'
    source['windows'][0]['provenance'] = 'ghp_' + 'a' * 30
    with pytest.raises(ValueError, match='credential'):
        packager.public_panel(source)


def test_seal_paths_cannot_escape(tmp_path):
    for value in ('../secret', '/secret', '.'):
        with pytest.raises(ValueError):
            packager.contained(tmp_path, value)
    (tmp_path / 'link').symlink_to('/etc/passwd')
    with pytest.raises(ValueError):
        packager.contained(tmp_path, 'link')

def test_retokenized_panel_caveat_discloses_provisional_legacy_extension():
    panel = dict(windows=[dict(tokenizer_length_extension=dict(reason="extended for tokenizer length"))],
                 retokenized_text_source=dict(caveat="not Step-5's own greedy path; model-generated-span property does not hold."))
    caveat = packager.panel_caveat(panel)
    assert 'extended for tokenizer length' in caveat
    assert 'not byte-comparable' in caveat
    assert 'model-generated-span property does not hold' in caveat
    assert 'not a Step-5-generated panel' in caveat
    assert '#302' in caveat


def test_own_model_text_has_no_borrowed_panel_caveat():
    panel = dict(checkpoint='official', generation_arm=dict(checkpoint='official', text_source='hf'), windows=[])
    assert packager.own_text(panel)
    assert packager.panel_caveat(panel) == ''
    panel['generation_arm']['text_source'] = 'api'
    assert packager.panel_caveat(panel) == ''
    panel['generation_arm']['checkpoint'] = 'borrowed'
    assert not packager.own_text(panel)
    assert '#302' in packager.panel_caveat(panel)


def test_text_checkpoint_guard_accepts_only_same_model_or_declared_higher_fidelity():
    from fidelity_windows import validate_text_source
    panel = dict(checkpoint='quant', generation_arm=dict(checkpoint='original'))
    with pytest.raises(ValueError, match='borrowed window text'):
        validate_text_source(panel)
    panel['text_source_declaration'] = dict(scored_checkpoint='quant', text_checkpoint='original',
        relationship='higher_fidelity_same_model', source_kind='original_checkpoint', evidence='vendor base_model mapping')
    validate_text_source(panel)
    assert packager.own_text(panel)
    panel['text_source_declaration']['scored_checkpoint'] = 'other'
    with pytest.raises(ValueError, match='borrowed window text'):
        validate_text_source(panel)


def test_sparse_launch_disclosure_preserves_tiles_and_rejects_missing_parameters():
    launch = dict(official_launch_configuration=False, num_stages=1, block_regions=16,
                  kernel='_sparse_attn_prefill_kernel')
    note = packager.sparse_launch_note(dict(sparse_attention_launch=launch))
    assert 'not the official' in note and 'num_stages=1' in note
    assert 'tiles and region-loop order remain unchanged' in note
    assert packager.sparse_launch_note({}) == ''
    for key, wrong in (('official_launch_configuration', True), ('num_stages', 3), ('block_regions', 8)):
        with pytest.raises(ValueError, match='unsupported'):
            packager.sparse_launch_note(dict(sparse_attention_launch={**launch, key: wrong}))


@pytest.mark.parametrize('decode_gen,prefill_gen', [(8306, 7065), (16000, 1000)])
def test_standard_generated_rows_are_informational_for_every_panel(decode_gen, prefill_gen):
    import re
    source = (ROOT / 'rust/crates/cuteafd-bench/src/fidelity_dataset.rs').read_text()
    windows = []
    for half, total in [('DECODE', decode_gen), ('PREFILL', prefill_gen)]:
        ids = re.findall(r'"([a-e][0-9]{2}|legacy)"', re.search(
            r'pub const STANDARD_' + half + r':[^=]+=\s*\[(.*?)\];', source, re.S).group(1))
        for index, wid in enumerate(ids):
            count = min(512, max(0, total - index * 512))
            windows.append(dict(id=wid, roles=['gen'] + ['ctx'] * (512 - count) + ['gen'] * count,
                                score_from=1))
    result = packager.standard_generated_rows(dict(windows=windows))
    assert result == dict(tier='standard-v2', decode_generated_positions=decode_gen,
                          prefill_generated_positions=prefill_gen,
                          max_min_ratio=max(decode_gen, prefill_gen) / min(decode_gen, prefill_gen),
                          informational_only=True)
    note = packager.standard_generated_note(result)
    assert f'decode {decode_gen:,}; prefill {prefill_gen:,}' in note
    assert 'Informational only' in note and 'not resampled' in note
    assert 'per-position rates' in note


def test_dense_root_wording_is_unchanged_and_v41_root_comes_from_the_manifest():
    # Dense/Step-5 goldens record no root arithmetic: their cards keep the original text byte for byte.
    assert packager.root_description({}, {'reference_snapshot': {'precision': 'official BF16'}}) == (
        'BF16 root', 'Checkpoint-native BF16/F32 tensors and family activation arithmetic, bounded attention, '
        'fixed GEMM rows 128, MTP/cache off.', 'Transformers source')
    # V4.1 has no BF16 master: the root precision is the set manifest's, the arithmetic the golden's own.
    meta = {'root_arithmetic': {'arithmetic': 'Official inference/model.py block by block.', 'source': 'official inference/model.py'}}
    panel = {'reference_snapshot': {'precision': 'official FP8 block dense + MXFP4 experts (no vendor BF16 master)'}}
    precision, arithmetic, source = packager.root_description(meta, panel)
    assert precision == 'root (official FP8 block dense + MXFP4 experts (no vendor BF16 master))'
    assert (arithmetic, source) == ('Official inference/model.py block by block.', 'official inference/model.py')


def test_sealed_f32_arithmetic_supplies_precision_without_mutating_panel():
    import copy
    panel = {'reference_snapshot': {}}
    original = copy.deepcopy(panel)
    meta = {'root_arithmetic': copy.deepcopy(packager.KIMI_F32_ROOT)}
    precision, arithmetic, source = packager.root_description(meta, panel)
    assert precision == 'root (native MXFP4 experts and shipped BF16/F32 dense tensors, computed in F32)'
    assert arithmetic == packager.KIMI_F32_ROOT['arithmetic']
    assert source == packager.KIMI_F32_ROOT['source']
    assert panel == original
    for key in ('arithmetic', 'source'):
        changed = copy.deepcopy(meta)
        changed['root_arithmetic'][key] += ' changed'
        with pytest.raises(ValueError, match='missing reference precision'):
            packager.root_description(changed, panel)
    with pytest.raises(ValueError, match='missing reference precision'):
        packager.root_description({'root_arithmetic': {'arithmetic': 'unknown', 'source': 'unknown'}}, panel)


def test_reference_role_metrics_and_card_explain_aggregate_nll():
    validation = dict(nll=3.4, scored_tokens=10, nll_gen=.2, nll_ctx=8.2,
                      gen_top1=.95, scored_tokens_gen=6, scored_tokens_ctx=4)
    metrics = packager.reference_role_metrics(validation)
    assert metrics == {key: value for key, value in validation.items() if key not in ('nll', 'scored_tokens')}
    note = packager.reference_role_note(metrics)
    assert 'Generated-row NLL 0.200000000' in note
    assert 'context-row NLL 8.200000000' in note
    assert '95.0000%' in note
    assert 'Aggregate NLL includes trailing-context rows' in note
    assert 'engine and reference score the same tokens' in note
    with pytest.raises(ValueError, match='aggregate NLL'):
        packager.reference_role_metrics({**validation, 'nll': 4})
    with pytest.raises(ValueError, match='counts mismatch'):
        packager.reference_role_metrics({**validation, 'scored_tokens_gen': 7})
    with pytest.raises(ValueError, match='generated top-1'):
        packager.reference_role_metrics({**validation, 'gen_top1': 1.1})


def test_reference_role_metrics_without_generated_rows():
    metrics = packager.reference_role_metrics(dict(nll=.5, scored_tokens=10,
        nll_gen=None, nll_ctx=.5, gen_top1=None, scored_tokens_gen=0, scored_tokens_ctx=10))
    assert 'n/a (no rows)' in packager.reference_role_note(metrics)
