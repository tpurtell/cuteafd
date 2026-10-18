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
    reports = set(re.findall(r'benchmarks/[^)\s]+/report.svg', (ROOT / 'README.md').read_text()))
    assert reports
    checked = 0
    for name in reports:
        report = json.loads((ROOT / name.replace('report.svg', 'report.json')).read_text())
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
