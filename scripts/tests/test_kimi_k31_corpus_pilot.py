"""Native corpus identity and bounded short-pilot admission behavior."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'quantization'))
from kimi_k3 import prepare_corpus as corpus
from kimi_k31.k2_core_k6 import pilot


def source(tmp_path, model_type='kimi_k31', revision=corpus.K31_REVISION):
    root = tmp_path / revision
    root.mkdir()
    for name in corpus.SOURCE_FILES:
        (root / name).write_text(json.dumps(dict(model_type=model_type)) if name == 'config.json' else name)
    return root


def test_k31_source_requires_reviewed_revision(tmp_path):
    root = source(tmp_path)
    assert corpus.source_contract(root) == dict(repo_id=corpus.K31_REPO,
        model_type='kimi_k31', revision=corpus.K31_REVISION)
    wrong = source(tmp_path, revision='bad-revision')
    with pytest.raises(ValueError, match='pinned'):
        corpus.source_contract(wrong)


def test_k31_corpus_cannot_reuse_k3_identity(tmp_path):
    root = source(tmp_path)
    manifest = dict(source=dict(snapshot=str(root), **corpus.source_contract(root),
        files={name: corpus.sha256_file(root / name) for name in corpus.SOURCE_FILES}))
    tokens = tmp_path / 'tokens'
    tokens.mkdir()
    (tokens / 'manifest.json').write_text(json.dumps(manifest))
    assert corpus.require_source(tokens, root) == manifest
    manifest['source']['model_type'] = 'kimi_k3'
    (tokens / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='source/tokenizer'):
        corpus.require_source(tokens, root)


def test_k31_panel_never_downloads_missing_payload(tmp_path):
    with pytest.raises(ValueError, match='SparkNest'):
        corpus.download_corpus(tmp_path / 'absent', allow_download=False)


def test_pilot_prefixes_keep_disjoint_record_identity():
    cal = [dict(id='cal', input_ids=list(range(300)))]
    held = [dict(id='held', input_ids=list(range(200)))]
    left, right = pilot.select_records(cal, held)
    assert left == dict(id='cal', input_ids=list(range(128)))
    assert right == dict(id='held', input_ids=list(range(64)))
    assert len(cal[0]['input_ids']) == 300
    with pytest.raises(ValueError, match='overlap'):
        pilot.select_records(cal, cal)
    with pytest.raises(ValueError, match='positive'):
        pilot.select_records(cal, held, True)


def report():
    return dict(schema=pilot.SCHEMA, status='passed', layers=[0, 3],
        native_checks=dict.fromkeys(pilot.CHECKS, True), identity=dict(code={'run.py': 'abc'}),
        results=[dict(bits=bits, serialized_reload_byte_exact=True,
            weight_nmse=.001, heldout_output_nmse=.002) for bits in (6, 6, 2, 2, 2)])


def test_pilot_admits_exact_complete_evidence():
    value = report()
    assert pilot.validate_qualification(value) is value


@pytest.mark.parametrize('mutation', ('missing-check', 'nonfinite', 'bad-reload', 'missing-projection'))
def test_pilot_rejects_incomplete_evidence(mutation):
    value = report()
    if mutation == 'missing-check':
        del value['native_checks'][pilot.CHECKS[0]]
    elif mutation == 'nonfinite':
        value['results'][0]['heldout_output_nmse'] = float('nan')
    elif mutation == 'bad-reload':
        value['results'][2]['serialized_reload_byte_exact'] = False
    else:
        value['results'].pop()
    with pytest.raises(ValueError):
        pilot.validate_qualification(value)
